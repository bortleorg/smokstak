//! XISF ingestion.
//!
//! XISF is a container of a signature, an XML header, and the pixel data as a
//! byte range further down the same file. It is worth reading for one reason
//! above the others -- it is what other calibration pipelines write. A set
//! calibrated elsewhere may exist as `.xisf` and nothing else, and without this
//! reader the only way to start from its calibrated lights is to export the
//! whole set again.
//!
//! Almost none of the interpretation lives here. The container says how the
//! bytes are laid out and this reader turns them into the same 16-bit mosaic
//! the FITS reader produces; from there `fits::build_frame` does the rest, on
//! the same keywords, because XISF writers copy the original FITS cards into
//! the file as `FITSKeyword` elements. Two things do differ:
//!
//! * **Orientation is not in question.** XISF defines the first stored row as
//!   the top of the image, so there is no `ROWORDER` guess to make and no flip
//!   to apply. A file that came from a bottom-up FITS was flipped when it was
//!   written, and the mosaic pattern that flip invalidated is recovered from
//!   the pixels exactly as it is for FITS.
//! * **The full scale is declared.** A floating-point FITS file leaves the
//!   reader to infer whether it is normalised or in 16-bit units; XISF states
//!   it in `bounds`, so it is read rather than guessed.
//!
//! The plate solve is the one place where XISF-native metadata has to be read.
//! A solver writing XISF may not write `CTYPE`/`CD` cards: the solution goes
//! into `PCL:AstrometricSolution:*` properties instead, and a
//! master with no readable solve would give up the registration seeding that
//! [`sr_core::wcs`] exists for.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use quick_xml::events::{BytesStart, Event};
use quick_xml::reader::Reader;

use sr_core::frame::RawFrame;
use sr_core::plane::Plane;
use sr_core::wcs::Wcs;
use sr_core::{Result, SrError};

use crate::fits::{self, Header};

/// The eight bytes every XISF file starts with, followed by the header length
/// and four reserved bytes.
const SIGNATURE: &[u8; 8] = b"XISF0100";
const PREAMBLE: usize = 16;

/// A header larger than this is a malformed file rather than a rich one. Real
/// ones reach a megabyte when a processing history is written in.
const MAX_HEADER_BYTES: u64 = 64 << 20;

/// How samples are stored in the data block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

impl SampleFormat {
    fn parse(s: &str) -> Option<SampleFormat> {
        Some(match s.trim() {
            "UInt8" => SampleFormat::UInt8,
            "UInt16" => SampleFormat::UInt16,
            "UInt32" => SampleFormat::UInt32,
            "UInt64" => SampleFormat::UInt64,
            "Float32" => SampleFormat::Float32,
            "Float64" => SampleFormat::Float64,
            _ => return None,
        })
    }

    /// Bytes per sample.
    fn width(self) -> usize {
        match self {
            SampleFormat::UInt8 => 1,
            SampleFormat::UInt16 => 2,
            SampleFormat::UInt32 | SampleFormat::Float32 => 4,
            SampleFormat::UInt64 | SampleFormat::Float64 => 8,
        }
    }

    /// The range an unscaled sample of this format spans, which is what
    /// `bounds` defaults to when the file does not give one.
    fn default_bounds(self) -> (f64, f64) {
        match self {
            SampleFormat::UInt8 => (0.0, 255.0),
            SampleFormat::UInt16 => (0.0, 65535.0),
            SampleFormat::UInt32 => (0.0, 4294967295.0),
            SampleFormat::UInt64 => (0.0, 18446744073709551615.0),
            SampleFormat::Float32 | SampleFormat::Float64 => (0.0, 1.0),
        }
    }
}

/// Where a block's bytes are, as the `location` attribute spells it.
#[derive(Clone, Debug)]
enum Location {
    /// A byte range elsewhere in the same file.
    Attachment { position: u64, size: u64 },
    /// The element's own text, or a child `Data` element's, in the named
    /// encoding.
    Inline { encoding: String },
}

/// The `compression` attribute: a codec, the size to expect back, and the item
/// size when the bytes were shuffled before being compressed.
#[derive(Clone, Debug)]
struct Compression {
    codec: String,
    uncompressed: usize,
    shuffle_item: Option<usize>,
}

/// One `<Image>` element, without its pixels.
#[derive(Clone, Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub format: SampleFormat,
    /// The values the stored range maps onto.
    pub bounds: (f64, f64),
    /// `Gray`, `RGB` or `CIELab`, as the file names it.
    pub colour_space: String,
    /// `Light`, `MasterLight`, `Flat` and so on, when the file says.
    pub image_type: String,
    /// Whether channels are stored one after another rather than interleaved.
    pub planar: bool,
    /// The original FITS cards, preserved by the writer, in file order.
    pub header: Header,
    /// The plate solve, from the FITS cards or from the XISF astrometric
    /// properties, whichever is there.
    pub wcs: Option<Wcs>,
    little_endian: bool,
    location: Location,
    compression: Option<Compression>,
    /// The block's bytes, when it was stored in the header rather than
    /// attached.
    inline: Vec<u8>,
}

impl Image {
    /// Samples in the whole block, across every channel.
    fn samples(&self) -> usize {
        self.width * self.height * self.channels
    }
}

/// Read the XML header and the first image it describes, without touching the
/// pixels.
pub fn read_header(path: &Path) -> Result<Image> {
    let xml = read_header_text(path)?;
    parse_header(&xml, &path.display().to_string())
}

/// The filter this file was taken through, for sorting a set without decoding
/// it.
pub fn peek_filter(path: &Path) -> Option<String> {
    read_header(path).ok().and_then(|i| {
        i.header
            .any_text(&["FILTER", "FILTER1", "FILTNAME"])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    })
}

/// Read every channel, normalised to `[0, 1]` by the file's declared bounds.
///
/// This is the way in for an image that is already colour -- a master, or a
/// registered frame -- which cannot become a [`RawFrame`] because there is no
/// mosaic left in it to reconstruct from.
pub fn read_planes(path: &Path) -> Result<(Image, Vec<Plane<f32>>)> {
    let image = read_header(path)?;
    let values = read_values(path, &image)?;
    let (w, h, n) = (image.width, image.height, image.width * image.height);

    let mut planes = Vec::with_capacity(image.channels);
    for c in 0..image.channels {
        let mut p = Plane::<f32>::new(w, h);
        if image.planar {
            p.data.copy_from_slice(&values[c * n..(c + 1) * n]);
        } else {
            for i in 0..n {
                p.data[i] = values[i * image.channels + c];
            }
        }
        planes.push(p);
    }
    Ok((image, planes))
}

/// Decode one single-channel XISF file into a mosaiced frame.
pub fn decode(path: &Path) -> Result<RawFrame> {
    let file = path.display().to_string();
    let image = read_header(path)?;
    if image.channels != 1 {
        return Err(SrError::Input(format!(
            "{file}: {} channels in colour space {}; this is an already-separated colour image, \
             and the reconstruction needs the undemosaiced mosaic",
            image.channels, image.colour_space
        )));
    }
    let (w, h) = (image.width, image.height);
    let values = read_values(path, &image)?;

    // Crop to even dimensions so the 2x2 mosaic tiles exactly, as elsewhere.
    // No row flip: XISF fixes the first stored row as the top of the image.
    let (cw, ch) = (w & !1, h & !1);
    let mut data = vec![0u16; cw * ch];
    for y in 0..ch {
        for x in 0..cw {
            let v = values[y * w + x] * 65535.0;
            data[y * cw + x] = if v.is_finite() {
                v.clamp(0.0, 65535.0) as u16
            } else {
                0
            };
        }
    }
    drop(values);

    let wcs = image.wcs;
    fits::build_frame(
        path,
        &image.header,
        fits::Mosaic {
            data,
            width: cw,
            height: ch,
            full_width: w,
            full_height: h,
            wcs,
        },
    )
}

/// The header text, checked against the signature and its declared length.
fn read_header_text(path: &Path) -> Result<String> {
    let file = path.display().to_string();
    let mut r = BufReader::new(File::open(path)?);
    let mut preamble = [0u8; PREAMBLE];
    r.read_exact(&mut preamble)
        .map_err(|e| SrError::Input(format!("{file}: too short to be an XISF file: {e}")))?;
    if &preamble[..8] != SIGNATURE {
        return Err(SrError::Input(format!(
            "{file}: does not start with the XISF signature; monolithic XISF 1.0 is the only form \
             handled"
        )));
    }
    let length = u32::from_le_bytes([preamble[8], preamble[9], preamble[10], preamble[11]]) as u64;
    if length == 0 || length > MAX_HEADER_BYTES {
        return Err(SrError::Input(format!(
            "{file}: declares a {length}-byte XML header, which is not a length a real file has"
        )));
    }
    let mut bytes = vec![0u8; length as usize];
    r.read_exact(&mut bytes).map_err(|e| {
        SrError::Input(format!(
            "{file}: the XML header is shorter than the {length} bytes declared: {e}"
        ))
    })?;
    String::from_utf8(bytes)
        .map_err(|e| SrError::Input(format!("{file}: the XML header is not valid UTF-8: {e}")))
}

/// One property, reduced to what this reader asks of it.
#[derive(Clone, Debug, Default)]
struct Property {
    /// The value of a scalar, as text.
    value: String,
    /// The raw little-endian bytes of a vector or a matrix.
    data: Vec<u8>,
}

/// Pull the first `<Image>` out of the XML, with the properties describing it.
///
/// Two texts can be open at once, and they are kept apart deliberately. An
/// image small enough to be stored in the header carries its bytes as its own
/// text, which arrives in pieces between its children -- and one of those
/// children is a property with a text of its own. Collecting both into one
/// buffer would lose the image.
fn parse_header(xml: &str, file: &str) -> Result<Image> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);

    let mut image: Option<PartialImage> = None;
    // Only the first image is read, and only its own children belong to it.
    let mut done = false;
    // The image's own text, collected while its element is open.
    let mut in_image_data = false;
    let mut image_text = String::new();
    // The property currently open, as its id and where its value is kept.
    let mut property: Option<(String, String)> = None;
    let mut text = String::new();

    loop {
        let event = reader
            .read_event()
            .map_err(|e| SrError::Input(format!("{file}: malformed XML header: {e}")))?;
        let (start, empty) = match &event {
            Event::Start(e) => (Some(e.clone()), false),
            Event::Empty(e) => (Some(e.clone()), true),
            _ => (None, false),
        };

        if let Some(e) = start {
            let name = String::from_utf8_lossy(e.local_name().as_ref()).to_string();
            let attrs = attributes(&e, file)?;
            match name.as_str() {
                "Image" if image.is_none() && !done => {
                    let partial = PartialImage::new(attrs, file)?;
                    let inline = matches!(partial.location, Location::Inline { .. });
                    image = Some(partial);
                    // An inline block is the element's own text, which arrives
                    // in pieces between the children.
                    if inline && !empty {
                        in_image_data = true;
                        image_text.clear();
                    }
                }
                "Data" if image.is_some() && !done && !empty => {
                    in_image_data = true;
                    image_text.clear();
                }
                "FITSKeyword" if !done => {
                    if let Some(img) = image.as_mut() {
                        let key = attrs.get("name").cloned().unwrap_or_default();
                        let value = attrs.get("value").cloned().unwrap_or_default();
                        if !key.trim().is_empty() {
                            img.header.cards.push((key.trim().to_string(), value));
                        }
                    }
                }
                "Property" if !done => {
                    if let Some(img) = image.as_mut() {
                        let id = attrs.get("id").cloned().unwrap_or_default();
                        match attrs.get("value") {
                            // A scalar states its value as an attribute.
                            Some(v) => img.properties.push((
                                id,
                                Property {
                                    value: v.clone(),
                                    data: Vec::new(),
                                },
                            )),
                            // Everything else carries it as element text: a
                            // string as itself, a vector or matrix encoded.
                            None if !empty => {
                                property =
                                    Some((id, attrs.get("location").cloned().unwrap_or_default()));
                                text.clear();
                            }
                            None => img.properties.push((id, Property::default())),
                        }
                    }
                }
                _ => {}
            }
            continue;
        }

        let chunk = match &event {
            Event::Text(t) => Some(t.unescape().unwrap_or_default().to_string()),
            Event::CData(t) => Some(String::from_utf8_lossy(t.as_ref()).to_string()),
            _ => None,
        };
        if let Some(c) = chunk {
            // A property's text belongs to the property; anything else inside
            // an inline image belongs to the image.
            if property.is_some() {
                text.push_str(&c);
            } else if in_image_data {
                image_text.push_str(&c);
            }
            continue;
        }

        match event {
            Event::Eof => break,
            Event::End(e) => {
                let name = local_name(&e);
                if name == "Property"
                    && let (Some((id, location)), Some(img)) = (property.take(), image.as_mut())
                {
                    let data = match location.strip_prefix("inline:") {
                        Some(encoding) => decode_text_block(encoding, &text),
                        None => Vec::new(),
                    };
                    img.properties.push((
                        id,
                        Property {
                            value: text.trim().to_string(),
                            data,
                        },
                    ));
                }
                if in_image_data && (name == "Image" || name == "Data") {
                    in_image_data = false;
                    if let Some(img) = image.as_mut()
                        && let Location::Inline { encoding } = &img.location
                    {
                        img.inline = decode_text_block(encoding, &image_text);
                    }
                }
                if name == "Image" && image.is_some() {
                    done = true;
                }
            }
            _ => {}
        }
    }

    let partial = image
        .ok_or_else(|| SrError::Input(format!("{file}: the XISF header describes no image")))?;
    partial.finish()
}

/// An `<Image>` as it is being filled in from its children.
struct PartialImage {
    width: usize,
    height: usize,
    channels: usize,
    format: SampleFormat,
    bounds: (f64, f64),
    colour_space: String,
    image_type: String,
    planar: bool,
    little_endian: bool,
    location: Location,
    compression: Option<Compression>,
    inline: Vec<u8>,
    header: Header,
    properties: Vec<(String, Property)>,
}

impl PartialImage {
    fn new(attrs: Attributes, file: &str) -> Result<PartialImage> {
        let geometry = attrs
            .get("geometry")
            .ok_or_else(|| SrError::Input(format!("{file}: the image has no geometry")))?;
        let dims: Vec<usize> = geometry
            .split(':')
            .map(|p| p.trim().parse::<usize>().unwrap_or(0))
            .collect();
        if dims.len() != 3 {
            return Err(SrError::Input(format!(
                "{file}: geometry {geometry:?} is not width:height:channels; only 2-D images are \
                 handled"
            )));
        }
        let (width, height, channels) = (dims[0], dims[1], dims[2]);
        if width < 4 || height < 4 {
            return Err(SrError::Input(format!(
                "{file}: implausible image size {width}x{height}"
            )));
        }
        if channels == 0 {
            return Err(SrError::Input(format!("{file}: the image has no channels")));
        }

        let format_name = attrs.get("sampleFormat").map(String::as_str).unwrap_or("");
        let format = SampleFormat::parse(format_name).ok_or_else(|| {
            SrError::Input(format!(
                "{file}: sampleFormat {format_name:?} is not a real-valued XISF sample format"
            ))
        })?;

        let bounds = match attrs.get("bounds") {
            Some(b) => {
                let v: Vec<f64> = b
                    .split(':')
                    .filter_map(|p| p.trim().parse::<f64>().ok())
                    .collect();
                if v.len() != 2 || v[1] <= v[0] {
                    return Err(SrError::Input(format!(
                        "{file}: bounds {b:?} do not name a rising range"
                    )));
                }
                (v[0], v[1])
            }
            None => format.default_bounds(),
        };

        Ok(PartialImage {
            width,
            height,
            channels,
            format,
            bounds,
            colour_space: attrs
                .get("colorSpace")
                .cloned()
                .unwrap_or_else(|| "Gray".into()),
            image_type: attrs.get("imageType").cloned().unwrap_or_default(),
            // Planar is the default, and what writers normally use.
            planar: !attrs
                .get("pixelStorage")
                .map(|s| s.eq_ignore_ascii_case("Normal"))
                .unwrap_or(false),
            little_endian: !attrs
                .get("byteOrder")
                .map(|s| s.eq_ignore_ascii_case("big"))
                .unwrap_or(false),
            location: parse_location(attrs.get("location").map(String::as_str), file)?,
            compression: parse_compression(attrs.get("compression").map(String::as_str), file)?,
            inline: Vec::new(),
            header: Header::default(),
            properties: Vec::new(),
        })
    }

    fn finish(mut self) -> Result<Image> {
        // Native XISF timestamps join preserved FITS cards in the same metadata
        // path; an explicit DATE-OBS always wins.
        if self.header.get("DATE-OBS").is_none()
            && let Some((_, p)) = self
                .properties
                .iter()
                .find(|(id, _)| id == "Observation:Time:Start")
            && !p.value.trim().is_empty()
        {
            self.header
                .cards
                .push(("DATE-OBS".into(), format!("'{}'", p.value.trim())));
        }
        // The FITS cards first: a file converted from FITS keeps a solve that
        // is already in this reader's terms. The XISF astrometric properties
        // are the fallback, for a solve written only there.
        let wcs =
            fits::plate_solve(&self.header).or_else(|| astrometric_solution(&self.properties));
        Ok(Image {
            width: self.width,
            height: self.height,
            channels: self.channels,
            format: self.format,
            bounds: self.bounds,
            colour_space: self.colour_space,
            image_type: self.image_type,
            planar: self.planar,
            header: self.header,
            wcs,
            little_endian: self.little_endian,
            location: self.location,
            compression: self.compression,
            inline: self.inline,
        })
    }
}

/// The plate solve stored as XISF `PCL:AstrometricSolution:*` properties.
///
/// Only the linear part of a gnomonic solution is read. A spline distortion
/// model may sit beside it in further properties; it is a sub-pixel correction,
/// and the same one this crate declines to read from FITS SIP coefficients.
///
/// One convention has to be crossed. These properties measure image coordinates from
/// the corner of the first pixel, so the centre of pixel zero is at 0.5 and its
/// reference point is half a pixel further along each axis than [`Wcs`] wants.
/// Its matrix is stored by rows, and refers to the array as stored -- which for
/// XISF is the image the right way up, so no flip enters.
fn astrometric_solution(properties: &[(String, Property)]) -> Option<Wcs> {
    let find = |id: &str| properties.iter().find(|(k, _)| k == id).map(|(_, v)| v);
    let projection = find("PCL:AstrometricSolution:ProjectionSystem")?;
    if !projection.value.trim().eq_ignore_ascii_case("Gnomonic") {
        return None;
    }
    let m = doubles(find("PCL:AstrometricSolution:LinearTransformationMatrix")?);
    let image = doubles(find("PCL:AstrometricSolution:ReferenceImageCoordinates")?);
    let sky = doubles(find(
        "PCL:AstrometricSolution:ReferenceCelestialCoordinates",
    )?);
    if m.len() != 4 || image.len() != 2 || sky.len() != 2 {
        return None;
    }
    let w = Wcs {
        crpix: (image[0] - 0.5, image[1] - 0.5),
        crval: (sky[0], sky[1]),
        cd: [[m[0], m[1]], [m[2], m[3]]],
    };
    w.is_plausible().then_some(w)
}

/// A property's bytes read as little-endian doubles.
fn doubles(p: &Property) -> Vec<f64> {
    p.data
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn parse_location(s: Option<&str>, file: &str) -> Result<Location> {
    let s = s.unwrap_or("").trim();
    if let Some(rest) = s.strip_prefix("attachment:") {
        let parts: Vec<&str> = rest.split(':').collect();
        let position = parts.first().and_then(|p| p.trim().parse::<u64>().ok());
        let size = parts.get(1).and_then(|p| p.trim().parse::<u64>().ok());
        return match (position, size) {
            (Some(position), Some(size)) => Ok(Location::Attachment { position, size }),
            _ => Err(SrError::Input(format!(
                "{file}: location {s:?} is not attachment:position:size"
            ))),
        };
    }
    if let Some(encoding) = s.strip_prefix("inline:") {
        return Ok(Location::Inline {
            encoding: encoding.trim().to_string(),
        });
    }
    if s == "embedded" {
        return Ok(Location::Inline {
            encoding: "base64".into(),
        });
    }
    Err(SrError::Input(format!(
        "{file}: the pixel data is at {s:?}; only data stored in the file itself is read, not data \
         referred to by URL or by path"
    )))
}

fn parse_compression(s: Option<&str>, file: &str) -> Result<Option<Compression>> {
    let Some(s) = s.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let parts: Vec<&str> = s.split(':').collect();
    let (codec, shuffled) = match parts[0].strip_suffix("+sh") {
        Some(c) => (c, true),
        None => (parts[0], false),
    };
    let uncompressed = parts
        .get(1)
        .and_then(|p| p.trim().parse::<usize>().ok())
        .ok_or_else(|| {
            SrError::Input(format!(
                "{file}: compression {s:?} does not give an uncompressed size"
            ))
        })?;
    let item = parts.get(2).and_then(|p| p.trim().parse::<usize>().ok());
    if shuffled && item.is_none() {
        return Err(SrError::Input(format!(
            "{file}: compression {s:?} says the bytes were shuffled but not by what item size"
        )));
    }
    Ok(Some(Compression {
        codec: codec.to_ascii_lowercase(),
        uncompressed,
        shuffle_item: if shuffled { item } else { None },
    }))
}

/// Read the pixel block and scale it to `[0, 1]` by the declared bounds.
fn read_values(path: &Path, image: &Image) -> Result<Vec<f32>> {
    let file = path.display().to_string();
    let want = image.samples();
    let width = image.format.width();
    let bytes = read_block(path, image)?;
    if bytes.len() < want * width {
        return Err(SrError::Input(format!(
            "{file}: the data block holds {} bytes, short of the {} that {}x{}x{} samples of {:?} \
             need",
            bytes.len(),
            want * width,
            image.width,
            image.height,
            image.channels,
            image.format
        )));
    }

    let (lo, hi) = image.bounds;
    let span = (hi - lo).max(1e-30);
    let big = !image.little_endian;
    let read: fn(&[u8], bool) -> f64 = match image.format {
        SampleFormat::UInt8 => |c, _| c[0] as f64,
        SampleFormat::UInt16 => |c, b| {
            let v = [c[0], c[1]];
            (if b {
                u16::from_be_bytes(v)
            } else {
                u16::from_le_bytes(v)
            }) as f64
        },
        SampleFormat::UInt32 => |c, b| {
            let v = [c[0], c[1], c[2], c[3]];
            (if b {
                u32::from_be_bytes(v)
            } else {
                u32::from_le_bytes(v)
            }) as f64
        },
        SampleFormat::UInt64 => |c, b| {
            let v = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
            (if b {
                u64::from_be_bytes(v)
            } else {
                u64::from_le_bytes(v)
            }) as f64
        },
        SampleFormat::Float32 => |c, b| {
            let v = [c[0], c[1], c[2], c[3]];
            (if b {
                f32::from_be_bytes(v)
            } else {
                f32::from_le_bytes(v)
            }) as f64
        },
        SampleFormat::Float64 => |c, b| {
            let v = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
            if b {
                f64::from_be_bytes(v)
            } else {
                f64::from_le_bytes(v)
            }
        },
    };

    let mut out = vec![0f32; want];
    for (o, c) in out.iter_mut().zip(bytes.chunks_exact(width)) {
        let v = (read(c, big) - lo) / span;
        *o = if v.is_finite() {
            v.clamp(0.0, 1.0) as f32
        } else {
            0.0
        };
    }
    Ok(out)
}

/// The raw bytes of the image's data block, decompressed and unshuffled.
fn read_block(path: &Path, image: &Image) -> Result<Vec<u8>> {
    let file = path.display().to_string();
    let stored = match &image.location {
        Location::Inline { .. } => image.inline.clone(),
        Location::Attachment { position, size } => {
            let mut f = File::open(path)?;
            f.seek(SeekFrom::Start(*position))?;
            let mut bytes = vec![0u8; *size as usize];
            f.read_exact(&mut bytes).map_err(|e| {
                SrError::Input(format!(
                    "{file}: the {size}-byte data block at {position} runs past the end of the \
                     file: {e}"
                ))
            })?;
            bytes
        }
    };
    let Some(c) = &image.compression else {
        return Ok(stored);
    };
    let plain = decompress(&stored, c, &file)?;
    if plain.len() != c.uncompressed {
        return Err(SrError::Input(format!(
            "{file}: the {} block decompressed to {} bytes and the header says {}",
            c.codec,
            plain.len(),
            c.uncompressed
        )));
    }
    Ok(match c.shuffle_item {
        Some(item) if item > 1 => unshuffle(&plain, item),
        _ => plain,
    })
}

fn decompress(stored: &[u8], c: &Compression, file: &str) -> Result<Vec<u8>> {
    match c.codec.as_str() {
        "zlib" => {
            let mut out = Vec::with_capacity(c.uncompressed);
            flate2::read::ZlibDecoder::new(stored)
                .read_to_end(&mut out)
                .map_err(|e| SrError::Input(format!("{file}: the zlib block is corrupt: {e}")))?;
            Ok(out)
        }
        "lz4" | "lz4hc" => lz4_flex::block::decompress(stored, c.uncompressed)
            .map_err(|e| SrError::Input(format!("{file}: the {} block is corrupt: {e}", c.codec))),
        other => Err(SrError::Input(format!(
            "{file}: the pixel data is compressed with {other}, which this reader does not \
             decompress; zlib, lz4 and lz4hc are the codecs it knows. Re-save the file with \
             compression off, or with one of those"
        ))),
    }
}

/// Undo XISF byte shuffling: the bytes were regrouped so that the first byte of
/// every item comes first, then every second byte, and so on, which puts the
/// slowly-varying high bytes of the samples together and gives the compressor
/// something to find. Any tail too short to make a whole item was left alone.
fn unshuffle(input: &[u8], item: usize) -> Vec<u8> {
    let items = input.len() / item;
    let mut out = vec![0u8; input.len()];
    let mut src = 0usize;
    for j in 0..item {
        let mut dst = j;
        for _ in 0..items {
            out[dst] = input[src];
            dst += item;
            src += 1;
        }
    }
    out[items * item..].copy_from_slice(&input[src..]);
    out
}

/// The bytes of an inline or embedded block, from the element's text.
fn decode_text_block(encoding: &str, text: &str) -> Vec<u8> {
    match encoding.trim() {
        "hex" => {
            let clean: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
            clean
                .as_chunks::<2>()
                .0
                .iter()
                .filter_map(|c| u8::from_str_radix(std::str::from_utf8(c).ok()?, 16).ok())
                .collect()
        }
        _ => base64_decode(text),
    }
}

/// Base64 as XISF uses it: the standard alphabet, whitespace ignored, padding
/// optional. Anything else ends the value rather than being skipped, so a
/// truncated block comes back short instead of coming back wrong.
fn base64_decode(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for b in s.bytes() {
        let six = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b' ' | b'\t' | b'\r' | b'\n' => continue,
            _ => break,
        } as u32;
        acc = (acc << 6) | six;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

type Attributes = HashMap<String, String>;

fn local_name(e: &quick_xml::events::BytesEnd) -> String {
    String::from_utf8_lossy(e.local_name().as_ref()).to_string()
}

fn attributes(e: &BytesStart, file: &str) -> Result<Attributes> {
    let mut out = Attributes::new();
    for a in e.attributes() {
        let a = a.map_err(|err| {
            SrError::Input(format!(
                "{file}: malformed attribute in the XML header: {err}"
            ))
        })?;
        let key = String::from_utf8_lossy(a.key.local_name().as_ref()).to_string();
        let value = a.unescape_value().unwrap_or_default().to_string();
        out.insert(key, value);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a monolithic XISF file in memory with its block attached.
    ///
    /// The attachment offset depends on the header length and the header
    /// length depends on how many digits the offset takes, so the offset is
    /// written zero-padded to a fixed width and the circle closes in one pass.
    fn synth(attrs: &str, children: &str, block: &[u8]) -> Vec<u8> {
        let head = |pos: usize| {
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
                 <xisf version=\"1.0\" xmlns=\"http://www.pixinsight.com/xisf\">\
                 <Image {attrs} location=\"attachment:{pos:010}:{}\">{children}</Image>\
                 </xisf>",
                block.len()
            )
        };
        let length = head(0).len();
        let xml = head(PREAMBLE + length);
        assert_eq!(xml.len(), length, "the offset changed the header length");
        let mut out = Vec::new();
        out.extend_from_slice(SIGNATURE);
        out.extend_from_slice(&(length as u32).to_le_bytes());
        out.extend_from_slice(&[0u8; 4]);
        out.extend_from_slice(xml.as_bytes());
        out.extend_from_slice(block);
        out
    }

    /// A throwaway directory that removes itself.
    struct Dir(std::path::PathBuf);

    impl Dir {
        fn new() -> Dir {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let p = std::env::temp_dir().join(format!("sr-xisf-{stamp}-{n}"));
            std::fs::create_dir_all(&p).unwrap();
            Dir(p)
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write(bytes: &[u8]) -> (Dir, std::path::PathBuf) {
        let dir = Dir::new();
        let path = dir.0.join("frame.xisf");
        std::fs::write(&path, bytes).unwrap();
        (dir, path)
    }

    /// A ramp with odd low bits, so the white-level detector does not mistake
    /// it for a converter narrower than the file.
    fn ramp(n: usize) -> Vec<u16> {
        (0..n)
            .map(|i| (i as u16).wrapping_mul(37).wrapping_add(101))
            .collect()
    }

    fn le16(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    /// The forward shuffle, which is the thing the reader has to undo.
    fn shuffle(plain: &[u8], item: usize) -> Vec<u8> {
        let items = plain.len() / item;
        let mut out = Vec::with_capacity(plain.len());
        for j in 0..item {
            for i in 0..items {
                out.push(plain[i * item + j]);
            }
        }
        out.extend_from_slice(&plain[items * item..]);
        out
    }

    #[test]
    fn a_sixteen_bit_image_comes_back_sample_for_sample() {
        let pixels = ramp(8 * 6);
        let bytes = synth(
            "geometry=\"8:6:1\" sampleFormat=\"UInt16\" colorSpace=\"Gray\"",
            "",
            &le16(&pixels),
        );
        let (_d, path) = write(&bytes);
        let frame = decode(&path).unwrap();
        assert_eq!((frame.width, frame.height), (8, 6));
        // UInt16 maps onto the pipeline's own 16-bit levels exactly, so this
        // path is lossless and can be checked value for value.
        for (i, want) in pixels.iter().enumerate() {
            let got = frame.samples.value_in_cell(i, i % 4) * frame.metadata.white_level;
            assert!(
                (got - *want as f32).abs() <= 1.0,
                "sample {i} came back {got} and was {want}"
            );
        }
    }

    #[test]
    fn float_samples_are_scaled_by_the_declared_bounds() {
        // The same bytes under two declared ranges must not give the same
        // image: bounds is the only thing that says what full scale is.
        let values: Vec<f32> = (0..8 * 6).map(|i| i as f32 / 47.0).collect();
        let block: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let unit = synth(
            "geometry=\"8:6:1\" sampleFormat=\"Float32\" bounds=\"0:1\"",
            "",
            &block,
        );
        let (_d, p) = write(&unit);
        let a = read_planes(&p).unwrap().1;

        let wide = synth(
            "geometry=\"8:6:1\" sampleFormat=\"Float32\" bounds=\"0:2\"",
            "",
            &block,
        );
        let (_d2, p2) = write(&wide);
        let b = read_planes(&p2).unwrap().1;

        for (i, want) in values.iter().enumerate() {
            assert!((a[0].data[i] - want).abs() < 1e-6, "unit bounds at {i}");
            assert!(
                (b[0].data[i] - want / 2.0).abs() < 1e-6,
                "wide bounds at {i}"
            );
        }
    }

    #[test]
    fn the_original_fits_keywords_reach_the_metadata() {
        let bytes = synth(
            "geometry=\"8:6:1\" sampleFormat=\"UInt16\"",
            "<FITSKeyword name=\"FILTER\" value=\"&apos;Ha&apos;\" comment=\"filter\"/>\
             <FITSKeyword name=\"EXPTIME\" value=\"300.0\" comment=\"seconds\"/>\
             <FITSKeyword name=\"INSTRUME\" value=\"&apos;ZWO ASI2600MC&apos;\" comment=\"\"/>\
             <FITSKeyword name=\"XPIXSZ\" value=\"3.76\" comment=\"\"/>",
            &le16(&ramp(8 * 6)),
        );
        let (_d, path) = write(&bytes);
        let frame = decode(&path).unwrap();
        assert_eq!(frame.metadata.filter.as_deref(), Some("Ha"));
        assert_eq!(frame.metadata.exposure_time, Some(300.0));
        assert_eq!(frame.metadata.model, "ASI2600MC");
        assert_eq!(frame.metadata.pixel_pitch_um, Some(3.76));
        assert_eq!(peek_filter(&path).as_deref(), Some("Ha"));
    }

    #[test]
    fn capture_time_uses_native_property_with_fits_precedence() {
        for (cards, expected) in [
            ("", "2026-09-22T01:02:03Z"),
            (
                "<FITSKeyword name=\"DATE-OBS\" value=\"&apos;2026-09-23T00:00:00&apos;\"/>",
                "2026-09-23T00:00:00",
            ),
        ] {
            let properties = format!(
                "{cards}<Property id=\"Observation:Time:Start\" type=\"TimePoint\" value=\"2026-09-22T01:02:03Z\"/>"
            );
            let bytes = synth(
                "geometry=\"8:6:1\" sampleFormat=\"UInt16\"",
                &properties,
                &le16(&ramp(8 * 6)),
            );
            let (_d, path) = write(&bytes);
            assert_eq!(
                decode(&path).unwrap().metadata.capture_time.as_deref(),
                Some(expected)
            );
        }
    }

    #[test]
    fn a_colour_image_is_refused_with_the_reason() {
        let bytes = synth(
            "geometry=\"8:6:3\" sampleFormat=\"UInt16\" colorSpace=\"RGB\"",
            "",
            &le16(&ramp(8 * 6 * 3)),
        );
        let (_d, path) = write(&bytes);
        let e = decode(&path).unwrap_err().to_string();
        assert!(e.contains("mosaic"), "{e}");
        // But it is still readable as the finished image it is.
        let (image, planes) = read_planes(&path).unwrap();
        assert_eq!(planes.len(), 3);
        assert_eq!(image.colour_space, "RGB");
    }

    #[test]
    fn planar_and_interleaved_channels_land_the_same_way() {
        let (w, h) = (8usize, 6usize);
        let planar: Vec<u16> = (0..3)
            .flat_map(|c| (0..w * h).map(move |i| (c * 1000 + i * 7 + 1) as u16))
            .collect();
        let mut interleaved = vec![0u16; w * h * 3];
        for c in 0..3 {
            for i in 0..w * h {
                interleaved[i * 3 + c] = planar[c * w * h + i];
            }
        }
        let a = synth(
            "geometry=\"8:6:3\" sampleFormat=\"UInt16\" colorSpace=\"RGB\"",
            "",
            &le16(&planar),
        );
        let b = synth(
            "geometry=\"8:6:3\" sampleFormat=\"UInt16\" colorSpace=\"RGB\" pixelStorage=\"Normal\"",
            "",
            &le16(&interleaved),
        );
        let (_d1, p1) = write(&a);
        let (_d2, p2) = write(&b);
        let x = read_planes(&p1).unwrap().1;
        let y = read_planes(&p2).unwrap().1;
        for c in 0..3 {
            assert_eq!(x[c].data, y[c].data, "channel {c}");
        }
    }

    #[test]
    fn big_endian_samples_are_read_the_other_way_round() {
        let pixels = ramp(8 * 6);
        let block: Vec<u8> = pixels.iter().flat_map(|s| s.to_be_bytes()).collect();
        let bytes = synth(
            "geometry=\"8:6:1\" sampleFormat=\"UInt16\" byteOrder=\"big\"",
            "",
            &block,
        );
        let (_d, path) = write(&bytes);
        let planes = read_planes(&path).unwrap().1;
        for (i, want) in pixels.iter().enumerate() {
            let got = planes[0].data[i] * 65535.0;
            assert!(
                (got - *want as f32).abs() <= 1.0,
                "sample {i}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn a_compressed_and_shuffled_block_is_put_back_together() {
        use std::io::Write;
        let pixels = ramp(8 * 6);
        let plain = le16(&pixels);
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&shuffle(&plain, 2)).unwrap();
        let block = enc.finish().unwrap();

        let attrs = format!(
            "geometry=\"8:6:1\" sampleFormat=\"UInt16\" compression=\"zlib+sh:{}:2\"",
            plain.len()
        );
        let bytes = synth(&attrs, "", &block);
        let (_d, path) = write(&bytes);
        let planes = read_planes(&path).unwrap().1;
        for (i, want) in pixels.iter().enumerate() {
            let got = planes[0].data[i] * 65535.0;
            assert!(
                (got - *want as f32).abs() <= 1.0,
                "sample {i}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn an_lz4_block_is_decompressed() {
        let pixels = ramp(8 * 6);
        let plain = le16(&pixels);
        let block = lz4_flex::block::compress(&plain);
        let attrs = format!(
            "geometry=\"8:6:1\" sampleFormat=\"UInt16\" compression=\"lz4hc:{}\"",
            plain.len()
        );
        let bytes = synth(&attrs, "", &block);
        let (_d, path) = write(&bytes);
        let planes = read_planes(&path).unwrap().1;
        assert!((planes[0].data[5] * 65535.0 - pixels[5] as f32).abs() <= 1.0);
    }

    #[test]
    fn an_unknown_codec_says_which_one_it_was() {
        let attrs = "geometry=\"8:6:1\" sampleFormat=\"UInt16\" compression=\"zstd:96\"";
        let bytes = synth(attrs, "", &[0u8; 40]);
        let (_d, path) = write(&bytes);
        let e = read_planes(&path).unwrap_err().to_string();
        assert!(e.contains("zstd"), "{e}");
    }

    #[test]
    fn shuffling_is_undone_including_the_tail_it_left_alone() {
        for item in [1usize, 2, 4, 8] {
            // A length that is not a whole number of items, so the bytes the
            // shuffle leaves in place are exercised too.
            let plain: Vec<u8> = (0..item * 13 + 3).map(|i| (i * 31 % 251) as u8).collect();
            assert_eq!(
                unshuffle(&shuffle(&plain, item), item),
                plain,
                "item size {item}"
            );
        }
    }

    #[test]
    fn base64_takes_padding_or_none_and_ignores_whitespace() {
        assert_eq!(base64_decode("TWE="), b"Ma");
        assert_eq!(base64_decode("TWE"), b"Ma");
        assert_eq!(base64_decode("TW Fu\n"), b"Man");
        assert_eq!(base64_decode(""), Vec::<u8>::new());
    }

    #[test]
    fn a_file_that_is_not_xisf_says_so_rather_than_reading_rubbish() {
        let (_d, path) = write(b"SIMPLE  =                    T");
        let e = read_header(&path).unwrap_err().to_string();
        assert!(e.contains("signature"), "{e}");
    }

    #[test]
    fn the_astrometric_solution_properties_are_read() {
        // An astrometric solution written into a 6248x4176 master of M45, taken
        // from the file verbatim. Checked against the sky rather than against
        // itself: these numbers put Merope, Atlas, Taygeta and Pleione within
        // about two pixels of where they are, at 4.46 arcsec per pixel, while
        // reading the matrix by columns instead puts them twelve to twenty
        // pixels out. That measurement is what fixes the convention here.
        let props = "\
            <Property id=\"PCL:AstrometricSolution:ProjectionSystem\" type=\"String\">Gnomonic</Property>\
            <Property id=\"PCL:AstrometricSolution:LinearTransformationMatrix\" type=\"F64Matrix\" \
             rows=\"2\" columns=\"2\" location=\"inline:base64\">zOjpRvTEAz/EGD57Mk1UP0hcaCg7SlS/acc4KBXFAz8=</Property>\
            <Property id=\"PCL:AstrometricSolution:ReferenceImageCoordinates\" type=\"F64Vector\" \
             length=\"2\" location=\"inline:base64\">IAHc7G9oqEB9aLxZ5U+gQA==</Property>\
            <Property id=\"PCL:AstrometricSolution:ReferenceCelestialCoordinates\" type=\"F64Vector\" \
             length=\"2\" location=\"inline:base64\">mMEXHkFfTECo8INDMRo4QA==</Property>";
        let bytes = synth(
            "geometry=\"8:6:1\" sampleFormat=\"UInt16\"",
            props,
            &le16(&ramp(8 * 6)),
        );
        let (_d, path) = write(&bytes);
        let image = read_header(&path).unwrap();
        let w = image.wcs.expect("the properties carry a solve");
        assert!((w.crval.0 - 56.744174).abs() < 1e-4, "RA {}", w.crval.0);
        assert!((w.crval.1 - 24.102314).abs() < 1e-4, "Dec {}", w.crval.1);
        assert!(
            (w.crpix.0 - 3123.7186).abs() < 1e-3,
            "crpix x {}",
            w.crpix.0
        );
        assert!(
            (w.crpix.1 - 2087.4479).abs() < 1e-3,
            "crpix y {}",
            w.crpix.1
        );
        assert!(
            (w.scale_arcsec() - 4.46158).abs() < 1e-4,
            "scale {}",
            w.scale_arcsec()
        );
        // The off-diagonal terms dominate, and they are the two that swap if
        // the matrix is read by columns.
        assert!(
            w.cd[0][1] > 0.0 && w.cd[1][0] < 0.0,
            "matrix read by columns: {:?}",
            w.cd
        );
    }

    #[test]
    fn a_solve_in_the_fits_cards_wins_over_the_one_in_the_properties() {
        // A file converted from FITS keeps cards that are already in this
        // reader's terms. The properties are the fallback, not the authority.
        let cards = "\
            <FITSKeyword name=\"CTYPE1\" value=\"&apos;RA---TAN&apos;\" comment=\"\"/>\
            <FITSKeyword name=\"CTYPE2\" value=\"&apos;DEC--TAN&apos;\" comment=\"\"/>\
            <FITSKeyword name=\"CRPIX1\" value=\"5.0\" comment=\"\"/>\
            <FITSKeyword name=\"CRPIX2\" value=\"4.0\" comment=\"\"/>\
            <FITSKeyword name=\"CRVAL1\" value=\"10.0\" comment=\"\"/>\
            <FITSKeyword name=\"CRVAL2\" value=\"20.0\" comment=\"\"/>\
            <FITSKeyword name=\"CD1_1\" value=\"0.001\" comment=\"\"/>\
            <FITSKeyword name=\"CD1_2\" value=\"0.0\" comment=\"\"/>\
            <FITSKeyword name=\"CD2_1\" value=\"0.0\" comment=\"\"/>\
            <FITSKeyword name=\"CD2_2\" value=\"0.001\" comment=\"\"/>\
            <Property id=\"PCL:AstrometricSolution:ProjectionSystem\" type=\"String\">Gnomonic</Property>\
            <Property id=\"PCL:AstrometricSolution:LinearTransformationMatrix\" type=\"F64Matrix\" \
             rows=\"2\" columns=\"2\" location=\"inline:base64\">zOjpRvTEAz/EGD57Mk1UP0hcaCg7SlS/acc4KBXFAz8=</Property>\
            <Property id=\"PCL:AstrometricSolution:ReferenceImageCoordinates\" type=\"F64Vector\" \
             length=\"2\" location=\"inline:base64\">IAHc7G9oqEB9aLxZ5U+gQA==</Property>\
            <Property id=\"PCL:AstrometricSolution:ReferenceCelestialCoordinates\" type=\"F64Vector\" \
             length=\"2\" location=\"inline:base64\">mMEXHkFfTECo8INDMRo4QA==</Property>";
        let bytes = synth(
            "geometry=\"8:6:1\" sampleFormat=\"UInt16\"",
            cards,
            &le16(&ramp(8 * 6)),
        );
        let (_d, path) = write(&bytes);
        let w = read_header(&path)
            .unwrap()
            .wcs
            .expect("the cards carry a solve");
        assert_eq!(w.crval, (10.0, 20.0));
        assert_eq!(w.crpix, (4.0, 3.0));
    }

    #[test]
    fn a_block_short_of_the_geometry_is_an_error_not_a_black_edge() {
        let bytes = synth(
            "geometry=\"8:6:1\" sampleFormat=\"UInt16\"",
            "",
            &le16(&ramp(20)),
        );
        let (_d, path) = write(&bytes);
        let e = read_planes(&path).unwrap_err().to_string();
        assert!(e.contains("short of"), "{e}");
    }

    fn base64_encode(bytes: &[u8]) -> String {
        const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for c in bytes.chunks(3) {
            let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
            let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
            for i in 0..4 {
                if i <= c.len() {
                    out.push(A[((n >> (18 - 6 * i)) & 63) as usize] as char);
                } else {
                    out.push('=');
                }
            }
        }
        out
    }

    /// An image small enough that the writer keeps it in the header rather than
    /// attaching it, with properties beside it. A property carries a text of its
    /// own, and collecting both texts into one buffer loses the image.
    #[test]
    fn an_inline_block_survives_a_property_with_text_beside_it() {
        let pixels = ramp(8 * 6);
        let payload = base64_encode(&le16(&pixels));
        let xml = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
             <xisf version=\"1.0\" xmlns=\"http://www.pixinsight.com/xisf\">\
             <Image geometry=\"8:6:1\" sampleFormat=\"UInt16\" location=\"inline:base64\">\
             <FITSKeyword name=\"FILTER\" value=\"&apos;Ha&apos;\" comment=\"\"/>\
             <Property id=\"PCL:ProcessingHistory\" type=\"String\">a long story</Property>\
             {payload}</Image></xisf>"
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SIGNATURE);
        bytes.extend_from_slice(&(xml.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        bytes.extend_from_slice(xml.as_bytes());
        let (_d, path) = write(&bytes);

        let planes = read_planes(&path).unwrap().1;
        for (i, want) in pixels.iter().enumerate() {
            let got = planes[0].data[i] * 65535.0;
            assert!(
                (got - *want as f32).abs() <= 1.0,
                "sample {i}: {got} vs {want}"
            );
        }
        // And neither text ended up in the other.
        let image = read_header(&path).unwrap();
        assert_eq!(image.header.any_text(&["FILTER"]).as_deref(), Some("Ha"));
        assert!(
            image.wcs.is_none(),
            "a processing history is not a plate solve"
        );
    }
}

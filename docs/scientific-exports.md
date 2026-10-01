# FITS and XISF master exports

The web app has FITS master and XISF master checkboxes under Also. Enable either or both before stacking. Per-filter runs produce one master per filter; memory-batched runs export the final combined master. The colour-composite section has its own FITS/XISF checkboxes.

CLI examples:

```powershell
smokstak stack lights.txt --output result.tif --fits --xisf
smokstak stack lights.txt --split-by-filter --output result.tif --fits
smokstak combine batch1 batch2 --output combined.tif --xisf
smokstak composite H=ha.linear.tif O=oiii.linear.tif S=sii.linear.tif --palette sho --output sho.tif --fits --xisf
```

These options add result.fits and/or result.xisf alongside the existing rendered TIFF. --float-tiff remains available separately. FITS/XISF stack masters contain the same 32-bit floating-point data as the existing unstretched linear TIFF, without output clipping or integer quantization. Existing linear processing options still apply; optional restoration is excluded just as it is from the linear TIFF. Composite masters retain the selected palette and any explicitly selected channel stretch, before the TIFF display transfer. Disable channel stretching for a linear composite.

FITS uses IEEE Float32, mono 2D or planar RGB 3D, big-endian samples and 2880-byte header/data blocks. Rows are stored top first, in the order the light frames were read, and marked ROWORDER = 'TOP-DOWN', so a reader that ignores ROWORDER shows the master the same way up as the frames it came from, and one that honours it agrees. XISF uses a single Float32 image with planar top-down data in an uncompressed little-endian attachment.

When the reference frame carries a plate solution, both masters carry one too: the reference's TAN solution moved onto the output grid (scaled for 2x, offset for a region of interest), as CTYPE/CRPIX/CRVAL/CD keywords in FITS and as FITSKeyword elements in XISF. SIP distortion terms are not carried, so treat it as a good seed for a re-solve rather than an astrometric reference. `combine` carries it through the accumulators; colour composites do not have one. The masters are not a copy of every acquisition header.

Implementation references: [NASA FITS array conventions](https://fits.gsfc.nasa.gov/users_guide/users_guide/node25.html), [FITS keywords](https://fits.gsfc.nasa.gov/users_guide/users_guide/node21.html), and the XISF 1.0 specification.

Validation: writer tests cover mono/RGB, channel order, row orientation, negative/above-one samples, and FITS/XISF headers. GUI tests cover scientific-sidecar overwrite detection and final batch flags. tools/verify_scientific_exports.py independently parses FITS/XML attachments and compares actual exported pixels to the linear TIFF.

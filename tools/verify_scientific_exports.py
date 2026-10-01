"""Verify FITS/XISF export samples against companion linear TIFFs.

Usage: python tools/verify_scientific_exports.py out/result [out/other]
Uses independent Python header/XML parsing and NumPy decoding; no app reader.
"""
import argparse
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import tifffile


def verify(base):
    expected=tifffile.imread(base.with_suffix('.linear.tif'))
    data=base.with_suffix('.fits').read_bytes()
    header={};end=None
    for i in range(0,len(data),80):
        card=data[i:i+80].decode('ascii')
        if card[:8].strip()=='END':end=((i+80+2879)//2880)*2880;break
        if card[8:10]=='= ':header[card[:8].strip()]=card[10:].strip()
    assert end is not None and len(data)%2880==0
    assert header['SIMPLE']=='T' and header['BITPIX']=='-32'
    assert header['ROWORDER']=="'TOP-DOWN'"
    w,h=int(header['NAXIS1']),int(header['NAXIS2']);c=int(header.get('NAXIS3','1'));n=w*h*c
    a=np.frombuffer(data,dtype='>f4',count=n,offset=end).reshape(c,h,w)
    a=a[0] if c==1 else a.transpose(1,2,0)
    assert np.array_equal(a,expected,equal_nan=True),f'{base}: FITS differs'
    data=base.with_suffix('.xisf').read_bytes();assert data[:8]==b'XISF0100'
    length=struct.unpack('<I',data[8:12])[0]
    image=ET.fromstring(data[16:16+length]).find('{http://www.pixinsight.com/xisf}Image')
    assert image.attrib['geometry']==f'{w}:{h}:{c}'
    assert image.attrib['sampleFormat']=='Float32' and image.attrib['pixelStorage']=='Planar'
    _,offset,size=image.attrib['location'].split(':');assert int(size)==n*4
    a=np.frombuffer(data,dtype='<f4',count=n,offset=int(offset)).reshape(c,h,w)
    a=a[0] if c==1 else a.transpose(1,2,0)
    assert np.array_equal(a,expected,equal_nan=True),f'{base}: XISF differs'
    return dict(shape=list(expected.shape),fits_exact=True,xisf_exact=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('base',type=Path,nargs='+');args=parser.parse_args()
    print(json.dumps({str(p):verify(p) for p in args.base},indent=2))

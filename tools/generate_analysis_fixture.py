"""Create mono FITS observations for analysis benchmarks, using NumPy.

Noise is independent across frames; --floor adds a fixed detector component.
At zero dither it remains a correlated floor. Stars support real registration.
"""
import argparse
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--width",type=int,default=1024)
    p.add_argument("--height",type=int,default=1024)
    p.add_argument("--frames",type=int,default=32)
    p.add_argument("--floor",type=float,default=0.0)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20260923)
    scene=np.full((args.height,args.width),.1,dtype=np.float32)
    stars=max(100,args.width*args.height//12000)
    for _ in range(stars):
        x=int(rng.integers(12,args.width-12));y=int(rng.integers(12,args.height-12))
        yy,xx=np.mgrid[-8:9,-8:9]
        scene[y-8:y+9,x-8:x+9]+=np.float32(rng.uniform(.15,.5))*np.exp(-(xx*xx+yy*yy)/6).astype(np.float32)
    if args.floor:
        scene+=rng.standard_normal(scene.shape,dtype=np.float32)*np.float32(args.floor)
    for i in range(args.frames):
        image=rng.standard_normal(scene.shape,dtype=np.float32)
        image*=.003;image+=scene
        cards=[("SIMPLE","T"),("BITPIX","-32"),("NAXIS","2"),("NAXIS1",str(args.width)),
               ("NAXIS2",str(args.height)),("ROWORDER","'TOP-DOWN'"),("FILTER","'L'"),
               ("EXPTIME","180"),("DATE-OBS",f"'2026-09-23T{i//60:02}:{i%60:02}:00'")]
        header="".join(f"{k:<8}= {v:<70}" for k,v in cards)+f"{'END':<80}"
        header=header.ljust((len(header)+2879)//2880*2880)
        with (args.output/f"{i:04}.fits").open("wb") as f:
            f.write(header.encode("ascii"));image.astype('>f4').tofile(f)
            f.write(b'\0'*((-image.nbytes)%2880))
        print(f"Wrote {i+1}/{args.frames}",flush=True)


if __name__=="__main__":
    main()

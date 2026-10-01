"""Reproducible analyze timing and sampled peak-memory matrix (no extra packages).

Run after cargo build --release. Outputs are isolated under --output.
Windows captures the process's high-water working set; other platforms report null.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time


def peak_bytes(process):
    if os.name != "nt":
        return None

    class Memory(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage")]
    counters = Memory()
    counters.cb = ctypes.sizeof(counters)
    if ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(int(process._handle)), ctypes.byref(counters), counters.cb):
        return counters.PeakWorkingSetSize
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/smokstak.exe"))
    parser.add_argument("--output", type=Path, default=Path("out/analysis-benchmark"))
    parser.add_argument("--filter")
    parser.add_argument("--counts", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--samples", type=int, nargs="+", default=[65536, 262144, 524288])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for count in args.counts:
        for samples in args.samples:
            directory = args.output / f"n{count}-s{samples}"
            directory.mkdir(exist_ok=True)
            # A fresh cache path makes the first pass cold on repeat benchmarks.
            cache = directory / f"cache-{time.time_ns()}"
            for temperature in ("cold", "warm"):
                report = directory / f"{temperature}.json"
                command = [str(args.binary.resolve()), "--threads", str(args.threads),
                           "--log", "warn", "analyze", str(args.input.resolve()),
                           "--max-frames", str(count), "--samples", str(samples),
                           "--cache-dir", str(cache), "--json", str(report),
                           "--html", str(report.with_suffix(".html"))]
                if args.filter:
                    command += ["--filter", args.filter]
                with report.with_suffix(".log").open("w", encoding="utf-8") as log:
                    start = time.perf_counter()
                    process = subprocess.Popen(command, stdout=log, stderr=log,
                                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    peak = 0
                    while process.poll() is None:
                        peak = max(peak, peak_bytes(process) or 0)
                        time.sleep(0.05)
                    wall = time.perf_counter() - start
                if process.returncode:
                    raise RuntimeError(f"Benchmark failed; see {report.with_suffix('.log')}")
                data = json.loads(report.read_text(encoding="utf-8"))
                row = dict(frames=count, samples=samples, cache=temperature, wall_seconds=wall,
                           peak_working_set_bytes=peak or None, timings=data["timings"],
                           summary=data["summary"], filters=[dict(filter=f["filter"],
                               sample_count=f["sample_count"], global_fit=f["global_fit"],
                               final_noise=f["integration_depth"][-1]["noise"] if f["integration_depth"] else None)
                               for f in data["filters"]])
                results.append(row)
                print(json.dumps(row), flush=True)
                (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
clean_capstone.py
- Normalizes loudness of all videos under ~/Desktop/dataForCapstone/{ambix,mono,none}
- Keeps video stream intact, adjusts audio using EBU R128 loudnorm (two-pass)
- Outputs to ~/Desktop/dataForCapstone/normalized/<subdir>/<name>_norm.<ext>
- Writes a CSV log with measured/target values.

Requirements:
  - ffmpeg and ffprobe available on PATH (brew install ffmpeg on macOS)
  - Python 3.8+

Optional:
  - Use --ambix-lossless to write multichannel/ambisonics as PCM in .mov (bigger files, analysis-friendly)
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# ---------- Utils ----------

def run(cmd: str, capture: bool = True) -> Tuple[int, str, str]:
    """Run a shell command. Returns (returncode, stdout, stderr)."""
    proc = subprocess.run(
        shlex.split(cmd),
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True
    )
    out = proc.stdout or ""
    err = proc.stderr or ""
    return proc.returncode, out, err

def ffprobe_streams(path: Path) -> Dict:
    """Return ffprobe stream info as dict."""
    cmd = f'ffprobe -v error -print_format json -show_streams -show_format "{path}"'
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{err}")
    return json.loads(out)

def choose_audio_codec(ext: str, channels: int, ambix_lossless: bool) -> Tuple[str, str]:
    """
    Choose audio codec and container extension.
    - Default: AAC @ 192k, keep original container extension.
    - If ambix_lossless and channels > 2, write PCM 16-bit and force .mov container
      (PCM in MP4 is poorly supported).
    """
    if ambix_lossless and channels > 2:
        return ("-c:a pcm_s16le", "mov")
    # default AAC
    return ("-c:a aac -b:a 192k", ext.lower().lstrip("."))

def two_pass_loudnorm_firstpass(infile: Path, target_I=-16.0, LRA=11.0, TP=-1.5) -> Dict:
    """
    First pass: compute loudnorm stats as JSON (from stderr).
    """
    filter_str = f'loudnorm=I={target_I}:LRA={LRA}:TP={TP}:print_format=json'
    cmd = f'ffmpeg -hide_banner -y -i "{infile}" -vn -af "{filter_str}" -f null -'
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"FFmpeg first pass failed for {infile}:\n{err}")
    # The JSON is in stderr, possibly with other lines. Extract last JSON block.
    json_text = extract_json_block(err)
    stats = json.loads(json_text)
    return stats

def extract_json_block(stderr_text: str) -> str:
    """
    Extract the last {...} JSON object present in a blob of text.
    """
    start = stderr_text.rfind("{")
    end = stderr_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Could not find JSON loudnorm block in FFmpeg output.")
    return stderr_text[start:end+1]

def two_pass_loudnorm_secondpass(
    infile: Path,
    outfile: Path,
    measured: Dict,
    target_I=-16.0, LRA=11.0, TP=-1.5,
    audio_codec: str = "-c:a aac -b:a 192k",
    sample_rate: Optional[int] = None
) -> None:
    """
    Second pass: apply loudnorm with measured parameters, copy video.
    """
    # Build measured args
    args = []
    for key in ["input_i", "input_tp", "input_lra", "input_thresh"]:
        if key not in measured:
            raise KeyError(f"Missing {key} in loudnorm measured stats.")
    args.append(f"measured_I={measured['input_i']}")
    args.append(f"measured_TP={measured['input_tp']}")
    args.append(f"measured_LRA={measured['input_lra']}")
    args.append(f"measured_thresh={measured['input_thresh']}")
    # optional offset
    if "offset" in measured:
        args.append(f"offset={measured['offset']}")
    args.append(f"I={target_I}")
    args.append(f"LRA={LRA}")
    args.append(f"TP={TP}")
    args.append("print_format=summary")

    af = "loudnorm=" + ":".join(args)
    sr_arg = f"-ar {sample_rate}" if sample_rate else ""
    cmd = (
        f'ffmpeg -hide_banner -y -i "{infile}" '
        f'-c:v copy {sr_arg} -af "{af}" {audio_codec} '
        f'"{outfile}"'
    )
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"FFmpeg second pass failed for {infile}:\n{err}")

# ---------- Main pipeline ----------

def process_file(
    in_path: Path,
    out_root: Path,
    target_I: float,
    LRA: float,
    TP: float,
    ambix_lossless: bool,
    resample_hz: Optional[int],
) -> Dict:
    """
    Normalize a single media file and return dict of stats for CSV log.
    """
    probe = ffprobe_streams(in_path)
    # Find audio stream info
    audio_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        return {"input": str(in_path), "status": "no_audio"}

    a = audio_streams[0]
    channels = int(a.get("channels", 2))
    sr = int(a.get("sample_rate", 48000)) if a.get("sample_rate") else None

    # Determine subdir from the first path component under the root
    # root/.../<subdir>/<file>
    # We'll mirror subdir under normalized/
    # Also compute output extension/codec choice
    orig_ext = in_path.suffix or ".mp4"
    audio_codec, out_ext = choose_audio_codec(orig_ext, channels, ambix_lossless)

    # Rebuild output path
    # out_root/<subdir>/<basename>_norm.<ext>
    # figure subdir relative to the given root
    # We assume the root is an ancestor.
    out_dir = out_root / in_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = in_path.stem + "_norm." + out_ext
    out_path = out_dir / out_name

    # First pass to measure
    stats = two_pass_loudnorm_firstpass(in_path, target_I, LRA, TP)

    # Second pass to apply with measured
    two_pass_loudnorm_secondpass(
        in_path,
        out_path,
        measured=stats,
        target_I=target_I,
        LRA=LRA,
        TP=TP,
        audio_codec=audio_codec,
        sample_rate=(resample_hz or sr),
    )

    # Prepare outcome
    record = {
        "input": str(in_path),
        "output": str(out_path),
        "channels": channels,
        "sample_rate_in": sr or "",
        "target_I": target_I,
        "target_TP": TP,
        "target_LRA": LRA,
        "measured_I": stats.get("input_i", ""),
        "measured_TP": stats.get("input_tp", ""),
        "measured_LRA": stats.get("input_lra", ""),
        "measured_thresh": stats.get("input_thresh", ""),
        "status": "ok",
    }
    return record

def main():
    p = argparse.ArgumentParser(description="Normalize loudness of capstone videos without changing speed/pitch.")
    p.add_argument("--root", default=str(Path.home() / "Desktop" / "dataForCapstone"),
                   help="Root folder containing ambix/ mono/ none/ (default: ~/Desktop/dataForCapstone)")
    p.add_argument("--out", default=None,
                   help="Output root (default: <root>/normalized)")
    p.add_argument("--target", type=float, default=-16.0, help="Target LUFS (default: -16.0)")
    p.add_argument("--tp", type=float, default=-1.5, help="True-peak ceiling in dBTP (default: -1.5)")
    p.add_argument("--lra", type=float, default=11.0, help="Target LRA (default: 11.0)")
    p.add_argument("--resample", type=int, default=48000, help="Force output audio sample rate (default: 48000)")
    p.add_argument("--ambix-lossless", action="store_true",
                   help="For >2 channels, write PCM in .mov (bigger files, analysis-friendly).")
    p.add_argument("--extensions", nargs="+", default=[".mp4", ".mov", ".mkv", ".m4v"],
                   help="Video extensions to include.")
    args = p.parse_args()

    root = Path(os.path.expanduser(args.root)).resolve()
    out_root = Path(os.path.expanduser(args.out)).resolve() if args.out else (root / "normalized")
    out_root.mkdir(parents=True, exist_ok=True)

    # Validate ffmpeg availability
    for tool in ("ffmpeg", "ffprobe"):
        code, out, err = run(f"{tool} -version")
        if code != 0:
            print(f"ERROR: {tool} not found on PATH. Install FFmpeg (e.g., 'brew install ffmpeg').", file=sys.stderr)
            sys.exit(1)

    subdirs = ["ambix", "mono", "none"]
    files = []
    for sub in subdirs:
        d = root / sub
        if not d.exists():
            print(f"Warning: missing folder {d}")
            continue
        for ext in args.extensions:
            files.extend(d.rglob(f"*{ext}"))

    if not files:
        print("No matching video files found. Check --root and --extensions.")
        sys.exit(0)

    log_rows = []
    print(f"Found {len(files)} files. Normalizing to {args.target} LUFS (TP {args.tp} dBTP)...")
    for i, f in enumerate(sorted(files)):
        rel = f.relative_to(root)
        print(f"[{i+1}/{len(files)}] {rel}")
        try:
            rec = process_file(
                f, out_root,
                target_I=args.target,
                LRA=args.lra,
                TP=args.tp,
                ambix_lossless=args.ambix_lossless,
                resample_hz=args.resample
            )
        except Exception as e:
            rec = {"input": str(f), "status": f"error: {e}"}
            print(f"  ✖ {e}", file=sys.stderr)
        else:
            print("  ✓ done")
        log_rows.append(rec)

    # Write CSV log
    csv_path = out_root / "normalization_log.csv"
    fieldnames = ["input","output","channels","sample_rate_in","target_I","target_TP","target_LRA",
                  "measured_I","measured_TP","measured_LRA","measured_thresh","status"]
    with open(csv_path, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=fieldnames)
        w.writeheader()
        for r in log_rows:
            w.writerow(r)

    print("\nAll set!")
    print(f"- Outputs: {out_root}")
    print(f"- Log:     {csv_path}")

if __name__ == "__main__":
    main()

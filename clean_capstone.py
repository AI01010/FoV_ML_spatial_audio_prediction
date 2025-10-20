#!/usr/bin/env python3
"""
clean_capstone.py (with optional frame extraction)
- Normalizes loudness of all videos under ~/Desktop/dataForCapstone/{ambix,mono,none}
- Keeps video stream intact, adjusts audio using EBU R128 loudnorm (two-pass)
- Outputs to ~/Desktop/dataForCapstone/normalized/<subdir>/<name>_norm.<ext>
- Writes a CSV log with measured/target values.
- (NEW) Optionally extracts every Nth frame from the ORIGINAL video into
        ~/Desktop/dataForCapstone/normalized_frames/<subdir>/<basename>/

Requirements:
  - ffmpeg and ffprobe available on PATH (brew install ffmpeg on macOS)
  - Python 3.8+
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
    proc = subprocess.run(
        shlex.split(cmd),
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""

def ffprobe_streams(path: Path) -> Dict:
    cmd = f'ffprobe -v error -print_format json -show_streams -show_format "{path}"'
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{err}")
    return json.loads(out)

def choose_audio_codec(ext: str, channels: int, ambix_lossless: bool) -> Tuple[str, str]:
    if ambix_lossless and channels > 2:
        return ("-c:a pcm_s16le", "mov")
    return ("-c:a aac -b:a 192k", ext.lower().lstrip("."))

def two_pass_loudnorm_firstpass(infile: Path, target_I=-16.0, LRA=11.0, TP=-1.5) -> Dict:
    filter_str = f'loudnorm=I={target_I}:LRA={LRA}:TP={TP}:print_format=json'
    cmd = f'ffmpeg -hide_banner -y -i "{infile}" -vn -af "{filter_str}" -f null -'
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"FFmpeg first pass failed for {infile}:\n{err}")
    json_text = extract_json_block(err)
    return json.loads(json_text)

def extract_json_block(stderr_text: str) -> str:
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
    args = [
        f"measured_I={measured['input_i']}",
        f"measured_TP={measured['input_tp']}",
        f"measured_LRA={measured['input_lra']}",
        f"measured_thresh={measured['input_thresh']}",
        f"I={target_I}",
        f"LRA={LRA}",
        f"TP={TP}",
        "print_format=summary",
    ]
    if "offset" in measured:
        args.insert(4, f"offset={measured['offset']}")

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

# ---------- Frame extraction ----------

def extract_frames_stride(infile: Path, out_dir: Path, stride: int, img_ext: str = "jpg", quality: int = 2) -> int:
    """
    Keep every Nth frame from the ORIGINAL video.
    select='not(mod(n\\,N))' picks 1 of every N frames by index.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = (out_dir / f"frame_%06d.{img_ext}").as_posix()
    vf = f"select='not(mod(n\\,{stride}))'"
    qflag = f"-q:v {quality}" if img_ext.lower() in ("jpg", "jpeg") else ""
    cmd = f'ffmpeg -hide_banner -y -i "{infile}" -vf {vf} -vsync vfr {qflag} "{pattern}"'
    code, out, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"FFmpeg frame extraction failed for {infile}:\n{err}")
    return len(list(out_dir.glob(f"frame_*.{img_ext}")))

# ---------- Main pipeline ----------

def process_file(
    in_path: Path,
    out_root: Path,
    frames_root: Optional[Path],
    target_I: float,
    LRA: float,
    TP: float,
    ambix_lossless: bool,
    resample_hz: Optional[int],
    do_extract_frames: bool,
    frame_stride: int,
    frame_img_ext: str,
) -> Dict:
    probe = ffprobe_streams(in_path)
    audio_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]

    # AUDIO normalize (if present)
    if not audio_streams:
        out_video_path = ""
        status = "no_audio"
        channels = ""
        sr = ""
    else:
        a = audio_streams[0]
        channels = int(a.get("channels", 2))
        sr = int(a.get("sample_rate", 48000)) if a.get("sample_rate") else None

        orig_ext = in_path.suffix or ".mp4"
        audio_codec, out_ext = choose_audio_codec(orig_ext, channels, ambix_lossless)

        subdir = in_path.parent.name
        out_dir = out_root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = in_path.stem + "_norm." + out_ext
        out_video_path = (out_dir / out_name)

        stats = two_pass_loudnorm_firstpass(in_path, target_I, LRA, TP)
        two_pass_loudnorm_secondpass(
            in_path,
            out_video_path,
            measured=stats,
            target_I=target_I,
            LRA=LRA,
            TP=TP,
            audio_codec=audio_codec,
            sample_rate=(resample_hz or sr),
        )
        status = "ok"

    # FRAMES from ORIGINAL source (not normalized copy)
    frames_count = ""
    frames_out_dir = ""
    if do_extract_frames and frames_root is not None:
        subdir = in_path.parent.name
        frames_out_dir = str(frames_root / subdir / in_path.stem)
        try:
            frames_count = extract_frames_stride(
                infile=in_path,
                out_dir=Path(frames_out_dir),
                stride=frame_stride,
                img_ext=frame_img_ext,
                quality=2,
            )
        except Exception as e:
            frames_count = f"error: {e}"
            status = "ok_frames_error" if status == "ok" else "no_audio_frames_error"

    # (Optional) re-measure for log clarity
    measured_I = measured_TP = measured_LRA = measured_thresh = ""
    if audio_streams:
        try:
            mstats = two_pass_loudnorm_firstpass(in_path, target_I, LRA, TP)
            measured_I = mstats.get("input_i", "")
            measured_TP = mstats.get("input_tp", "")
            measured_LRA = mstats.get("input_lra", "")
            measured_thresh = mstats.get("input_thresh", "")
        except Exception:
            pass

    return {
        "input": str(in_path),
        "output": str(out_video_path) if audio_streams else "",
        "channels": channels if audio_streams else "",
        "sample_rate_in": sr if audio_streams else "",
        "target_I": target_I,
        "target_TP": TP,
        "target_LRA": LRA,
        "measured_I": measured_I,
        "measured_TP": measured_TP,
        "measured_LRA": measured_LRA,
        "measured_thresh": measured_thresh,
        "frames_stride": frame_stride if do_extract_frames else "",
        "frames_output_dir": frames_out_dir,
        "frames_count": frames_count,
        "status": status,
    }

def main():
    p = argparse.ArgumentParser(description="Normalize loudness (no speed/pitch change) and optionally extract frames.")
    p.add_argument("--root", default=str(Path.home() / "Desktop" / "dataForCapstone"),
                   help="Root folder containing ambix/ mono/ none/ (default: ~/Desktop/dataForCapstone)")
    p.add_argument("--out", default=None,
                   help="Output root for normalized videos (default: <root>/normalized)")
    p.add_argument("--target", type=float, default=-16.0, help="Target LUFS (default: -16.0)")
    p.add_argument("--tp", type=float, default=-1.5, help="True-peak ceiling in dBTP (default: -1.5)")
    p.add_argument("--lra", type=float, default=11.0, help="Target LRA (default: 11.0)")
    p.add_argument("--resample", type=int, default=48000, help="Force output audio sample rate (default: 48000)")
    p.add_argument("--ambix-lossless", action="store_true",
                   help="For >2 channels, write PCM in .mov (bigger files, analysis-friendly).")
    p.add_argument("--extensions", nargs="+", default=[".mp4", ".mov", ".mkv", ".m4v"],
                   help="Video extensions to include.")

    # NEW: frame extraction options
    p.add_argument("--extract-frames", action="store_true",
                   help="If set, extract frames from the ORIGINAL video.")
    p.add_argument("--frame-stride", type=int, default=5,
                   help="Keep every Nth frame (default: 5). Lower = more frames.")
    p.add_argument("--frame-ext", default="jpg", choices=["jpg","jpeg","png","webp"],
                   help="Image format for frames (default: jpg).")
    p.add_argument("--frames-out", default=None,
                   help="Root for extracted frames (default: <root>/normalized_frames)")

    args = p.parse_args()

    root = Path(os.path.expanduser(args.root)).resolve()
    out_root = Path(os.path.expanduser(args.out)).resolve() if args.out else (root / "normalized")
    out_root.mkdir(parents=True, exist_ok=True)

    frames_root = None
    if args.extract_frames:
        frames_root = Path(os.path.expanduser(args.frames_out)).resolve() if args.frames_out else (root / "normalized_frames")
        frames_root.mkdir(parents=True, exist_ok=True)

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
    if args.extract_frames:
        print(f"Also extracting frames: keep every {args.frame_stride}th frame as .{args.frame_ext}")

    for i, f in enumerate(sorted(files)):
        rel = f.relative_to(root)
        print(f"[{i+1}/{len(files)}] {rel}")
        try:
            rec = process_file(
                in_path=f,
                out_root=out_root,
                frames_root=frames_root,
                target_I=args.target,
                LRA=args.lra,
                TP=args.tp,
                ambix_lossless=args.ambix_lossless,
                resample_hz=args.resample,
                do_extract_frames=args.extract_frames,
                frame_stride=args.frame_stride,
                frame_img_ext=args.frame_ext,
            )
        except Exception as e:
            rec = {"input": str(f), "status": f"error: {e}"}
            print(f"  ✖ {e}", file=sys.stderr)
        else:
            print("  ✓ done")
        log_rows.append(rec)

    # Write CSV log
    csv_path = out_root / "normalization_log.csv"
    fieldnames = [
        "input","output","channels","sample_rate_in","target_I","target_TP","target_LRA",
        "measured_I","measured_TP","measured_LRA","measured_thresh",
        "frames_stride","frames_output_dir","frames_count","status"
    ]
    for r in log_rows:
        for k in fieldnames:
            r.setdefault(k, "")
    with open(csv_path, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=fieldnames)
        w.writeheader()
        for r in log_rows:
            w.writerow(r)

    print("\nAll set!")
    print(f"- Normalized videos: {out_root}")
    if frames_root:
        print(f"- Extracted frames: {frames_root}")
    print(f"- Log:               {csv_path}")

if __name__ == "__main__":
    main()

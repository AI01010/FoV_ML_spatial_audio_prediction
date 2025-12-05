import os, csv, subprocess, sys, json, glob

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, "Desktop", "dataForCapstone")
NORM_BASE = os.path.join(BASE, "normalized")
LOG_PATH = os.path.join(NORM_BASE, "normalization_log.csv")
OUT_BASE = os.path.join(BASE, "recovered_full_audio")
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm")
AUDIO_EXTS = (".wav", ".flac", ".m4a", ".aac", ".opus", ".ogg")

# Set True if you want to ALWAYS output 4 channels even when the source is mono/stereo (this will upmix with pan).
FORCE_UPMIX_WHEN_LT4 = False

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def looks_like_video(p): return p.lower().endswith(VIDEO_EXTS)
def looks_like_audio(p): return p.lower().endswith(AUDIO_EXTS)

def detect_columns(header):
    h = [x.strip().lower() for x in header]
    orig_keys = ("orig", "source", "input", "before")
    norm_keys = ("norm", "output", "dest", "after")
    oi = ni = None
    for i, name in enumerate(h):
        if any(k in name for k in orig_keys) and oi is None: oi = i
        if any(k in name for k in norm_keys) and ni is None: ni = i
    if oi is None or ni is None:
        oi, ni = (0,1) if len(h) > 1 else (0,0)
    return oi, ni

def run_ffprobe_json(args):
    out = subprocess.check_output(
        ["ffprobe","-v","error","-of","json"] + args,
        stderr=subprocess.STDOUT
    ).decode("utf-8","ignore")
    return json.loads(out)

def ffprobe_audio_streams(path):
    try:
        data = run_ffprobe_json(
            ["-select_streams","a","-show_entries","stream=index,codec_name,channels,channel_layout", path]
        )
        streams = data.get("streams", [])
        # Build audio-position (0..N-1) because ffmpeg -map uses that with a:
        a_pos = -1
        res = []
        for s in streams:
            a_pos += 1
            res.append({
                "global_index": int(s.get("index", 0)),
                "a_pos": a_pos,  # <- index among audio streams for -map 1:a:<a_pos>
                "channels": int(s.get("channels", 0)) if s.get("channels") is not None else 0,
                "codec_name": s.get("codec_name",""),
                "channel_layout": s.get("channel_layout","")
            })
        return res
    except Exception:
        return []

def pick_best_audio_stream(streams):
    if not streams: return None
    return sorted(streams, key=lambda s: (-s["channels"], s["a_pos"]))[0]

def find_candidate_norm(norm_path, orig_path):
    """
    If norm_path is a directory or doesn't exist, try to infer the normalized file:
    <stem>_norm.<ext> under normalized/{ambix|mono} matching the original's stem.
    """
    if norm_path and os.path.isfile(norm_path):
        return norm_path

    # If norm_path is a directory, use it as the root; otherwise search both ambix/mono under normalized
    roots = []
    if norm_path and os.path.isdir(norm_path):
        roots.append(norm_path)
    else:
        roots += [os.path.join(NORM_BASE, "ambix"), os.path.join(NORM_BASE, "mono")]

    stem = os.path.splitext(os.path.basename(orig_path))[0]
    # Try common video extensions with _norm suffix
    patterns = [os.path.join(root, f"{stem}_norm{ext}") for root in roots for ext in VIDEO_EXTS]
    for pat in patterns:
        if os.path.isfile(pat):
            return pat

    # Fallback: any file starting with stem and containing 'norm'
    globs = [os.path.join(root, f"{stem}*norm*") for root in roots]
    for g in globs:
        for cand in glob.glob(g):
            if os.path.isfile(cand) and looks_like_video(cand):
                return cand
    return norm_path if (norm_path and os.path.isfile(norm_path)) else None

def search_external_4ch_audio(orig_stem):
    """
    Look for a separate 4ch audio file with same stem under /ambix and /mono.
    Prefer files with >=4 channels.
    """
    candidates = []
    for sub in ("ambix","mono"):
        root = os.path.join(BASE, sub)
        for ext in AUDIO_EXTS:
            cand = os.path.join(root, orig_stem + ext)
            if os.path.isfile(cand):
                candidates.append(cand)

    best = None
    best_ch = 0
    for cand in candidates:
        streams = ffprobe_audio_streams(cand)
        s = pick_best_audio_stream(streams)
        if s and s["channels"] > best_ch:
            best = (cand, s)
            best_ch = s["channels"]

    return best  # (path, streaminfo) or None

def main():
    if not os.path.isfile(LOG_PATH):
        print(f"❌ Could not find log at {LOG_PATH}")
        sys.exit(1)

    ensure_dir(OUT_BASE)

    done = skipped = 0
    with open(LOG_PATH, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        print("❌ normalization_log.csv is empty.")
        sys.exit(1)

    header = rows[0]
    has_header = any(any(c.isalpha() for c in cell) for cell in header)
    if has_header:
        oi, ni = detect_columns(header)
        data = rows[1:]
    else:
        oi, ni = 0, 1
        data = rows

    for r in data:
        if len(r) <= max(oi, ni):
            continue

        orig_path = r[oi].strip()
        norm_path = r[ni].strip()

        # Allow relative paths for original
        if not os.path.isabs(orig_path):
            for sub in ("ambix","mono"):
                cand = os.path.join(BASE, sub, orig_path)
                if os.path.isfile(cand):
                    orig_path = cand
                    break

        if not os.path.isfile(orig_path):
            print(f"❌ Skip (orig not found): {orig_path}")
            skipped += 1
            continue

        # If norm is missing/dir, try to infer the normalized file next to normalized/{ambix|mono}
        norm_found = find_candidate_norm(norm_path, orig_path)
        if not norm_found or not os.path.isfile(norm_found):
            print(f"❌ Skip (norm not found): orig={orig_path}\n   tried norm={norm_path}")
            skipped += 1
            continue
        norm_path = norm_found

        if not (looks_like_video(orig_path) and looks_like_video(norm_path)):
            print(f"⚠️ Skip non-video pair:\n   orig: {orig_path}\n   norm: {norm_path}")
            skipped += 1
            continue

        # Output subfolder (based on normalized path)
        sub = "ambix" if "/ambix/" in norm_path.replace("\\","/") else ("mono" if "/mono/" in norm_path.replace("\\","/") else "misc")
        out_dir = os.path.join(OUT_BASE, sub)
        ensure_dir(out_dir)
        out_file = os.path.join(out_dir, os.path.basename(norm_path))

        stem = os.path.splitext(os.path.basename(orig_path))[0]

        # 1) Prefer a separate 4ch audio file with same stem in /ambix or /mono
        ext4 = search_external_4ch_audio(stem)
        audio_src = None
        audio_info = None

        if ext4:
            audio_src, audio_info = ext4
            print(f"🎯 Using external audio: {os.path.basename(audio_src)} ({audio_info['channels']}ch)")
        else:
            # 2) Fall back to audio inside original container
            streams = ffprobe_audio_streams(orig_path)
            best = pick_best_audio_stream(streams)
            if not best:
                print(f"❌ No audio streams in original: {orig_path}")
                skipped += 1
                continue
            audio_src = orig_path
            audio_info = best
            print(f"🔊 Using audio from original: {os.path.basename(orig_path)} "
                  f"(a:{best['a_pos']} {best['channels']}ch {best['codec_name']} {best.get('channel_layout','')})")

        a_pos = audio_info["a_pos"]
        ch = audio_info["channels"]

        # If we truly don't have ≥4ch and you want to force 4ch, we'll upmix:
        upmix = FORCE_UPMIX_WHEN_LT4 and (ch < 4)

        # Try copy first
        cmd_copy = [
            "ffmpeg","-y",
            "-i", norm_path,            # 0 = normalized video
            "-i", audio_src,            # 1 = audio source (external or original)
            "-map","0:v:0",
            "-map", f"1:a:{a_pos}",
            "-c:v","copy",
            "-c:a","copy",
            "-movflags","+faststart",
            out_file
        ]
        if upmix:
            # Can't copy+upmix; skip copy try
            cmd_copy = None

        try:
            if cmd_copy:
                subprocess.run(cmd_copy, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"✅ Wrote (copied audio): {out_file}")
                done += 1
                continue
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors='ignore').splitlines()[-1][:200]
            print(f"ℹ️ Stream copy failed (codec/container or channel mismatch). Retrying with AAC. ffmpeg: {err}")

        # Re-encode path
        # If we need to upmix to 4ch from mono/stereo, use pan filter. Else just set -ac to match existing ch if ≥4.
        reencode = [
            "ffmpeg","-y",
            "-i", norm_path,
            "-i", audio_src,
            "-map","0:v:0",
            "-map", f"1:a:{a_pos}",
            "-c:v","copy",
            "-c:a","aac",
            "-ar","48000",
            "-b:a","512k",
            "-movflags","+faststart"
        ]
        if upmix:
            if ch == 1:
                # duplicate mono to 4 channels
                pan = "pan=4c|c0=c0|c1=c0|c2=c0|c3=c0"
            else:
                # stereo to 4ch (simple duplication to back channels)
                pan = "pan=4c|c0=FL|c1=FR|c2=FL|c3=FR"
            reencode += ["-filter:a", pan]
        else:
            # If source already ≥4ch, keep 4ch
            if ch >= 4:
                reencode += ["-ac","4"]
            else:
                # keep original channel count (no fake 4ch)
                reencode += ["-ac", str(max(1, ch))]

        reencode += [out_file]

        try:
            subprocess.run(reencode, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ch_out = "4ch" if (upmix or ch >= 4) else f"{ch}ch"
            print(f"✅ Wrote (AAC {ch_out}): {out_file}")
            done += 1
        except subprocess.CalledProcessError as e2:
            print(f"❌ FFmpeg failed for {os.path.basename(norm_path)}\n{e2.stderr.decode(errors='ignore')[:800]}...")
            skipped += 1

    print(f"\nFinished. Created {done} files → {OUT_BASE}. Skipped {skipped}.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render one sample per Phonon voice, plus an 8 kHz mu-law copy of how it sounds over a phone line.

Uses the audited, pinned ptts CLI through uvx, so a mistyped package name can't be fetched.

    python scripts/phonon_voice_samples.py                      # English voices, default 31M model
    python scripts/phonon_voice_samples.py --model hf://gradium/phonon-40m-en
    python scripts/phonon_voice_samples.py --lang all --out samples-all
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PTTS = ["uvx", "ptts==0.4.1"]
DEFAULT_TEXT = (
    "It's really important that I speak with you as soon as possible about an urgent "
    "matter...please call me back at your earliest convenience! Thank you and have a great day!"
)


def list_voices() -> list[dict[str, str]]:
    """Return the voice catalog as dicts; ptts prints name, lang, id, description separated by tabs."""
    out = subprocess.run([*PTTS, "--list-voices"], capture_output=True, text=True, check=True).stdout
    voices = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 4:
            voices.append(dict(zip(("name", "lang", "id", "description"), parts)))
    return voices


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def to_phone_quality(src: Path, dst: Path) -> None:
    """Downsample to what a caller hears: 8 kHz mono G.711 mu-law, the format Twilio plays."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", str(dst)],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--lang", default="en", help="voice language to render (en, fr, de, es, pt, or all)")
    parser.add_argument("--model", help="ptts model, e.g. hf://gradium/phonon-40m-en (default: ptts's 31M English)")
    parser.add_argument("--only", help="comma-separated voice names to render, e.g. 'Zoey,Hadley'")
    parser.add_argument("--out", type=Path, default=Path.home() / "code" / "phonon")
    parser.add_argument("--no-phone", action="store_true", help="skip the 8 kHz mu-law copies")
    args = parser.parse_args()

    phone = not args.no_phone and shutil.which("ffmpeg") is not None
    if not args.no_phone and not phone:
        print("ffmpeg not found; skipping phone-quality copies", file=sys.stderr)

    voices = [v for v in list_voices() if args.lang == "all" or v["lang"] == args.lang]
    if args.only:
        wanted = {n.strip().lower() for n in args.only.split(",")}
        voices = [v for v in voices if v["name"].lower() in wanted]
    if not voices:
        print(f"no voices for language {args.lang!r}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for i, voice in enumerate(voices, 1):
        base = f"{voice['lang']}_{slug(voice['name'])}"
        wav = args.out / f"{base}.wav"
        print(f"[{i}/{len(voices)}] {voice['name']} ({voice['lang']})", flush=True)

        if not wav.exists():
            # Pass the catalog ID rather than the name: names contain spaces and may not be unique.
            cmd = [*PTTS, "--lang", voice["lang"], "--voice", voice["id"], "-o", str(wav)]
            if args.model:
                cmd += ["--model", args.model]
            start = time.monotonic()
            result = subprocess.run([*cmd, args.text], capture_output=True, text=True)
            if result.returncode != 0:
                failures.append(voice["name"])
                print(f"  failed: {result.stderr.strip().splitlines()[-1:]}", file=sys.stderr)
                continue
            print(f"  {time.monotonic() - start:.1f}s", flush=True)

        phone_wav = args.out / f"{base}_phone.wav"
        if phone and not phone_wav.exists():
            to_phone_quality(wav, phone_wav)
        rows.append({**voice, "file": wav.name, "phone_file": phone_wav.name if phone else ""})

    with open(args.out / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "lang", "id", "description", "file", "phone_file"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} samples in {args.out}/ (index.csv lists each voice's description)")
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

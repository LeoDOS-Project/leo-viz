#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/convert-pdf-slides-to-svg.sh [PDF] [OUT_DIR]

Defaults:
  PDF     assets/presentations/SpaceCoMP-BGITC-Stages.pdf
  OUT_DIR assets/presentations/spacecomp-primer

Converts a PDF slide deck to numbered SVG files matching the in-app
presentation convention:

  01.svg
  02.svg
  ...

The script also rewrites src/slides.rs so the deck metadata matches exactly
the generated slide count. If ImageMagick is available, large embedded PNG
images are re-encoded as JPEG to keep native/web preloading responsive.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

pdf="${1:-assets/presentations/SpaceCoMP-BGITC-Stages.pdf}"
out_dir="${2:-assets/presentations/spacecomp-primer}"
slides_rs="src/slides.rs"

if ! command -v pdftocairo >/dev/null 2>&1; then
  echo "error: pdftocairo not found. Install Poppler first." >&2
  exit 1
fi

if ! command -v pdfinfo >/dev/null 2>&1; then
  echo "error: pdfinfo not found. Install Poppler first." >&2
  exit 1
fi

if [[ ! -f "$pdf" ]]; then
  echo "error: PDF not found: $pdf" >&2
  exit 1
fi

if [[ ! -f "$slides_rs" ]]; then
  echo "error: run this script from the walker-delta repository root" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

mkdir -p "$out_dir"

page_count="$(pdfinfo "$pdf" | awk '/^Pages:/ { print $2 }')"
if [[ -z "$page_count" || ! "$page_count" =~ ^[0-9]+$ ]]; then
  echo "error: could not determine PDF page count" >&2
  exit 1
fi

find "$out_dir" -maxdepth 1 -type f -name '*.svg' -delete

for page in $(seq 1 "$page_count"); do
  printf -v out_file "%s/%02d.svg" "$out_dir" "$page"
  printf -v tmp_file "%s/slide-%02d.svg" "$tmp_dir" "$page"
  pdftocairo -svg -f "$page" -l "$page" "$pdf" "$tmp_file"
  mv "$tmp_file" "$out_file"
done

if command -v magick >/dev/null 2>&1; then
  python3 - "$out_dir" <<'PY'
from pathlib import Path
import base64
import re
import subprocess
import sys
import tempfile

root = Path(sys.argv[1])
pattern = re.compile(r"data:image/png;base64,([A-Za-z0-9+/=\n\r]+)")
threshold = 750_000
quality = "85"
changed = 0

for path in sorted(root.glob("*.svg")):
    text = path.read_text()
    replacements = []
    for match in pattern.finditer(text):
        b64 = "".join(match.group(1).split())
        raw = base64.b64decode(b64)
        if len(raw) < threshold:
            continue

        with tempfile.TemporaryDirectory() as td:
            png = Path(td) / "in.png"
            jpg = Path(td) / "out.jpg"
            png.write_bytes(raw)
            subprocess.run(
                [
                    "magick",
                    str(png),
                    "-auto-orient",
                    "-background",
                    "white",
                    "-alpha",
                    "remove",
                    "-alpha",
                    "off",
                    "-strip",
                    "-quality",
                    quality,
                    f"jpg:{jpg}",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            jpg_raw = jpg.read_bytes()

        if len(jpg_raw) >= len(raw) * 0.95:
            continue

        replacements.append(
            (
                match.start(),
                match.end(),
                "data:image/jpeg;base64,"
                + base64.b64encode(jpg_raw).decode("ascii"),
            )
        )

    if not replacements:
        continue

    out = []
    last = 0
    for start, end, repl in replacements:
        out.append(text[last:start])
        out.append(repl)
        last = end
    out.append(text[last:])
    path.write_text("".join(out))
    changed += 1

print(f"Optimized embedded raster images in {changed} SVG slides")
PY
else
  echo "Skipped SVG raster optimization: magick not found"
fi

python3 - "$slides_rs" "$page_count" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
count = int(sys.argv[2])
text = path.read_text()

pattern = re.compile(r"pub const SPACECOMP_PRIMER_SLIDE_COUNT: usize = \d+;")
new_text, n = pattern.subn(f"pub const SPACECOMP_PRIMER_SLIDE_COUNT: usize = {count};", text)
if n != 1:
    raise SystemExit("error: could not find SPACECOMP_PRIMER_SLIDE_COUNT in src/slides.rs")

path.write_text(new_text)
PY

echo "Generated $page_count SVG slides in $out_dir"
echo "Updated $slides_rs"

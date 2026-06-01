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

The script also rewrites src/slides.rs so the embedded deck includes exactly
the generated slide files.
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

python3 - "$slides_rs" "$page_count" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
count = int(sys.argv[2])
text = path.read_text()

replacement = "\n".join(
    f'    include_bytes!("../assets/presentations/spacecomp-primer/{i:02}.svg"),'
    for i in range(1, count + 1)
)

pattern = re.compile(
    r"const SPACECOMP_PRIMER_SLIDES: &\[\&\[u8\]\] = &\[\n"
    r".*?"
    r"\n\];",
    re.S,
)

new_text, n = pattern.subn(
    f"const SPACECOMP_PRIMER_SLIDES: &[&[u8]] = &[\n{replacement}\n];",
    text,
)
if n != 1:
    raise SystemExit("error: could not find SPACECOMP_PRIMER_SLIDES in src/slides.rs")

path.write_text(new_text)
PY

echo "Generated $page_count SVG slides in $out_dir"
echo "Updated $slides_rs"

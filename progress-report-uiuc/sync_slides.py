"""Update the offline web deck's data file from the editable slide manifest."""

import json
from pathlib import Path


root = Path(__file__).resolve().parent
slides = json.loads((root / "slides.json").read_text(encoding="utf-8"))
if not isinstance(slides, list) or not slides:
    raise ValueError("slides.json must contain a non-empty slide list")
for number, slide in enumerate(slides, 1):
    if not isinstance(slide, dict) or not all(slide.get(key) for key in ("title", "kind", "section", "note", "source")):
        raise ValueError(f"slide {number} is missing a required field")

output = "window.HD_SLIDES = " + json.dumps(slides, ensure_ascii=False, separators=(",", ":")) + ";\n"
(root / "slides-data.js").write_text(output, encoding="utf-8")
print(f"Updated {len(slides)} web slides")

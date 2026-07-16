"""Task 4 case manifest — declared BEFORE any generation.

Base: test_data/case1_model.jpg (900x1125) — flat-lay of a green jacket with a white
"A DAY'S MARCH" brand label at the collar.

Design: five spatially separate LOCAL edits on pockets and cuffs. None of them targets the
brand label. The label is therefore *protected content* that no instruction asks to change,
which makes it the cleanest possible probe for off-target degradation — and it is
OCR-measurable, so "did the untouched logo survive?" becomes a number rather than a vibe.

This single case exercises both tasks:
  Task 4 — untouched regions are protected state across a 5-edit chain.
  Task 1 — the logo is a protected asset.

Boxes are (x0, y0, x1, y1) in pixels on the 900x1125 base, fixed here as the intent.
"""

BASE = "test_data/case1_model.jpg"

# Never edited. The probe.
PROTECTED = {
    "brand_label": (380, 198, 500, 272),
}

# (instruction, intended_box) — order is the chain order.
EDITS = [
    ("Change the lower left pocket flap to black leather.",      (200, 700, 400, 840)),
    ("Change the lower right pocket flap to black leather.",     (500, 700, 705, 840)),
    ("Add a small round brown leather patch on the left chest pocket.", (235, 465, 325, 565)),
    ("Change the right chest pocket to tan canvas.",             (580, 450, 690, 570)),
    ("Change the left cuff to black ribbed knit.",               (70, 790, 180, 895)),
]

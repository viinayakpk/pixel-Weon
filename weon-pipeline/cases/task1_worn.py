"""Task 1, worn case: the identity manifest, declared BEFORE any generation.

Why this file exists separately from the experiment: an inventory written after seeing the output
is not an inventory, it is a rationalisation. The Task 1 product-shot experiment declared ONE logo
box on a product with TWO marks, and the certificate was structurally unable to notice the second
one going missing. This is that mistake's fix, and it must be timestamped before the run.

Why the sneaker and not the jacket: the general brief's product is model + environment + garment ->
the model WEARING the garment. The jacket's "A DAY'S MARCH" collar label is an INTERNAL label that
should normally be hidden once the jacket is worn — rewarding its visibility would reward an
incorrect result. The ARIGATO sneaker is externally visible when worn, carries two marks in two
material modes, and is the data the brief itself supplied for Task 1.

Framing: a clearly synthetic model from approximately knee down. The general brief permits
"clearly synthetic/licensed" people and forbids photos of real people without consent. Knee-down
keeps the shoe large enough in frame for the upper wordmark to be measurable at all — a full-body
shot would render it a few pixels wide and make every measurement meaningless.
"""

PACKSHOT = "case1_packshot.jpg"

# Every identity-bearing attribute we expect to survive the photoshoot, declared up front.
# `checkable` records HONESTLY whether this pipeline can verify it, rather than implying it can.
MANIFEST = [
    {"id": "upper_arigato", "expected": 1, "material_mode": "albedo (gold on brown suede)",
     "packshot_box": [793, 1309, 1091, 1448],
     "spelling": "ARIGATO", "letters": 7,
     "geometry": "fine-stroked serif, gold, small, on the lateral side below the lace holes",
     "checkable": "yes — stroke IoU vs the extracted asset, plus a spelling specialist"},

    {"id": "midsole_arigato", "expected": 1, "material_mode": "relief (debossed rubber, same colour)",
     "packshot_box": [200, 1660, 700, 1810],
     "spelling": "ARIGATO", "letters": 7,
     "geometry": "debossed into the dark midsole, lateral side, toward the heel",
     "checkable": ("presence by eye only. This pipeline cannot represent or repair a relief mark; "
                   "the earlier automatic presence detector scored a blank band higher than one "
                   "that visibly reads ARIGATO. Declared so its loss is visible, not because we "
                   "can score it.")},

    {"id": "panel_layout", "expected": 1, "material_mode": "colour blocking",
     "geometry": ("dark brown suede body; rust/orange suede overlays at the toe and heel; "
                  "white/cream heel tab"),
     "checkable": "qualitative only — no automatic panel check is implemented"},

    {"id": "sole_silhouette", "expected": 1, "material_mode": "structure + colour",
     "geometry": "dark brown midsole over a gum/tan outsole with a ridged tread strip",
     "checkable": "qualitative only"},

    {"id": "lacing", "expected": 1, "material_mode": "structure",
     "geometry": "dark brown flat laces, ~5 visible eyelet pairs, metal-tipped aglet",
     "checkable": "qualitative only — eyelet count by eye"},

    {"id": "stitching", "expected": 1, "material_mode": "structure",
     "geometry": "visible contrast stitching around the panel seams and the toe overlay",
     "checkable": "qualitative only"},
]

# Declared BEFORE generation so the run cannot be steered toward a flattering framing.
SCENE = ("Full-colour editorial fashion photograph, cropped from the knee down, of a person "
         "standing on a sunlit concrete pavement wearing plain straight-leg dark denim jeans and "
         "these exact sneakers. Natural daylight from the left, shallow depth of field, camera at "
         "ankle height, the shoes large and sharp in frame. Photographic, not illustrated.")

# The person must be unmistakably synthetic per the brief; no real person is used or implied.
SYNTHETIC_NOTE = (" The person is a synthetic 3D-rendered mannequin-like figure. Do not depict a "
                  "recognisable real individual. Only legs and footwear are in frame.")

# What "success" would mean, written down before we can be tempted to move it.
SUCCESS_CRITERIA = {
    "primary": ("Is the upper ARIGATO wordmark present, correctly spelled, and geometrically "
                "close to the packshot asset in a WORN shot?"),
    "secondary": "Is the midsole ARIGATO still present at all?",
    "mechanism_question": ("Does the deterministic protected-mark repair transfer from the "
                           "product shot to the worn shot, or does it break on the new pose, "
                           "scale and lighting?"),
    "hard_stop": ("If no usable worn baseline exists after 2 composition attempts or ~90 minutes, "
                  "stop and report the failure. Do not iterate toward a flattering sample."),
}

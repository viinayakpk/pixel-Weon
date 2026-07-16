# Can an automatic locator supply the write support?

Every support in this project was declared by hand. That is defensible for five known edits and
useless at scale, so this is the open question the whole approach depends on.

An earlier attempt asked *"where is the right chest pocket?"*, got a box back, and concluded the
locator worked. That conclusion was wrong twice over: the returned box landed near the **centre
placket**, with **zero pixel overlap** against *both* declared chest-pocket boxes and roughly
**6.2× their area** — and `box_2d` is a rectangle, so it was never a segmentation mask to begin
with. This experiment replaces that conclusion with a measurement.

## Protocol

- **Enumerate, don't confirm.** "Where is X?" presupposes X was found and gives the model no way to
  say it cannot see one. The prompt asks for *every* pocket, with no count supplied, so miscounting
  becomes visible.
- **Score both readings of "right".** On a flat-lay the *wearer's* right chest pocket appears on the
  **image-left**. The turn-4 instruction said "right chest pocket" and named neither convention.
  Both declared boxes are scored; a hit on either counts.
- **`box_2d` is a rectangle, not a mask.** Masks were requested separately.
- **Look before believing.** The overlay was rendered and inspected
  ([`locator_overlay.png`](locator_overlay.png)).

## Result: 2 of 4 pockets, and it does not say so

The jacket has **four** pockets. The locator returned **two**.

| returned | box (px) | IoU vs image-right chest | IoU vs image-left chest | mask |
|---|---|---:|---:|---|
| lower pocket on image-left | (197, 707, 388, 906) | 0.0000 (0 px) | 0.0000 (0 px) | yes |
| lower pocket on image-right | (485, 707, 684, 915) | 0.0000 (0 px) | 0.0000 (0 px) | yes |

Both **lower flap** pockets were found, with accurate segmentation masks. Both **chest welt**
pockets were omitted entirely — not mislocated, **absent from the response**.

The omission is verified, not assumed: both declared chest boxes contain a real, visible welt
opening with stitching ([`missed_pockets_verification.png`](missed_pockets_verification.png)). The
locator missed two real pockets. My declarations were not the fiction.

## Why this matters more than a mislocated box

The response was confident, well-formed JSON with correct labels and good masks for what it did
return. **Nothing in it signals incompleteness.** A pipeline consuming this output has no way to
learn that half the pockets are missing — the failure mode is *silent omission*, which is exactly
the class of failure this project exists to catch. Silence is not success.

It also explains the earlier placket answer. Forced to name a pocket it cannot see, the model
invents a location; allowed to enumerate, it omits. Those are the same underlying failure, and only
the enumerating form makes it legible.

The two it found are large and flapped; the two it missed are welts defined by a **seam line alone**
— no colour contrast, no flap, no cast shadow. That rhymes with the albedo-versus-relief probe
elsewhere in this project (features carried by geometry rather than colour are the ones that go
unrepresented), but two pockets is an observation, not evidence for a mechanism.

## Consequence: re-composition was NOT run

The plan was to re-composite the already-paid turn-4 candidate through a verified polygon — free,
no new generation call. It did not run, because **localization failed and the precondition was
never met.** Re-compositing through an unverified region would polish a mistake, which is the exact
error the collar ablation already identified: feathering a mis-grounded mask improves a seam
diagnostic while the support is still in the wrong place.

Reporting the blocked step is the result. The hand-declared supports stay hand-declared, and
"automatic grounding" remains the largest unsolved dependency of this approach rather than a
solved component.

## Limits

- One image, one grounder (`gemini-2.5-flash`), one prompt, one call. This is not a benchmark of
  Gemini's segmentation, and a different prompt or a stronger model may well enumerate all four.
- It establishes that *this* locator, on *this* garment, cannot be trusted to supply supports
  without human verification — which is enough to block the automation, and nothing more.

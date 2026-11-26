# Logo prompts

Two prompts live here:

- **A — for an image model** (Midjourney / Seedream / GPT-image / Flux), to explore
  constructions I could not get to work by hand.
- **B — for a coding LLM**, to edit the SVG geometry directly.

What exists already, and what "improve" means, is in `assets/logo/README.md`.
The short version: the mark is a geometric monoline **∂**; the open problem is
how to fold a **wave** into it without producing a *wobbling* ∂.

---

## 0. Before you paste: image models cannot read SVG

Rasterise first, then feed the PNG as a reference / image-to-image input.

```bash
rsvg-convert -w 1024 -h 1024 assets/logo/ctpwa-mark.svg -o /tmp/ctpwa-ref.png
rsvg-convert -w 1024 -h 768  assets/logo/ctpwa-logo.svg -o /tmp/ctpwa-lockup-ref.png
```

Attach **`ctpwa-ref.png`** (the mark) and, if the tool accepts more than one
reference, **`ctpwa-lockup-ref.png`** (the mark + wave lockup). Without a
reference image use the text-only variant in §2.

---

## A. Prompt for an image model

### 1. With the reference image attached

```
Redesign this logo mark. Keep its identity. Improve its craft.

WHAT IT IS
A monoline logo mark for "CTPWA", a CUDA + PyTorch toolkit for covariant-tensor
partial-wave analysis in particle physics. The figure is the mathematical partial
sign, U+2202: a small open loop at the upper left, a near-vertical back descending
on the right, and a wide bowl sweeping left along the bottom, the bowl's inner end
tucking back up into the back. One continuous stroke, uniform weight, round caps,
no fill, no text.

GOAL
Find a construction in which a WAVE is structural to this figure, so the mark reads
as "partial wave" without ever ceasing to read as the partial sign. Note the
difficulty honestly: a transverse sine needs a horizontal run and this letterform
has none, so merely rippling the existing stroke produces a quivering partial sign
rather than a wave. Look for the wave inside the geometry - in the sweep of the
bowl, in the descent of the back, in the aperture on the left, or in the way the
loop closes - rather than laid on top as decoration.

HARD CONSTRAINTS
- flat 2D vector; a single ink; no gradient, no 3D, no bevel, no shadow, no texture
- monoline: one uniform stroke weight throughout, round caps and joins
- must remain legible as the partial sign at 16 x 16 px
- must survive as pure black on white
- no lettering, no numerals, no text, no invented glyphs anywhere in the image
- square 1:1, optically centred, generous margin, plain white background

DESIGN LANGUAGE to borrow from, not to copy
The geometric economy of the Julia three-dot mark. The single negative-space cut of
the ROOT logo, which is one glyph carved out of a solid disc. The isometric lattice
discipline of TensorFlow. The flat confidence of the Scientific Python press kit.
Every one of them commits to one idea and one stroke, then stops.

DELIVER
Four clearly distinct constructions in a 2x2 grid, no captions and no labels.
They must differ in WHERE the wave lives, not in colour or ornament.

AVOID
atom with orbiting electrons, gears, brains, circuit traces, robots, rockets,
lightbulbs, shields, globes, a generic glowing blue orb, binary rain, lens flares,
decorative swirls, watercolour, hand-drawn wobble, fake text, watermark.
```

### 2. Text-only variant

Same prompt with the reference image dropped and the first paragraph replaced by:

```
Design a flat vector logo mark for "CTPWA", a CUDA + PyTorch toolkit for
covariant-tensor partial-wave analysis in particle physics. The mark must be the
mathematical partial sign, U+2202: a small open loop at the upper left, a
near-vertical back descending on the right, and a wide bowl sweeping left along the
bottom, its inner end tucking back up into the back - drawn as one continuous
monoline stroke with round caps.
```

…then append everything from `GOAL` onward unchanged.

### 3. Critique prompt (run on whatever comes back)

```
You are a logo critic, not a designer. Given this mark, answer in that order, and
do not soften any answer:

1. At 32 px, is it still a partial sign or has it become an abstract squiggle?
2. Converted to pure black on white, does it lose any information?
3. Is the wave a structural part of the figure, or has a sine simply been laid on
   top of a letterform? State which, and where the two meet.
4. Name the single most overused cliche in this image.
5. Does it contain any letter or numeral that was not intended? Point to it.
6. What one edit would most improve it, and what would that edit damage?
```

---

## B. Prompt for a coding LLM editing the SVG

Give it the current file contents (`assets/logo/ctpwa-mark.svg`) plus this:

```
You are editing a hand-authored SVG logo. Treat the existing geometry as the
source of truth and change as little as possible.

CURRENT MARK
viewBox 0 0 128 128, one <path>, fill="none", stroke="currentColor",
stroke-width 13, stroke-linecap/linejoin="round". Eight cubic segments, C1
continuous at every joint:
  M42 20 C47 11 62 8 74 13 C84 18 89 32 89 47 C89 66 89 84 81 97
  C73 110 58 115 46 113 C34 111 23 101 21 86 C19 71 25 60 34 56
  C46 51 64 53 78 59 C82 61 86 59 88 54

READ IT AS
the partial sign. Tip at (42,20). Top sweep right into a near-vertical back at
x=89, y=47..97. Bowl bottom around y=113. Left wall rising to y=86..56. Bowl top
edge running right from (34,56) to (78,59) and tucking into the back at (88,54).
The aperture is the gap on the left between the tip's descent and the back.

TASK
Produce 3 variants that make the figure read as "partial wave". For each, name
which part of the skeleton carries the wave and why that placement does not
destroy the letterform. A transverse sine needs a horizontal run; if you place one
where there is no horizontal run, say so and explain what you did instead.

CONSTRAINTS - all mandatory
- single <path> per variant where possible; no <text>, no gradient, no filter,
  no mask, no embedded raster, no external font, no script
- fill="none", stroke="currentColor", stroke-linecap/linejoin="round"
- keep the rendered stroke at >= 12 user units on the 128 box, or the mark dies
  at 16 px
- preserve C1 continuity at every joint; no corners in the middle of a stroke
- symmetric about the vertical axis is NOT required; optical centring is
- keep the bounding box within 8..120 on both axes

DELIVER, for each variant
1. the complete SVG file contents
2. one sentence naming the construction
3. the exact coordinates you changed from the list above
4. the shell commands you ran to rasterise it at 16, 32 and 512 px, and what you
   saw at 16 px
```

Verify anything it returns with:

```bash
rsvg-convert -w 16  variant.svg -o /tmp/v16.png
rsvg-convert -w 512 variant.svg -o /tmp/v512.png
```

---

## C. Pushing the covariant-tensor direction (hard, and worth knowing why)

Read the "Covariant tensor" section of `assets/logo/README.md` first — it records
what was already tried and rejected, so you do not pay for it twice. The short
diagnosis: *Partial* gave a character (∂) whose spoken name is the word itself.
*Covariant Tensor* gives no character at all — every tensor symbol is an expression
(a letter plus indices), and covariance is index *position*, a sub-baseline
typographic fact with no silhouette. Any picture you draw therefore lands in the
generic icon space, where collisions are the norm.

Attach the rasterised mark as reference and ask:

```
Explore a second logo element for "CTPWA" that expresses COVARIANT TENSOR, to sit
alongside the existing partial-sign mark.

THE EASY ANSWERS ARE ALREADY REJECTED
Do not return these without solving the problem each one has:
- a node with legs (tensor contraction diagram) - reads as a network or a molecule
- a fan of legs from a point (tensor rank) - reads as a fountain, a hand, a shell
- two parallel waves in phase ("co-variant" = varying together) - reads as an audio
  equaliser or a water mark
- a circle with a cross, the tensor-product sign - the most overloaded symbol in
  machine learning, and it means the product, not covariance
Each is more literal than the mark I already have and each is a worse logo, because
a picture competes with every other picture while a character competes with nothing.

WHAT A GOOD ANSWER MUST DO
Be unambiguous WITHOUT a caption. The test: show it alone, at 32 px, to someone who
knows the notation, and they should name the concept. If it needs a label it has
failed.

HARD CONSTRAINTS
- flat 2D vector, one ink, no gradient, no 3D, no shadow
- monoline, uniform stroke weight, round caps and joins
- legible at 16 px; a triangle or a circle survives, an open curve with a thin
  aperture does not
- no lettering, no numerals, no text, no subscript characters set as type
- square 1:1, optically centred - measure the ink box, do not eyeball it

DELIVER
Three constructions in a row, each with one sentence naming the concept it encodes
and one sentence stating honestly what it is ambiguous with. The second sentence is
the important one; a construction that cannot name its own collision has not been
thought through.
```

When something comes back, rasterise it for real rather than trusting a vector
preview, and measure the ink box rather than eyeballing the centring. The local
tools are `tmp/logo-recon/verify2.py` (ink box and margins of a shipped file) and
`tmp/logo-recon/compare.py` (true small-size rasters side by side).

---

## Acceptance checklist

Judge the output on these, in this order. Design taste does not enter until all
six pass.

1. **16 px** — still the partial sign, not a blob.
2. **Pure black** — no information lives in colour or layering.
3. **Distinct from the neighbours** — set it beside ROOT, Julia and TensorFlow at
   the same size; it must not read as any of them.
4. **The identifying stroke is untouched.** Every symbol is recognised by one
   *extra* mark, not by its body: a radical sign is the stroke **plus the
   vinculum** (measured from ROOT's own SVG — the bar at the upper right is what
   makes it a radical rather than a check); the ∂ is the stroke **plus the
   aperture**, and without the gap on the left it is just a ring. Locate that
   stroke first and say which one it is, then check it survived. Every failed
   attempt in this project deformed a body; none of them protected the identifying
   stroke, which is why all of them produced a quivering ∂.
5. **One sentence** — "the partial sign whose bowl is a wave" has to be sayable.
   If the construction needs a paragraph, it is decoration.
6. **The full set** — favicon, monochrome, light and dark, lockup. A mark without
   its variants is not finished.

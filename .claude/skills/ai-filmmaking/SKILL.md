---
name: ai-filmmaking
description: Generate production-grade prompts for AI filmmaking — character reference sheets, 9-panel cinematic storyboards, and Seedance 2.0 video shot prompts. Use this skill whenever the user wants to create an AI-generated film, short, music video, commercial, trailer, scene, or video sequence, or asks for prompts for image and video models like Nano Banana Pro, Midjourney, Flux, GPT Image 2, Seedance 2.0, Veo, or Runway. Also triggers when the user describes a character or story they want to render visually, asks for a storyboard or shot list, or mentions cinematic prompts, text-to-video, image-to-video, or character/style locking across shots. Use even when the user doesn't say the word "prompt" — if they're describing a video they want to make and need help structuring it, use this skill.
---

# AI Filmmaking

A toolkit for generating production-grade prompts across the AI filmmaking pipeline. It chains three templates together: a **Character Sheet** to lock identity, a **9-panel Storyboard Grid** to design a continuous scene, and **Seedance 2.0 Video Prompts** to render each shot.

The templates are deliberately genre-agnostic. The same skeleton produces photorealistic sci-fi, Pixar-style 3D, anime, music video, noir, documentary, influencer vlog, or stylized commercial work — genre is a swappable parameter, not a hardcoded assumption.

## Workflow

Most users want all three templates in sequence (character → storyboard → video). Some come in mid-pipeline ("I have a character sheet, just give me Seedance prompts for these panels"). Figure out where the user is and deliver what they need — don't ship all three when they only asked for one.

When a brief comes in:

1. **Identify the gaps in the creative spec.** Before generating anything, you need enough to write concrete copy. The minimum:
   - Genre / visual style (live-action cinematic? Pixar 3D? anime? music video? influencer vlog? photorealistic sci-fi?)
   - Lead character(s) — a sentence each on age range, build, wardrobe, key prop
   - Setting / location
   - The scene's emotional beat or story turn (what happens, in one sentence)
   - Aspect ratio (default `16:9` for cinema, `9:16` for vertical / music video / social)
   - Exclusions — anything that must NOT appear in frame

2. **Ask 2–3 sharp questions only if the brief is genuinely vague.** Don't fish for every field. If the user already wrote a paragraph of detail, just start generating. Only ask about what's load-bearing.

3. **Pick the template(s) the user actually needs.** Don't deliver a character sheet if they came in with character images already.

4. **Fill every bracket.** Don't ship a skeleton with `[BRACKETED]` placeholders. Replace every field with concrete, evocative copy. The whole point of this skill is to do the writing for them.

5. **Deliver in fenced code blocks.** These prompts get pasted straight into Nano Banana Pro, Midjourney, Seedance, etc. Clean, paste-ready text only — no commentary mixed inside the block.

## Character Lock — the most important rule

The single biggest cause of failure in AI filmmaking is character drift: the same person looking like a different person between shots. To prevent this:

**Reuse the same character description verbatim in every template that features that character.** Don't paraphrase between templates. Don't trust the model to "remember" between generations. The text description is the identity anchor.

When you generate the second and third templates, copy the character description forward unchanged. That continuity is what carries identity across the pipeline.

If reference images exist, reference them *and* keep the text description as a belt-and-braces backup. In Seedance 2.0 prompts, use `@image1`, `@image2`, etc. — see the numbering rule under Template 3.

## Keep prompts lean

A second, equally important rule: **prompts should be short and evocative, not exhaustive.** The model doesn't need a wardrobe spec sheet to lock a character — it needs the handful of visual details that make that character recognizable. Word clutter doesn't make outputs more accurate; it dilutes the signal and crowds the model's attention.

The same goes for panel beats and shot descriptions. Write like a screenwriter, not a novelist. Short, declarative phrases. `Viper lunges, palm strike to sternum` — not `Viper takes a sudden step forward and extends her right arm in a forceful palm-strike motion targeting Raven's chest area`.

This applies to **every template**. Even the Character Sheet — which is the identity anchor for the whole pipeline — should not exceed ~30–60 words of character description in description-only mode, and should add no prose description at all when a reference image is attached. See Template 1 for why scene contamination is worse than under-specification.

## Style consistency

The same palette, grain, lighting language, and lens descriptors should appear in every shot of a sequence. Inconsistent style descriptors are why worlds drift between generations as badly as characters do. If shot 1 establishes "35mm film grain, muted tones, golden-hour key light," that exact block belongs in shots 2 through 9.

Aspect ratio is part of the prompt — state it in every shot, not just the first.

---

## Template 1 — Character Sheet

**Use for:** Nano Banana Pro, GPT Image 2, Midjourney, Flux, or any high-fidelity image model. This sheet becomes the identity reference for every subsequent shot.

**Output:** an 8-shot grid — top row is four full-body views (front, side, three-quarter, back), bottom row is four matching close-ups.

The character sheet has two modes depending on whether the user provided a reference image. Pick the right one — using Mode B when an image is available is one of the biggest sources of word bloat in this skill.

### Two critical rules before you write anything

**1. If a reference image is attached, do not re-describe it in prose.** The image *is* the description. Adding a paragraph that re-narrates what the model can already see fights the visual anchor and dilutes attention. State the rendering style you want applied — that's it.

**2. Keep lighting neutral on the character sheet.** The sheet is a reference meant to be reused across many scenes. If you bake scene-specific lighting into it ("cold moonlight ambient with warm red flare rim-light"), you lock the character to one scene and contaminate every downstream shot. Scene lighting belongs in Template 2 and Template 3, not here. The sheet should be lit neutrally — clean, even, studio-style — so it serves any scene.

Also: always include `Background should be simple and not distracting from character design.` The point of the sheet is the character; busy backgrounds steal attention from the identity anchor.

### Mode A — Reference image(s) attached

Use when the user has provided photos or AI-generated reference shots of the character. The image carries the visual; you just specify rendering style.

```
Create a professional character reference sheet for attached character with {use attached image(s) as strong reference, 1:1 similarity, [STYLE DESCRIPTORS — e.g., photorealistic, live-action, lifelike]}. Divide the sheet into four different vertical columns, each representing a different angle, for a total of eight shots. The entire top row must show full-body views from head to toe facing four different directions: the front, the side, a three-quarter view, and the back. All subjects in the top row must be fully visible including feet, with no cropping at the ankles, knees, or head. The bottom row must contain four close-up shots of the face (including front and profile views) corresponding to each of the full-body shots above. The style must be [STYLE BLOCK — e.g., photorealistic, life-like live action shot on a DSLR camera with 35mm film and muted color tones; do not make it look like a 3D render]. Background should be simple and not distracting from character design. Aspect ratio = 16:9.
```

### Mode B — Description only (no reference image)

Use when the user has no reference photo and the character has to be specified entirely in prose. Even here, keep it tight: a comma-separated list of identity-locking traits, not a novelist's paragraph. **~30–60 words is the target; over 100 is a failure.**

Include only what makes the character recognizable: age range, build, skin, hair, eyes, the wardrobe in brief, and the key prop. Skip narrative atmosphere (dirt smudges from a backstory, damp hair from the rain in the scene, sparks streaming from the flare). Those are scene effects, not identity, and they belong in the shot prompts.

> Good (~55 words): `a woman in her late twenties, lean weather-hardened build, pale skin, watchful dark-brown eyes, dark brown shoulder-length wavy hair; wears a charcoal arctic parka with wolf-fur-lined hood, canvas harness straps over the chest, insulated cargo trousers, scuffed black snow boots; right hand grips a lit red emergency flare`
>
> Bloated (~155 words, scene contamination): `a woman in her late twenties, lean and weather-hardened build, pale wind-burned skin with a smudge of dirt and a faint blood streak across her right cheekbone, watchful dark-brown eyes, defined brows, slightly chapped lips, shoulder-length dark brown wavy hair partially tucked into her hood with a few damp strands framing her face. She wears a heavy charcoal-black arctic parka, oversized hood lined with thick brown-and-grey wolf fur pulled up around her head, weathered canvas-and-leather harness straps across the chest, a dark wool scarf bunched at the neck, layered thermal underlayers, insulated cargo trousers, and scuffed black snow boots. In her right hand she grips a lit red emergency road flare, crimson sparks and pink-red smoke streaming from the tip and casting warm red rim-light across her face and fur hood.`

The bloated version is what happens when the model treats the description as creative writing instead of identity locking. Sparks, blood streaks, damp hair, and rim-light cast onto the hood are all scene-specific — they fight identity reuse across shots.

```
Create a professional character reference sheet for [CHARACTER DESCRIPTION — tight comma-separated identity traits: age, build, skin, hair, eyes; wardrobe in brief; key prop. Target 30–60 words. No scene effects, no atmosphere, no narrative damage]. Divide the sheet into four different vertical columns, each representing a different angle, for a total of eight shots. The entire top row must show full-body views from head to toe facing four different directions: the front, the side, a three-quarter view, and the back. All subjects in the top row must be fully visible including feet, with no cropping at the ankles, knees, or head. The bottom row must contain four close-up shots of the face (including front and profile views) corresponding to each of the full-body shots above. The style must be [STYLE BLOCK — e.g., photorealistic, life-like live action shot on a DSLR camera with 35mm film and muted color tones; do not make it look like a 3D render]. Background should be simple and not distracting from character design. Aspect ratio = 16:9.
```

### Adapting the style block to genre

- **Photorealistic / live-action:** `photorealistic, life-like live action shot on a DSLR camera with 35mm film and muted color tones, do not make it look like a 3D render`
- **Pixar / 3D animation:** `stylized 3D render in the visual language of modern Pixar features, soft global illumination, expressive proportions, no photoreal rendering`
- **Anime / 2D:** `2D anime cel-shading in the style of [reference studio], clean line work, painterly backgrounds, no 3D render`
- **Noir:** `high-contrast black and white film, harsh chiaroscuro lighting, 35mm grain`
- **Influencer / iPhone vlog:** `natural daylight, iPhone selfie-camera aesthetic, soft skin tones, no cinematic grade`

The structure stays the same; the aesthetic descriptor swaps. Avoid descriptors that imply scene lighting (e.g. "moonlit," "firelit," "neon-soaked") — those belong in the shot prompts, not the character sheet.

---

## Template 2 — Cinematic Storyboard Grid

**Use for:** Nano Banana Pro, GPT Image 2, Midjourney, Flux, or any high-fidelity image model.

**Output:** A single image containing a 3×3 grid of 9 sequential panels depicting one continuous scene, with a production-notes strip under each panel.

The whole point of this template is to render a coherent moment in time across 9 beats with locked characters and locked geography. Treat the panels as one continuous take broken into sequential frames — not nine unrelated images.

### Filling the template, the right way

**Character descriptions: one tight sentence per character.** The full identity spec lives in the Character Sheet. Here you only need the handful of details that lock the look across panels — distinguishing features, hair, key clothing color, key prop. Don't repeat the wardrobe inventory.

> Good: `RAVEN: Mid-40s, lean Asian man, silver-grey spiky hair, sleeveless ochre kung-fu robe, gold arm cuffs, barefoot.`
>
> Bloated: `RAVEN: A man approximately 45 years of age with an athletic build of average height, featuring silver-grey spiky hair styled upward, wearing a traditional sleeveless ochre-colored kung-fu robe made of cotton with intricate gold arm cuffs and going barefoot on the wooden floor.`

**Panel beats: screenwriter-style, not novelist-style.** Each beat should be a tight slug line — what's in frame, who's where, what happens. Three to twelve words is the sweet spot. Don't narrate; direct.

> Good: `Panel 2 (top-center): Viper lunges, palm strike to sternum. Whip-pan with the motion.`
>
> Bloated: `Panel 2 (top-center): Viper takes a sudden aggressive step forward and extends her right arm forcefully in a palm-strike motion aimed at Raven's sternum while the camera whip-pans dynamically to follow the action.`

### Annotation strip format

Each panel's annotation strip carries three short lines in clean uppercase, formatted as screenplay slug lines. The first two lines are fixed; the third adapts to genre:

```
CAM:    [camera framing and movement]
MOVE:   [what the subject does — short, punchy]
MOOD:   [emotional / atmospheric beat]
```

The third line substitutes depending on what matters most for the genre:
- **MOOD** — default, for drama / dance / cinematic narrative
- **VOICE** — for vlog / influencer / dialogue-driven scenes (carries the spoken line)
- **STYLE** — for martial arts / action / fight choreography (carries the stance or technique)

Examples of strong annotation triplets, drawn from real reference boards:

```
CAM: SLOW LOW ORBIT. WIDE.
MOVE: STANDOFF. CHERRY PETALS DRIFT BETWEEN.
MOOD: TENSION. STILLNESS BEFORE STRIKE.
```

```
CAM: SELFIE CAM. CLOSE.
MOVE: HOLDS BOTTLE UP TO CAMERA.
VOICE: "OKAY GUYS, I FOUND IT."
```

```
CAM: WHIP-PAN WITH LUNGE. MEDIUM.
MOVE: VIPER LUNGES, PALM STRIKE TO STERNUM.
STYLE: VIPER — SILAT (CEKAK SILAT) / VIPER'S BITE.
```

```
CAM: RISING ORBIT. PUSH-IN.
MOVE: FULL-BODY EXTENSION. ARMS REACH SKY.
MOOD: AWAKENING. SOFT PIANO. BUILD.
```

Note the cadence: 2–6 words per line, often broken into short period-separated phrases. That rhythm is what makes them read like a director's notes instead of a paragraph.

**Readability matters.** Request the annotation text in a clean, high-contrast sans-serif font legible at the rendered grid size. The whole point of the strip is that the user can read it — bad legibility kills the board's usefulness as a production document.

### The template

```
Create a cinematic storyboard sheet in a 3x3 grid format (9 panels arranged in 3 rows x 3 columns) depicting ONE CONTINUOUS [SCENE TYPE] between [NUMBER] characters.

Style: Cinematic, [GENRE/REFERENCE TONE], [STYLE DESCRIPTORS — e.g., live-action, photorealistic, lifelike, 35mm film grain]. Aspect ratio = 16:9 page layout. No text, no captions, no panel numbers inside panels, only thin clean separators between panels. UNDER EACH panel a thin off-white annotation strip with three short lines of production notes in a clean, high-contrast sans-serif font (must be legible at rendered size): CAM (camera framing/movement), MOVE (subject action), and MOOD (atmosphere). Substitute VOICE for MOOD on vlog/dialogue scenes, or STYLE for MOOD on action/martial-arts scenes. Notes must read as short, declarative slug lines — not full sentences.

CHARACTER LOCK — all characters must appear IDENTICAL across all 9 panels (same face, same build, same clothing, same props). Use the descriptions below as the source of truth. If reference images are attached, treat them as additional identity anchors and match them precisely.

[CHARACTER A]: [one tight sentence — distinguishing features, hair, key clothing color, key prop].
[CHARACTER B]: [same — one tight sentence].
[Add CHARACTER C / D only if needed.]

This is a CONTINUOUS [SCENE TYPE] — one [encounter / moment / exchange], one location, one unbroken flow of time. Same [LOCATION & ATMOSPHERE]. [EXCLUSIONS — anything that must NOT appear].

Camera moves naturally around the action as if shot in a single continuous take broken into 9 sequential beats.

Narrative — [SEQUENCE NAME] (read left-to-right, top-to-bottom):
Panel 1 (top-left): [Beat 1 — short, declarative].
Panel 2 (top-center): [Beat 2].
Panel 3 (top-right): [Beat 3].
Panel 4 (middle-left): [Beat 4].
Panel 5 (middle-center): [Beat 5].
Panel 6 (middle-right): [Beat 6].
Panel 7 (bottom-left): [Beat 7].
Panel 8 (bottom-center): [Beat 8].
Panel 9 (bottom-right): [Beat 9].
```

**When writing the 9 beats, think like a director.** Each panel should advance the action by a meaningful unit of time *and* shift the camera framing — wide → medium → close-up → over-the-shoulder, etc. Nine variants of "they're talking from the same angle" wastes the grid. Vary the framing, vary the action, build to a climactic beat in panels 7–9.

---

## Template 3 — Seedance 2.0 Video Prompts

Seedance prompts come in **three variants** depending on what reference material the user has. Pick the one that matches the situation — don't default to Variant A if they already have a storyboard, because Variants B and C give much higher fidelity when a grid exists.

### `@image` numbering — read this first

When multiple reference images are involved, **each one gets a unique number**: `@image1`, `@image2`, `@image3`. Two character sheets are NOT both `@image1` — that collapses them into one reference and breaks identity lock. The convention:

- Character A's sheet → `@image1`
- Character B's sheet → `@image2`
- Storyboard grid (if used alongside character sheets) → next available number (`@image3` when there are two sheets, `@image1` when the grid is the only reference)

If you use the same number for two different references, characters will fuse together in the output. Get this right.

### Duration default — always 15 seconds

Seedance 2.0 caps at **15 seconds per generation**. Always default to the full 15 seconds — anything shorter wastes available runtime. A 10- or 12-second prompt leaves 3–5 seconds of paid generation unused, and the user can always trim in post if they don't need it.

Only use a shorter duration when the user **explicitly asks for one** (e.g. "make me a 6-second teaser", "5-second loop", "8 seconds for a TikTok intro"). Otherwise, fill the templates with `15 seconds` and let the TIMELINE cover the full `0:00–0:15` range.

### Variant A — Text-driven shots, optional character sheet references

Use when: the user has individual character sheets (or no references) and wants composable shot prompts they can iterate on freely. This is the right variant for text-to-video with light visual anchoring.

```
FORMAT: 15 seconds / [NUMBER] CUTS / [GENRE + TONE] / [AUDIO INSTRUCTION]

SUBJECT 1: [Physical description, defining traits, style]. [Reference @image1 if a sheet exists].
SUBJECT 2: [Optional — second subject or environmental force. Reference @image2 if a second sheet exists].

ENVIRONMENT: [Location, time of day, weather, lighting, atmosphere].

AUDIO / MOOD: [Music direction or absence]. [Key sounds, ambient layers, sonic textures].

TIMELINE (must cover full 0:00–0:15):
0:00–0:05: [Shot type or camera movement] — [What is in frame and what happens].
0:05–0:10: [Shot type or camera movement] — [What is in frame and what happens].
0:10–0:15: [Shot type or camera movement] — [What is in frame and what happens].
```

### Variant B — Storyboard grid as the main reference

Use when: the user has a 9-panel storyboard and wants the whole sequence rendered as one continuous video. The grid carries character identity, geography, framing, and pacing — your job is to give Seedance the through-line.

```
Use the provided cinematic storyboard grid @image1 as the main visual and motion reference. Create a 15-second cinematic sequence. Read the storyboard panels as sequential shots, not as one image. Follow the panel order, camera logic, motion arrows and camera framing consistently. Handheld camera moments to boost realism. NO TEXT ON SCREEN, NO MUSIC.

Storyline:
[Fill briefly from the storyboard — a compressed through-line, not a panel-by-panel novelization. The grid is the reference; the storyline is just the narrative spine.]
```

### Variant C — Character sheets + storyboard grid

Use when: the user has both character sheets AND a storyboard grid. This is the highest-fidelity setup — characters are locked by the sheets, motion and framing are locked by the grid.

```
Character 1: @image1
Character 2: @image2

Use the provided character sheets and cinematic storyboard grid @image3 as the main visual and motion reference. Create a 15-second cinematic sequence. Read the storyboard panels as sequential shots, not as one image. Follow the panel order, camera logic, motion arrows and camera framing consistently and temporally. NO TEXT ON SCREEN, NO MUSIC.

Storyline:
[Fill briefly from the storyboard.]
```

If there's only one character sheet, drop `Character 2` and renumber the grid to `@image2`. If there are three character sheets, they become `@image1`/`@image2`/`@image3` and the grid becomes `@image4`. The structure stays; only the numbers shift.

### Audio default: no music

Seedance prompts should **not** include music unless the user explicitly asks for it or specifies a track. Default the AUDIO line (Variant A) or include `NO MUSIC` (Variants B and C). Music baked in at generation time is hard to remove cleanly, so let the user score the cut themselves unless they say otherwise.

The exception is when the user actively asks for music or names a track / mood — then bake it in and follow their direction.

### Dialog scenes (primarily Variant A)

If the user wants spoken dialog, bake it into the TIMELINE in quoted brackets, with the speaker's emotion and micro-gestures fused into the line. For a second character's response, **always start with "She replies:" or "He replies:"** — the word "reply" signals consecutive-order speech to the video model and prevents speakers from collapsing into each other.

Example:

```
0:00–0:03: Medium shot — She says (excited, eyes wide): "This is the best gift I ever received!" (pointing at the box).
0:03–0:06: Reverse over-the-shoulder — He replies (quietly amused): "I wasn't sure you'd open it."
```

### Multi-shot sequences without a storyboard

If the user wants a multi-shot sequence and doesn't have a storyboard, generate one Variant A prompt per shot, keeping the SUBJECT, ENVIRONMENT, and AUDIO blocks identical across all prompts. Only the TIMELINE and camera framing should change shot-to-shot. (If they DO have a storyboard, Variant B or C is the better path — much less drift than chaining nine separate text prompts.)

---

## Delivery format

Deliver each prompt in its own fenced code block so the user can copy-paste cleanly. Any commentary, options, or follow-up questions live **outside** the code blocks. A prompt block should contain only the prompt — nothing else.

If you're delivering multiple prompts (e.g., one Seedance Variant A prompt per storyboard panel), label them clearly above each block:

> `### Panel 1 — Seedance prompt`
>
> ```
> ...prompt text...
> ```

The user will likely paste each one into a separate generation, so they need to tell them apart at a glance.

## Common pitfalls

- **Shipping `[BRACKETED]` placeholders.** If `[CHARACTER DESCRIPTION]` makes it into the final output, the skill has failed. Fill every field.
- **Bloating character descriptions in the Storyboard template.** One tight sentence per character, not three. Identity lives in the Character Sheet (image or short description); the Storyboard just needs a brief reminder.
- **Bloating the Character Sheet description when a reference image is attached.** The image *is* the description — don't re-narrate it. Mode A is short by design.
- **Baking scene effects into the Character Sheet.** Blood streaks, dirt smudges, damp hair, sparks streaming from the prop, scene-specific rim-light cast onto the character — all of these contaminate the identity reference. Lighting should be neutral; transient details belong in the shot prompts.
- **Narrating panel beats instead of directing them.** "Viper lunges, palm strike to sternum" — not a sentence with three subordinate clauses.
- **Annotation strips written as full sentences.** They should read like screenplay slug lines: short, period-separated, uppercase, 2–6 words per line.
- **Using MOOD when the scene calls for VOICE or STYLE.** Vlogs need the spoken line. Fight scenes need the stance. Pick the right third line for the genre.
- **Collapsing multiple references onto `@image1`.** Each reference gets its own number. Character A's sheet ≠ Character B's sheet ≠ storyboard grid.
- **Using Variant A when the user has a storyboard.** If a grid exists, Variant B or C is much higher fidelity than chaining text prompts.
- **Forgetting aspect ratio on shots 2–9.** State it in every prompt.
- **Adding music to Seedance prompts the user didn't ask for.** Default to ambient / `NO MUSIC`.
- **Defaulting Seedance duration below 15 seconds.** Seedance 2.0 maxes at 15s — anything shorter wastes paid runtime. Use 15s unless the user explicitly asked for less.
- **Editorializing inside the code block.** Notes go outside. The prompt is what gets pasted.

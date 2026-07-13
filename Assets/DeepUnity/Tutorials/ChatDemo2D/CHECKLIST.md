# ChatDemo2D — Orchestrator Checklist (Stardew-style farm rebuild)

A cozy Stardew-flavored mini farm built entirely from **Kenney CC0 "tiny" packs** (Tiny Farm /
Tiny Town / Tiny Dungeon + Kenney Fonts + Interface Sounds): a working farm loop (hoe → plant →
water → timed growth → harvest, 3 crops), a day/night tint cycle, a critter pen, and **two**
LLM-driven villagers (Gemma3-270M default / Qwen3.5-0.8B option) speaking through streaming
Chatterbox-Turbo TTS. Same plumbing quality bar as ChatDemo3D.

## State of the folder (done without Unity)

- **Archived earlier:** the disliked old demo lives untouched at `Tutorials/ChatDemo2D_OLD`
  (folder+meta shell-renamed, GUIDs preserved). `EditorBuildSettings.asset` had `m_Scenes: []` —
  no old-scene reference existed.
- **Pivot cleanup:** the interim "reused village art" iteration was rejected and its copied art
  (map/knight/tavern-owner/bird/plank/Cinzel) was **deleted** from `ChatDemo2D/Art`, along with
  the scripts that animated it (`DirectionalSpriteAnimator2D`, `SpriteLoop2D`, `AmbientBird2D`)
  and the stale `Generated/Cinzel2D SDF.asset`. `_OLD` still holds its own originals.
- **Now in the folder:** 58 semantically-named Kenney tiles in `Art/Tiles/` (fresh GUIDs),
  `Art/Fonts/Kenney Mini.ttf`, Kenney UI sounds, `Art/LICENSES.txt` (all CC0);
  11 runtime scripts in `Scripts/`; the farm scene builder in `Editor/ChatDemo2DBuilder.cs`.
- **Stale files that the builder overwrites:** `ChatDemo2D.unity` and parts of `Generated/`
  were produced by the pre-pivot builder run. Opening the scene BEFORE rebuilding shows missing
  scripts/sprites — expected; just build first.

## Step 1 — Build the scene

Deterministic + idempotent (imports configured in place, `Generated/` rebuilt or reused, scene
file rebuilt from scratch; ground layer is baked to `Generated/GroundBaked.png` — no Tilemap
package needed).

**Headless** (first run imports the new art + compiles scripts before executing):

```
"C:\Program Files\Unity\Hub\Editor\2022.3.43f1\Editor\Unity.exe" ^
  -projectPath "C:\dev\DeepUnity" -batchmode ^
  -executeMethod DeepUnity.Tutorials.ChatDemo2D.EditorTools.ChatDemo2DBuilder.BuildBatch ^
  -logFile "C:\dev\DeepUnity\chatdemo2d_build.log"
```

No `-quit` — `BuildBatch` exits itself (0 = OK; log tail `[ChatDemo2DBuilder] BATCH OK`).
**Or in the editor:** menu `DeepUnity/Tutorials/Build ChatDemo2D Scene`.

## Step 2 — Open `ChatDemo2D.unity` and play

### Farm loop (the core to verify)
1. **Scene:** 40x30-tile meadow — farmhouse (slate roof) top-left, general store (red roof) +
   sign top-right, well mid-path, fenced field (gate at the top) with 3x8 plot grid, critter pen
   with 2 hens/cow/sheep ambling around, tree border, decor. Player (farmer sprite) spawns on
   the farmhouse path.
2. **Movement:** WASD/arrows; hop animation while moving, X-flip on direction; fences, buildings,
   trees, well and map edges block.
3. **Toolbar (bottom-left), keys 1-6 or mouse wheel:** 1 hoe, 2 water bucket, 3-5 seed bags
   (carrot/turnip/tomato), 6 harvest. Selected slot frame turns gold.
4. **Targeting:** stand next to/inside the field facing a cell — a white outline highlights the
   targeted plot (only within reach). No highlight while the chat is open.
5. **Farm actions (Space or left click):**
   - Hoe an untilled cell → tilled soil patch appears (soft tick sound).
   - Seeds on tilled soil → sprout appears.
   - Water → soil turns dark/wet. **Growth only ticks while wet**; each stage advance dries the
     soil again (re-water after every growth spurt — Old Hobb explains this in character).
   - Stages: sprout → growing → ripe. Carrot ~22 s/stage, turnip ~32 s, tomato ~45 s (of watered
     time). Two waterings minimum per crop.
   - Harvest (6) on a ripe crop → counter (top-left of toolbar column) increments, plot returns
     to dry tilled soil, replantable.
6. **Day/night:** clock top-right ("Day 1  08:00", one day = 4 real minutes). World tint runs
   clear morning → amber dusk (~19:00) → deep blue night (~21:30-4:30) → dawn blush. UI is
   never tinted; crop growth is unaffected (ambience only, per approved scope).

### Chat NPCs (the plumbing to verify — mirrors ChatDemo3D)
7. **Old Hobb** (farmer sprite, nameplate) stands by the field gate; **Granny Marla** (granny
   sprite) outside the store. On scene start the selected LLM prewarms its kernels (a few early
   frame hitches are expected) and the shared Chatterbox TTS weight stream begins.
8. Walk within ~2 units → "[ E ] Talk" prompt. **E** opens the dialogue: movement locks, the two
   face each other, camera glides in (~0.85 s, ortho 5.5 → 3.4, pair framed above the panel),
   chat window slides up. Say button pulses `. . .` while the model loads (~2-3 s cold); typing
   is live immediately.
9. Ask something (Enter or Say): reply streams token-by-token AND is spoken clause-by-clause
   while still generating (2D audio, spatialBlend 0). Hobb = `conds_elder` voice, pitch 0.95,
   gruff farm-wisdom persona (he coaches the actual farm mechanics); Marla = `conds` voice,
   pitch 1.04, warm storekeeper persona. The talking NPC bobs faster while replying.
10. **Escape or Leave** closes at any time, even mid-reply (speech cuts; the model stays
    resident while you're inside the prefetch circle — walking out of it is what unloads).
    Both NPCs share one window — verify talking to one, leaving, then talking to the other
    stamps the right title/persona.
11. Farming input is dead while chatting (WASD types into the field; Space doesn't swing tools).

### Resolutions
12. 16:9 and ultrawide: chat panel + prompt are center-anchored fixed-size (scaler 1920x1080,
    match 0.5) — extra width just shows more farm; toolbar bottom-left, clock top-right, hint
    bottom-right stay in their corners.

## Known limitations / notes

- **Tiny-series characters are single-frame**: walk/talk animation is procedural (hop/bob +
  X-flip) by design — matches the pack's minimal style.
- **Critters** wander a pure-transform rect inside the pen (no physics); the player can walk
  into the pen through its gate and through critters — cosmetic only.
- **No inventory/economy** (approved scope): seeds are free, harvests just count up.
- **Gemma3-270M INT4 collapses** (known project-wide result) — keep INT8/FP16 (INT8 default for
  LLM and TTS).
- **TTS swap point:** `NPCInteractor2D.cs` carries the marked `TTS ENGINE SWAP POINT` block —
  when the CosyVoice port lands, swap the component construction in `Start()`; the
  FeedText/FlushText/StopSpeaking call sites are engine-agnostic.
- Both NPCs default to Gemma3-270M and therefore SHARE one pooled instance (LLMPool refcounts
  per model+quant — overlapping prefetch circles never double-load). Granny remembers her
  conversation (ContinueWhereLeftOff; the old KeepAliveInBackground mode was removed — residency
  belongs to the prefetch zone). The TTS engine is shared statically too.
- No background music (nothing CC0 staged); UI clicks/type ticks + farm-action ticks are wired.
- If the field/pen fence visuals ever need reshaping, edit the tile rects in
  `ChatDemo2DBuilder` (`DIRT_RECTS`, `FenceRect`, `TileSpan` calls) and re-run — colliders and
  visuals derive from the same tile coordinates.

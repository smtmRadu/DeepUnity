# AnyaChatDemo — ARCHIVED (2026-07-17)

A realistic-talking-head experiment: a full-GPU chat NPC (Qwen3.5-0.8B int8 + pocket-tts) rendered
as a portrait "video call" with a facially-rigged human. Archived after concluding that convincing
realism needs a modern head model (MetaHuman-in-Unity became legal June 2025) — the *motion* stack
built here is model-agnostic and carries over to any ARKit-rigged replacement.

## What works (and is reusable)

- **Video→face mocap pipeline**: any video of a face → MediaPipe FaceLandmarker → per-frame
  ARKit-52 blendshape weights + head pose → `Art/anya_idle_mocap.bytes` → replayed by
  `Scripts/AnyaMocapTrack.cs` (+`AnyaMocapIdle`). Real human idle motion (blinks, saccades, head
  sway) with intensity/head/smoothing knobs. 51/52 channels map 1:1 to the rig by name.
- `Scripts/AnyaLifeLayer.cs` — procedural idle fallback (deterministic, non-repeating).
- `Art/AnyaSkin.shader` — Built-in RP fake-SSS skin (wrap diffuse + subsurface terminator tint +
  pore detail normals + fresnel sheen).
- `Editor/AnyaChatDemoBuilder.cs` — builds both scenes (chat + face preview) from scratch.
- `Editor/AnyaFilmStrip.cs` — renders idle/mocap to MSAA frame sequences for review.

## Missing art (excluded from git — ~100 MB)

The Microsoft Rocketbox avatar (MIT) is NOT tracked. To restore, download into
`Art/Female_Adult_01/`:

- `Export/Female_Adult_01_facial.fbx` and `Textures/f001_*.tga` from
  `https://github.com/microsoft/Microsoft-Rocketbox` under
  `Assets/Avatars/Adults/Female_Adult_01/` (raw URLs work, e.g.
  `https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Female_Adult_01/Export/Female_Adult_01_facial.fbx`).
  Textures used: head/body color+normal+specular, head normal_wrinkle, opacity_color.
- Attribution/license: see `Art/NOTICE.md`.

Then run `DeepUnity/Anya/Build Chat Scene` or `Build Face-Preview Scene` to rebuild.

## Why archived

Rocketbox (2010) tops out at "game character" realism regardless of shading. The verified 2026
upgrade paths (researched, in priority order): MetaHuman→Unity (legal since June 2025, ARKit-52
extractable, realism ~9/10), 2D neural talking head on a quad (LivePortrait-class, photoreal,
fixed framing), Character Creator 5 (turnkey, ~7/10). NVIDIA Audio2Face-3D (open-source, outputs
ARKit-52 from audio) is the planned talking layer whenever this is revived.

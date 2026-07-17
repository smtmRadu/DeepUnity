# Anya — character asset attribution

The realistic character used in this demo ("Anya") is **Female_Adult_01** from the
**Microsoft Rocketbox Avatar Library**, released by Microsoft under the **MIT License**.

- Source: https://github.com/microsoft/Microsoft-Rocketbox
- License: MIT (see the repo's `LICENSE.md`)
- Files used here: `Export/Female_Adult_01_facial.fbx` (the facial-rig variant — 175 blendshape
  channels incl. ARKit-style visemes, used to drive lip-sync) and the head/body TGA textures
  (`Textures/f001_*`).

The facial FBX is the *facial* variant specifically because it carries the viseme/expression
blendshapes the lip-sync (uLipSync, driven by the pocket-tts audio) maps onto.

Only this one avatar was vendored (not the whole ~10 GB library). Swap in any other Rocketbox
`*_facial.fbx` the same way if you want a different face.

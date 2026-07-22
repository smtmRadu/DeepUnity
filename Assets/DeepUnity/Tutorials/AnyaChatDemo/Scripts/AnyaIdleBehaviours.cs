using UnityEngine;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    // The default behaviour set for Anya's camera-anchored idle. Each class is one drop-in
    // AnyaBehaviour unit; AnyaBehaviourIdle composes them. Evaluation order matters only for the
    // gaze chain: LookAway (shared glance) -> HeadMotion (head aim + share of glance) ->
    // CameraGaze (eyes compensate the head and carry the rest of the glance).

    /// <summary>
    /// PROCEDURAL breathing on the STATIC body — the only body motion after the body-animator
    /// round was reverted (no Animator, no clip stance). A slow sinusoid (same 0.25 Hz family as
    /// the head's breathing bob) lifts the clavicles a touch and pitches the chest a whisper.
    /// Applied REST-RELATIVE in character space every frame (local rotation reset to the cached
    /// rest first), so it never accumulates and the standing pose stays exactly the calibrated
    /// LowerArms rest. <see cref="Amount"/> is AnyaBehaviourIdle's <c>breathingAmount</c> dial:
    /// 1.0 ≈ the full clip-like breathing of the rejected body round (clavicle ±4°, chest ±1.5°),
    /// the shipped default 0.03 ≈ clavicle ±0.12° — a barely-there collar/shoulder life.
    /// Deterministic function of t like every other unit.
    /// </summary>
    public class AnyaBreathingBehaviour : AnyaBehaviour
    {
        public float Hz = 0.25f;              // calm resting breath
        public float Amount = 0.03f;          // set per-frame by AnyaBehaviourIdle (inspector dial)
        public float ChestPitchFull = 1.5f;   // deg at Amount = 1
        public float ClavicleLiftFull = 4f;   // deg at Amount = 1

        Transform chest, clavL, clavR;
        Quaternion chestRest, clavLRest, clavRRest;   // LOCAL rest rotations, cached at Init

        public override void Init(AnyaFaceRig rig)
        {
            chest = Find(rig.Root, "Bip01 Spine2");
            if (chest == null) chest = Find(rig.Root, "Bip01 Spine1");
            if (chest == null) chest = Find(rig.Root, "Bip01 Spine");
            clavL = Find(rig.Root, "Bip01 L Clavicle");
            clavR = Find(rig.Root, "Bip01 R Clavicle");
            if (chest != null) chestRest = chest.localRotation;
            if (clavL != null) clavLRest = clavL.localRotation;
            if (clavR != null) clavRRest = clavR.localRotation;
        }

        public override void Evaluate(AnyaFaceRig rig, in AnyaIdleFrame f)
        {
            float a = Mathf.Clamp01(Amount);
            if (a <= 0f) return;
            float s = Mathf.Sin(f.t * 2f * Mathf.PI * Hz);   // -1..1 breath cycle around the rest pose

            // chest first (the clavicles hang off it): inhale = a whisper of chest-up, about the
            // character's right axis (positive pitch = down, hence -s)
            if (chest != null)
            {
                chest.localRotation = chestRest;   // rest-relative: never accumulates
                chest.rotation = Quaternion.AngleAxis(-s * ChestPitchFull * a, rig.Root.right) * chest.rotation;
            }
            // clavicles: shoulder tips lift on the inhale — roll about the character's forward
            // axis, mirrored per side (AngleAxis(+90, fwd) maps +X to +Y: the -X/left tip needs
            // the negative angle to rise)
            if (clavL != null)
            {
                clavL.localRotation = clavLRest;
                clavL.rotation = Quaternion.AngleAxis(-s * ClavicleLiftFull * a, rig.Root.forward) * clavL.rotation;
            }
            if (clavR != null)
            {
                clavR.localRotation = clavRRest;
                clavR.rotation = Quaternion.AngleAxis(s * ClavicleLiftFull * a, rig.Root.forward) * clavR.rotation;
            }
        }

        static Transform Find(Transform root, string name)
        {
            foreach (var t in root.GetComponentsInChildren<Transform>(true))
                if (t.name == name) return t;
            return null;
        }
    }

    /// <summary>
    /// The "looking point" scheduler: most of the time the look point IS the camera (glance = 0).
    /// Every several seconds she picks a nearby off-camera point, glances at it, holds ~1 s and
    /// returns. Writes the SAME glance on two envelopes: the fast saccade one
    /// (<see cref="AnyaFaceRig.GlanceX"/>/<c>GlanceY</c>, for the EYES — ~80 ms, that speed is
    /// correct for eyes) and a slow eased one (<see cref="AnyaFaceRig.HeadGlanceX"/>/<c>HeadGlanceY</c>,
    /// for the head share — real head reorientation takes ~0.4-0.6 s; running the head on the
    /// saccade envelope is what read as a robotic snap). Fully suppressed while speaking.
    /// </summary>
    public class AnyaLookAwayBehaviour : AnyaBehaviour
    {
        public const int Seed = 4101;
        public const float MinGap = 5f, MaxGap = 11f;    // seconds between glances — "rarely"
        const float OutDur = 0.08f, BackDur = 0.12f;     // EYE saccade: out fast, return slightly slower
        const float HeadOut = 0.45f, HeadBack = 0.65f;   // HEAD share: slow, eased both ways

        public static float Hold(int idx) => 0.55f + AnyaFaceRig.Hash01(Seed + 77 + idx * 31) * 0.95f;

        public static Vector2 Offset(int idx)
        {
            float ang = AnyaFaceRig.Hash01(Seed + 11 + idx * 17) * Mathf.PI * 2f;
            float mag = 0.5f + AnyaFaceRig.Hash01(Seed + 23 + idx * 29) * 0.4f;
            return new Vector2(Mathf.Cos(ang), Mathf.Sin(ang) * 0.55f) * mag;   // less vertical range
        }

        public override void Evaluate(AnyaFaceRig rig, in AnyaIdleFrame f)
        {
            // "recalling" gaze while the LLM composes the reply: a sustained up-left look (her
            // left) with a slow wander — the classic memory-access glance filling the reply
            // latency. Rides the same glance channels, so eyes lead and the head eases after.
            if (f.think > 0.001f)
            {
                float wx = (Mathf.PerlinNoise(f.t * 0.30f, 31.7f) - 0.5f) * 0.22f;
                float wy = (Mathf.PerlinNoise(f.t * 0.27f, 37.1f) - 0.5f) * 0.18f;
                float ex = (0.55f + wx) * f.think;
                float ey = (0.40f + wy) * f.think;
                rig.GlanceX += ex; rig.GlanceY += ey;
                rig.HeadGlanceX += ex * 1.5f;   // the head joins the recall more than a normal glance
                rig.HeadGlanceY += ey * 1.5f;
            }

            AnyaFaceRig.EventAt(Seed, f.t, MinGap, MaxGap, out float ts, out int idx);
            if (idx < 0 || ts >= 900f) return;
            float hold = Hold(idx);
            float backStart = OutDur + hold;   // when the eyes start returning to the camera

            // eyes: saccade out, hold, saccade back
            float e;
            if (ts < OutDur) e = AnyaFaceRig.Smooth01(ts / OutDur);
            else if (ts < backStart) e = 1f;
            else e = 1f - AnyaFaceRig.Smooth01((ts - backStart) / BackDur);

            // head: same event, but eased reorientation — starts with the eyes, arrives much later,
            // and drifts back home more slowly than the eyes snap back
            float he;
            if (ts < HeadOut) he = AnyaFaceRig.Smooth01(ts / HeadOut);
            else if (ts < backStart) he = 1f;
            else he = 1f - AnyaFaceRig.Smooth01((ts - backStart) / HeadBack);

            if (e <= 0f && he <= 0f) return;
            // speaking OR thinking -> the scheduled random glances collapse (the camera lock owns
            // the gaze while speaking; the recall gaze owns it while thinking)
            Vector2 off = Offset(idx) * ((1f - f.speak) * (1f - f.think));
            rig.GlanceX += off.x * e;
            rig.GlanceY += off.y * e;
            rig.HeadGlanceX += off.x * he;
            rig.HeadGlanceY += off.y * he;
        }
    }

    /// <summary>
    /// Head: aims at the CAMERA as its base pose (camYaw/camPitch), plus gentle sway + breathing,
    /// a SMALL share of the current look-away glance, and occasional damped nods / subtle tilts.
    /// Roll and tilt are deliberately much weaker than the old life layer (no more random head
    /// incline); nods/tilts are silenced while speaking so the head stays on the camera.
    /// </summary>
    public class AnyaHeadMotionBehaviour : AnyaBehaviour
    {
        public float SwayPitch = 2.0f;     // full Perlin range in degrees (i.e. about ±1.0°)
        public float SwayYaw = 2.4f;
        public float SwayRoll = 1.0f;      // was 2.2 + big tilts in the old layer — heavily damped
        public float BreathPitch = 0.45f;  // breathing bob amplitude, deg
        public float BreathHz = 0.25f;
        public float GlanceHeadDeg = 2.4f; // head share of a look-away — eyes carry most of it
        public float NodAmp = 3.4f;        // was 5.5..8 — damped
        public float TiltAmp = 1.5f;       // was 4.5..7 — heavily damped
        public float LeanRange = 0.0045f;  // fore/aft translational drift, m (±~0.45 cm) — halved (user: leaning was too much)
        public float SpeakLeanIn = 0.003f; // extra lean toward the camera while speaking (engagement) — halved with it

        public override void Evaluate(AnyaFaceRig rig, in AnyaIdleFrame f)
        {
            // base: look straight at the camera — MINUS the torso's current contribution. The body
            // clip's stance bends the spine a little; without subtracting it, the head aims off-
            // lens by exactly that constant bias (which the eyes cannot fully hide); with it, the
            // head rides the breathing yet keeps pointing at the camera.
            float pitch = f.camPitch - rig.BodyPitch, yaw = f.camYaw - rig.BodyYaw, roll = 0f;

            // continuous idle sway (incommensurate Perlin) + breathing
            pitch += (Mathf.PerlinNoise(f.t * 0.13f, 0.0f) - 0.5f) * SwayPitch;
            yaw += (Mathf.PerlinNoise(f.t * 0.11f, 5.0f) - 0.5f) * SwayYaw;
            roll += (Mathf.PerlinNoise(f.t * 0.09f, 9.0f) - 0.5f) * SwayRoll;
            pitch += Mathf.Sin(f.t * 2f * Mathf.PI * BreathHz) * BreathPitch;

            // fore/aft translation — the rotational sway's complement ("not only sideways"):
            // a very slow Perlin drift toward/away from the camera, a whisper of breathing z,
            // and a tiny lean-IN while she speaks. Spring-smoothed in the rig apply.
            rig.Lean += (Mathf.PerlinNoise(f.t * 0.07f, 13.0f) - 0.5f) * 2f * LeanRange
                      + Mathf.Sin(f.t * 2f * Mathf.PI * BreathHz) * 0.0015f
                      + f.speak * SpeakLeanIn;

            // small head share of the look-away — on the SLOW eased envelope (HeadGlance), never the
            // eye-saccade one (same direction as the eyes; sign: +glanceX = her left = -yaw)
            yaw += -rig.HeadGlanceX * GlanceHeadDeg;
            pitch += -rig.HeadGlanceY * GlanceHeadDeg * 0.7f;

            // occasional NOD (chin dip + slight overshoot) or subtle curious TILT — never while speaking
            AnyaFaceRig.EventAt(9107, f.t, 6f, 13f, out float ts, out int idx);
            if (ts < 900f)
            {
                float quiet = 1f - f.speak;
                bool tilt = AnyaFaceRig.Hash01(9300 + idx) < 0.35f;
                if (!tilt && ts < 1.0f)
                {
                    float e = Mathf.Sin(Mathf.Clamp01(ts / 1.0f) * Mathf.PI);
                    float overshoot = ts > 0.7f ? -0.25f * Mathf.Sin((ts - 0.7f) / 0.3f * Mathf.PI) : 0f;
                    pitch += (e + overshoot) * (NodAmp + 1.6f * AnyaFaceRig.Hash01(9400 + idx)) * quiet;
                }
                else if (tilt && ts < 1.6f)
                {
                    float e = Mathf.Sin(Mathf.Clamp01(ts / 1.6f) * Mathf.PI);
                    float dir = AnyaFaceRig.Hash01(9500 + idx) < 0.5f ? 1f : -1f;
                    roll += e * dir * (TiltAmp + 0.9f * AnyaFaceRig.Hash01(9600 + idx)) * quiet;
                }
            }

            rig.Pitch += pitch; rig.Yaw += yaw; rig.Roll += roll;
        }
    }

    /// <summary>
    /// Eyes: the camera-anchored gaze. Whatever the head is doing (sway, breathing, nod), the eyes
    /// counter-rotate to stay ON the look point (VOR-like), so at rest and while speaking she looks
    /// straight into the camera. The shared glance offset is added on top — the eyes carry the bulk
    /// of every look-away. Fixational micro-tremor + slow drift keep the gaze from being glassy.
    /// Must run AFTER AnyaHeadMotionBehaviour (reads rig.Yaw/Pitch).
    /// </summary>
    public class AnyaCameraGazeBehaviour : AnyaBehaviour
    {
        public float EyeDegPerUnit = 11f;   // head-residual degrees that map to gaze 1.0

        int lookInL, lookInR, lookOutL, lookOutR, lookUpL, lookUpR, lookDownL, lookDownR;

        public override void Init(AnyaFaceRig rig)
        {
            lookInL = rig.Shape("EyeLookInLeft"); lookInR = rig.Shape("EyeLookInRight");
            lookOutL = rig.Shape("EyeLookOutLeft"); lookOutR = rig.Shape("EyeLookOutRight");
            lookUpL = rig.Shape("EyeLookUpLeft"); lookUpR = rig.Shape("EyeLookUpRight");
            lookDownL = rig.Shape("EyeLookDownLeft"); lookDownR = rig.Shape("EyeLookDownRight");
        }

        public override void Evaluate(AnyaFaceRig rig, in AnyaIdleFrame f)
        {
            // aim at the camera compensating the head pose the bone ACTUALLY has (post-spring,
            // last applied) — the eyes track the real head, so during a glance/nod the eyes move
            // first and the head eases after them (eyes lead, head follows).
            // residual yaw toward her right (+) -> gaze right = negative gx (gx>0 = her left)
            float gx = -(f.camYaw - rig.AppliedYaw) / EyeDegPerUnit;
            float gy = (rig.AppliedPitch - f.camPitch) / EyeDegPerUnit;

            // the look-away — the eyes carry MOST of the offset (head only took ~2°)
            gx += rig.GlanceX;
            gy += rig.GlanceY;

            // fixational micro-tremor + slow sub-degree drift so the eyes are never laser-locked
            gx += (Mathf.PerlinNoise(f.t * 11.3f, 4.1f) - 0.5f) * 0.05f
                + (Mathf.PerlinNoise(f.t * 0.4f, 17.3f) - 0.5f) * 0.06f;
            gy += (Mathf.PerlinNoise(f.t * 9.7f, 8.6f) - 0.5f) * 0.04f
                + (Mathf.PerlinNoise(f.t * 0.33f, 23.9f) - 0.5f) * 0.05f;

            gx = Mathf.Clamp(gx, -1.2f, 1.2f);
            gy = Mathf.Clamp(gy, -1.2f, 1.2f);

            const float H = 24f, V = 20f;   // max eye-look weight horizontal / vertical
            if (gx > 0) { rig.Add(lookOutL, gx * H); rig.Add(lookInR, gx * H); }   // her left
            else { rig.Add(lookInL, -gx * H); rig.Add(lookOutR, -gx * H); }        // her right
            if (gy > 0) { rig.Add(lookUpL, gy * V); rig.Add(lookUpR, gy * V); }
            else { rig.Add(lookDownL, -gy * V); rig.Add(lookDownR, -gy * V); }

            rig.GazeX = gx; rig.GazeY = gy;
        }
    }

    /// <summary>
    /// Blinks: irregular, faster close than open, slight L/R offset, a gaze-evoked blink on ~35%
    /// of look-away onsets, and a RARE double blink (~13%) — an occasional "she's being nice"
    /// tell, deliberately uncommon so it stays a charm rather than a tic. Never triples.
    /// </summary>
    public class AnyaBlinkBehaviour : AnyaBehaviour
    {
        int blinkL, blinkR;

        public override void Init(AnyaFaceRig rig)
        {
            blinkL = rig.Shape("EyeBlinkLeft"); blinkR = rig.Shape("EyeBlinkRight");
        }

        public override void Evaluate(AnyaFaceRig rig, in AnyaIdleFrame f)
        {
            AnyaFaceRig.EventAt(2207, f.t, 2.4f, 5.6f, out float ts, out int idx);
            float amt = BlinkCurve(ts);
            // rare double blink, second close a touch softer
            if (idx >= 0 && AnyaFaceRig.Hash01(3300 + idx) < 0.13f)
                amt = Mathf.Max(amt, BlinkCurve(ts - 0.16f) * 0.90f);

            // natural gaze-evoked blink on some look-away onsets
            AnyaFaceRig.EventAt(AnyaLookAwayBehaviour.Seed, f.t,
                                AnyaLookAwayBehaviour.MinGap, AnyaLookAwayBehaviour.MaxGap,
                                out float sts, out int sidx);
            if (sidx >= 0 && AnyaFaceRig.Hash01(4400 + sidx) < 0.35f)
                amt = Mathf.Max(amt, BlinkCurve(sts - 0.02f));

            float lead = AnyaFaceRig.Hash01(5500 + idx) * 0.02f;   // tiny L/R timing asymmetry
            rig.Add(blinkL, BlinkCurve(ts - lead) < amt ? amt * 100f : BlinkCurve(ts - lead) * 100f);
            rig.Add(blinkR, amt * 100f);
        }

        // faster close than open; 0 outside the blink window
        static float BlinkCurve(float ts)
        {
            const float dur = 0.14f;
            if (ts < 0f || ts > dur) return 0f;
            float x = ts / dur;
            float p = x < 0.4f ? (x / 0.4f) : 1f - (x - 0.4f) / 0.6f;
            return AnyaFaceRig.Smooth01(p);
        }
    }

    /// <summary>Periodic genuine (Duchenne) smiles: mouth corners + cheek raise + eye squint.</summary>
    public class AnyaSmileBehaviour : AnyaBehaviour
    {
        int smileL, smileR, cheekL, cheekR, squintL, squintR, browInner, dimpL, dimpR, jawOpen;

        public override void Init(AnyaFaceRig rig)
        {
            smileL = rig.Shape("MouthSmileLeft"); smileR = rig.Shape("MouthSmileRight");
            cheekL = rig.Shape("CheekSquintLeft"); cheekR = rig.Shape("CheekSquintRight");
            squintL = rig.Shape("EyeSquintLeft"); squintR = rig.Shape("EyeSquintRight");
            browInner = rig.Shape("BrowInnerUp");
            dimpL = rig.Shape("MouthDimpleLeft"); dimpR = rig.Shape("MouthDimpleRight");
            jawOpen = rig.Shape("JawOpen");
        }

        public override void Evaluate(AnyaFaceRig rig, in AnyaIdleFrame f)
        {
            AnyaFaceRig.EventAt(6101, f.t, 6f, 13f, out float ts, out int idx);
            if (ts > 3.2f || ts >= 900f) return;
            float env;
            if (ts < 0.55f) env = AnyaFaceRig.Smooth01(ts / 0.55f);        // bloom
            else if (ts < 2.05f) env = 1f;                                 // hold
            else env = 1f - AnyaFaceRig.Smooth01((ts - 2.05f) / 1.15f);    // fade
            float warmth = 0.7f + 0.3f * AnyaFaceRig.Hash01(6200 + idx);
            env *= warmth;
            // a text-conditioned talking smile is active (AnyaTalkingExpressionsBehaviour runs
            // earlier this frame) -> yield, so the two smile sources never stack unnaturally
            env *= 1f - rig.TalkSmile;
            if (env <= 0.001f) return;
            float asym = 0.88f + 0.12f * AnyaFaceRig.Hash01(6300 + idx);   // one corner leads

            rig.Add(smileL, env * 62f);
            rig.Add(smileR, env * 62f * asym);
            rig.Add(cheekL, env * 46f);
            rig.Add(cheekR, env * 46f * asym);
            rig.Add(squintL, env * 24f);                                   // Duchenne marker
            rig.Add(squintR, env * 24f * asym);
            rig.Add(browInner, env * 8f);
            rig.Add(dimpL, env * 12f);
            rig.Add(dimpR, env * 12f);
            rig.Add(jawOpen, env * 6f * AnyaFaceRig.Hash01(6400 + idx));   // occasional warm open smile
        }
    }

    /// <summary>Occasional attentive brow flash (quick up-down with slight eye widening).</summary>
    public class AnyaBrowFlashBehaviour : AnyaBehaviour
    {
        int browInner, browOutL, browOutR, wideL, wideR;

        public override void Init(AnyaFaceRig rig)
        {
            browInner = rig.Shape("BrowInnerUp");
            browOutL = rig.Shape("BrowOuterUpLeft"); browOutR = rig.Shape("BrowOuterUpRight");
            wideL = rig.Shape("EyeWideLeft"); wideR = rig.Shape("EyeWideRight");
        }

        public override void Evaluate(AnyaFaceRig rig, in AnyaIdleFrame f)
        {
            AnyaFaceRig.EventAt(7103, f.t, 3.6f, 8.5f, out float ts, out int idx);
            if (ts > 0.7f || ts >= 900f) return;
            float e = Mathf.Sin(Mathf.Clamp01(ts / 0.7f) * Mathf.PI);
            float amp = 12f + 10f * AnyaFaceRig.Hash01(7200 + idx);
            rig.Add(browInner, e * amp);
            rig.Add(browOutL, e * amp * 0.8f);
            rig.Add(browOutR, e * amp * 0.8f);
            rig.Add(wideL, e * amp * 0.35f);
            rig.Add(wideR, e * amp * 0.35f);
        }
    }

    /// <summary>
    /// Lips at rest: slightly parted with slow breathing variation, plus a rare lip-press/swallow
    /// micro-gesture. Only touches LipsPart (or a whisper of JawOpen as fallback) and MouthPress,
    /// so AnyaLipSync's channels (JawOpen/Funnel/Pucker/Stretch) stay free to blend on top.
    /// </summary>
    public class AnyaRestMouthBehaviour : AnyaBehaviour
    {
        int lipsPart, jawOpen, pressL, pressR;

        public override void Init(AnyaFaceRig rig)
        {
            lipsPart = rig.Shape("LipsPart");
            jawOpen = rig.Shape("JawOpen");
            pressL = rig.Shape("MouthPressLeft"); pressR = rig.Shape("MouthPressRight");
        }

        public override void Evaluate(AnyaFaceRig rig, in AnyaIdleFrame f)
        {
            float part = 2.2f + 1.4f * Mathf.PerlinNoise(f.t * 0.35f, 2.7f);
            if (lipsPart >= 0) rig.Add(lipsPart, part); else rig.Add(jawOpen, part * 0.6f);

            AnyaFaceRig.EventAt(8104, f.t, 7f, 16f, out float ts, out int idx);
            if (ts < 0.5f)
            {
                float e = Mathf.Sin(Mathf.Clamp01(ts / 0.5f) * Mathf.PI)
                        * (10f + 8f * AnyaFaceRig.Hash01(8200 + idx));
                rig.Add(pressL, e); rig.Add(pressR, e);
            }
        }
    }
}

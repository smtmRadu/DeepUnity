using UnityEngine;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    /// <summary>
    /// Runtime preview of Anya's procedural idle — NO LLM, NO text-to-speech. Press Play and she just
    /// sits there being alive: saccadic gaze, irregular blinks, head sway + occasional nods/tilts, and
    /// periodic genuine (Duchenne) smiles. All of it comes from <see cref="AnyaLifeLayer"/>, the exact
    /// same code the edit-mode filmstrip renders, so what you see here is what the filmstrip shows.
    /// The real demo layers TTS-driven lip-sync and LLM emotion on top of this same idle.
    /// </summary>
    public class AnyaFaceDemo : MonoBehaviour
    {
        readonly AnyaLifeLayer life = new AnyaLifeLayer();
        float t0;

        void Start()
        {
            var smr = GetComponentInChildren<SkinnedMeshRenderer>();
            if (smr == null || smr.sharedMesh == null) { enabled = false; return; }
            life.Init(smr);
            t0 = Time.time;
        }

        void LateUpdate()
        {
            // LateUpdate so our head-bone pose wins over Unity's animation/skinning pass this frame
            if (life.Ready) life.Evaluate(Time.time - t0);
        }
    }
}

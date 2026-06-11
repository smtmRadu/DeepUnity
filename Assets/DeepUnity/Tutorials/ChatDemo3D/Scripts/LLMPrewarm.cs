using System.Collections;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Driver-compiles every LLM compute kernel during the first seconds of the scene — one
    /// kernel per frame, but the big ones (DeltaNet) cost a visibly long single frame each.
    /// Running it here, right after scene load while the player is still orienting, moves that
    /// unavoidable hitch away from the first chat open mid-game (Qwen3_5Model.Warmup skips the
    /// compiles when they already ran — the static prewarm flag is per-session).
    /// This demo uses Qwen; a Gemma-based scene would call Gemma3Modeling.Gemma3Model.PrewarmKernels()
    /// the same way instead (don't prewarm a model the scene never runs — wasted frames).
    /// </summary>
    public class LLMPrewarm : MonoBehaviour
    {
        private IEnumerator Start()
        {
            yield return null;   // let the first scene frame present before spending budget
            LLM.CurrentPhase = "kernel-prewarm";
            var e = Qwen3_5Modeling.Qwen3_5Model.PrewarmKernels();
            while (e.MoveNext()) yield return e.Current;
            LLM.CurrentPhase = "idle";
        }
    }
}

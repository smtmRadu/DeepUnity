using System.Collections.Generic;

namespace DeepUnity
{
    /// <summary>
    /// Refcounted per-model instance pool: every consumer asking for the same
    /// (model id, weight quant, KV quant) gets THE SAME LLM instance — one weight stream, one
    /// VRAM copy. This is what prevents the double-load stutter when two NPCs share a model and
    /// their residency zones overlap (e.g. Velmire and the witch both on Qwen3.5-0.8B int8:
    /// walking from one to the other must NOT start a second 900 MB stream).
    ///
    /// Because the KV cache lives on the shared instance, the pool also tracks WHOSE
    /// conversation currently occupies it: a consumer only trusts its live KV (tier-a reuse)
    /// while it is still the owner — another NPC's InitializeChat steals ownership, and the
    /// previous owner falls back to its disk restore / transcript re-prefill path.
    /// </summary>
    public static class LLMPool
    {
        sealed class Slot { public LLM llm; public int refs; public object convOwner; }

        static readonly Dictionary<string, Slot> slots = new Dictionary<string, Slot>();
        static readonly Dictionary<LLM, string> keys = new Dictionary<LLM, string>();

#if UNITY_EDITOR
        // Statics survive when domain reload is disabled in Enter Play Mode options — never let
        // a released instance from the previous play session be handed out again.
        static LLMPool()
        {
            UnityEditor.EditorApplication.playModeStateChanged += s =>
            {
                if (s == UnityEditor.PlayModeStateChange.ExitingPlayMode)
                {
                    slots.Clear();
                    keys.Clear();
                }
            };
        }
#endif

        /// <summary>One shared instance per (id, quant, kvQuant); construction goes through
        /// <see cref="LLMRegistry.Create"/> on first acquire. Main-thread only.</summary>
        public static LLM Acquire(string id, LLMQuant quant, KVQuant kvQuant)
        {
            string key = $"{id}|{quant}|{kvQuant}";
            if (slots.TryGetValue(key, out var s) && s.llm != null)
            {
                s.refs++;
                return s.llm;
            }
            var llm = LLMRegistry.Create(id, quant, kvQuant);
            slots[key] = new Slot { llm = llm, refs = 1 };
            keys[llm] = key;
            return llm;
        }

        /// <summary>Drops one reference; the instance leaves the GPU IMMEDIATELY when the last
        /// holder lets go (leaving the prefetch zone = unload, no grace period — per the demo's
        /// residency contract). Instances not created through the pool are released directly.</summary>
        public static void Release(LLM llm)
        {
            if (llm == null) return;
            if (!keys.TryGetValue(llm, out string key)) { llm.Release(); return; }
            var s = slots[key];
            if (--s.refs > 0) return;
            slots.Remove(key);
            keys.Remove(llm);
            // Trickle-free on the dispatcher: unload STARTS now (zone exit = unload, no grace),
            // but the buffers free ~64 MB/frame so the driver never has to digest a monolithic
            // teardown right before the next model's allocations (measured 250-550 ms stalls).
            // If play mode exits mid-trickle the remainder is reclaimed by the domain reload.
            DeepUnityDispatcher.Run(llm.ReleaseSlow());
        }

        /// <summary>Marks <paramref name="owner"/>'s conversation as the one living in the
        /// shared instance's KV cache — call right after a successful InitializeChat/restore.</summary>
        public static void ClaimConversation(LLM llm, object owner)
        {
            if (llm != null && keys.TryGetValue(llm, out string key))
                slots[key].convOwner = owner;
        }

        /// <summary>True while <paramref name="owner"/>'s conversation still occupies the shared
        /// instance's KV (nobody re-initialized it since the claim). Non-pooled instances have a
        /// single user by definition, so they always report true.</summary>
        public static bool OwnsConversation(LLM llm, object owner)
        {
            if (llm == null) return false;
            if (!keys.TryGetValue(llm, out string key)) return true;
            return ReferenceEquals(slots[key].convOwner, owner);
        }
    }
}

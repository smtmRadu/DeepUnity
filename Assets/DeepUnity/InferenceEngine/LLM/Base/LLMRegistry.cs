using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;

namespace DeepUnity
{
    /// <summary>
    /// Marks a STATIC, parameterless method returning <see cref="LLMRegistry.Entry"/> as a
    /// self-registering LLM catalog entry. Put one next to each concrete model class — the
    /// registry discovers it by reflection, so a freshly ported LLM shows up in every
    /// model-picker (NPC inspector, tools) automatically, with no central enum to extend.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class LLMEntryAttribute : Attribute
    {
        /// <summary>Sort key in dropdowns (ties break alphabetically by id).</summary>
        public int Order { get; }
        public LLMEntryAttribute(int order = 0) { Order = order; }
    }

    /// <summary>
    /// Auto-discovered catalog of every chat-capable LLM in the engine. Entries are declared
    /// AT the model classes via <see cref="LLMEntryAttribute"/> and found by scanning the
    /// engine assembly once — game code addresses models by their stable string id
    /// (serialization-friendly: reordering or adding models never breaks scene data).
    /// </summary>
    public static class LLMRegistry
    {
        public sealed class Entry
        {
            /// <summary>Stable identifier — doubles as the inspector label ("Qwen3.5-0.8B").</summary>
            public string id;
            /// <summary>Builds the model: (weight quant, KV-cache quant, max context length) → instance.</summary>
            public Func<LLMQuant, KVQuant, int, LLM> create;
            /// <summary>Optional scene-start prewarm (kernel compiles + tokenizer parse).</summary>
            public Func<IEnumerator> prewarm;
            /// <summary>Dropdown ordering (from the attribute).</summary>
            public int order;
        }

        static List<Entry> entries;
        static string[] ids;

        public static IReadOnlyList<Entry> Entries { get { Scan(); return entries; } }
        public static string[] Ids { get { Scan(); return ids; } }

        public static Entry Find(string id)
        {
            Scan();
            foreach (var e in entries)
                if (e.id == id) return e;
            return null;
        }

        /// <summary>Builds the model registered under <paramref name="id"/>. An unknown id
        /// (typo, model removed) falls back to the first entry with a console warning instead
        /// of crashing the NPC.</summary>
        public static LLM Create(string id, LLMQuant quant, KVQuant kvQuant, int maxContextLength = 8192)
        {
            Scan();
            var e = Find(id);
            if (e == null)
            {
                if (entries.Count == 0)
                    throw new InvalidOperationException("LLMRegistry: no [LLMEntry] methods found in the engine assembly.");
                ConsoleMessage.Warning($"LLMRegistry: unknown model id '{id}' — falling back to '{entries[0].id}'.");
                e = entries[0];
            }
            return e.create(quant, kvQuant, maxContextLength);
        }

        // One reflection pass over the engine assembly (everything lives in Assembly-CSharp
        // alongside LLM). Static + NonPublic included so entries can sit as private helpers
        // right next to their model class.
        static void Scan()
        {
            if (entries != null) return;
            entries = new List<Entry>();
            foreach (var type in typeof(LLM).Assembly.GetTypes())
            {
                foreach (var m in type.GetMethods(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic))
                {
                    var attr = m.GetCustomAttribute<LLMEntryAttribute>();
                    if (attr == null) continue;
                    try
                    {
                        if (m.Invoke(null, null) is Entry e && !string.IsNullOrEmpty(e.id) && e.create != null)
                        {
                            e.order = attr.Order;
                            entries.Add(e);
                        }
                    }
                    catch (Exception ex)
                    {
                        ConsoleMessage.Warning($"LLMRegistry: entry {type.Name}.{m.Name} threw during scan: {ex.Message}");
                    }
                }
            }
            entries.Sort((a, b) => a.order != b.order
                ? a.order.CompareTo(b.order)
                : string.Compare(a.id, b.id, StringComparison.Ordinal));
            ids = new string[entries.Count];
            for (int i = 0; i < entries.Count; i++) ids[i] = entries[i].id;
        }
    }
}

using System;
using System.Collections.Generic;
using System.IO;

namespace DeepUnity
{
    namespace QwenASRModeling
    {
        // CPU-side tensor store over the exported weights folder (manifest.tsv + .bin), for the
        // pure-C# reference implementation (QwenASRCPU) and the net8.0 parity harness.
        // Pure C# (no UnityEngine). fp16 files are widened to float via a 65536-entry table;
        // tensors load lazily and are cached (0.6B fully touched ≈ 3.1 GB RAM — harness only).
        // The GPU path never uses this class (QwenASRWeights streams the same files instead).
        public class QwenASRTensors
        {
            public readonly string Root;
            readonly Dictionary<string, (string file, string dtype, int numel, int[] shape)> entries
                = new Dictionary<string, (string, string, int, int[])>();
            readonly Dictionary<string, float[]> cache = new Dictionary<string, float[]>();
            static float[] halfTable;

            public QwenASRTensors(string weightsDir)
            {
                Root = weightsDir;
                foreach (string line in File.ReadAllLines(Path.Combine(weightsDir, "manifest.tsv")))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    string[] p = line.Split('\t');
                    string[] sh = p[4].Split(',');
                    int[] shape = new int[sh.Length];
                    for (int i = 0; i < sh.Length; i++) shape[i] = int.Parse(sh[i]);
                    entries[p[0]] = (p[1], p[2], int.Parse(p[3]), shape);
                }
                if (halfTable == null)
                {
                    halfTable = new float[65536];
                    for (int i = 0; i < 65536; i++) halfTable[i] = HalfToFloat((ushort)i);
                }
            }

            public bool Has(string name) => entries.ContainsKey(name);
            public int[] Shape(string name) => entries[name].shape;

            /// <summary>Tensor as widened float[] (row-major). Cached after first load.</summary>
            public float[] F(string name)
            {
                if (cache.TryGetValue(name, out float[] hit)) return hit;
                if (!entries.TryGetValue(name, out var e))
                    throw new KeyNotFoundException($"QwenASR manifest has no tensor '{name}'.");
                if (e.dtype != "f16")
                    throw new NotSupportedException($"{name}: dtype {e.dtype} (harness runs the fp16 export).");
                byte[] raw = File.ReadAllBytes(Path.Combine(Root, e.file));
                float[] r = new float[e.numel];
                for (int i = 0; i < e.numel; i++)
                    r[i] = halfTable[(ushort)(raw[2 * i] | (raw[2 * i + 1] << 8))];
                cache[name] = r;
                return r;
            }

            /// <summary>Tied embedding/lm_head [151936, hidden] reassembled from the 16 fp16 shards.</summary>
            public float[] Embedding(int vocab, int hidden)
            {
                const string key = "dec/embed_tokens";
                if (cache.TryGetValue(key, out float[] hit)) return hit;
                float[] full = new float[(long)vocab * hidden];
                int rows = vocab / 16;
                for (int k = 0; k < 16; k++)
                {
                    float[] part = F($"dec/embed_tokens/part_{k}");
                    Array.Copy(part, 0, full, (long)k * rows * hidden, part.Length);
                    cache.Remove($"dec/embed_tokens/part_{k}");   // drop the shard copy
                }
                cache[key] = full;
                return full;
            }

            public void Evict(string name) => cache.Remove(name);

            // IEEE-754 half -> float (bit-exact widening; table-driven above).
            static float HalfToFloat(ushort h)
            {
                int sign = (h >> 15) & 1;
                int exp = (h >> 10) & 0x1F;
                int man = h & 0x3FF;
                int f;
                if (exp == 0)
                {
                    if (man == 0) f = sign << 31;
                    else
                    {
                        exp = 127 - 15 + 1;
                        while ((man & 0x400) == 0) { man <<= 1; exp--; }
                        man &= 0x3FF;
                        f = (sign << 31) | (exp << 23) | (man << 13);
                    }
                }
                else if (exp == 0x1F)
                    f = (sign << 31) | 0x7F800000 | (man << 13);
                else
                    f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
                return BitConverter.Int32BitsToSingle(f);
            }
        }
    }
}

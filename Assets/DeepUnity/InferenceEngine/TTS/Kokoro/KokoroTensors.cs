using System;
using System.Collections.Generic;
using System.IO;

namespace DeepUnity
{
    namespace KokoroModeling
    {
        // CPU-side reader for the Kokoro fp16 manifest export (weights_kokoro_fp16/manifest.tsv,
        // written by validation/import_kokoro.py in the ChatterboxWeights TSV schema).
        // Pure C# (no UnityEngine) so the dotnet parity harness and Unity share it.
        // Used by: KokoroCPU (fp32 reference/low-end backend + the runtime CPU biLSTMs).
        // The GPU streaming path reuses the generic ChatterboxWeights loader (same format).
        public class KokoroTensors
        {
            readonly Dictionary<string, (string file, int[] shape)> entries =
                new Dictionary<string, (string, int[])>(StringComparer.Ordinal);
            readonly Dictionary<string, float[]> cache = new Dictionary<string, float[]>(StringComparer.Ordinal);
            readonly string root;

            public KokoroTensors(string weightsDir)
            {
                root = weightsDir;
                string manifest = Path.Combine(weightsDir, "manifest.tsv");
                if (!File.Exists(manifest))
                    throw new FileNotFoundException(
                        $"manifest.tsv missing in '{weightsDir}' (run validation/import_kokoro.py).");
                foreach (string line in File.ReadAllLines(manifest))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    string[] p = line.Split('\t');       // name file dtype numel shape-csv
                    string[] sh = p[4].Split(',');
                    int[] shape = new int[sh.Length];
                    for (int i = 0; i < sh.Length; i++) shape[i] = int.Parse(sh[i]);
                    entries[p[0]] = (p[1], shape);
                }
            }

            public bool Has(string name) => entries.ContainsKey(name);
            public int[] Shape(string name) => entries[name].shape;

            /// <summary>fp16 .bin -> fp32 array (cached). Lock-guarded: KokoroModel runs CPU
            /// stages (LSTMs, NSF) on concurrent worker tasks that share this store.</summary>
            public float[] D(string name)
            {
                lock (cache)
                {
                    if (cache.TryGetValue(name, out float[] c)) return c;
                    if (!entries.TryGetValue(name, out var e))
                        throw new KeyNotFoundException($"Kokoro manifest has no tensor '{name}'.");
                    byte[] raw = File.ReadAllBytes(Path.Combine(root, e.file));
                    float[] r = new float[raw.Length / 2];
                    for (int i = 0; i < r.Length; i++)
                        r[i] = HalfToFloat((ushort)(raw[2 * i] | (raw[2 * i + 1] << 8)));
                    cache[name] = r;
                    return r;
                }
            }

            public static float HalfToFloat(ushort h)
            {
                int sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
                float s = sign == 1 ? -1f : 1f;
                if (exp == 0) return man == 0 ? sign * 0f : s * man * 5.9604645e-8f;      // subnormal: m*2^-24
                if (exp == 31) return man == 0 ? s * float.PositiveInfinity : float.NaN;
                return s * (1024 + man) * (float)Math.Pow(2, exp - 25);
            }

            /// <summary>Phoneme vocab from vocab.txt (line i = symbol for id i; NO trimming).</summary>
            public Dictionary<char, int> LoadVocab()
            {
                var v = new Dictionary<char, int>();
                string[] lines = File.ReadAllText(Path.Combine(root, "vocab.txt")).Split('\n');
                for (int i = 0; i < lines.Length; i++)
                    if (lines[i].Length == 1 && lines[i] != "$" && !v.ContainsKey(lines[i][0]))
                        v[lines[i][0]] = i;
                return v;
            }
        }
    }
}

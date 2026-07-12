using System;
using System.Collections.Generic;
using System.IO;

namespace DeepUnity
{
    namespace ParakeetModeling
    {
        // CPU-side reader for the Parakeet fp16 manifest export
        // (weights_parakeet_tdt_0.6b_{v2,v3}_fp16/manifest.tsv, written by
        // validation/import_parakeet.py in the ChatterboxWeights TSV schema).
        // Pure C# (no UnityEngine) so the dotnet parity harness and Unity share it.
        // Used by: ParakeetCPU (harness reference encoder + the runtime CPU TDT decode side:
        // dec/* and joint/head tensors). The GPU streaming path uses ParakeetWeights.
        public class ParakeetTensors
        {
            readonly Dictionary<string, (string file, int[] shape)> entries =
                new Dictionary<string, (string, int[])>(StringComparer.Ordinal);
            readonly Dictionary<string, float[]> cache = new Dictionary<string, float[]>(StringComparer.Ordinal);
            public readonly string Root;

            public ParakeetTensors(string weightsDir)
            {
                Root = weightsDir;
                string manifest = Path.Combine(weightsDir, "manifest.tsv");
                if (!File.Exists(manifest))
                    throw new FileNotFoundException(
                        $"manifest.tsv missing in '{weightsDir}' (run validation/import_parakeet.py).");
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

            /// <summary>fp16 .bin -> fp32 array (cached).</summary>
            public float[] D(string name)
            {
                if (cache.TryGetValue(name, out float[] c)) return c;
                if (!entries.TryGetValue(name, out var e))
                    throw new KeyNotFoundException($"Parakeet manifest has no tensor '{name}'.");
                byte[] raw = File.ReadAllBytes(Path.Combine(Root, e.file));
                float[] r = new float[raw.Length / 2];
                for (int i = 0; i < r.Length; i++)
                    r[i] = HalfToFloat((ushort)(raw[2 * i] | (raw[2 * i + 1] << 8)));
                cache[name] = r;
                return r;
            }

            /// <summary>Drop a cached tensor (the harness/runtime frees encoder tensors it
            /// no longer needs; the decode-side tensors stay cached for the app's lifetime).</summary>
            public void Evict(string name) => cache.Remove(name);

            public static float HalfToFloat(ushort h)
            {
                int sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
                float s = sign == 1 ? -1f : 1f;
                if (exp == 0) return man == 0 ? s * 0f : s * man * 5.9604645e-8f;   // subnormal m*2^-24
                if (exp == 31) return man == 0 ? s * float.PositiveInfinity : float.NaN;
                return s * (1f + man / 1024f) * Pow2(exp - 15);
            }

            static float Pow2(int e)
            {
                float r = 1f;
                for (int i = 0; i < (e < 0 ? -e : e); i++) r = e < 0 ? r * 0.5f : r * 2f;
                return r;
            }
        }
    }
}

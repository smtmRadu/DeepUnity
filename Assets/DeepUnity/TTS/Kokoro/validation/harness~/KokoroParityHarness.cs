#if !UNITY_5_3_OR_NEWER
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using DeepUnity.KokoroModeling;

// B1 harness (outside Unity — KokoroG2P/KokoroTensors/KokoroCPU are pure C#):
//   1. G2P gate: Phonemize(text_i) byte-equal to dump/t{i}_phonemes.txt + corpus sweep.
//   2. Model parity: KokoroCPU stages vs python dumps (fp16 weights vs fp32 ref -> corr/maxabs).
//      t0 additionally grades dec_x + wav with INJECTED rand_ini/sine_noise.
class Program
{
    const string KOKORO = "C:/dev/DeepUnity/Assets/DeepUnity/TTS/Kokoro";
    const string WEIGHTS = "C:/dev/DeepUnity/Assets/Resources/Weights/weights_kokoro_fp16";

    // ---------------- minimal .npy loader (v1, little-endian, C-order) ----------------
    static Array LoadNpy(string path, out int[] shape)
    {
        byte[] all = File.ReadAllBytes(path);
        int headerLen = BitConverter.ToUInt16(all, 8);
        int dataStart = 10 + headerLen;
        string header = Encoding.ASCII.GetString(all, 10, headerLen);
        bool fortran = header.Contains("'fortran_order': True");
        string shapeStr = header.Substring(header.IndexOf("'shape':", StringComparison.Ordinal) + 8);
        shapeStr = shapeStr.Substring(shapeStr.IndexOf('(') + 1);
        shapeStr = shapeStr.Substring(0, shapeStr.IndexOf(')'));
        var dims = new List<int>();
        foreach (string s in shapeStr.Split(','))
            if (!string.IsNullOrWhiteSpace(s)) dims.Add(int.Parse(s.Trim()));
        if (dims.Count == 0) dims.Add(1);
        shape = dims.ToArray();
        long count = 1; foreach (int d in shape) count *= d;
        if (header.Contains("f4"))
        {
            float[] r = new float[count];
            Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4);
            if (fortran && shape.Length >= 2)
            {   // dumps are [1,A,B]; F-order flat = B-major -> convert to C-order
                int A = shape[shape.Length - 2], B = shape[shape.Length - 1];
                float[] c = new float[count];
                for (int a = 0; a < A; a++)
                    for (int b2 = 0; b2 < B; b2++) c[a * B + b2] = r[b2 * A + a];
                return c;
            }
            return r;
        }
        long[] l = new long[count];
        Buffer.BlockCopy(all, dataStart, l, 0, (int)count * 8);
        return l;
    }

    static float[] F(string n) => (float[])LoadNpy($"{KOKORO}/validation/dump/{n}.npy", out _);
    static long[] I(string n) => (long[])LoadNpy($"{KOKORO}/validation/dump/{n}.npy", out _);

    static (double corr, double maxAbs) Diff(float[] a, float[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        double ma = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (int i = 0; i < n; i++)
        {
            double d = Math.Abs(a[i] - b[i]);
            if (d > ma) ma = d;
            sa += a[i]; sb += b[i]; saa += (double)a[i] * a[i];
            sbb += (double)b[i] * b[i]; sab += (double)a[i] * b[i];
        }
        double cov = sab / n - (sa / n) * (sb / n);
        double va = saa / n - (sa / n) * (sa / n), vb = sbb / n - (sb / n) * (sb / n);
        return (cov / Math.Sqrt(Math.Max(va * vb, 1e-30)), ma);
    }

    static int fails = 0;
    static void Grade(string name, float[] mine, float[] reference, double minCorr)
    {
        var (corr, ma) = Diff(mine, reference);
        bool ok = corr >= minCorr && mine.Length == reference.Length;
        if (!ok) fails++;
        Console.WriteLine($"  {name,-10} corr={corr:F6} maxabs={ma:E2} len {mine.Length}/{reference.Length} {(ok ? "OK" : "FAIL")}");
    }

    static int Main()
    {
        Console.OutputEncoding = Encoding.UTF8;
        var g2p = new DeepUnity.KokoroModeling.KokoroG2P(KOKORO + "/KokoroG2P");

        // ---------------- 1. G2P gates ----------------
        for (int i = 0; i < 3; i++)
        {
            string meta = File.ReadAllText($"{KOKORO}/validation/dump/t{i}_meta.json");
            string text = Regex.Match(meta, "\"text\": \"(.*?)\",\n").Groups[1].Value.Replace("\\\"", "\"");
            string expected = File.ReadAllText($"{KOKORO}/validation/dump/t{i}_phonemes.txt");
            bool ok = g2p.Phonemize(text) == expected;
            if (!ok) fails++;
            Console.WriteLine($"G2P t{i}: {(ok ? "EXACT MATCH" : "MISMATCH")}");
        }
        int cok = 0, ctotal = 0;
        foreach (string line in File.ReadAllLines($"{KOKORO}/validation/dump/g2p_corpus.tsv"))
        {
            string[] p = line.Split('\t');
            if (p.Length != 2) continue;
            ctotal++;
            string cgot = g2p.Phonemize(p[0]);
            if (cgot == p[1]) cok++;
            else Console.WriteLine($"MISMATCH: {p[0]}\n  exp: {p[1]}\n  got: {cgot}");
        }
        Console.WriteLine($"G2P corpus: {cok}/{ctotal} exact");
        if (cok != ctotal) fails++;

        // ---------------- 2. model parity ----------------
        var cpu = new KokoroCPU(new KokoroTensors(WEIGHTS));
        for (int i = 0; i < 3; i++)
        {
            Console.WriteLine($"--- t{i} stages ---");
            long[] idsL = I($"t{i}_input_ids");
            int[] ids = Array.ConvertAll(idsL, v => (int)v);
            float[] refS = F($"t{i}_ref_s");
            var sw = System.Diagnostics.Stopwatch.StartNew();

            Func<int, float[]> u01, n01;
            if (i == 0)
            {   // inject the python noise -> wav directly comparable
                u01 = n => F("t0_rand_ini");
                n01 = n => F("t0_sine_noise");
            }
            else
            {   // fresh RNG (no reference noise dumped for t1/t2 — stages only)
                var rng = new Random(1234);
                u01 = n => { var a = new float[n]; for (int j = 0; j < n; j++) a[j] = (float)rng.NextDouble(); return a; };
                n01 = n =>
                {
                    var a = new float[n];
                    for (int j = 0; j < n; j += 2)
                    {
                        double r1 = 1 - rng.NextDouble(), r2 = rng.NextDouble();
                        double m = Math.Sqrt(-2 * Math.Log(r1));
                        a[j] = (float)(m * Math.Cos(2 * Math.PI * r2));
                        if (j + 1 < n) a[j + 1] = (float)(m * Math.Sin(2 * Math.PI * r2));
                    }
                    return a;
                };
            }
            var S = cpu.Forward(ids, refS, 1f, u01, n01);
            sw.Stop();

            Grade("bert_dur", S.bertDur, F($"t{i}_bert_dur"), 0.999);
            Grade("d_en", S.dEn, F($"t{i}_d_en"), 0.999);
            Grade("d", S.d, F($"t{i}_d"), 0.999);
            Grade("duration", S.duration, F($"t{i}_duration"), 0.999);
            long[] pdRef = I($"t{i}_pred_dur");
            int pdDiff = 0;
            for (int t = 0; t < S.predDur.Length; t++) if (S.predDur[t] != pdRef[t]) pdDiff++;
            Console.WriteLine($"  pred_dur   {pdDiff} of {S.predDur.Length} tokens differ");
            if (pdDiff > 2) fails++;
            if (pdDiff > 0)
            {   // CHECKLIST B2-D provision: override with reference durations, regrade downstream
                Console.WriteLine("  (override with reference pred_dur for downstream stages)");
                S.predDur = Array.ConvertAll(pdRef, v => (int)v);
                float[] sd = new float[128], sp = new float[128];
                Array.Copy(refS, 0, sd, 0, 128);
                Array.Copy(refS, 128, sp, 0, 128);
                S = cpu.ForwardFromDurations(S, ids, sd, sp, u01, n01);
            }
            Grade("en", S.en, F($"t{i}_en"), 0.995);
            Grade("F0_pred", S.F0, F($"t{i}_F0_pred"), 0.995);
            Grade("N_pred", S.N, F($"t{i}_N_pred"), 0.995);
            Grade("t_en", S.tEn, F($"t{i}_t_en"), 0.999);
            Grade("asr", S.asr, F($"t{i}_asr"), 0.999);
            if (i == 0)
            {
                Grade("dec_x", S.decX, F("t0_dec_x"), 0.99);
                Grade("wav", S.wav, F("t0_wav"), 0.99);
            }
            Console.WriteLine($"  forward time: {sw.ElapsedMilliseconds} ms (audio {S.wav.Length / 24000f:F2}s)");
        }
        Console.WriteLine(fails == 0 ? "\nALL PASS" : $"\n{fails} FAILURES");
        return fails;
    }
}
#endif

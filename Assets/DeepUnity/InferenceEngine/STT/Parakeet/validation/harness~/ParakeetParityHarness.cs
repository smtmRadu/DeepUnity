#if !UNITY_5_3_OR_NEWER
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using DeepUnity.ParakeetModeling;

// E1 harness (outside Unity — ParakeetCPU/ParakeetTensors/ParakeetTokenizer are pure C#):
// grades every pipeline stage of the fp16-weight C# implementation against the fp32 HF
// reference dumps (validation/reference_dumps/{v3,v2}/<clip>/, from dump_reference.py),
// then checks the emitted token/duration sequences EXACTLY and the final transcripts.
// This validates all math the ParakeetCS.compute kernels mirror, before Unity is involved.
class Program
{
    const string ROOT = "C:/dev/DeepUnity/Assets/DeepUnity/InferenceEngine/STT/Parakeet";
    const string RES = "C:/dev/DeepUnity/Assets/Resources/Weights";
    static readonly string[] Clips = { "clip1_hello", "clip2_numbers", "clip3_game" };

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
            if (fortran && shape.Length == 2)
            {   // e.g. mel.npy (slice of a permuted torch tensor): F-order flat -> C-order
                int A = shape[0], B = shape[1];
                float[] c = new float[count];
                for (int a = 0; a < A; a++)
                    for (int b2 = 0; b2 < B; b2++) c[a * B + b2] = r[b2 * A + a];
                return c;
            }
            return r;
        }
        if (header.Contains("i4"))
        {
            int[] r = new int[count];
            Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4);
            return r;
        }
        throw new NotSupportedException($"npy dtype in {path}");
    }

    // ---------------- minimal 16-bit PCM mono wav reader ----------------
    static float[] LoadWav(string path)
    {
        byte[] b = File.ReadAllBytes(path);
        int pos = 12;
        while (pos < b.Length)
        {
            string id = Encoding.ASCII.GetString(b, pos, 4);
            int size = BitConverter.ToInt32(b, pos + 4);
            if (id == "data")
            {
                int n = size / 2;
                float[] s = new float[n];
                for (int i = 0; i < n; i++)
                    s[i] = BitConverter.ToInt16(b, pos + 8 + 2 * i) / 32768f;
                return s;
            }
            pos += 8 + size + (size & 1);
        }
        throw new InvalidDataException($"no data chunk in {path}");
    }

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
        Console.WriteLine($"  {name,-12} corr={corr:F6} maxabs={ma:E2} len {mine.Length}/{reference.Length} {(ok ? "OK" : "FAIL")}");
    }

    static void RunVariant(ParakeetVariant variant, string tag)
    {
        Console.WriteLine($"===== {tag} =====");
        string weights = $"{RES}/weights_parakeet_tdt_0.6b_{tag}_fp16";
        var cpu = new ParakeetCPU(new ParakeetTensors(weights), variant);
        var tok = new ParakeetTokenizer(weights);

        foreach (string clip in Clips)
        {
            string dump = $"{ROOT}/validation/reference_dumps/{tag}/{clip}";
            float[] F(string n) => (float[])LoadNpy($"{dump}/{n}.npy", out _);
            int[] I(string n) => (int[])LoadNpy($"{dump}/{n}.npy", out _);

            Console.WriteLine($"--- {clip} ---");
            float[] samples = LoadWav($"{ROOT}/validation/clips/{clip}.wav");
            var sw = System.Diagnostics.Stopwatch.StartNew();

            float[] mel = cpu.Mel(samples, out int tMel);
            Grade("mel", mel, F("mel"), 0.9995);

            float[] sub = cpu.Subsample(mel, tMel, out int tEnc);
            Grade("sub_out", sub, F("sub_out"), 0.999);

            Grade("pos_emb", cpu.PosEmb(tEnc), F("pos_emb"), 0.9999);

            float[] enc = cpu.Encoder(sub, tEnc, out float[] layer0);
            Grade("enc_layer0", layer0, F("enc_layer0"), 0.999);
            Grade("enc_out", enc, F("enc_out"), 0.99);

            float[] proj = cpu.EncProj(enc, tEnc);
            Grade("enc_proj", proj, F("enc_proj"), 0.99);

            var res = cpu.Decode(proj, tEnc);
            sw.Stop();

            // first-8 joint logits
            float[] refLogits = F("joint_logits_first8");
            int per = refLogits.Length / 8;
            float[] mineLogits = new float[Math.Min(res.FirstLogits.Count, 8) * per];
            for (int i = 0; i < res.FirstLogits.Count && i < 8; i++)
                Array.Copy(res.FirstLogits[i], 0, mineLogits, i * per, per);
            Grade("joint_first8", mineLogits, refLogits, 0.99);

            // exact sequence checks
            int[] refTok = I("tokens"), refDur = I("durations"), refFrm = I("frames");
            int tokDiff = res.Tokens.Count == refTok.Length ? 0 : 999;
            if (tokDiff == 0)
                for (int i = 0; i < refTok.Length; i++)
                    if (res.Tokens[i] != refTok[i] || res.Durs[i] != refDur[i] || res.Frames[i] != refFrm[i]) tokDiff++;
            if (tokDiff != 0) fails++;
            Console.WriteLine($"  tokens/durs  {(tokDiff == 0 ? $"EXACT ({refTok.Length} emissions)" : $"{tokDiff} DIFF (mine {res.Tokens.Count} vs ref {refTok.Length})")}");

            string transcript = tok.Decode(res.Tokens);
            string expected = File.ReadAllLines($"{dump}/transcript.txt")[0];
            bool tOk = transcript == expected;
            if (!tOk) fails++;
            Console.WriteLine($"  transcript   {(tOk ? "EXACT MATCH" : $"MISMATCH\n    exp: {expected}\n    got: {transcript}")}");
            Console.WriteLine($"  time: {sw.ElapsedMilliseconds} ms (clip {samples.Length / 16000f:F2}s, T_enc {tEnc}, {res.Steps} steps)");
        }
    }

    static int Main()
    {
        Console.OutputEncoding = Encoding.UTF8;
        RunVariant(ParakeetVariant.V3, "v3");
        RunVariant(ParakeetVariant.V2, "v2");
        Console.WriteLine(fails == 0 ? "\nALL PASS" : $"\n{fails} FAILURES");
        return fails;
    }
}
#endif

#if !UNITY_5_3_OR_NEWER
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using DeepUnity;
using DeepUnity.QwenASRModeling;

// D1 parity harness (outside Unity — QwenASRConfig/Tokenizer/Tensors/CPU are pure C#).
// Grades the pure-C# reference pipeline stage-by-stage against the D0 python dumps
// (validation/reference_dumps/<size>/<clip>/), fp16 weights vs fp32 reference → corr/maxabs.
// Stages: mel → prompt ids (exact) → enc_out → proj_out → logits_step0 (argmax must match)
// → greedy tokens (exact) → transcript (string equal). Exit code = number of failed gates.
//
//   cd validation/harness && dotnet run -c Release            (0.6b, all clips)
//   dotnet run -c Release -- 1.7b clip1_hello                  (size / clip filters)
class Program
{
    const string ROOT = "C:/dev/DeepUnity/Assets/DeepUnity/InferenceEngine/STT/QwenASR";
    const string RES  = "C:/dev/DeepUnity/Assets/Resources/Weights";

    // ---------------- minimal .npy loader (v1, little-endian, C-order) ----------------
    static Array LoadNpy(string path, out int[] shape)
    {
        byte[] all = File.ReadAllBytes(path);
        int headerLen = BitConverter.ToUInt16(all, 8);
        int dataStart = 10 + headerLen;
        string header = Encoding.ASCII.GetString(all, 10, headerLen);
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
            return r;
        }
        if (header.Contains("i4") || header.Contains("u4"))
        {
            int[] r = new int[count];
            Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4);
            return r;
        }
        long[] l = new long[count];
        Buffer.BlockCopy(all, dataStart, l, 0, (int)count * 8);
        return l;
    }

    static float[] LoadWav16kMono(string path)
    {
        byte[] b = File.ReadAllBytes(path);
        // minimal RIFF parse: find the 'data' chunk (fmt assumed 16-bit PCM mono 16 kHz — D0 clips)
        int pos = 12;
        while (pos < b.Length - 8)
        {
            string id = Encoding.ASCII.GetString(b, pos, 4);
            int size = BitConverter.ToInt32(b, pos + 4);
            if (id == "data")
            {
                int n = size / 2;
                float[] r = new float[n];
                for (int i = 0; i < n; i++)
                    r[i] = BitConverter.ToInt16(b, pos + 8 + 2 * i) / 32768f;
                return r;
            }
            pos += 8 + size + (size & 1);
        }
        throw new IOException($"no data chunk in {path}");
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
        Console.WriteLine($"  {name,-14} corr={corr:F6} maxabs={ma:E2} len {mine.Length}/{reference.Length} {(ok ? "OK" : "FAIL")}");
    }

    static void GradeExact(string name, IList<int> mine, int[] reference)
    {
        bool ok = mine.Count == reference.Length;
        if (ok) for (int i = 0; i < reference.Length; i++) if (mine[i] != reference[i]) { ok = false; break; }
        if (!ok) fails++;
        Console.WriteLine($"  {name,-14} len {mine.Count}/{reference.Length} {(ok ? "EXACT MATCH" : "MISMATCH")}");
        if (!ok && mine.Count > 0)
            Console.WriteLine($"    mine[0..8]: {string.Join(",", new List<int>(mine).GetRange(0, Math.Min(8, mine.Count)))}");
    }

    static int Main(string[] rawArgs)
    {
        Console.OutputEncoding = Encoding.UTF8;
        string size = Array.Find(rawArgs, a => a == "1.7b") ?? "0.6b";
        string clipFilter = Array.Find(rawArgs, a => a.StartsWith("clip"));

        QwenASRConfig.ApplySize(size == "1.7b" ? QwenASRSize.B1_7 : QwenASRSize.B0_6);
        string weights = $"{RES}/weights_qwen3asr_{size}_fp16";
        var tensors = new QwenASRTensors(weights);
        var tok = new QwenASRTokenizer(Path.Combine(weights, "tokenizer"));

        string dumpRoot = $"{ROOT}/validation/reference_dumps/{size}";
        foreach (string clipDir in Directory.GetDirectories(dumpRoot))
        {
            string clip = Path.GetFileName(clipDir);
            if (clipFilter != null && clip != clipFilter) continue;
            Console.WriteLine($"--- {size} / {clip} ---");
            var sw = System.Diagnostics.Stopwatch.StartNew();

            var cpu = new QwenASRCPU(tensors);
            float[] wav = LoadWav16kMono($"{ROOT}/validation/clips/{clip}.wav");

            // §1 mel
            float[] mel = cpu.Mel(wav, out int validFrames, out int paddedFrames);
            float[] melRef = (float[])LoadNpy($"{clipDir}/mel.npy", out int[] melShape);
            Grade("mel", mel, melRef, 0.9999);

            // §5 prompt ids (dump used language=None → empty system)
            int nAudio = QwenASRConfig.AudioTokenCount(validFrames);
            int[] promptIds = QwenASRCPU.BuildPromptIds(tok, nAudio);
            int[] idsRef = Array.ConvertAll((long[])LoadNpy($"{clipDir}/input_ids.npy", out _), v => (int)v);
            GradeExact("prompt_ids", promptIds, idsRef);

            // §2 encoder
            float[] encOut = cpu.Encode(mel, validFrames, paddedFrames, out int nTok);
            float[] encRef = (float[])LoadNpy($"{clipDir}/enc_out.npy", out _);
            Grade("enc_out", encOut, encRef, 0.999);

            // §3 projector
            float[] projOut = cpu.Project(encOut, nTok);
            float[] projRef = (float[])LoadNpy($"{clipDir}/proj_out.npy", out _);
            Grade("proj_out", projOut, projRef, 0.999);

            // §4 first-step logits + §6 greedy
            float[] logits = cpu.PrefillLogits(promptIds, projOut);
            float[] logitsRef = (float[])LoadNpy($"{clipDir}/logits_step0.npy", out _);
            Grade("logits_step0", logits, logitsRef, 0.999);
            int amMine = 0; for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[amMine]) amMine = i;
            int amRef = 0; for (int i = 1; i < logitsRef.Length; i++) if (logitsRef[i] > logitsRef[amRef]) amRef = i;
            bool amOk = amMine == amRef;
            if (!amOk) fails++;
            Console.WriteLine($"  {"step0_argmax",-14} mine={amMine} ref={amRef} {(amOk ? "OK" : "FAIL")}");

            var greedy = cpu.Greedy(promptIds, projOut);
            int[] tokRef = Array.ConvertAll((long[])LoadNpy($"{clipDir}/tokens_greedy.npy", out _), v => (int)v);
            GradeExact("tokens_greedy", greedy, tokRef);

            string transcript = QwenASRCPU.ParseTranscript(tok, greedy);
            string transRef = File.ReadAllText($"{clipDir}/transcript.txt").Trim();
            bool tOk = transcript == transRef;
            if (!tOk) fails++;
            Console.WriteLine($"  {"transcript",-14} {(tOk ? "EXACT MATCH" : "MISMATCH")}  \"{transcript}\"");
            if (!tOk) Console.WriteLine($"    expected: \"{transRef}\"");
            Console.WriteLine($"  ({sw.Elapsed.TotalSeconds:F1}s)");
        }

        Console.WriteLine(fails == 0 ? "\nALL GATES PASS" : $"\n{fails} GATE(S) FAILED");
        return fails;
    }
}
#endif

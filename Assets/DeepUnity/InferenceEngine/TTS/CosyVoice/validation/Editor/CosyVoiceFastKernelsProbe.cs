using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace CosyVoiceModeling
    {
        // #31 fast-kernel parity + A/B probes — cosyvoice-deepopt (DEEPOPT.md §6).
        // Gate structure follows GemmaCpmGemvParityProbe / the PocketTTS FastKernels rounds:
        // isolated tight gate (fast arm vs legacy arm, same weights, same run) + full-path
        // gates vs the Python reference dumps + same-run A/B timing. Editor-mode synchronous
        // (ClaudeBridge invoke / -executeMethod batch statics). Covers:
        //   RunLmPrefill[Int8]  CosyVoiceLM.FastPrefill  (QKV/O/GateUp/Down GemmCoal + head)
        //   RunFlow[Int8]       CosyVoiceFlow.FastDit31  (AdaLNStats/DitQKVCoal/DitLinearCoal/
        //                                                 RopeQKPair/PackEstIn)
        //   RunHift             HiFTVocoder.FastConv     (Conv1DTileTC + fused prologues)
        //   RunChain            all-fast vs all-legacy flow->HiFT (deterministic, no sampler)
        // Plus LEGACY menu twins that re-run the existing A1/A2/A3 dump probes with the new
        // flags OFF — proves the fallback paths survived (checklist item 8).
        public static class CosyVoiceFastKernelsProbe
        {
            const string DUMP_DIR = "Assets/DeepUnity/InferenceEngine/TTS/CosyVoice/validation/dump";
            const string FP16_DIR = "Assets/Resources/Weights/weights_cosyvoice3_fp16";
            const string INT8_DIR = "Assets/Resources/Weights/weights_cosyvoice3_int8";

            static readonly StringBuilder report = new StringBuilder();
            static bool failed;

            static void Log(string line)
            {
                report.AppendLine(line);
                Debug.Log("[CosyVoiceFastKernels] " + line);
            }

            static void Gate(bool pass, string what)
            {
                if (!pass) { failed = true; Log($"   FAIL ({what})"); }
            }

            static void Finish(string name)
            {
                Log("");
                Log(failed ? "## RESULT: FAIL" : "## RESULT: PASS");
                File.WriteAllText($"ProbeLogs/cosyvoice_fastkernels_{name}.md", report.ToString());
                File.WriteAllText($"ProbeLogs/cosyvoice_fastkernels_{name}.done", failed ? "FAIL" : "PASS");
            }

            // ============================ LM prefill (lever 3) ============================

            [MenuItem("DeepUnity/CosyVoice/#31 LM Prefill Parity + A-B")]
            public static void RunLmPrefill() => LmCore(FP16_DIR);

            [MenuItem("DeepUnity/CosyVoice/#31 LM Prefill Parity + A-B INT8")]
            public static void RunLmPrefillInt8() => LmCore(INT8_DIR);

            const int LM_GREEDY_STEPS = 8;

            static void LmCore(string weightsDir)
            {
                report.Clear(); failed = false;
                Directory.CreateDirectory("ProbeLogs");
                CosyVoiceWeights weights = null;
                bool prevFast = CosyVoiceLM.FastPrefill;
                try
                {
                    Log($"# CosyVoice3 #31 — LM prefill GemmCoal parity + A/B — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    Log($"weights: {weightsDir} (FastLM ON in both arms; only FastPrefill flips)");

                    int[] promptText = Ints("prompt_text_tokens");
                    int[] text = Ints("text_tokens");
                    int[] promptSpeech = Ints("prompt_speech_tokens");
                    int[] textFull = new int[promptText.Length + text.Length];
                    promptText.CopyTo(textFull, 0);
                    text.CopyTo(textFull, promptText.Length);

                    weights = new CosyVoiceWeights(weightsDir, beginLoad: false);
                    weights.LoadBlocking("llm/");
                    Log($"llm/* resident; prefill length = {2 + textFull.Length + promptSpeech.Length} tokens");

                    var (logitsA, toksA, stepsA, msA) = LmArm(weights, fast: false, textFull, promptSpeech);
                    var (logitsB, toksB, stepsB, msB) = LmArm(weights, fast: true, textFull, promptSpeech);

                    // step-0 logits: rel maxAbs <= 2e-3, corr >= 0.999999, argmax MATCH
                    var (dMax, _, dCorr) = Diff(logitsB, logitsA);
                    float refMag = MaxAbs(logitsA);
                    float relMax = dMax / Mathf.Max(refMag, 1e-9f);
                    int argA = ArgMax(logitsA), argB = ArgMax(logitsB);
                    Log($"## step-0 logits (fast vs legacy): maxAbs {dMax:F5} (rel {relMax:E2})  corr {dCorr:F8}");
                    Log($"   argmax: fast {argB} vs legacy {argA} — {(argA == argB ? "MATCH" : "MISMATCH")}");
                    Gate(relMax <= 2e-3f, "rel maxAbs > 2e-3");
                    Gate(dCorr >= 0.999999f, "corr < 0.999999");
                    Gate(argA == argB, "argmax mismatch");

                    // 8 greedy continuation steps: >= 7/8 token match + per-step corr >= 0.99999
                    int match = 0;
                    float minStepCorr = 1f;
                    for (int i = 0; i < LM_GREEDY_STEPS; i++)
                    {
                        if (toksA[i] == toksB[i]) match++;
                        var (_, _, sc) = Diff(stepsB[i], stepsA[i]);
                        minStepCorr = Mathf.Min(minStepCorr, sc);
                        Log($"   greedy step {i}: tok fast {toksB[i]} vs legacy {toksA[i]}" +
                            $"{(toksA[i] == toksB[i] ? "" : "  <-- DIVERGES")}  logits corr {sc:F7}");
                    }
                    Log($"## greedy continuation: {match}/{LM_GREEDY_STEPS} tokens match, min step corr {minStepCorr:F7}");
                    Gate(match >= LM_GREEDY_STEPS - 1, $"token match {match}/{LM_GREEDY_STEPS} < {LM_GREEDY_STEPS - 1}");
                    Gate(minStepCorr >= 0.99999f, "step logits corr < 0.99999");

                    Log($"## [perf] prefill: legacy {msA:F0} ms | fast {msB:F0} ms  ({msA / Mathf.Max(msB, 1e-3f):F2}x)");
                    Log("   (predicted ~2.5-3x per DEEPOPT §4.3 — prediction, not a claim)");
                }
                catch (Exception e)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    CosyVoiceLM.FastPrefill = prevFast;
                    weights?.Dispose();
                    Finish("lm");
                }
            }

            static (float[] logits0, int[] toks, float[][] stepLogits, float ms) LmArm(
                CosyVoiceWeights weights, bool fast, int[] textFull, int[] promptSpeech)
            {
                CosyVoiceLM.FastPrefill = fast;
                var lm = new CosyVoiceLM(weights);
                try
                {
                    int L = lm.BuildPrefillEmbeds(textFull, promptSpeech);
                    IEnumerator pf = lm.PrefillYielding(L);
                    while (pf.MoveNext()) { }
                    float ms = lm.PrefillMs;
                    float[] logits0 = lm.ReadLogits();

                    int[] toks = new int[LM_GREEDY_STEPS];
                    float[][] stepLogits = new float[LM_GREEDY_STEPS][];
                    float[] cur = logits0;
                    for (int i = 0; i < LM_GREEDY_STEPS; i++)
                    {
                        toks[i] = ArgMax(cur);
                        IEnumerator d = lm.DecodeStepYielding(toks[i]);
                        while (d.MoveNext()) { }
                        cur = lm.ReadLogits();
                        stepLogits[i] = cur;
                    }
                    return (logits0, toks, stepLogits, ms);
                }
                finally { lm.Dispose(); }
            }

            // ============================ Flow DiT (lever 2) ==============================

            [MenuItem("DeepUnity/CosyVoice/#31 DiT Fast Parity + A-B")]
            public static void RunFlow() => FlowCore(FP16_DIR);

            [MenuItem("DeepUnity/CosyVoice/#31 DiT Fast Parity + A-B INT8")]
            public static void RunFlowInt8() => FlowCore(INT8_DIR);

            static void FlowCore(string weightsDir)
            {
                report.Clear(); failed = false;
                Directory.CreateDirectory("ProbeLogs");
                CosyVoiceWeights weights = null;
                bool prevFast = CosyVoiceFlow.FastDit31;
                try
                {
                    Log($"# CosyVoice3 #31 — DiT FastDit31 parity + A/B — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    Log($"weights: {weightsDir}");

                    int[] speechTokens = Ints("speech_tokens");
                    float[] refH = Floats("flow_h_lookahead", out int[] hs);
                    float[] refDx = Floats("dit_dxdt_step0", out int[] ds);
                    float[] refMel = Floats("flow_mel", out int[] ms);

                    weights = new CosyVoiceWeights(weightsDir, beginLoad: false);
                    weights.LoadBlocking("flow/");

                    var (melA, _, wallA, issueA) = FlowArm(weights, fast: false, speechTokens, taps: null);
                    var tapsB = new Dictionary<string, float[]>();
                    var (melB, framesB, wallB, issueB) = FlowArm(weights, fast: true, speechTokens, tapsB);

                    // A/B mel gate (same weights): maxAbs <= 5e-3 (fp16), corr >= 0.9999
                    var (mMax, mMae, mCorr) = Diff(melB, melA);
                    Log($"## mel fast-vs-legacy: maxAbs {mMax:F5}  MAE {mMae:F6}  corr {mCorr:F6}");
                    Gate(mMax <= 5e-3f, "A/B mel maxAbs > 5e-3");
                    Gate(mCorr >= 0.9999f, "A/B mel corr < 0.9999");

                    // fast arm vs the Python dumps (the existing A2 gates)
                    var (hMax, _, hCorr) = Diff(tapsB["h_lookahead"], refH);
                    Log($"## fast vs dump h_lookahead: maxAbs {hMax:F4}  corr {hCorr:F6}");
                    Gate(hCorr >= 0.999f, "h corr < 0.999");
                    float[] dxCond = TransposeCT(SliceRow(refDx, 0, ds[1] * ds[2]), ds[1], ds[2]);
                    float[] dxUncond = TransposeCT(SliceRow(refDx, 1, ds[1] * ds[2]), ds[1], ds[2]);
                    var (_, _, cCorr) = Diff(tapsB["dxdt_cond_s0"], dxCond);
                    var (_, _, uCorr) = Diff(tapsB["dxdt_uncond_s0"], dxUncond);
                    Log($"## fast vs dump dxdt step0: cond corr {cCorr:F6}  uncond corr {uCorr:F6}");
                    Gate(cCorr >= 0.99f, "dxdt cond corr < 0.99");
                    Gate(uCorr >= 0.99f, "dxdt uncond corr < 0.99");
                    float[] refMelTC = TransposeCT(refMel, ms[1], ms[2]);
                    var (_, _, rCorr) = Diff(melB, refMelTC);
                    Log($"## fast vs dump mel: [{framesB},80] vs [{ms[2]},80]  corr {rCorr:F6}");
                    Gate(framesB == ms[2] && rCorr >= 0.99f, "dump mel gate (len or corr < 0.99)");

                    int nt = CosyVoiceConfig.CFM_TIMESTEPS;
                    Log($"## [perf] solve wall: legacy {wallA:F0} ms (issue {issueA:F0}) | fast {wallB:F0} ms (issue {issueB:F0})  ({wallA / Mathf.Max(wallB, 1e-3f):F2}x)");
                    Log($"   analytic dispatches/solve (offline, DEEPOPT §2.3): legacy {299 * nt} (299/step) vs fast {184 * nt} (184/step)");
                    Log("   (predicted 1.4-1.8x per DEEPOPT §4.2 — prediction, not a claim; if GemmCoal loses to LTB2, apply the §4.2 LTB2-fusion fallback)");
                }
                catch (Exception e)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    CosyVoiceFlow.FastDit31 = prevFast;
                    weights?.Dispose();
                    Finish("flow");
                }
            }

            static (float[] mel, int frames, float wallMs, float issueMs) FlowArm(
                CosyVoiceWeights weights, bool fast, int[] speechTokens, Dictionary<string, float[]> taps)
            {
                CosyVoiceFlow.FastDit31 = fast;
                var flow = new CosyVoiceFlow(weights);
                try
                {
                    if (taps != null)
                        flow.DebugTap = (name, buf, count) =>
                        {
                            float[] a = new float[count];
                            buf.GetData(a, 0, 0, count);
                            taps[name] = a;
                        };
                    ComputeBuffer outMel = null; int pm = 0, outFrames = 0;
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    IEnumerator it = flow.SynthesizeMelYielding(speechTokens, (m, p, n) => { outMel = m; pm = p; outFrames = n; });
                    while (it.MoveNext()) { }
                    sw.Stop();
                    float[] mel = new float[outFrames * CosyVoiceConfig.MEL_DIM];
                    outMel.GetData(mel, 0, pm * CosyVoiceConfig.MEL_DIM, mel.Length);
                    return (mel, outFrames, (float)sw.Elapsed.TotalMilliseconds, flow.IssueMs);
                }
                finally { flow.Dispose(); }
            }

            // ============================ HiFT convs (lever 1) ============================

            [MenuItem("DeepUnity/CosyVoice/#31 HiFT Conv Parity + A-B")]
            public static void RunHift()
            {
                report.Clear(); failed = false;
                Directory.CreateDirectory("ProbeLogs");
                CosyVoiceWeights weights = null;
                ComputeBuffer melBuf = null;
                bool prevFast = HiFTVocoder.FastConv;
                try
                {
                    Log($"# CosyVoice3 #31 — HiFT Conv1DTileTC parity + A/B — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    Log($"weights: {FP16_DIR}");

                    float[] melRaw = Floats("flow_mel", out int[] msh);
                    int Tg = msh[2];
                    float[] mel = TransposeCT(melRaw, msh[1], Tg);
                    float[] refSrc = Floats("hift_source", out _);
                    float[] refWav = Floats("wav", out _);
                    float[] refF0 = Floats("hift_f0", out _);

                    weights = new CosyVoiceWeights(FP16_DIR, beginLoad: false);
                    weights.LoadBlocking("hift/");
                    melBuf = new ComputeBuffer(Tg * CosyVoiceConfig.MEL_DIM, 4, ComputeBufferType.Structured);
                    melBuf.SetData(mel);

                    var tapsA = new Dictionary<string, float[]>();
                    var tapsB = new Dictionary<string, float[]>();
                    var (wavA, f0A, msA) = HiftArm(weights, fast: false, melBuf, Tg, refSrc, tapsA);
                    var (wavB, f0B, msB) = HiftArm(weights, fast: true, melBuf, Tg, refSrc, tapsB);

                    // F0 (fp64-sensitive chain — the condnet convs are tile-reordered): A/B tight
                    // gate + the A1 dump gate (corr >= 0.999)
                    var (fMax, _, fCorrAB) = Diff(f0B, f0A);
                    var (_, _, fCorrRef) = Diff(f0B, refF0);
                    Log($"## F0: fast-vs-legacy maxAbs {fMax:F5} corr {fCorrAB:F7}; fast-vs-dump corr {fCorrRef:F6}");
                    Gate(fCorrAB >= 0.99999f, "F0 A/B corr < 0.99999");
                    Gate(fCorrRef >= 0.999f, "F0 dump corr < 0.999 (A1 gate)");

                    // per-stage fast-vs-legacy gates: corr >= 0.99999 on every common tap
                    Log("## per-stage fast-vs-legacy (DebugTap):");
                    float minStage = 1f;
                    foreach (var kv in tapsA)
                    {
                        if (!tapsB.TryGetValue(kv.Key, out float[] b)) continue;
                        var (sMax, _, sCorr) = Diff(b, kv.Value);
                        minStage = Mathf.Min(minStage, sCorr);
                        Log($"   stage {kv.Key,-10} maxAbs {sMax:F5}  corr {sCorr:F7}" +
                            (sCorr < 0.99999f ? "  <-- DIVERGES" : ""));
                    }
                    Gate(minStage >= 0.99999f, $"stage corr {minStage:F7} < 0.99999");

                    // final wav A/B: corr >= 0.999, maxAbs <= 2e-2
                    var (wMax, _, wCorr) = Diff(wavB, wavA);
                    Log($"## wav fast-vs-legacy: maxAbs {wMax:F5}  corr {wCorr:F6}");
                    Gate(wCorr >= 0.999f && wMax <= 2e-2f, "A/B wav gate (corr < 0.999 or maxAbs > 2e-2)");

                    // fast arm vs the Python dump (the existing A1 wav gate + stage refs if dumped)
                    var (_, _, dCorr) = Diff(wavB, refWav);
                    Log($"## fast vs dump wav: corr {dCorr:F6} ({wavB.Length} vs {refWav.Length} samples)");
                    Gate(wavB.Length == refWav.Length && dCorr >= 0.99f, "dump wav gate (len or corr < 0.99)");
                    string stagesDir = Path.Combine(DUMP_DIR, "hift_stages");
                    if (Directory.Exists(stagesDir))
                    {
                        // informational, matching A1's convention (DIVERGES marker, no hard fail —
                        // the A/B stage gates above are the tight gates for the tile kernels)
                        Log("## fast vs dump stages (informational, A1 convention):");
                        foreach (var kv in tapsB)
                        {
                            string p = Path.Combine(stagesDir, kv.Key + ".npy");
                            if (!File.Exists(p)) continue;
                            float[] rf = (float[])LoadNpy(p, out int[] rs);
                            float[] refTC = TransposeCT(rf, rs[1], rs[2]);
                            var (_, _, sCorr) = Diff(kv.Value, refTC);
                            Log($"   stage {kv.Key,-10} corr {sCorr:F6}" + (sCorr < 0.995f ? "  <-- DIVERGES" : ""));
                        }
                    }

                    SaveWav("ProbeLogs/cosyvoice_fasthift_unity.wav", wavB, CosyVoiceConfig.SAMPLE_RATE);
                    Log($"## [perf] vocode: legacy {msA:F0} ms | fast {msB:F0} ms  ({msA / Mathf.Max(msB, 1e-3f):F2}x)");
                    Log("   analytic dispatches/offline chunk (DEEPOPT §2.4): legacy ~279 vs fast ~135");
                    Log("   (predicted 5-8x per DEEPOPT §4.1 — prediction, not a claim); audio -> ProbeLogs/cosyvoice_fasthift_unity.wav");
                }
                catch (Exception e)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    HiFTVocoder.FastConv = prevFast;
                    melBuf?.Release();
                    weights?.Dispose();
                    Finish("hift");
                }
            }

            static (float[] wav, float[] f0, float ms) HiftArm(CosyVoiceWeights weights, bool fast,
                ComputeBuffer melBuf, int Tg, float[] refSrc, Dictionary<string, float[]> taps)
            {
                HiFTVocoder.FastConv = fast;
                var voc = new HiFTVocoder(weights) { InjectSource = refSrc };
                try
                {
                    if (taps != null)
                        voc.DebugTap = (name, buf, count) =>
                        {
                            float[] a = new float[count];
                            buf.GetData(a, 0, 0, count);
                            taps[name] = a;
                        };
                    float[] wav = null;
                    IEnumerator it = voc.VocodeYielding(melBuf, Tg, w => wav = w);
                    while (it.MoveNext()) { }
                    float[] f0 = new float[Tg];
                    voc.DebugF0.GetData(f0, 0, 0, Tg);
                    return (wav, f0, voc.VocoderMs);
                }
                finally { voc.Dispose(); }
            }

            // ============================ Flow+HiFT chain A/B =============================

            [MenuItem("DeepUnity/CosyVoice/#31 Flow+HiFT E2E A-B")]
            public static void RunChain()
            {
                report.Clear(); failed = false;
                Directory.CreateDirectory("ProbeLogs");
                CosyVoiceWeights weights = null;
                bool prevDit = CosyVoiceFlow.FastDit31, prevConv = HiFTVocoder.FastConv;
                try
                {
                    Log($"# CosyVoice3 #31 — flow+HiFT chain all-fast vs all-legacy — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    Log($"weights: {FP16_DIR} (dump tokens -> mel -> wav; injected NSF source, no sampler -> deterministic)");

                    int[] speechTokens = Ints("speech_tokens");
                    float[] refSrc = Floats("hift_source", out _);

                    weights = new CosyVoiceWeights(FP16_DIR, beginLoad: false);
                    weights.LoadBlocking("flow/");
                    weights.LoadBlocking("hift/");

                    var (wavA, msA) = ChainArm(weights, fast: false, speechTokens, refSrc);
                    var (wavB, msB) = ChainArm(weights, fast: true, speechTokens, refSrc);

                    var (cMax, _, cCorr) = Diff(wavB, wavA);
                    Log($"## chain wav fast-vs-legacy: {wavB.Length} vs {wavA.Length} samples; maxAbs {cMax:F5}  corr {cCorr:F6}");
                    Gate(wavA.Length == wavB.Length, "length mismatch");
                    Gate(cCorr >= 0.999f, "wav corr < 0.999");
                    SaveWav("ProbeLogs/cosyvoice_fastchain_unity.wav", wavB, CosyVoiceConfig.SAMPLE_RATE);
                    Log($"## [perf] chain total: legacy {msA:F0} ms | fast {msB:F0} ms  ({msA / Mathf.Max(msB, 1e-3f):F2}x)");
                    Log("   audio -> ProbeLogs/cosyvoice_fastchain_unity.wav");
                }
                catch (Exception e)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    CosyVoiceFlow.FastDit31 = prevDit;
                    HiFTVocoder.FastConv = prevConv;
                    weights?.Dispose();
                    Finish("chain");
                }
            }

            static (float[] wav, float ms) ChainArm(CosyVoiceWeights weights, bool fast,
                int[] speechTokens, float[] refSrc)
            {
                CosyVoiceFlow.FastDit31 = fast;
                HiFTVocoder.FastConv = fast;
                CosyVoiceFlow flow = null;
                HiFTVocoder voc = null;
                ComputeBuffer melBuf = null;
                try
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    flow = new CosyVoiceFlow(weights);
                    ComputeBuffer outMel = null; int pm = 0, outFrames = 0;
                    IEnumerator it = flow.SynthesizeMelYielding(speechTokens, (m, p, n) => { outMel = m; pm = p; outFrames = n; });
                    while (it.MoveNext()) { }
                    float[] mel = new float[outFrames * CosyVoiceConfig.MEL_DIM];
                    outMel.GetData(mel, 0, pm * CosyVoiceConfig.MEL_DIM, mel.Length);
                    melBuf = new ComputeBuffer(outFrames * CosyVoiceConfig.MEL_DIM, 4, ComputeBufferType.Structured);
                    melBuf.SetData(mel);
                    voc = new HiFTVocoder(weights) { InjectSource = refSrc };
                    float[] wav = null;
                    IEnumerator vit = voc.VocodeYielding(melBuf, outFrames, w => wav = w);
                    while (vit.MoveNext()) { }
                    sw.Stop();
                    return (wav, (float)sw.Elapsed.TotalMilliseconds);
                }
                finally { voc?.Dispose(); melBuf?.Release(); flow?.Dispose(); }
            }

            // ============== LEGACY twins: existing dump probes with the flags OFF =========
            // Bisect arms for checklist item 8 — prove the fallback paths survived.

            [MenuItem("DeepUnity/CosyVoice/A1 HiFT Parity LEGACY (FastConv off)")]
            public static void RunHiftLegacy()
            {
                bool prev = HiFTVocoder.FastConv;
                HiFTVocoder.FastConv = false;
                try { CosyVoiceHiftProbe.Run(); } finally { HiFTVocoder.FastConv = prev; }
            }

            [MenuItem("DeepUnity/CosyVoice/A2 DiT Flow Parity LEGACY (FastDit31 off)")]
            public static void RunFlowLegacy()
            {
                bool prev = CosyVoiceFlow.FastDit31;
                CosyVoiceFlow.FastDit31 = false;
                try { CosyVoiceFlowProbe.Run(); } finally { CosyVoiceFlow.FastDit31 = prev; }
            }

            [MenuItem("DeepUnity/CosyVoice/A3 LM Parity LEGACY (FastPrefill off)")]
            public static void RunLmPrefillLegacy()
            {
                bool prev = CosyVoiceLM.FastPrefill;
                CosyVoiceLM.FastPrefill = false;
                try { CosyVoiceLmProbe.Run(); } finally { CosyVoiceLM.FastPrefill = prev; }
            }

            // ---------------- npy / diff helpers (self-contained per probe file) ----------------
            static Array LoadNpy(string path, out int[] shape)
            {
                byte[] all = File.ReadAllBytes(path);
                if (all[0] != 0x93) throw new Exception($"not npy: {path}");
                int major = all[6];
                int headerLen = major >= 2 ? BitConverter.ToInt32(all, 8) : BitConverter.ToUInt16(all, 8);
                int dataStart = (major >= 2 ? 12 : 10) + headerLen;
                string header = Encoding.ASCII.GetString(all, major >= 2 ? 12 : 10, headerLen);

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
                if (header.Contains("i8"))
                {
                    long[] r = new long[count];
                    Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 8);
                    return r;
                }
                if (header.Contains("i4"))
                {
                    int[] r = new int[count];
                    Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4);
                    return r;
                }
                throw new Exception($"unsupported npy dtype in {path}: {header}");
            }

            static float[] Floats(string name, out int[] shape)
                => (float[])LoadNpy(Path.Combine(DUMP_DIR, name + ".npy"), out shape);

            static int[] Ints(string name)
            {
                Array a = LoadNpy(Path.Combine(DUMP_DIR, name + ".npy"), out _);
                if (a is int[] i) return i;
                if (a is long[] l)
                {
                    int[] r = new int[l.Length];
                    for (int j = 0; j < l.Length; j++) r[j] = (int)l[j];
                    return r;
                }
                float[] f = (float[])a;
                int[] r2 = new int[f.Length];
                for (int j = 0; j < f.Length; j++) r2[j] = (int)Math.Round(f[j]);
                return r2;
            }

            static float[] SliceRow(float[] src, int row, int rowLen)
            {
                float[] r = new float[rowLen];
                Array.Copy(src, row * rowLen, r, 0, rowLen);
                return r;
            }

            static (float maxAbs, float mae, float corr) Diff(float[] a, float[] b)
            {
                int n = Mathf.Min(a.Length, b.Length);
                double maxAbs = 0, mae = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
                for (int i = 0; i < n; i++)
                {
                    double d = Math.Abs(a[i] - b[i]);
                    maxAbs = Math.Max(maxAbs, d); mae += d;
                    sa += a[i]; sb += b[i]; saa += (double)a[i] * a[i]; sbb += (double)b[i] * b[i]; sab += (double)a[i] * b[i];
                }
                double cov = sab / n - (sa / n) * (sb / n);
                double va = saa / n - (sa / n) * (sa / n), vb = sbb / n - (sb / n) * (sb / n);
                return ((float)maxAbs, (float)(mae / n), (float)(cov / Math.Sqrt(Math.Max(va * vb, 1e-20))));
            }

            static float MaxAbs(float[] a)
            {
                float m = 0f;
                foreach (float v in a) m = Mathf.Max(m, Mathf.Abs(v));
                return m;
            }

            static int ArgMax(float[] a)
            {
                int bi = 0; float bv = a[0];
                for (int i = 1; i < a.Length; i++) if (a[i] > bv) { bv = a[i]; bi = i; }
                return bi;
            }

            static float[] TransposeCT(float[] src, int C, int T)   // [C,T] -> [T,C]
            {
                float[] r = new float[C * T];
                for (int c = 0; c < C; c++)
                    for (int t = 0; t < T; t++)
                        r[t * C + c] = src[c * T + t];
                return r;
            }

            static void SaveWav(string path, float[] samples, int sr)
            {
                using var fs = new FileStream(path, FileMode.Create);
                using var w = new BinaryWriter(fs);
                int byteLen = samples.Length * 2;
                w.Write(Encoding.ASCII.GetBytes("RIFF")); w.Write(36 + byteLen);
                w.Write(Encoding.ASCII.GetBytes("WAVEfmt ")); w.Write(16);
                w.Write((short)1); w.Write((short)1); w.Write(sr); w.Write(sr * 2);
                w.Write((short)2); w.Write((short)16);
                w.Write(Encoding.ASCII.GetBytes("data")); w.Write(byteLen);
                foreach (float s in samples)
                    w.Write((short)Mathf.Clamp(Mathf.RoundToInt(s * 32767f), short.MinValue, short.MaxValue));
            }
        }
    }
}

#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    // Editor entry points for the two LLM benchmark probes — boot-vs-framedrop (LMBootKnobProbe)
    // and prefill speed (LMPrefillProbe). Each run boots ONE model kind + quant; pick the combo
    // from the menu (interactive) or pass -model/-quant on a batch command line:
    //
    //   Unity.exe -projectPath <proj> -batchmode -executeMethod DeepUnity.LMBenchmarkProbeRunner.RunBootKnobProbe -model qwen -quant int8
    //   Unity.exe -projectPath <proj> -batchmode -executeMethod DeepUnity.LMBenchmarkProbeRunner.RunPrefillProbe  -model gemma -quant fp16
    //
    // (NO -nographics: compute shaders need a graphics device.) Defaults: qwen, fp16.
    public static class LMBenchmarkProbeRunner
    {
        const string TempScenePath = "Assets/__lm_benchmark_probe_tmp.unity";

        // ---- batch entry points (read -model/-quant from the command line) --------------------
        public static void RunBootKnobProbe()    => LaunchBoot(ParseModel(), ParseQuant());
        public static void RunPrefillProbe()     => LaunchPrefill(ParseModel(), ParseQuant());
        public static void RunDecodeDecayProbe() => LaunchDecodeDecay(ParseModel(), ParseQuant());

        // ---- interactive menu items -----------------------------------------------------------
        [MenuItem("Tools/DeepUnity/Benchmarks/Boot Knob Probe/Qwen3.5 FP16")] static void B_Q_FP16() => LaunchBoot(ProbeModelKind.Qwen3_5_0_8B, LLMQuant.FP16);
        [MenuItem("Tools/DeepUnity/Benchmarks/Boot Knob Probe/Qwen3.5 INT8")] static void B_Q_INT8() => LaunchBoot(ProbeModelKind.Qwen3_5_0_8B, LLMQuant.INT8);
        [MenuItem("Tools/DeepUnity/Benchmarks/Boot Knob Probe/Qwen3.5 INT4")] static void B_Q_INT4() => LaunchBoot(ProbeModelKind.Qwen3_5_0_8B, LLMQuant.INT4);
        [MenuItem("Tools/DeepUnity/Benchmarks/Boot Knob Probe/Gemma3 FP16")]  static void B_G_FP16() => LaunchBoot(ProbeModelKind.Gemma3_270M, LLMQuant.FP16);
        [MenuItem("Tools/DeepUnity/Benchmarks/Boot Knob Probe/Gemma3 INT8")]  static void B_G_INT8() => LaunchBoot(ProbeModelKind.Gemma3_270M, LLMQuant.INT8);
        [MenuItem("Tools/DeepUnity/Benchmarks/Boot Knob Probe/Gemma3 INT4")]  static void B_G_INT4() => LaunchBoot(ProbeModelKind.Gemma3_270M, LLMQuant.INT4);

        [MenuItem("Tools/DeepUnity/Benchmarks/Prefill Probe/Qwen3.5 FP16")] static void P_Q_FP16() => LaunchPrefill(ProbeModelKind.Qwen3_5_0_8B, LLMQuant.FP16);
        [MenuItem("Tools/DeepUnity/Benchmarks/Prefill Probe/Qwen3.5 INT8")] static void P_Q_INT8() => LaunchPrefill(ProbeModelKind.Qwen3_5_0_8B, LLMQuant.INT8);
        [MenuItem("Tools/DeepUnity/Benchmarks/Prefill Probe/Qwen3.5 INT4")] static void P_Q_INT4() => LaunchPrefill(ProbeModelKind.Qwen3_5_0_8B, LLMQuant.INT4);
        [MenuItem("Tools/DeepUnity/Benchmarks/Prefill Probe/Gemma3 FP16")]  static void P_G_FP16() => LaunchPrefill(ProbeModelKind.Gemma3_270M, LLMQuant.FP16);
        [MenuItem("Tools/DeepUnity/Benchmarks/Prefill Probe/Gemma3 INT8")]  static void P_G_INT8() => LaunchPrefill(ProbeModelKind.Gemma3_270M, LLMQuant.INT8);
        [MenuItem("Tools/DeepUnity/Benchmarks/Prefill Probe/Gemma3 INT4")]  static void P_G_INT4() => LaunchPrefill(ProbeModelKind.Gemma3_270M, LLMQuant.INT4);

        [MenuItem("Tools/DeepUnity/Benchmarks/Decode Decay Probe/Qwen3.5 FP16")] static void D_Q_FP16() => LaunchDecodeDecay(ProbeModelKind.Qwen3_5_0_8B, LLMQuant.FP16);
        [MenuItem("Tools/DeepUnity/Benchmarks/Decode Decay Probe/Qwen3.5 INT8")] static void D_Q_INT8() => LaunchDecodeDecay(ProbeModelKind.Qwen3_5_0_8B, LLMQuant.INT8);
        [MenuItem("Tools/DeepUnity/Benchmarks/Decode Decay Probe/Qwen3.5 INT4")] static void D_Q_INT4() => LaunchDecodeDecay(ProbeModelKind.Qwen3_5_0_8B, LLMQuant.INT4);
        [MenuItem("Tools/DeepUnity/Benchmarks/Decode Decay Probe/Gemma3 FP16")]  static void D_G_FP16() => LaunchDecodeDecay(ProbeModelKind.Gemma3_270M, LLMQuant.FP16);
        [MenuItem("Tools/DeepUnity/Benchmarks/Decode Decay Probe/Gemma3 INT8")]  static void D_G_INT8() => LaunchDecodeDecay(ProbeModelKind.Gemma3_270M, LLMQuant.INT8);
        [MenuItem("Tools/DeepUnity/Benchmarks/Decode Decay Probe/Gemma3 INT4")]  static void D_G_INT4() => LaunchDecodeDecay(ProbeModelKind.Gemma3_270M, LLMQuant.INT4);

        // ---- launchers ------------------------------------------------------------------------
        static void LaunchBoot(ProbeModelKind kind, LLMQuant quant)
        {
            Launch($"bootknob_{LMProbeCommon.ModelLabel(kind)}_{quant}", dir =>
            {
                var go = new GameObject("LMBootKnobProbe");
                var probe = go.AddComponent<LMBootKnobProbe>();
                probe.model = kind;
                probe.quant = quant;
                probe.reportDirectory = dir;
            });
        }

        static void LaunchPrefill(ProbeModelKind kind, LLMQuant quant)
        {
            Launch($"prefill_{LMProbeCommon.ModelLabel(kind)}_{quant}", dir =>
            {
                var go = new GameObject("LMPrefillProbe");
                var probe = go.AddComponent<LMPrefillProbe>();
                probe.model = kind;
                probe.quant = quant;
                probe.reportDirectory = dir;
            });
        }

        static void LaunchDecodeDecay(ProbeModelKind kind, LLMQuant quant)
        {
            // Gemma3 KV capacity is 2048; keep maxTokens safely under it. Qwen3.5 is 8192.
            int maxTokens = kind == ProbeModelKind.Gemma3_270M ? 2000 : 4096;
            Launch($"decodedecay_{LMProbeCommon.ModelLabel(kind)}_{quant}", dir =>
            {
                var go = new GameObject("LMDecodeDecayProbe");
                var probe = go.AddComponent<LMDecodeDecayProbe>();
                probe.model = kind;
                probe.quant = quant;
                probe.maxTokens = maxTokens;
                probe.reportDirectory = dir;
            });
        }

        static void Launch(string tag, Action<string> addProbe)
        {
            try
            {
                string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
                string dir = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", $"{tag}_{runId}");
                Directory.CreateDirectory(dir);

                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                var camGo = new GameObject("Camera");
                camGo.AddComponent<Camera>();
                addProbe(dir);
                EditorSceneManager.SaveScene(scene, TempScenePath);

                Debug.Log($"[LMBenchmarkProbeRunner] {tag} — report dir: {dir}");
                EditorApplication.isPlaying = true;
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                EditorApplication.Exit(1);
            }
        }

        // ---- command-line parsing -------------------------------------------------------------
        static ProbeModelKind ParseModel()
        {
            string v = ArgValue("-model")?.ToLowerInvariant();
            if (v != null && v.Contains("gemma")) return ProbeModelKind.Gemma3_270M;
            return ProbeModelKind.Qwen3_5_0_8B; // default
        }

        static LLMQuant ParseQuant()
        {
            string v = ArgValue("-quant")?.ToLowerInvariant();
            if (v == "int8") return LLMQuant.INT8;
            if (v == "int4") return LLMQuant.INT4;
            return LLMQuant.FP16; // default
        }

        static string ArgValue(string flag)
        {
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
                if (string.Equals(args[i], flag, StringComparison.OrdinalIgnoreCase))
                    return args[i + 1];
            return null;
        }
    }
}
#endif

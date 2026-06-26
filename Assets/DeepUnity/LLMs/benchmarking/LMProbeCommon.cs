using System.Collections;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>Which exported model a benchmark probe should build (one per editor run).</summary>
    public enum ProbeModelKind
    {
        Qwen3_5_0_8B,
        Gemma3_270M,
    }

    /// <summary>
    /// Shared bits for the LLM benchmark probes (LMBootKnobProbe, LMPrefillProbe): a model-agnostic
    /// factory over {model kind} x {quant}, the matching kernel-prewarm dispatch, and the GPU/CPU
    /// system-info block every report stamps so a result can be traced back to the machine it came
    /// from. Each probe is configured for ONE model kind + quant per editor run — these helpers just
    /// remove the per-probe switch boilerplate.
    /// </summary>
    public static class LMProbeCommon
    {
        /// <summary>Constructs the selected model at the given quant. Boot proceeds across frames; poll IsReady.</summary>
        public static LLM Build(ProbeModelKind kind, LLMQuant quant) => kind switch
        {
            ProbeModelKind.Qwen3_5_0_8B => new Qwen3_5ForCausalLM(Qwen3_5Size.B0_8, quant),
            ProbeModelKind.Gemma3_270M  => new Gemma3ForCausalLM(quant),
            _ => throw new System.ArgumentOutOfRangeException(nameof(kind), kind, "unknown probe model"),
        };

        /// <summary>
        /// Compiles every compute kernel once (the driver's one-time first-dispatch cost) WITHOUT a
        /// model instance. The boot probe runs this before its knob sweep so the kernel-compile
        /// spikes (which would only hit the first config anyway) don't contaminate the per-config
        /// weight-streaming frame measurements.
        /// </summary>
        public static IEnumerator Prewarm(ProbeModelKind kind) => kind switch
        {
            ProbeModelKind.Qwen3_5_0_8B => Qwen3_5ForCausalLM.Prewarm(),
            ProbeModelKind.Gemma3_270M  => Gemma3ForCausalLM.Prewarm(),
            _ => throw new System.ArgumentOutOfRangeException(nameof(kind), kind, "unknown probe model"),
        };

        public static string ModelLabel(ProbeModelKind kind) => kind switch
        {
            ProbeModelKind.Qwen3_5_0_8B => "qwen3.5-0.8B",
            ProbeModelKind.Gemma3_270M  => "gemma3-270M",
            _ => kind.ToString(),
        };

        /// <summary>
        /// Markdown block identifying the machine — GPU (name, API, VRAM), CPU (name, logical cores,
        /// base clock), RAM and OS. Stamped at the top of every probe report so results from
        /// different machines (Victus / Pavilion / a pod) are never confused.
        /// </summary>
        public static string SystemInfoBlock()
        {
            var sb = new StringBuilder();
            sb.AppendLine("## Machine");
            sb.AppendLine();
            sb.AppendLine($"- GPU: {SystemInfo.graphicsDeviceName} ({SystemInfo.graphicsDeviceType}, {SystemInfo.graphicsMemorySize} MB)");
            sb.AppendLine($"- GPU driver/API: {SystemInfo.graphicsDeviceVersion}");
            sb.AppendLine($"- CPU: {SystemInfo.processorType} ({SystemInfo.processorCount} logical cores @ {SystemInfo.processorFrequency} MHz)");
            sb.AppendLine($"- RAM: {SystemInfo.systemMemorySize} MB");
            sb.AppendLine($"- Device: {SystemInfo.deviceName} | OS: {SystemInfo.operatingSystem}");
            sb.AppendLine($"- Unity: {Application.unityVersion}");
            return sb.ToString();
        }

        /// <summary>Compact one-line machine tag for CSV / filenames (GPU + CPU short).</summary>
        public static string MachineTag() =>
            $"{SystemInfo.graphicsDeviceName} | {SystemInfo.processorType}";

        /// <summary>Minimal JSON string escaping (quotes, backslashes, control chars).</summary>
        public static string JsonStr(string s)
        {
            if (s == null) return "null";
            var sb = new StringBuilder("\"");
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default: if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4")); else sb.Append(c); break;
                }
            }
            return sb.Append('"').ToString();
        }

        /// <summary>
        /// The machine as a JSON object (no trailing comma) — same fields as <see cref="SystemInfoBlock"/>,
        /// embedded under a "machine" key in each probe's summary.json so results can be aggregated and
        /// traced back to the GPU/CPU they came from.
        /// </summary>
        public static string MachineJson()
        {
            var sb = new StringBuilder();
            sb.Append("{");
            sb.Append("\"gpu\":").Append(JsonStr(SystemInfo.graphicsDeviceName)).Append(',');
            sb.Append("\"gpu_api\":").Append(JsonStr(SystemInfo.graphicsDeviceType.ToString())).Append(',');
            sb.Append("\"gpu_driver\":").Append(JsonStr(SystemInfo.graphicsDeviceVersion)).Append(',');
            sb.Append("\"vram_mb\":").Append(SystemInfo.graphicsMemorySize).Append(',');
            sb.Append("\"cpu\":").Append(JsonStr(SystemInfo.processorType)).Append(',');
            sb.Append("\"cpu_cores\":").Append(SystemInfo.processorCount).Append(',');
            sb.Append("\"cpu_mhz\":").Append(SystemInfo.processorFrequency).Append(',');
            sb.Append("\"ram_mb\":").Append(SystemInfo.systemMemorySize).Append(',');
            sb.Append("\"device\":").Append(JsonStr(SystemInfo.deviceName)).Append(',');
            sb.Append("\"os\":").Append(JsonStr(SystemInfo.operatingSystem)).Append(',');
            sb.Append("\"unity\":").Append(JsonStr(Application.unityVersion));
            sb.Append("}");
            return sb.ToString();
        }
    }
}

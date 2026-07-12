namespace DeepUnity
{
    // ONE log format for every model's GPU residency traffic (LLM / TTS / STT), so loading,
    // slow-loading, defetching and releasing read identically across the whole engine:
    //   [GPU] kokoro_int8 SLOW prefetch started — 143 MB at 2.4 MB/frame (~1.0 s @60fps)
    //   [GPU] kokoro_int8 BOOSTED to full speed — 3.2 MB/frame (~0.4 s left @60fps)
    //   [GPU] kokoro_int8 fully streamed, resident — 143 MB in 1.21 s (73 frames, worst slice 6.2 ms)
    //   [GPU] kokoro_int8 defetching (slow)
    //   [GPU] kokoro_int8 released — 143 MB freed
    // `model` is the real weights-folder stem (weights_<stem>) — never a display alias.
    public static class ResidencyLog
    {
        /// <summary>Trim a weights path down to its identifying stem.</summary>
        public static string Label(string paramsPath)
        {
            string n = System.IO.Path.GetFileName(
                paramsPath.TrimEnd('/', '\\'));
            return n.StartsWith("weights_") ? n.Substring("weights_".Length) : n;
        }

        public static string Mode(long bytesPerFrame) =>
            bytesPerFrame <= 0 ? "paused"
            : bytesPerFrame < LLM.UploadBudgetBytes ? "slow"
            : bytesPerFrame > LLM.UploadBudgetBytes ? "boost" : "normal";

        // "SLOW prefetch started" vs plain full-speed "streaming started" — the reader must be
        // able to tell latent walk-up loading from a boot-time full-speed stream at a glance.
        public static void Loading(string model, long totalBytes, long bytesPerFrame)
        {
            string size = totalBytes > 0 ? $"{totalBytes / 1e6:0} MB at " : "";
            string eta = totalBytes > 0 && bytesPerFrame > 0
                ? $" (~{totalBytes / (double)bytesPerFrame / 60.0:0.0} s @60fps)" : "";
            string verb = Mode(bytesPerFrame) == "slow" ? "SLOW prefetch started"
                        : Mode(bytesPerFrame) == "paused" ? "prefetch parked (budget 0)"
                        : "streaming started (full speed)";
            ConsoleMessage.Info($"[GPU] {model} {verb} — {size}{bytesPerFrame / 1e6:0.0} MB/frame{eta}");
        }

        public static void Resident(string model, long bytes, double ms, int frames, double worstSliceMs = -1)
            => ConsoleMessage.Info($"[GPU] {model} fully streamed, resident — {(bytes > 0 ? $"{bytes / 1e6:0} MB in " : "")}{ms / 1000.0:0.00} s " +
                                   $"({frames} frames{(worstSliceMs < 0 ? "" : $", worst slice {worstSliceMs:0.0} ms")})");

        /// <summary>Live budget change mid-load (SlowPrefetch / BoostFetch / PausePrefetch) —
        /// spelled out per direction so a boost landing is unmissable in the console.</summary>
        public static void Budget(string model, long bytesPerFrame, long remainingBytes)
        {
            string eta = bytesPerFrame > 0 && remainingBytes > 0
                ? $" (~{remainingBytes / (double)bytesPerFrame / 60.0:0.0} s left @60fps)" : "";
            string verb = Mode(bytesPerFrame) == "slow" ? "SLOW prefetch retargeted"
                        : Mode(bytesPerFrame) == "paused" ? "prefetch parked (budget 0)"
                        : "BOOSTED to full speed";
            ConsoleMessage.Info($"[GPU] {model} {verb} — {bytesPerFrame / 1e6:0.0} MB/frame{eta}");
        }

        public static void Defetching(string model, long bytesPerFrame)
            => ConsoleMessage.Info($"[GPU] {model} defetching ({(bytesPerFrame > 0 ? "slow" : "instant")})");

        public static void Released(string model, long bytes)
            => ConsoleMessage.Info($"[GPU] {model} released — {bytes / 1e6:0} MB freed");
    }
}

using System;
using System.Collections;

namespace DeepUnity
{
    // Abstract STT family base (ModelBase -> STT -> QwenASR/Parakeet...).
    // Skeleton only for now — filled in when the STT workstreams' build-outs land.
    // Contract: push-to-talk style transcription of a 16 kHz mono clip captured from
    // Unity's Microphone API (utterance-level; streaming variants may extend later).
    public abstract class STT : ModelBase
    {
        /// <summary>Expected input sample rate (Hz), typically 16000.</summary>
        public abstract int InputSampleRate { get; }

        /// <summary>Transcribe a mono utterance (samples at <see cref="InputSampleRate"/>).
        /// onTranscript receives the final text (null/empty on failure).</summary>
        public abstract IEnumerator Transcribe(float[] samples, Action<string> onTranscript);
    }
}

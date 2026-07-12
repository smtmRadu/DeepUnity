using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    // Abstract TTS family base (ModelBase -> TTS -> Chatterbox/CosyVoice/Kokoro...).
    // The shared surface mirrors what ChatterboxTTS pioneered: coroutine synthesis with an
    // AudioClip convenience wrapper, a raw-samples path for streaming consumers (ring buffers),
    // and a per-token/streaming tap where the model family supports it.
    // New TTS models are born extending this; ChatterboxTTS is rebased in the WS-F legacy pass.
    public abstract class TTS : ModelBase
    {
        /// <summary>Output sample rate (mono PCM).</summary>
        public abstract int SampleRate { get; }

        /// <summary>Rolling synthesis speed indicator (family-defined; 0 when idle).</summary>
        public float TokensPerSecond { get; protected set; }

        /// <summary>Full text -> mono samples at <see cref="SampleRate"/> via onWav
        /// (null on failure). Yields per frame; safe to pump alongside gameplay.</summary>
        public abstract IEnumerator Synthesize(string text, Action<float[]> onWav);

        /// <summary>Full text -> AudioClip convenience (built on <see cref="Synthesize"/>).</summary>
        public virtual IEnumerator Speak(string text, Action<AudioClip> onClip)
        {
            float[] wav = null;
            var e = Synthesize(text, w => wav = w);
            while (e.MoveNext()) yield return e.Current;
            if (wav == null) { onClip?.Invoke(null); yield break; }
            AudioClip clip = AudioClip.Create(GetType().Name, wav.Length, 1, SampleRate, false);
            clip.SetData(wav, 0);
            onClip?.Invoke(clip);
        }
    }
}

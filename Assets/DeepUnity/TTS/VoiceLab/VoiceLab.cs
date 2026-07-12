using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity
{
    // VoiceLab — the TTS audition scene (Assets/DeepUnity/TTS/VoiceLab/VoiceLab.unity, built by
    // VoiceLabBuilder). Pick an engine + baked voice, tweak pitch/speed, type a line, hear it,
    // and SAVE the preset to voice_presets.json (also logs the exact NPCChatBase inspector
    // values to copy: ttsModel / ttsVoice / voicePitch).
    //
    // Engine plumbing (reuses the untouched voice components + engines):
    //   - Voices are discovered by parsing the selected engine's weights manifest.tsv:
    //     Kokoro/CosyVoice3 = "voices/<name>" rows, Chatterbox = "<name>/t3_speaker_emb" rows
    //     ("conds", "conds_elder").
    //   - All three engines bind their voice at CONSTRUCTION (KokoroTTS/CosyVoiceTTS/ChatterboxTTS
    //     ctors take `voice`; the prompt/voicepack fields are readonly, and the shared-TTS statics
    //     on the voice components cache the first engine built). So ANY engine or voice change =
    //     StopSpeaking -> destroy the voice component -> Release() the engine -> SetSharedTTS(null)
    //     -> construct a fresh engine bound to the new voice -> SetSharedTTS(it) -> AddComponent a
    //     fresh voice component. Kokoro re-streams ~150 MB (seconds); CosyVoice3/Chatterbox
    //     re-stream their full GB-scale weights — their APIs expose no voice-only rebind, so a
    //     voice switch there costs a full reload (status label shows the progress).
    //   - One engine resident at a time (this is an audition tool, not a game scene).
    [RequireComponent(typeof(AudioSource))]
    public class VoiceLab : MonoBehaviour
    {
        public enum Engine { Kokoro = 0, CosyVoice3 = 1, Chatterbox = 2 }

        // sample lines (short/medium/long buttons; medium is the scene prefill)
        public const string SAMPLE_SHORT = "Hello there, traveler.";
        public const string SAMPLE_MEDIUM = "The old mill by the river went quiet for twenty years, " +
                                            "yet tonight its wheel is turning again.";
        public const string SAMPLE_LONG = "Long ago, before the roads had names, this valley belonged to " +
                                          "the wind and the wolves. My grandmother swore the mountain hummed " +
                                          "a song of its own on winter nights. Sit down, pour yourself some " +
                                          "tea, and I will tell you how it goes.";

        // weights folders — int8 preferred, fp16 fallback (same Resources layout the engines use)
        const string KOKORO_INT8 = "Assets/Resources/Weights/weights_kokoro_int8";
        const string KOKORO_FP16 = "Assets/Resources/Weights/weights_kokoro_fp16";
        const string COSY_INT8   = "Assets/Resources/Weights/weights_cosyvoice3_int8";
        const string COSY_FP16   = "Assets/Resources/Weights/weights_cosyvoice3_fp16";
        const string CB_INT8     = "Assets/Resources/Weights/weights_chatterbox_turbo_int8";
        const string CB_FP16     = "Assets/Resources/Weights/weights_chatterbox_turbo_fp16";
        const string PRESETS_PATH = "Assets/DeepUnity/TTS/VoiceLab/voice_presets.json";

        // labels match NPCChatBase.TtsModel member names — the saved preset/log is copy-paste truth
        static readonly string[] ENGINE_LABELS = { "Kokoro", "CosyVoice3", "Chatterbox" };
        // RTF from the engines' TokensPerSecond: Kokoro counts 600-sample frames (24 kHz -> 40/s of
        // audio); CosyVoice3/Chatterbox speech tokens run at 25 Hz. RTF = tokensPerAudioSec / TPS.
        static readonly float[] TOKENS_PER_AUDIO_SEC = { 40f, 25f, 25f };
        static readonly string[] PREFERRED_VOICE = { "af_heart", "default", "conds" };

        [Header("UI (wired by VoiceLabBuilder)")]
        [SerializeField] TMP_Dropdown engineDropdown;
        [SerializeField] TMP_Dropdown voiceDropdown;
        [SerializeField] Slider pitchSlider;
        [SerializeField] Slider speedSlider;
        [SerializeField] TMP_Text pitchValue;
        [SerializeField] TMP_Text speedValue;
        [SerializeField] CanvasGroup speedRow;
        [SerializeField] TMP_InputField textInput;
        [SerializeField] TMP_InputField presetNameInput;
        [SerializeField] Button speakButton;
        [SerializeField] Button stopButton;
        [SerializeField] Button saveButton;
        [SerializeField] Button sampleShortButton;
        [SerializeField] Button sampleMediumButton;
        [SerializeField] Button sampleLongButton;
        [SerializeField] TMP_Text statusLabel;

        // per-engine catalog, index = (int)Engine
        readonly string[] weightsDir = new string[3];
        readonly List<string>[] voices = new List<string>[3];
        readonly bool[] available = new bool[3];
        LLMQuant chatterboxQuant = LLMQuant.INT8;

        Engine engine = Engine.Kokoro;
        string voiceName;

        // engines (one resident at a time) + their scene-facing voice components
        KokoroTTS kokoroTts;
        CosyVoiceModeling.CosyVoiceTTS cosyTts;
        ChatterboxTTS chatterTts;
        KokoroVoice kk;
        CosyVoiceModeling.CosyVoiceVoice cv;
        ChatterboxVoice cb;

        AudioSource source;

        // audition metrics
        float speakPressedAt, lastTtfa = -1f, lastRtf = -1f;
        bool ttfaPending;
        float statusTimer;
        string note = "";
        bool settingUi;

        // ---------------- preset file ----------------
        [Serializable]
        public class VoicePreset
        {
            public string name;
            public string engine;   // NPCChatBase.TtsModel member name
            public string voice;    // ttsVoice
            public float pitch;     // voicePitch
            public float speed;     // KokoroVoice.speed (1 for the other engines)
            public string note;     // the NPCChatBase inspector line, human-readable
        }

        [Serializable]
        public class PresetFile { public List<VoicePreset> presets = new List<VoicePreset>(); }

        // ---------------- lifecycle ----------------

        void Awake()
        {
            source = GetComponent<AudioSource>();
            source.playOnAwake = false;
            source.spatialBlend = 0f;

            DetectEngines();
            WireUi();

            int first = -1;
            for (int i = 0; i < available.Length; i++)
                if (available[i]) { first = i; break; }
            if (available[(int)Engine.Kokoro]) first = (int)Engine.Kokoro;   // Kokoro is the priority pick

            if (first < 0)
            {
                note = "No TTS weights exported on this machine — expected manifests under " +
                       "Assets/Resources/Weights/weights_*.";
                if (speakButton != null) speakButton.interactable = false;
                if (voiceDropdown != null) voiceDropdown.interactable = false;
                RefreshStatus();
                return;
            }
            if (engineDropdown != null) engineDropdown.SetValueWithoutNotify(first);
            SelectEngine((Engine)first);
        }

        void Update()
        {
            // time-to-first-audio: from the SPEAK press to the first audible sample
            if (ttfaPending && IsAudibleNow())
            {
                lastTtfa = Time.realtimeSinceStartup - speakPressedAt;
                ttfaPending = false;
            }

            // RTF: sample the engine's TokensPerSecond while a synthesis is in flight (the
            // engines zero it when idle) and keep the last reading
            float tps = CurrentTps();
            if (tps > 0.01f) lastRtf = TOKENS_PER_AUDIO_SEC[(int)engine] / tps;

            statusTimer -= Time.unscaledDeltaTime;
            if (statusTimer <= 0f) { statusTimer = 0.2f; RefreshStatus(); }
        }

        void OnDestroy() => ReleaseEngines();   // components die with the GO; free the GPU weights

        // ---------------- engine catalog ----------------

        void DetectEngines()
        {
            weightsDir[(int)Engine.Kokoro] = Directory.Exists(KOKORO_INT8) ? KOKORO_INT8 : KOKORO_FP16;
            weightsDir[(int)Engine.CosyVoice3] = Directory.Exists(COSY_INT8) ? COSY_INT8 : COSY_FP16;
            bool cbInt8 = Directory.Exists(CB_INT8);
            chatterboxQuant = cbInt8 ? LLMQuant.INT8 : LLMQuant.FP16;
            weightsDir[(int)Engine.Chatterbox] = cbInt8 ? CB_INT8 : CB_FP16;

            for (int i = 0; i < 3; i++)
            {
                voices[i] = ParseVoices(weightsDir[i], (Engine)i);
                available[i] = voices[i] != null && voices[i].Count > 0;
            }
        }

        // Baked voices from the weights manifest. Kokoro rows: "voices/<name>"; CosyVoice3 rows:
        // "voices/<name>/<file>" (grouped by <name>); Chatterbox voices are conds groups, keyed
        // by their "<name>/t3_speaker_emb" row ("conds", "conds_elder", ...).
        static List<string> ParseVoices(string dir, Engine e)
        {
            string manifest = dir + "/manifest.tsv";
            if (!File.Exists(manifest)) return null;

            var found = new List<string>();
            foreach (string line in File.ReadLines(manifest))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                int tab = line.IndexOf('\t');
                string key = tab < 0 ? line : line.Substring(0, tab);
                string[] parts = key.Split('/');

                string v = null;
                if (e == Engine.Chatterbox)
                {
                    if (parts.Length == 2 && parts[1] == "t3_speaker_emb") v = parts[0];
                }
                else if (parts.Length >= 2 && parts[0] == "voices")
                {
                    v = parts[1];
                }
                if (v != null && !found.Contains(v)) found.Add(v);
            }
            return found;
        }

        // ---------------- UI wiring ----------------

        void WireUi()
        {
            if (engineDropdown != null)
            {
                var opts = new List<string>(3);
                for (int i = 0; i < 3; i++)
                    opts.Add(available[i] ? ENGINE_LABELS[i] : ENGINE_LABELS[i] + "  (not exported)");
                engineDropdown.ClearOptions();
                engineDropdown.AddOptions(opts);
                engineDropdown.onValueChanged.AddListener(i => { if (!settingUi) SelectEngine((Engine)i); });
            }
            if (voiceDropdown != null)
                voiceDropdown.onValueChanged.AddListener(i =>
                {
                    if (settingUi || !available[(int)engine]) return;
                    string v = voices[(int)engine][i];
                    if (v != voiceName) ApplyVoice(v);
                });

            if (pitchSlider != null)
                pitchSlider.onValueChanged.AddListener(v =>
                {
                    if (pitchValue != null) pitchValue.text = v.ToString("0.00");
                    // pitch is live on all engines — the components push it to the AudioSource
                    if (kk != null) kk.pitch = v;
                    if (cv != null) cv.pitch = v;
                    if (cb != null) cb.pitch = v;
                });
            if (speedSlider != null)
                speedSlider.onValueChanged.AddListener(v =>
                {
                    if (speedValue != null) speedValue.text = v.ToString("0.00");
                    if (kk != null) kk.speed = v;   // Kokoro-only; applies from the next clause
                });
            if (pitchValue != null && pitchSlider != null) pitchValue.text = pitchSlider.value.ToString("0.00");
            if (speedValue != null && speedSlider != null) speedValue.text = speedSlider.value.ToString("0.00");

            if (speakButton != null) speakButton.onClick.AddListener(OnSpeakPressed);
            if (stopButton != null) stopButton.onClick.AddListener(() => { StopActive(); note = ""; RefreshStatus(); });
            if (saveButton != null) saveButton.onClick.AddListener(OnSavePressed);
            if (sampleShortButton != null) sampleShortButton.onClick.AddListener(() => SetText(SAMPLE_SHORT));
            if (sampleMediumButton != null) sampleMediumButton.onClick.AddListener(() => SetText(SAMPLE_MEDIUM));
            if (sampleLongButton != null) sampleLongButton.onClick.AddListener(() => SetText(SAMPLE_LONG));
        }

        void SetText(string t) { if (textInput != null) textInput.text = t; }

        // ---------------- engine/voice switching ----------------

        void SelectEngine(Engine e)
        {
            engine = e;
            bool ok = available[(int)e];

            settingUi = true;
            if (voiceDropdown != null)
            {
                voiceDropdown.ClearOptions();
                if (ok) voiceDropdown.AddOptions(voices[(int)e]);
                voiceDropdown.interactable = ok;
            }
            int idx = 0;
            if (ok)
            {
                idx = Mathf.Max(0, voices[(int)e].IndexOf(PREFERRED_VOICE[(int)e]));
                if (voiceDropdown != null) voiceDropdown.SetValueWithoutNotify(idx);
            }
            if (voiceDropdown != null) voiceDropdown.RefreshShownValue();
            settingUi = false;

            bool kokoro = e == Engine.Kokoro;
            if (speedRow != null) speedRow.alpha = kokoro ? 1f : 0.35f;      // speed is Kokoro-only
            if (speedSlider != null) speedSlider.interactable = kokoro;
            if (speakButton != null) speakButton.interactable = ok;

            if (!ok)
            {
                DestroyVoiceComponents();
                ReleaseEngines();
                voiceName = null;
                lastTtfa = lastRtf = -1f;
                note = ENGINE_LABELS[(int)e] + " weights are not exported on this machine " +
                       "(missing manifest under Assets/Resources/Weights/).";
                RefreshStatus();
                return;
            }
            note = "";
            ApplyVoice(voices[(int)e][idx]);
        }

        void ApplyVoice(string v)
        {
            voiceName = v;
            DestroyVoiceComponents();
            ReleaseEngines();
            lastTtfa = lastRtf = -1f;
            ttfaPending = false;
            BuildEngineAndVoice();
            RefreshStatus();
        }

        // Fresh engine bound to the chosen voice + fresh voice component bound to that engine.
        // SetSharedTTS BEFORE AddComponent so the component's EnsureTts adopts our instance
        // instead of constructing its own.
        void BuildEngineAndVoice()
        {
            string dir = weightsDir[(int)engine];
            float pitch = pitchSlider != null ? pitchSlider.value : 1f;

            switch (engine)
            {
                case Engine.Kokoro:
                    kokoroTts = new KokoroTTS(dir, voice: voiceName);
                    KokoroVoice.SetSharedTTS(kokoroTts);
                    kk = gameObject.AddComponent<KokoroVoice>();
                    kk.streaming = true;                 // speed only applies on the streaming path
                    kk.weightsPath = dir;
                    kk.voiceName = voiceName;
                    kk.pitch = pitch;
                    kk.speed = speedSlider != null ? speedSlider.value : 1f;
                    break;

                case Engine.CosyVoice3:
                    cosyTts = new CosyVoiceModeling.CosyVoiceTTS(dir, voiceName);
                    CosyVoiceModeling.CosyVoiceVoice.SetSharedTTS(cosyTts);
                    cv = gameObject.AddComponent<CosyVoiceModeling.CosyVoiceVoice>();
                    cv.weightsPath = dir;
                    cv.voiceName = voiceName;
                    cv.pitch = pitch;
                    cv.prebufferSeconds = 2.5f;          // RTF ~2.9 pre-A6: one early gap beats stutter
                    break;

                default:   // Chatterbox — ctor resolves the weights dir itself from the quant
                    chatterTts = new ChatterboxTTS(voice: voiceName, quantization: chatterboxQuant);
                    ChatterboxVoice.SetSharedTTS(chatterTts);
                    cb = gameObject.AddComponent<ChatterboxVoice>();
                    cb.streaming = true;
                    cb.voiceName = voiceName;
                    cb.quantization = chatterboxQuant;
                    cb.pitch = pitch;
                    break;
            }
        }

        void DestroyVoiceComponents()
        {
            if (kk != null) { kk.StopSpeaking(); Destroy(kk); kk = null; }
            if (cv != null) { cv.StopSpeaking(); Destroy(cv); cv = null; }
            if (cb != null) { cb.StopSpeaking(); Destroy(cb); cb = null; }
            if (source != null) { source.Stop(); source.clip = null; source.loop = false; }
        }

        void ReleaseEngines()
        {
            if (kokoroTts != null) { kokoroTts.Release(); kokoroTts = null; KokoroVoice.SetSharedTTS(null); }
            if (cosyTts != null) { cosyTts.Release(); cosyTts = null; CosyVoiceModeling.CosyVoiceVoice.SetSharedTTS(null); }
            if (chatterTts != null) { chatterTts.Release(); chatterTts = null; ChatterboxVoice.SetSharedTTS(null); }
        }

        // ---------------- speak / stop ----------------

        void OnSpeakPressed()
        {
            if (!available[(int)engine] || voiceName == null)
            {
                note = "Select an exported engine first.";
                RefreshStatus();
                return;
            }
            if (!ActiveIsReady())
            {
                note = "Engine still streaming weights — wait for Ready, then hit SPEAK.";
                RefreshStatus();
                return;
            }
            string text = textInput != null ? textInput.text.Trim() : "";
            if (text.Length == 0)
            {
                note = "Type some text (or hit a sample button) first.";
                RefreshStatus();
                return;
            }

            StopActive();
            note = "";
            speakPressedAt = Time.realtimeSinceStartup;
            ttfaPending = true;
            lastTtfa = -1f;
            lastRtf = -1f;

            switch (engine)
            {
                case Engine.Kokoro:     kk.Say(text); break;
                case Engine.CosyVoice3: cv.Say(text); break;
                default:                cb.Say(text); break;
            }
        }

        void StopActive()
        {
            ttfaPending = false;
            if (kk != null) kk.StopSpeaking();
            if (cv != null) cv.StopSpeaking();
            if (cb != null) cb.StopSpeaking();
        }

        // ---------------- preset save ----------------

        void OnSavePressed()
        {
            if (voiceName == null)
            {
                note = "Nothing to save — select an exported engine/voice first.";
                RefreshStatus();
                return;
            }
            string presetName = presetNameInput != null ? presetNameInput.text.Trim() : "";
            if (presetName.Length == 0)
            {
                note = "Give the preset a name first.";
                RefreshStatus();
                return;
            }

            float pitch = Mathf.Round((pitchSlider != null ? pitchSlider.value : 1f) * 100f) / 100f;
            float speed = engine == Engine.Kokoro && speedSlider != null
                        ? Mathf.Round(speedSlider.value * 100f) / 100f : 1f;
            string engineName = ENGINE_LABELS[(int)engine];

            PresetFile file = LoadPresetFile();
            VoicePreset p = file.presets.Find(x => x.name == presetName);
            if (p == null) { p = new VoicePreset(); file.presets.Add(p); }
            p.name = presetName;
            p.engine = engineName;
            p.voice = voiceName;
            p.pitch = pitch;
            p.speed = speed;
            p.note = $"NPCChatBase: ttsModel={engineName} ttsVoice={voiceName} voicePitch={pitch:0.##}";

            File.WriteAllText(PRESETS_PATH, JsonUtility.ToJson(file, true));
#if UNITY_EDITOR
            UnityEditor.AssetDatabase.ImportAsset(PRESETS_PATH);
#endif

            string quantNote = engine == Engine.Chatterbox ? $" | ttsQuantization: {chatterboxQuant}" : "";
            string speedNote = engine == Engine.Kokoro && Mathf.Abs(speed - 1f) > 0.005f
                ? $"  (speed {speed:0.00} is KokoroVoice.speed — NPCChatBase has no speed field)" : "";
            Debug.Log($"[VoiceLab] Preset '{presetName}' saved to {PRESETS_PATH}");
            Debug.Log($"[VoiceLab] NPCChatBase inspector values -> ttsModel: {engineName} | " +
                      $"ttsVoice: {voiceName} | voicePitch: {pitch:0.00}{quantNote}{speedNote}");
            note = $"Preset '{presetName}' saved (values logged to the Console).";
            RefreshStatus();
        }

        PresetFile LoadPresetFile()
        {
            try
            {
                if (File.Exists(PRESETS_PATH))
                {
                    PresetFile f = JsonUtility.FromJson<PresetFile>(File.ReadAllText(PRESETS_PATH));
                    if (f != null && f.presets != null) return f;
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("[VoiceLab] voice_presets.json unreadable — starting a fresh file. " + e.Message);
            }
            return new PresetFile();
        }

        // ---------------- status ----------------

        bool ActiveIsReady()
        {
            switch (engine)
            {
                case Engine.Kokoro:     return kk != null && kk.IsReady;
                case Engine.CosyVoice3: return cv != null && cv.IsReady;
                default:                return cb != null && cb.IsReady;
            }
        }

        bool ActiveIsSpeaking()
        {
            switch (engine)
            {
                case Engine.Kokoro:     return kk != null && kk.IsSpeaking;
                case Engine.CosyVoice3: return cv != null && cv.IsSpeaking;
                default:                return cb != null && cb.IsSpeaking;
            }
        }

        bool IsAudibleNow()
        {
            switch (engine)
            {
                case Engine.Kokoro:     return kk != null && kk.IsAudioPlaying;
                // CosyVoice/Chatterbox pause their streaming source when starved -> isPlaying == audible
                case Engine.CosyVoice3: return cv != null && source != null && source.isPlaying;
                default:                return cb != null && source != null && source.isPlaying;
            }
        }

        float CurrentTps()
        {
            switch (engine)
            {
                case Engine.Kokoro:     return kokoroTts != null ? kokoroTts.TokensPerSecond : 0f;
                case Engine.CosyVoice3: return cosyTts != null ? cosyTts.TokensPerSecond : 0f;
                default:                return chatterTts != null ? chatterTts.TokensPerSecond : 0f;
            }
        }

        static string Mb(long bytes) => (bytes / (1024f * 1024f)).ToString("0.0");

        void RefreshStatus()
        {
            if (statusLabel == null) return;
            var sb = new StringBuilder(256);

            sb.Append("Engine: ").Append(ENGINE_LABELS[(int)engine])
              .Append("    Voice: ").Append(voiceName ?? "-")
              .Append("    Weights: ").Append(Path.GetFileName(weightsDir[(int)engine]));
            sb.Append('\n');

            // load state: Kokoro/CosyVoice are ModelBase (Residency + byte progress); Chatterbox
            // predates ModelBase and only exposes IsReady
            ModelBase m = engine == Engine.Kokoro ? (ModelBase)kokoroTts
                        : engine == Engine.CosyVoice3 ? cosyTts : null;
            if (m != null)
                sb.Append("Load: ").Append(m.Residency).Append("  ")
                  .Append((m.LoadProgress * 100f).ToString("0")).Append("%  (")
                  .Append(Mb(m.UploadedWeightBytes)).Append(" / ").Append(Mb(m.TotalWeightBytes)).Append(" MB)");
            else if (chatterTts != null)
                sb.Append("Load: ").Append(chatterTts.IsReady
                    ? "Ready" : "streaming weights... (ChatterboxTTS exposes no byte counter)");
            else
                sb.Append("Load: -");
            sb.Append('\n');

            sb.Append("State: ").Append(ActiveIsSpeaking() ? "SPEAKING" : ActiveIsReady() ? "ready" : "loading");
            if (lastTtfa >= 0f) sb.Append("    time-to-first-audio ").Append(lastTtfa.ToString("0.00")).Append("s");
            if (lastRtf > 0f)
            {
                sb.Append("    RTF~").Append(lastRtf.ToString("0.00"))
                  .Append(" (").Append((1f / lastRtf).ToString("0.0")).Append("x realtime");
                if (engine == Engine.Chatterbox) sb.Append(", T3 decode");
                sb.Append(')');
            }

            if (!string.IsNullOrEmpty(note)) sb.Append('\n').Append(note);
            statusLabel.text = sb.ToString();
        }
    }
}

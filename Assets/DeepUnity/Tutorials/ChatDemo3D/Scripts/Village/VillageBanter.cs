using System;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Plays the strolling pair's scripted back-and-forth: alternating authored lines, each
    /// spoken out loud through that villager's own pocket-tts voice (spatial, at the NPC) with
    /// the text typing itself into the bubble above their head in step with the audio (the
    /// bubble follows the voice's OnClauseSpoken events — same sync the dialogue window uses).
    ///
    /// Scripted rather than generated on purpose: this is the ambient Witcher-style walla layer.
    /// The GPU's LLM budget belongs to the REAL conversation the player can start with E at any
    /// moment — including mid-line, which is why every gate here re-checks InConversation.
    /// A line only starts when both voices are streamed in (the walk-up prefetch zone does that)
    /// and the player is close enough to actually hear it.
    /// </summary>
    public class VillageBanter : MonoBehaviour
    {
        [Serializable]
        public struct Line
        {
            public int speaker;         // 0 = strollerA, 1 = strollerB
            [TextArea] public string text;
        }

        [SerializeField] private VillageStroller strollerA;
        [SerializeField] private VillageStroller strollerB;
        [SerializeField] private VillageSpeechBubble bubbleA;
        [SerializeField] private VillageSpeechBubble bubbleB;
        [SerializeField] private Line[] lines;
        [Tooltip("Banter only runs with the player within this range of the pair (hearing distance).")]
        [SerializeField] private float earshotRadius = 18f;
        [Tooltip("Beat between one line ending and the reply starting, seconds.")]
        [SerializeField] private float linePause = 0.9f;

        Transform playerT;
        int next;              // index into lines
        float nextLineAt;      // Time.time gate
        bool suspended;
        VillageStroller lastSpeaker;

        void Start()
        {
            var playerGO = GameObject.FindWithTag("Player");
            if (playerGO != null) playerT = playerGO.transform;
            nextLineAt = Time.time + 2f;
        }

        void Update()
        {
            if (suspended || lines == null || lines.Length == 0) return;
            if (strollerA == null || strollerB == null) return;
            // ANY open dialogue silences the walla — the fishmonger's included: his replies are
            // already synthesizing on the shared TTS engine, and a second and third concurrent
            // voice is exactly what a 4 GB card cannot afford (and what the audio duck is for)
            if (NPCChatBase.AnyConversationOpen) return;
            if (strollerA.InConversation || strollerB.InConversation) return;

            // hold (don't advance the script) while a line is still sounding
            if (lastSpeaker != null && lastSpeaker.VoiceBusy)
            {
                nextLineAt = Time.time + linePause;
                return;
            }

            if (Time.time < nextLineAt) return;

            // both voices resident (the prefetch zone streams them during the walk-up) and the
            // player near enough for spatial audio to carry — otherwise the lines would play to
            // an empty street and drain the ring for nothing
            if (!strollerA.VoiceReady || !strollerB.VoiceReady) return;
            if (playerT != null)
            {
                Vector3 d = playerT.position - strollerA.transform.position;
                d.y = 0f;
                if (d.sqrMagnitude > earshotRadius * earshotRadius) return;
            }

            Line line = lines[next];
            next = (next + 1) % lines.Length;

            VillageStroller speaker = line.speaker == 0 ? strollerA : strollerB;
            VillageSpeechBubble bubble = line.speaker == 0 ? bubbleA : bubbleB;
            lastSpeaker = speaker;
            bubble?.BeginUtterance();
            speaker.SpeakAmbient(line.text);
            // guard against HasPendingSpeech lagging Say() by a frame — without this the next
            // Update could read the voice as idle and fire the reply on top of this line
            nextLineAt = Time.time + 1f;
        }

        public void Suspend()
        {
            suspended = true;
            bubbleA?.HideNow();
            bubbleB?.HideNow();
        }

        public void Resume()
        {
            suspended = false;
            nextLineAt = Time.time + linePause + 1.5f;   // let the walk resume before the talk does
        }
    }
}

using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    // End-to-end play-mode test of the 2D farm demo (real ChatDemo2D scene):
    //   T1  enter Hobb's prefetch circle  -> Gemma slow-prefetches, frames stay clean
    //   T2  talk to Hobb (LlmOnly)        -> dialogue opens, reply streams token-by-token
    //   T3  GIVE-ITEMS flow               -> basket (injected) -> Give -> hidden prompt ->
    //                                        thank-you reply -> COINS land after it ends
    //   T4  leave the circle              -> Hobb's LLM unloads immediately (ResetEveryTime)
    //   T5  Marla (ContinueWhereLeftOff)  -> talk, leave her circle, LLM RELEASES (residency is
    //                                        the zone's job; the old KeepAlive mode was removed)
    //   T6  Hobb circle re-entry          -> LLM re-lands normally after both released
    // Report: ProbeLogs/npc_e2e_2d.md + .done marker (via NpcE2E2DRunner).
    public class NpcE2E2DProbe : MonoBehaviour
    {
        const string ReportPath = "ProbeLogs/npc_e2e_2d.md";
        const string MarkerPath = "ProbeLogs/npc_e2e_2d.done";

        readonly StringBuilder sb = new StringBuilder();
        readonly List<string> errors = new List<string>();
        readonly List<float> frames = new List<float>();
        bool recording;
        int failures;

        Transform player;
        Rigidbody2D playerRb;
        NPCChatBase hobb, marla;
        INPCChatWindow window;
        Tutorials.ChatDemo2D.FarmingSystem farm;

        void Awake() => Application.logMessageReceived += OnLog;
        void OnDestroy() => Application.logMessageReceived -= OnLog;

        void OnLog(string msg, string stack, LogType type)
        {
            if (type == LogType.Exception || type == LogType.Error)
                if (errors.Count < 40) errors.Add($"{type}: {msg.Substring(0, Mathf.Min(160, msg.Length))}");
        }

        void Update()
        {
            if (recording) frames.Add(Time.unscaledDeltaTime * 1000f);
        }

        IEnumerator Start()
        {
            sb.AppendLine("# 2D farm demo end-to-end probe — " + System.DateTime.Now.ToString("yyyy-MM-dd HH:mm"));
            yield return null;

            var playerGO = GameObject.FindWithTag("Player");
            player = playerGO != null ? playerGO.transform : null;
            playerRb = playerGO != null ? playerGO.GetComponent<Rigidbody2D>() : null;
            foreach (var npc in FindObjectsOfType<NPCChatBase>(true))
            {
                if (npc.gameObject.name.Contains("Hobb")) hobb = npc;
                if (npc.gameObject.name.Contains("Marla")) marla = npc;
            }
            window = FindObjectOfType<Tutorials.ChatDemo2D.ChatWindow2D>(true);
            farm = FindObjectOfType<Tutorials.ChatDemo2D.FarmingSystem>(true);

            if (player == null || hobb == null || marla == null || window == null || farm == null)
            {
                Fail($"scene wiring: player={(player != null)} hobb={(hobb != null)} marla={(marla != null)} " +
                     $"window={(window != null)} farm={(farm != null)}");
                Finish(); yield break;
            }

            yield return PhaseR("T0 boot settle (prewarm window)", 5f, null);

            // T1 — enter Hobb's prefetch circle (r=7): Gemma must slow-prefetch cleanly
            Teleport(ZonePoint(hobb.transform, 5f));
            float t0 = Time.unscaledTime;
            yield return PhaseR("T1 Hobb circle entry -> Gemma lands", 30f, () => hobb.LlmReady);
            Check(hobb.LlmReady, $"T1 Hobb LLM ready in {Time.unscaledTime - t0:0.0} s");

            // T2 — walk into the talk trigger and have a normal text-only exchange
            Teleport(ZonePoint(hobb.transform, 1.2f));
            yield return new WaitForSecondsRealtime(1.5f);
            hobb.StartInteraction();
            yield return PhaseR("T2a Hobb dialogue open", 40f,
                                () => hobb.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(hobb.State == NPCChatBase.NPCState.WaitingInInteraction, "T2a dialogue reached Waiting");
            window.InputField.text = "Good morning! One short sentence please.";
            hobb.AskNPC();
            yield return PhaseR("T2b Hobb reply (token stream)", 90f,
                                () => hobb.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(hobb.State == NPCChatBase.NPCState.WaitingInInteraction, "T2b reply finished");

            // T3 — GIVE ITEMS: inject a harvest into the basket, give it, expect thanks + coins
            var harvestedField = typeof(Tutorials.ChatDemo2D.FarmingSystem)
                .GetField("harvested", BindingFlags.NonPublic | BindingFlags.Instance);
            var basket = (int[])harvestedField.GetValue(farm);
            basket[0] = 2; basket[1] = 1; basket[2] = 0;   // 2 carrots + 1 turnip = 2*2+1*3 = 7 g
            Check(farm.HasAnyHarvest, "T3a injected basket registers as harvest");
            int coinsBefore = farm.Coins;

            var hobb2d = (Tutorials.ChatDemo2D.NPCInteractor2D)hobb;
            hobb2d.GiveItems();
            Check(!farm.HasAnyHarvest, "T3b basket emptied on Give");
            yield return PhaseR("T3c thank-you reply", 90f,
                                () => hobb.State == NPCChatBase.NPCState.WaitingInInteraction);
            // coins land in OnReplyFinished — same frame as the state flip
            Check(farm.Coins - coinsBefore == 7,
                  $"T3d coins paid AFTER the thank-you ({farm.Coins - coinsBefore} g, expected 7)");

            hobb.CloseInteraction();
            yield return new WaitForSecondsRealtime(0.5f);

            // T4 — leave the circle: ResetEveryTime Hobb must release immediately
            Teleport(hobb.transform.position + (Vector3)(Vector2.down * 20f));
            yield return PhaseR("T4 circle exit -> immediate unload", 12f, () => !hobb.LlmLoaded);
            Check(!hobb.LlmLoaded, "T4 Hobb LLM released after leaving the circle");

            // T5 — Marla is ContinueWhereLeftOff: talk, leave her circle — the LLM RELEASES like
            // everyone else's (residency belongs to the zone; the conversation persists to disk)
            Teleport(ZonePoint(marla.transform, 5f));
            yield return PhaseR("T5a Marla circle entry -> Gemma lands", 30f, () => marla.LlmReady);
            Teleport(ZonePoint(marla.transform, 1.2f));
            yield return new WaitForSecondsRealtime(1.5f);
            marla.StartInteraction();
            yield return PhaseR("T5b Marla dialogue open", 40f,
                                () => marla.State == NPCChatBase.NPCState.WaitingInInteraction);
            window.InputField.text = "What tea do you have today? One sentence.";
            marla.AskNPC();
            yield return PhaseR("T5c Marla reply", 90f,
                                () => marla.State == NPCChatBase.NPCState.WaitingInInteraction);
            Check(marla.State == NPCChatBase.NPCState.WaitingInInteraction, "T5c Marla reply finished");
            marla.CloseInteraction();
            yield return new WaitForSecondsRealtime(0.5f);
            Teleport(marla.transform.position + (Vector3)(Vector2.down * 20f));
            yield return PhaseR("T5d Marla circle exit -> release (KV save may defer it)", 20f, () => !marla.LlmLoaded);
            Check(!marla.LlmLoaded, "T5d ContinueWhereLeftOff: Marla's LLM released after leaving her circle");

            // T6 — re-entry after both released: the LLM re-lands through the normal prefetch
            Teleport(ZonePoint(hobb.transform, 5f));
            float shareT = Time.unscaledTime;
            yield return PhaseR("T6 Hobb re-entry (fresh prefetch)", 40f, () => hobb.LlmReady);
            Check(hobb.LlmReady, $"T6 Hobb ready in {Time.unscaledTime - shareT:0.00} s after re-entry");

            Finish();
        }

        IEnumerator PhaseR(string name, float timeout, System.Func<bool> until)
        {
            frames.Clear(); recording = true;
            float start = Time.unscaledTime;
            while (Time.unscaledTime - start < timeout)
            {
                if (until != null && until()) break;
                yield return null;
            }
            recording = false;
            bool timedOut = until != null && !until();
            int over17 = 0, over33 = 0; float worst = 0;
            foreach (float f in frames)
            {
                if (f > 16.7f) over17++;
                if (f > 33.4f) over33++;
                if (f > worst) worst = f;
            }
            sb.AppendLine($"\n## {name}");
            sb.AppendLine($"- {Time.unscaledTime - start:0.00} s, {frames.Count} frames | >16.7 ms: {over17} | >33.4 ms: {over33} | worst {worst:0.0} ms{(timedOut ? "  **TIMEOUT**" : "")}");
            if (timedOut) failures++;
        }

        void Check(bool ok, string what)
        {
            sb.AppendLine($"- {(ok ? "PASS" : "**FAIL**")}: {what}");
            if (!ok) failures++;
        }

        void Fail(string what) { sb.AppendLine($"- **FAIL**: {what}"); failures++; }

        void Teleport(Vector3 pos)
        {
            if (playerRb != null) playerRb.position = pos;
            player.position = pos;
        }

        Vector3 ZonePoint(Transform npc, float dist)
        {
            Vector2 d = (Vector2)player.position - (Vector2)npc.position;
            Vector2 dir = d.sqrMagnitude < 0.01f ? Vector2.down : d.normalized;
            return npc.position + (Vector3)(dir * dist);
        }

        void Finish()
        {
            sb.AppendLine($"\n## Console errors/exceptions during the run: {errors.Count}");
            foreach (var e in errors) sb.AppendLine($"- {e}");
            sb.AppendLine($"\n## VERDICT: {(failures == 0 && errors.Count == 0 ? "ALL PASS" : $"{failures} failures, {errors.Count} errors")}");
            Directory.CreateDirectory("ProbeLogs");
            File.WriteAllText(ReportPath, sb.ToString());
            File.WriteAllText(MarkerPath, "done");
            Debug.Log($"[NpcE2E2DProbe] report -> {ReportPath}");
        }
    }
}

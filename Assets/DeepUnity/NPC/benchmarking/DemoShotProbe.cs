using System.Collections;
using System.IO;
using System.Reflection;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepUnity
{
    // Play-mode screenshot rig for the dissertation: drives a REAL conversation in a demo scene
    // (teleport to the NPC, open the dialogue, ask a question) and renders the main camera to a
    // high-res PNG while the reply is streaming into the window, then again when it finished.
    // Batch-mode realities this works around (vs the E2E probes it descends from):
    //   * no game view  -> captures are manual Camera.Render() into a 1920x1080 RT;
    //   * overlay UI would be invisible to that render -> every ScreenSpaceOverlay canvas is
    //     switched to ScreenSpaceCamera for the run;
    //   * no audio device -> the RUNNER flips every NPC to LlmOnly first, so the window types
    //     per generated token instead of following a voice that would never play;
    //   * nothing renders between captures -> every Animator is forced to AlwaysAnimate, or
    //     culled rigs would pose-freeze and screenshot in bind pose.
    // Spawned into the opened scene by DemoShotRunner; exits the editor process when done.
    public class DemoShotProbe : MonoBehaviour
    {
        public string npcNameContains;
        [TextArea] public string question;
        public string shotPrefix;
        public bool villageBanterShot;     // stage the strolling pair + bubble before the dialogue
        public float midShotDelay = 2.5f;  // seconds into the streaming reply for the "mid" frame

        const string OutDir = "ProbeLogs/demo_shots";
        const int W = 1920, H = 1080;

        Transform player;
        CharacterController playerCC;
        Rigidbody2D playerRb;
        NPCChatBase npc;
        NPCDialogueWindow window;
        Camera cam;
        bool canvasesSwitched;
        int exitCode;

        IEnumerator Start()
        {
            yield return null;
            yield return null;   // scene Start()s + frame-0 prewarm done

            foreach (var a in FindObjectsOfType<Animator>(true))
                a.cullingMode = AnimatorCullingMode.AlwaysAnimate;

            // diagnostics/meta HUD out of the dissertation figures (user 2026-08-10): the FPS
            // counter and the souls purse belong to the playable demo, not to a paper screenshot
            foreach (var fps in FindObjectsOfType<Tutorials.ChatDemo3D.FpsCounter>(true))
                fps.gameObject.SetActive(false);
            foreach (var t in FindObjectsOfType<Transform>(true))
                if (t.name == "SoulsCounter") t.gameObject.SetActive(false);

            var playerGO = GameObject.FindWithTag("Player");
            player = playerGO != null ? playerGO.transform : null;
            playerCC = playerGO != null ? playerGO.GetComponent<CharacterController>() : null;
            playerRb = playerGO != null ? playerGO.GetComponent<Rigidbody2D>() : null;
            foreach (var n in FindObjectsOfType<NPCChatBase>(true))
                if (n.gameObject.name.Contains(npcNameContains)) { npc = n; break; }
            window = FindObjectOfType<NPCDialogueWindow>(true);
            cam = Camera.main;

            if (player == null || npc == null || window == null || cam == null)
            {
                Debug.LogError($"[DemoShotProbe] wiring: player={player != null} npc={npc != null} " +
                               $"window={window != null} cam={cam != null}");
                Quit(2);
                yield break;
            }

            Directory.CreateDirectory(OutDir);

            if (villageBanterShot)
                yield return BanterShots();

            // walk up: inside the prefetch zone AND the talk trigger, then wait for the model
            Teleport(NearPoint(1.5f));
            float t0 = Time.unscaledTime;
            yield return Until(() => npc.LlmReady, 300f, "LlmReady");
            Debug.Log($"[DemoShotProbe] LLM ready in {Time.unscaledTime - t0:0.0}s");
            yield return new WaitForSecondsRealtime(1.5f);   // physics tick: trigger sees the player

            npc.StartInteraction();
            yield return Until(() => npc.State == NPCChatBase.NPCState.WaitingInInteraction, 120f, "dialogue open");
            yield return new WaitForSecondsRealtime(1.6f);   // camera blend to the dialogue framing

            window.InputField.text = question;
            npc.AskNPC();
            yield return Until(() => npc.State == NPCChatBase.NPCState.TalkingInInteraction, 90f, "reply start");
            yield return new WaitForSecondsRealtime(midShotDelay);
            Capture(shotPrefix + "_mid");

            yield return Until(() => npc.State == NPCChatBase.NPCState.WaitingInInteraction, 240f, "reply done");
            yield return new WaitForSecondsRealtime(0.4f);
            Capture(shotPrefix + "_done");

            Debug.Log("[DemoShotProbe] DONE");
            Quit(exitCode);
        }

        // -------------------------------------------------------------- staged village walla
        // No audio device means no OnClauseSpoken, so the bubble is fed its clause directly (the
        // same private entry the voice event calls) and types itself out exactly as in play.
        IEnumerator BanterShots()
        {
            Tutorials.ChatDemo3D.VillageStroller odo = null;
            foreach (var s in FindObjectsOfType<Tutorials.ChatDemo3D.VillageStroller>())
                if (s.gameObject.name.Contains("Odo")) { odo = s; break; }
            if (odo == null) { Debug.LogError("[DemoShotProbe] no Odo for the banter shot"); exitCode = 3; yield break; }

            // park the player far from the loop so PlayerBlocksPath never pauses the group
            Teleport(odo.transform.position + Vector3.up * 0.1f + new Vector3(12f, 0f, -14f));
            yield return new WaitForSecondsRealtime(6f);   // let the stroll settle into its rhythm

            var bubble = odo.GetComponentInChildren<Tutorials.ChatDemo3D.VillageSpeechBubble>(true);
            if (bubble != null)
            {
                bubble.BeginUtterance();
                var onClause = bubble.GetType().GetMethod("OnClause", BindingFlags.Instance | BindingFlags.NonPublic);
                onClause?.Invoke(bubble, new object[] { "Fish again at Bram's stall. Third week running, I swear.", 3.4f });
                yield return new WaitForSecondsRealtime(2.4f);   // mid-reveal (word-snapped): the sync is the point
            }

            var rig = cam.GetComponent<Tutorials.ChatDemo3D.SoulsCameraRig>();
            if (rig != null) rig.enabled = false;

            Vector3 mid = odo.transform.position + Vector3.up * 1.15f;
            cam.transform.position = odo.transform.position + odo.transform.forward * 4.3f
                                     - odo.transform.right * 1.1f + Vector3.up * 1.6f;
            cam.transform.rotation = Quaternion.LookRotation(mid - cam.transform.position);
            yield return null;   // the bubble billboards in LateUpdate — give it a frame to face us
            Capture(shotPrefix + "_banter");

            // second angle from the flank: the pair, the rope, the cow trailing
            cam.transform.position = odo.transform.position + odo.transform.forward * 2.6f
                                     + odo.transform.right * 3.6f + Vector3.up * 1.5f;
            cam.transform.rotation = Quaternion.LookRotation(
                (odo.transform.position - odo.transform.forward * 1.2f + Vector3.up * 1.0f) - cam.transform.position);
            yield return null;
            Capture(shotPrefix + "_banter_side");

            if (rig != null) rig.enabled = true;
        }

        // -------------------------------------------------------------- plumbing

        IEnumerator Until(System.Func<bool> cond, float timeout, string what)
        {
            float start = Time.unscaledTime;
            while (!cond() && Time.unscaledTime - start < timeout)
                yield return null;
            if (!cond())
            {
                Debug.LogError($"[DemoShotProbe] TIMEOUT waiting for {what} ({timeout:0}s)");
                Capture(shotPrefix + "_timeout_" + what.Replace(' ', '_'));
                Quit(4);
            }
        }

        void Teleport(Vector3 pos)
        {
            if (playerCC != null) playerCC.enabled = false;
            if (playerRb != null) { player.position = new Vector3(pos.x, pos.y, player.position.z); }
            else player.position = pos;
            if (playerCC != null) playerCC.enabled = true;
        }

        Vector3 NearPoint(float dist)
        {
            Vector3 d = player.position - npc.transform.position;
            if (playerRb != null) d.z = 0f; else d.y = 0f;
            d = d.sqrMagnitude < 0.01f ? (playerRb != null ? Vector3.left : Vector3.back) : d.normalized;
            Vector3 p = npc.transform.position + d * dist;
            if (playerRb != null) p.z = player.position.z; else p.y = player.position.y;
            return p;
        }

        void Capture(string name)
        {
            if (!canvasesSwitched)
            {
                // overlay canvases don't exist to a manual Camera.Render — re-home them on the camera
                foreach (var c in FindObjectsOfType<Canvas>(true))
                    if (c.isRootCanvas && c.renderMode == RenderMode.ScreenSpaceOverlay)
                    {
                        c.renderMode = RenderMode.ScreenSpaceCamera;
                        c.worldCamera = cam;
                        c.planeDistance = 1f;
                    }
                canvasesSwitched = true;
            }
            Canvas.ForceUpdateCanvases();

            var rt = new RenderTexture(W, H, 24) { antiAliasing = 4 };
            cam.targetTexture = rt;
            cam.Render();
            RenderTexture.active = rt;
            var tex = new Texture2D(W, H, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, W, H), 0, 0);
            tex.Apply();
            File.WriteAllBytes($"{OutDir}/{name}.png", tex.EncodeToPNG());
            Destroy(tex);
            RenderTexture.active = null;
            cam.targetTexture = null;
            rt.Release();
            Debug.Log($"[DemoShotProbe] shot -> {OutDir}/{name}.png");
        }

        void Quit(int code)
        {
#if UNITY_EDITOR
            EditorApplication.Exit(code);
#endif
        }
    }
}

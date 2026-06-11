using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

/// <summary>
/// File-based command bridge so an external tool (Claude Code) can drive the OPEN editor
/// without batch mode. It polls <project>/ClaudeBridge/ for cmd_*.json files, executes them
/// on the editor main thread and writes result_*.json next to them.
///
/// Command format (JSON):
///   { "action": "ping" }
///   { "action": "refresh" }                                       // AssetDatabase.Refresh (recompiles changed scripts)
///   { "action": "invoke", "method": "Full.Namespace.Type.StaticMethod" }
///   { "action": "menu",   "method": "DeepUnity/Build ChatDemo3D Scene" }
///   { "action": "screenshot", "path": "ProbeLogs/live.png",
///     "pos": [0,2,-9.6], "euler": [6,0,0], "width": 1600, "height": 900, "poseAnimators": true }
///
/// Results: { "status": "ok"|"error", "output": "..." } (output includes captured Debug.Log lines).
/// </summary>
[InitializeOnLoad]
public static class ClaudeBridge
{
    static string BridgeDir => Path.Combine(Directory.GetCurrentDirectory(), "ClaudeBridge");
    static double nextPoll;

    static ClaudeBridge()
    {
        EditorApplication.update += Poll;
    }

    static void Poll()
    {
        if (EditorApplication.timeSinceStartup < nextPoll || EditorApplication.isCompiling) return;
        nextPoll = EditorApplication.timeSinceStartup + 0.5;
        if (!Directory.Exists(BridgeDir)) return;

        foreach (var cmdPath in Directory.GetFiles(BridgeDir, "cmd_*.json"))
        {
            string id = Path.GetFileNameWithoutExtension(cmdPath).Substring(4);
            string resultPath = Path.Combine(BridgeDir, "result_" + id + ".json");
            string json;
            try { json = File.ReadAllText(cmdPath); File.Delete(cmdPath); }   // consume first: a crash must not loop the command
            catch (IOException) { continue; }                                  // writer still holds the file — next poll

            var log = new StringBuilder();
            Application.LogCallback capture = (msg, stack, type) => log.AppendLine($"[{type}] {msg}");
            Application.logMessageReceived += capture;
            try
            {
                var cmd = JsonUtility.FromJson<Cmd>(json);
                string output = Execute(cmd);
                File.WriteAllText(resultPath, "{\"status\":\"ok\",\"output\":" + Quote(output + "\n" + log) + "}");
            }
            catch (Exception e)
            {
                File.WriteAllText(resultPath, "{\"status\":\"error\",\"output\":" + Quote(e + "\n" + log) + "}");
            }
            finally
            {
                Application.logMessageReceived -= capture;
            }
        }
    }

    [Serializable]
    class Cmd
    {
        public string action;
        public string method;
        public string path;
        public float[] pos;
        public float[] euler;
        public int width = 1600;
        public int height = 900;
        public bool poseAnimators;
        public bool addLight;   // temp camera-aligned light so night scenes are inspectable
    }

    static string Execute(Cmd cmd)
    {
        switch (cmd.action)
        {
            case "ping":
                return $"pong | Unity {Application.unityVersion} | scene {EditorSceneManager.GetActiveScene().path}";

            case "refresh":
                AssetDatabase.Refresh();
                return "refresh requested (a recompile/domain reload may follow — re-ping before the next command)";

            case "menu":
                return EditorApplication.ExecuteMenuItem(cmd.method)
                    ? "menu item executed: " + cmd.method
                    : throw new Exception("menu item not found: " + cmd.method);

            case "invoke":
            {
                int split = cmd.method.LastIndexOf('.');
                string typeName = cmd.method.Substring(0, split), methodName = cmd.method.Substring(split + 1);
                var type = AppDomain.CurrentDomain.GetAssemblies()
                                    .Select(a => a.GetType(typeName)).FirstOrDefault(t => t != null)
                           ?? throw new Exception("type not found: " + typeName);
                var mi = type.GetMethod(methodName, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                         ?? throw new Exception("static method not found: " + cmd.method);
                object ret = mi.Invoke(null, null);
                return "invoked " + cmd.method + (ret != null ? " -> " + ret : "");
            }

            case "screenshot":
                return Screenshot(cmd);

            default:
                throw new Exception("unknown action: " + cmd.action);
        }
    }

    static string Screenshot(Cmd cmd)
    {
        // humanoid rigs T-pose in edit mode; sample each animator's default state through a
        // PlayableGraph (needs the runtimeAnimatorController to be assigned, else it no-ops)
        if (cmd.poseAnimators)
            foreach (var anim in UnityEngine.Object.FindObjectsOfType<Animator>())
            {
                var ctrl = anim.runtimeAnimatorController as AnimatorController;
                var st = ctrl != null ? ctrl.layers[0].stateMachine.defaultState : null;
                if (st == null || !(st.motion is AnimationClip clip)) continue;
                anim.Rebind();   // a freshly built (never-played) scene needs this or the graph no-ops
                var graph = PlayableGraph.Create("pose");
                var output = AnimationPlayableOutput.Create(graph, "pose", anim);
                var playable = AnimationClipPlayable.Create(graph, clip);
                output.SetSourcePlayable(playable);
                playable.SetTime(0.4);
                graph.Evaluate(0f);
                graph.Destroy();
            }

        var camGO = new GameObject("ClaudeBridgeProbeCam");
        GameObject lightGO = null;
        try
        {
            if (cmd.addLight)
            {
                lightGO = new GameObject("ClaudeBridgeProbeLight");
                var l = lightGO.AddComponent<Light>();
                l.type = LightType.Directional;
                l.intensity = 1.2f;
                l.color = Color.white;
                lightGO.transform.rotation = Quaternion.Euler(cmd.euler[0] + 18f, cmd.euler[1] + 12f, 0f);
            }
            var cam = camGO.AddComponent<Camera>();
            cam.fieldOfView = 55f;
            cam.nearClipPlane = 0.05f;
            cam.farClipPlane = 500f;
            camGO.transform.position = new Vector3(cmd.pos[0], cmd.pos[1], cmd.pos[2]);
            camGO.transform.rotation = Quaternion.Euler(cmd.euler[0], cmd.euler[1], cmd.euler[2]);

            var rt = new RenderTexture(cmd.width, cmd.height, 24);
            cam.targetTexture = rt;
            cam.Render();
            RenderTexture.active = rt;
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();
            RenderTexture.active = null;
            cam.targetTexture = null;

            string outPath = Path.IsPathRooted(cmd.path) ? cmd.path : Path.Combine(Directory.GetCurrentDirectory(), cmd.path);
            Directory.CreateDirectory(Path.GetDirectoryName(outPath));
            File.WriteAllBytes(outPath, tex.EncodeToPNG());

            UnityEngine.Object.DestroyImmediate(tex);
            rt.Release();
            UnityEngine.Object.DestroyImmediate(rt);
            return "screenshot saved: " + outPath;
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(camGO);
            if (lightGO != null) UnityEngine.Object.DestroyImmediate(lightGO);
        }
    }

    static string Quote(string s)
    {
        var sb = new StringBuilder("\"");
        foreach (char c in s)
            sb.Append(c switch
            {
                '"' => "\\\"",
                '\\' => "\\\\",
                '\n' => "\\n",
                '\r' => "\\r",
                '\t' => "\\t",
                _ when c < ' ' => "\\u" + ((int)c).ToString("x4"),
                _ => c.ToString(),
            });
        return sb.Append('"').ToString();
    }
}

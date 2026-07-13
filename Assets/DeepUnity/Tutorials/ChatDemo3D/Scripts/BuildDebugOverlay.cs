using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// TEMPORARY build-diagnostics overlay: collects every Debug.Log/Warning/Error/Exception via
    /// Application.logMessageReceived and draws the recent ones on screen in PLAYER BUILDS ONLY
    /// (the editor has the Console). Pops open automatically on the first error/exception;
    /// F2 toggles it manually (errors-only vs everything with F3). Bootstraps itself on scene
    /// load — no scene object or builder wiring needed. Delete this file when done debugging.
    /// </summary>
    public class BuildDebugOverlay : MonoBehaviour
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Bootstrap()
        {
            if (Application.isEditor) return;   // editor has the Console — builds only
            var go = new GameObject("BuildDebugOverlay(temp)");
            DontDestroyOnLoad(go);
            go.AddComponent<BuildDebugOverlay>();
        }

        struct Line { public string text; public LogType type; public int count; }
        readonly List<Line> lines = new List<Line>(MAX_LINES + 1);
        const int MAX_LINES = 24;

        bool visible;            // auto-set on first error; F2 toggles
        bool errorsOnly = true;  // F3 toggles warnings/logs too
        int errors, warnings;
        Vector2 scroll;

        void OnEnable() => Application.logMessageReceived += OnLog;
        void OnDisable() => Application.logMessageReceived -= OnLog;

        void OnLog(string condition, string stackTrace, LogType type)
        {
            bool isError = type == LogType.Error || type == LogType.Exception || type == LogType.Assert;
            if (isError) { errors++; visible = true; }
            else if (type == LogType.Warning) warnings++;

            // first stack frame only — enough to locate the throw without flooding the screen
            string text = condition;
            if (type == LogType.Exception && !string.IsNullOrEmpty(stackTrace))
            {
                int nl = stackTrace.IndexOf('\n');
                text += "\n    " + (nl > 0 ? stackTrace.Substring(0, nl) : stackTrace).Trim();
            }

            lock (lines)
            {
                // collapse consecutive repeats (ConsoleMessage warnings can spam per frame)
                if (lines.Count > 0 && lines[lines.Count - 1].text == text)
                {
                    var last = lines[lines.Count - 1]; last.count++;
                    lines[lines.Count - 1] = last;
                    return;
                }
                lines.Add(new Line { text = text, type = type, count = 1 });
                if (lines.Count > MAX_LINES) lines.RemoveAt(0);
            }
        }

        void Update()
        {
            if (Input.GetKeyDown(KeyCode.F2)) visible = !visible;
            if (Input.GetKeyDown(KeyCode.F3)) errorsOnly = !errorsOnly;
        }

        void OnGUI()
        {
            if (!visible)
            {
                if (errors > 0)
                    GUI.Label(new Rect(8, Screen.height - 24, 600, 22),
                              $"<color=#ff5544>{errors} error(s)</color> — F2 to show log", Rich(13));
                return;
            }

            float w = Mathf.Min(Screen.width - 16, 1100f);
            float h = Screen.height * 0.45f;
            GUI.Box(new Rect(8, 8, w, h), GUIContent.none);
            GUILayout.BeginArea(new Rect(12, 12, w - 8, h - 8));
            GUILayout.Label($"<b>Build log</b>   errors {errors} · warnings {warnings}   " +
                            "(F2 hide, F3 " + (errorsOnly ? "show all" : "errors only") + ")", Rich(14));
            scroll = GUILayout.BeginScrollView(scroll);
            lock (lines)
            {
                for (int i = 0; i < lines.Count; i++)
                {
                    var l = lines[i];
                    bool isError = l.type == LogType.Error || l.type == LogType.Exception || l.type == LogType.Assert;
                    if (errorsOnly && !isError) continue;
                    string color = isError ? "#ff5544" : l.type == LogType.Warning ? "#ffcc44" : "#cccccc";
                    string suffix = l.count > 1 ? $"  <i>(×{l.count})</i>" : "";
                    GUILayout.Label($"<color={color}>{l.text}</color>{suffix}", Rich(12));
                }
            }
            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        static GUIStyle _style;
        static GUIStyle Rich(int size)
        {
            _style ??= new GUIStyle(GUI.skin.label) { richText = true, wordWrap = true };
            _style.fontSize = size;
            return _style;
        }
    }
}

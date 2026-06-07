using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// Runtime-spawned floating label that billboards above an <see cref="Agent"/> and shows its live
    /// episode reward, so you can watch every agent's performance in real time during training.
    ///
    /// Fully self-contained and heuristic — no scene/prefab setup:
    ///  • builds its own world-space Canvas + TMP text (with a dim backdrop for readability on any environment);
    ///  • auto-detects the agent's visual size by scanning EVERY child renderer
    ///    (MeshRenderer / SkinnedMeshRenderer / SpriteRenderer / …), so it generalizes across 2D and 3D agents;
    ///  • places itself along the camera's "up" from the agent's bounding-sphere center, so it is ALWAYS above
    ///    the agent relative to the camera and always clears the mesh from any view angle;
    ///  • copies the camera rotation each frame → upright, non-mirrored billboard;
    ///  • scales to the agent's size so the text reads nicely whether the agent is tiny or huge.
    ///
    /// Attached automatically by <see cref="Agent"/> in Learn mode when <c>Hyperparameters.showAgentRewardLabels</c>
    /// is on (default off). The factors below can be tweaked live in the inspector while playing.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class AgentRewardLabel : MonoBehaviour
    {
        [Tooltip("Label world-height as a fraction of the agent's largest bounds dimension.")]
        public float sizeFactor = 0.166f;
        [Tooltip("Extra gap between the agent and the label, as a fraction of the agent's size.")]
        public float paddingFactor = 0.25f;
        [Tooltip("How often (seconds, unscaled) the child renderer set is re-scanned (handles meshes added/removed at runtime).")]
        public float rescanInterval = 1f;
        public Color positiveColor = new Color(0.45f, 1f, 0.5f);
        public Color negativeColor = new Color(1f, 0.5f, 0.45f);
        public Color zeroColor = Color.white;

        const float REF_W = 512f, REF_H = 160f;     // canvas reference size in px (font auto-fits this rect)
        const float MIN_AGENT_SIZE = 0.1f;          // floor so tiny / zero-bounds agents still get a readable label

        // Active camera, resolved once per frame and shared by all labels. Re-evaluated every frame so the
        // labels follow camera swaps — a camera that was switched off is never kept.
        static Camera _activeCam;
        static int _activeCamFrame = -1;

        Agent _agent;
        Transform _root;
        TextMeshProUGUI _tmp;
        Renderer[] _renderers;
        float _rescanElapsed;
        string _lastText;

        /// <summary>Attaches a reward label to the given agent and returns it.</summary>
        public static AgentRewardLabel Attach(Agent agent)
        {
            var label = agent.gameObject.AddComponent<AgentRewardLabel>();
            label._agent = agent;
            return label;
        }

        void Start()
        {
            if (_agent == null) _agent = GetComponent<Agent>();
            Rescan();
            Build();
        }

        // Scan the agent and ALL of its descendant GameObjects for renderers — this is the size heuristic.
        void Rescan() => _renderers = GetComponentsInChildren<Renderer>(includeInactive: false);

        void Build()
        {
            var go = new GameObject($"RewardLabel ({name})", typeof(RectTransform), typeof(Canvas));
            _root = go.transform;                                  // intentionally NOT parented to the agent →
            var rt = (RectTransform)_root;                         // immune to the agent's own scale / rotation
            rt.sizeDelta = new Vector2(REF_W, REF_H);
            go.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;

            // Dim backdrop so white text is readable over any background.
            var bgGo = new GameObject("BG", typeof(RectTransform), typeof(Image));
            bgGo.transform.SetParent(_root, false);
            Stretch((RectTransform)bgGo.transform);
            var bg = bgGo.GetComponent<Image>();
            bg.color = new Color(0f, 0f, 0f, 0.4f);
            bg.raycastTarget = false;

            var txtGo = new GameObject("Text", typeof(RectTransform));
            txtGo.transform.SetParent(_root, false);
            Stretch((RectTransform)txtGo.transform);
            _tmp = txtGo.AddComponent<TextMeshProUGUI>();
            _tmp.alignment = TextAlignmentOptions.Center;
            _tmp.enableAutoSizing = true;                          // font auto-fits the rect → "prints nicely"
            _tmp.fontSizeMin = 1f;
            _tmp.fontSizeMax = 160f;
            _tmp.raycastTarget = false;
            _tmp.text = "0.00";
        }

        static void Stretch(RectTransform rt)
        {
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
        }

        void LateUpdate()
        {
            if (_root == null || _agent == null) return;

            Camera cam = ActiveCamera();
            if (cam == null) return;

            _rescanElapsed += Time.unscaledDeltaTime;
            if (_rescanElapsed >= rescanInterval) { _rescanElapsed = 0f; Rescan(); }

            // World AABB over the cached child renderers (cheap: each .bounds is precomputed).
            Bounds b = new Bounds(transform.position, Vector3.zero);
            bool has = false;
            if (_renderers != null)
                foreach (var r in _renderers)
                {
                    if (r == null || !r.enabled) continue;
                    if (!has) { b = r.bounds; has = true; } else b.Encapsulate(r.bounds);
                }
            if (!has) b = new Bounds(transform.position, Vector3.one * MIN_AGENT_SIZE);

            float radius = Mathf.Max(MIN_AGENT_SIZE, b.extents.magnitude);            // clears the mesh from any view angle
            float agentSize = Mathf.Max(MIN_AGENT_SIZE, b.size.x, b.size.y, b.size.z);
            float worldHeight = agentSize * sizeFactor;

            // Offset along the camera's up so the label is ALWAYS above the agent relative to the screen.
            _root.position = b.center + cam.transform.up * (radius + worldHeight * 0.5f + agentSize * paddingFactor);
            _root.rotation = cam.transform.rotation;                                  // upright, non-mirrored billboard
            _root.localScale = Vector3.one * (worldHeight / REF_H);

            // Push to TMP only when the value changes → no redundant mesh rebuilds across many agents.
            float reward = _agent.EpisodeCumulativeReward;
            string s = reward.ToString("0.00");
            if (s != _lastText)
            {
                _lastText = s;
                _tmp.text = s;
                _tmp.color = reward > 0f ? positiveColor : (reward < 0f ? negativeColor : zeroColor);
            }
        }

        // Resolves the currently active camera, cached per frame and shared by all labels. Prefers the tagged
        // main camera; otherwise the enabled camera that renders on top (highest depth). Camera.main and
        // Camera.allCameras both return only active+enabled cameras, so a switched-off camera is never kept.
        static Camera ActiveCamera()
        {
            if (_activeCamFrame == Time.frameCount && _activeCam != null)
                return _activeCam;
            _activeCamFrame = Time.frameCount;

            Camera cam = Camera.main;
            if (cam == null)
            {
                Camera[] cams = Camera.allCameras;
                for (int i = 0; i < cams.Length; i++)
                    if (cam == null || cams[i].depth >= cam.depth) cam = cams[i];
            }
            _activeCam = cam;
            return cam;
        }

        void OnDestroy()
        {
            if (_root != null) Destroy(_root.gameObject);
        }
    }
}

#if UNITY_EDITOR
using System.IO;
using TMPro;
using UnityEditor;
using UnityEditor.Events;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine.Rendering.PostProcessing;
using DeepUnity.Tutorials.ChatDemo2D;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    /// <summary>
    /// Builds the "Anya" talking-head demo: a realistic Rocketbox character framed like a portrait
    /// webcam, 3-point lit, on a dark backdrop. Stage 1 (this method) is purely the VISUAL — model,
    /// scale-normalize, skin material, camera framing, lights. Stage 2 (chat + Qwen/pocket-tts +
    /// lip-sync) is layered on after the face is confirmed to render well.
    /// </summary>
    public static class AnyaChatDemoBuilder
    {
        const string ROOT = "Assets/DeepUnity/Tutorials/AnyaChatDemo";
        const string ART = ROOT + "/Art/Female_Adult_01";
        const string FBX = ART + "/Export/Female_Adult_01_facial.fbx";
        const string TEX = ART + "/Textures/";
        const string SCENE = ROOT + "/AnyaChatDemo.unity";
        const string PREVIEW = ROOT + "/AnyaFacePreview.unity";

        // bridge-invokable play toggles so I can drive a live preview headlessly
        [MenuItem("DeepUnity/Anya/_EnterPlay")] public static void EnterPlay() { if (!EditorApplication.isPlaying) EditorApplication.isPlaying = true; }
        [MenuItem("DeepUnity/Anya/_ExitPlay")] public static void ExitPlay() { if (EditorApplication.isPlaying) EditorApplication.isPlaying = false; }

        [MenuItem("DeepUnity/Anya/Build Chat Scene (Qwen + pocket-tts)")]
        public static void BuildScene()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var anya = BuildVisual(out _);
            BuildChatAndNpc(anya);
            Directory.CreateDirectory(ROOT);
            EditorSceneManager.SaveScene(scene, SCENE);
            AssetDatabase.SaveAssets();
            Debug.Log($"[Anya] CHAT scene built (Qwen3.5-0.8B int8 + pocket-tts). Saved to {SCENE}");
        }

        [MenuItem("DeepUnity/Anya/Build Face-Preview Scene (no LLM)")]
        public static void BuildFacePreview()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var anya = BuildVisual(out _);
            // prefer REAL captured facial mocap when a track exists; procedural life layer otherwise
            var track = AssetDatabase.LoadAssetAtPath<TextAsset>(ROOT + "/Art/anya_idle_mocap.bytes");
            if (track != null)
            {
                var idle = anya.AddComponent<AnyaMocapIdle>();
                SetObj(idle, "track", track);
            }
            else anya.AddComponent<AnyaFaceDemo>();
            Directory.CreateDirectory(ROOT);
            EditorSceneManager.SaveScene(scene, PREVIEW);
            AssetDatabase.SaveAssets();
            Debug.Log($"[Anya] FACE-PREVIEW scene built ({(track != null ? "REAL MOCAP idle" : "procedural idle")}, NO LLM/TTS). " +
                      $"Open {PREVIEW} and press Play.");
        }

        // shared visual: dark studio, the Rocketbox character (scale-normalized + skinned), a portrait
        // "webcam" camera, a dark backdrop and 3-point lighting. Returns the character root.
        static GameObject BuildVisual(out float faceY)
        {
            faceY = 1.46f;
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.16f, 0.16f, 0.19f);
            RenderSettings.fog = false;

            var fbx = AssetDatabase.LoadAssetAtPath<GameObject>(FBX);
            if (fbx == null) { Debug.LogError($"[Anya] FBX missing at {FBX}"); return null; }
            NormalizeNormalMaps();
            var anya = (GameObject)PrefabUtility.InstantiatePrefab(fbx);
            anya.name = "Anya";
            anya.transform.position = Vector3.zero;
            anya.transform.rotation = Quaternion.identity;   // faces +Z toward the camera

            var smr = anya.GetComponentInChildren<SkinnedMeshRenderer>(true);
            var b = smr.bounds;
            if (b.size.y > 0.001f) anya.transform.localScale *= 1.70f / b.size.y;   // normalize to ~1.70 m
            smr.updateWhenOffscreen = true;
            b = smr.bounds;
            faceY = Mathf.Lerp(b.center.y, b.max.y, 0.72f);   // eyes/nose height

            ApplySkin(smr);
            AnyaBodyPose.LowerArms(anya.transform);   // FBX imports in A-pose; drop the arms to a natural rest

            var camGO = new GameObject("Main Camera", typeof(Camera), typeof(AudioListener));
            camGO.tag = "MainCamera";
            var cam = camGO.GetComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.05f, 0.055f, 0.07f);
            cam.fieldOfView = 34f;
            cam.nearClipPlane = 0.05f;
            cam.allowMSAA = true;
            camGO.transform.position = new Vector3(0f, faceY + 0.02f, 0.72f);
            camGO.transform.LookAt(new Vector3(0f, faceY, 0f));
            AddPostFX(camGO, 0.72f);   // ACES + AO + portrait DoF + SMAA — the "modern render" look

            var backdrop = GameObject.CreatePrimitive(PrimitiveType.Quad);
            backdrop.name = "Backdrop";
            Object.DestroyImmediate(backdrop.GetComponent<Collider>());
            backdrop.transform.position = new Vector3(0f, faceY, -0.9f);
            backdrop.transform.localScale = new Vector3(6f, 6f, 1f);
            var bmat = new Material(Shader.Find("Standard"));
            bmat.color = new Color(0.07f, 0.08f, 0.11f); bmat.SetFloat("_Glossiness", 0.1f);
            backdrop.GetComponent<MeshRenderer>().sharedMaterial = bmat;

            // intensities tuned for LINEAR color space + ACES tonemapping
            MakeDirLight("Key",  new Vector3(28f, 205f, 0f), new Color(1.0f, 0.96f, 0.9f), 1.45f, LightShadows.Soft);
            MakeDirLight("Fill", new Vector3(12f, 150f, 0f), new Color(0.75f, 0.8f, 0.95f), 0.55f, LightShadows.None);
            MakeDirLight("Rim",  new Vector3(35f, 10f, 0f),  new Color(0.9f, 0.92f, 1.0f), 0.35f, LightShadows.None);

            // catchlight: a dim point light near the camera so the eyes get a moist sparkle
            var catchGO = new GameObject("Catchlight", typeof(Light));
            catchGO.transform.position = new Vector3(0.12f, faceY + 0.10f, 0.55f);
            var cl = catchGO.GetComponent<Light>();
            cl.type = LightType.Point; cl.range = 1.4f; cl.intensity = 0.55f;
            cl.color = new Color(1f, 0.98f, 0.95f); cl.shadows = LightShadows.None;
            return anya;
        }

        // Rocketbox skin: Standard (Specular setup) with color/normal/specular. One material reused
        // across the mesh's slots for v1 (head UVs texture the face correctly; body is mostly
        // off-frame in the portrait). Refined per-slot later if needed.
        static void ApplySkin(SkinnedMeshRenderer smr)
        {
            // head: custom fake-SSS skin shader (wrap diffuse + subsurface terminator tint + pore
            // detail normals + fresnel sheen) — Standard lighting reads as plastic on skin
            var skin = new Material(Shader.Find("DeepUnity/AnyaSkin"));
            skin.name = "Anya_Head";
            skin.SetTexture("_MainTex", Load(TEX + "f001_head_color.tga"));
            skin.SetTexture("_BumpMap", Load(TEX + "f001_head_normal.tga"));
            skin.SetTexture("_SpecGlossMap", Load(TEX + "f001_head_specular.tga"));
            var pore = Load(TEX + "skin_pore_normal.png");
            if (pore != null) skin.SetTexture("_DetailNormal", pore);
            // body material: its OWN texture set (previously reused the head texture — mangled torso);
            // mostly clothing, so kill the skin-like gloss (satin-shirt artifact)
            var body = MakeSpecMat("Anya_Body", "f001_body");
            body.SetFloat("_Glossiness", 0.15f);
            body.SetFloat("_GlossMapScale", 0.35f);   // with a SpecGlossMap, smoothness = map alpha * this
            body.SetColor("_SpecColor", new Color(0.08f, 0.08f, 0.08f));

            // eyelashes / brows / hair cards: alpha-CUTOUT off the opacity atlas (else they render as
            // opaque skin-textured slabs over the eyes — the defect in v1)
            var cut = new Material(Shader.Find("Standard"));
            cut.name = "Anya_Opacity";
            var op = Load(TEX + "f001_opacity_color.tga");
            if (op != null) cut.SetTexture("_MainTex", op);
            cut.SetFloat("_Mode", 1f);   // Cutout
            cut.SetFloat("_Cutoff", 0.35f);
            cut.EnableKeyword("_ALPHATEST_ON");
            cut.renderQueue = 2450;

            var mats = smr.sharedMaterials;
            var sb = new System.Text.StringBuilder("[Anya] material slots: ");
            for (int i = 0; i < mats.Length; i++)
            {
                string n = (mats[i] != null ? mats[i].name : "null").ToLowerInvariant();
                sb.Append($"[{i}]='{(mats[i] != null ? mats[i].name : "null")}' ");
                bool isCut = n.Contains("opacity") || n.Contains("lash") || n.Contains("hair")
                          || n.Contains("brow") || n.Contains("eyelash") || n.Contains("transp");
                mats[i] = isCut ? cut : (n.Contains("body") ? body : skin);
            }
            Debug.Log(sb.ToString());
            smr.sharedMaterials = mats;
        }

        // Standard (Specular setup) material from a Rocketbox texture triplet (color/normal/specular)
        static Material MakeSpecMat(string name, string texPrefix)
        {
            var m = new Material(Shader.Find("Standard (Specular setup)"));
            m.name = name;
            m.SetTexture("_MainTex", Load(TEX + texPrefix + "_color.tga"));
            var nrm = Load(TEX + texPrefix + "_normal.tga");
            if (nrm != null) { m.EnableKeyword("_NORMALMAP"); m.SetTexture("_BumpMap", nrm); }
            var spec = Load(TEX + texPrefix + "_specular.tga");
            if (spec != null) { m.EnableKeyword("_SPECGLOSSMAP"); m.SetTexture("_SpecGlossMap", spec); }
            m.SetColor("_SpecColor", new Color(0.2f, 0.2f, 0.2f));
            m.SetFloat("_Glossiness", 0.35f);
            return m;
        }

        static Texture2D Load(string p) => AssetDatabase.LoadAssetAtPath<Texture2D>(p);

        // mark the normal maps as NormalMap type so Unity unpacks them correctly
        static void NormalizeNormalMaps()
        {
            foreach (var n in new[] { "f001_head_normal.tga", "f001_head_normal_wrinkle.tga", "f001_body_normal.tga", "skin_pore_normal.png" })
            {
                var imp = AssetImporter.GetAtPath(TEX + n) as TextureImporter;
                if (imp != null && imp.textureType != TextureImporterType.NormalMap)
                {
                    imp.textureType = TextureImporterType.NormalMap;
                    imp.SaveAndReimport();
                }
            }
        }

        // The "modern render" pass: SMAA + a global PostProcess volume with ACES tonemapping, gentle
        // bloom, ambient occlusion, portrait depth-of-field (face sharp, backdrop soft) and a vignette.
        // Profile is saved as an asset so the scene keeps it.
        static void AddPostFX(GameObject camGO, float focusDist)
        {
            var layer = camGO.AddComponent<PostProcessLayer>();
            var res = AssetDatabase.LoadAssetAtPath<PostProcessResources>(
                "Packages/com.unity.postprocessing/PostProcessing/PostProcessResources.asset");
            if (res != null) layer.Init(res);
            layer.volumeLayer = 1;   // Default
            layer.volumeTrigger = camGO.transform;
            layer.antialiasingMode = PostProcessLayer.Antialiasing.SubpixelMorphologicalAntialiasing;

            var profile = ScriptableObject.CreateInstance<PostProcessProfile>();

            var cg = profile.AddSettings<ColorGrading>();
            cg.gradingMode.Override(GradingMode.HighDefinitionRange);
            cg.tonemapper.Override(Tonemapper.ACES);
            cg.postExposure.Override(0.35f);
            cg.contrast.Override(8f);
            cg.saturation.Override(3f);

            var bloom = profile.AddSettings<Bloom>();
            bloom.intensity.Override(0.6f);
            bloom.threshold.Override(1.15f);
            bloom.softKnee.Override(0.6f);

            var ao = profile.AddSettings<AmbientOcclusion>();
            ao.mode.Override(AmbientOcclusionMode.MultiScaleVolumetricObscurance);
            ao.intensity.Override(0.5f);

            var dof = profile.AddSettings<DepthOfField>();
            dof.focusDistance.Override(focusDist);
            dof.aperture.Override(5f);
            dof.focalLength.Override(55f);

            var vig = profile.AddSettings<Vignette>();
            vig.intensity.Override(0.24f);
            vig.smoothness.Override(0.45f);

            AssetDatabase.DeleteAsset(ROOT + "/AnyaPostProfile.asset");
            AssetDatabase.CreateAsset(profile, ROOT + "/AnyaPostProfile.asset");

            var volGO = new GameObject("PostFX");
            volGO.layer = 0;
            var vol = volGO.AddComponent<PostProcessVolume>();
            vol.isGlobal = true;
            vol.sharedProfile = profile;
        }

        static void MakeDirLight(string name, Vector3 euler, Color color, float intensity, LightShadows shadows)
        {
            var go = new GameObject(name, typeof(Light));
            go.transform.rotation = Quaternion.Euler(euler);
            var l = go.GetComponent<Light>();
            l.type = LightType.Directional;
            l.color = color;
            l.intensity = intensity;
            l.shadows = shadows;
        }

        // ---------------------------------------------------------------- chat UI + NPC
        const string PERSONA =
            "You are Anya, a warm, curious and down-to-earth young woman talking to someone face to " +
            "face over a video call. Speak naturally in the first person, like a real person — relaxed " +
            "and friendly. Keep your replies short, one to three sentences. Never mention being an AI, " +
            "a model, or a program; you are simply Anya.";

        static void BuildChatAndNpc(GameObject anya)
        {
            Color cream = new Color(0.95f, 0.92f, 0.85f);
            Color gold = new Color(0.92f, 0.78f, 0.45f);
            Color barBg = new Color(0.07f, 0.08f, 0.11f, 0.85f);

            var canvasGO = new GameObject("UI", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            var canvas = canvasGO.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGO.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
            scaler.matchWidthOrHeight = 0.5f;
            new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));

            // bottom chat panel (same shape as ChatDemo2D: floating lines over a dark input strip)
            var panel = MakeRect("ChatWindow2D", canvasGO.transform, new Vector2(0.5f, 0), new Vector2(0.5f, 0),
                                 new Vector2(1120, 320), new Vector2(0, 18));
            ((RectTransform)panel.transform).pivot = new Vector2(0.5f, 0);
            var group = panel.AddComponent<CanvasGroup>();

            var lines = MakeRect("Lines", panel.transform, new Vector2(0, 0), new Vector2(1, 0),
                                 new Vector2(-60, 210), new Vector2(0, 100));
            ((RectTransform)lines.transform).pivot = new Vector2(0.5f, 0);

            var lineGO = MakeRect("LineTemplate", lines.transform, new Vector2(0, 0), new Vector2(1, 0),
                                  new Vector2(0, 30), Vector2.zero);
            ((RectTransform)lineGO.transform).pivot = new Vector2(0.5f, 0);
            lineGO.AddComponent<CanvasGroup>();
            var lineTmp = lineGO.AddComponent<TextMeshProUGUI>();
            lineTmp.fontSize = 28; lineTmp.color = Color.white;
            lineTmp.alignment = TextAlignmentOptions.BottomLeft; lineTmp.raycastTarget = false;

            var title = MakeTMP("Title", panel.transform, "Anya", 22, gold, TextAlignmentOptions.Left,
                                new Vector2(0, 0), new Vector2(0, 0), new Vector2(320, 26), new Vector2(210, 74));
            var info = MakeTMP("InfoText", panel.transform, "", 20, new Color(0.86f, 0.82f, 0.72f, 0.9f),
                               TextAlignmentOptions.Center, new Vector2(0, 0), new Vector2(1, 0), new Vector2(-160, 26), new Vector2(0, 72));
            info.GetComponent<TMP_Text>().fontStyle = FontStyles.Italic;

            var row = MakeRect("InputRow", panel.transform, new Vector2(0, 0), new Vector2(1, 0),
                               new Vector2(-60, 54), new Vector2(0, 34));
            var strip = row.AddComponent<Image>(); strip.color = barBg;
            var hlg = row.AddComponent<HorizontalLayoutGroup>();
            hlg.padding = new RectOffset(16, 12, 6, 6); hlg.spacing = 10;
            hlg.childControlWidth = true; hlg.childControlHeight = true;
            hlg.childForceExpandWidth = false; hlg.childForceExpandHeight = true;

            var inputGO = BuildInputField(row.transform, cream, gold, out var inputField);
            inputGO.AddComponent<LayoutElement>().flexibleWidth = 1f;
            var sendBtn = BuildButton(row.transform, "Send", cream, 100);

            var win = panel.AddComponent<ChatWindow2D>();
            SetObj(win, "canvasGroup", group);
            SetObj(win, "linesContainer", (RectTransform)lines.transform);
            SetObj(win, "lineTemplate", lineGO);
            SetObj(win, "inputField", inputField);
            SetObj(win, "sendButton", sendBtn.GetComponent<Button>());
            SetObj(win, "infoText", info.GetComponent<TMP_Text>());
            SetObj(win, "titleText", title.GetComponent<TMP_Text>());

            var npc = anya.AddComponent<NPCInteractorAnya>();
            SetObj(npc, "chatWindow", win);
            SetStr(npc, "npc_name", "Anya");
            SetStr(npc, "system_prompt", PERSONA);
            SetStr(npc, "approach_text", "");
            SetStr(npc, "model", "Qwen3.5-0.8B");
            SetEnum(npc, "quantization", (int)LLMQuant.INT8);
            SetEnum(npc, "conversationMode", (int)NPCChatBase.ConversationMode.LlmPlusTts);
            SetEnum(npc, "historyMode", (int)NPCChatBase.HistoryMode.ResetEveryTime);
            SetEnum(npc, "ttsModel", (int)NPCChatBase.TtsModel.PocketTTS);
            SetStr(npc, "ttsVoice", "jean");
            SetI(npc, "maxContextLength", 8192);
            SetB(npc, "cacheKVCache", false);

            UnityEventTools.AddPersistentListener(sendBtn.GetComponent<Button>().onClick, new UnityAction(npc.AskNPC));
            UnityEventTools.AddVoidPersistentListener(inputField.onSubmit, new UnityAction(npc.AskNPC));
            Debug.Log("[Anya] chat + NPC wired (Qwen3.5-0.8B int8 + pocket-tts).");
        }

        static GameObject MakeRect(string name, Transform parent, Vector2 aMin, Vector2 aMax, Vector2 sizeDelta, Vector2 pos)
        {
            var go = new GameObject(name, typeof(RectTransform));
            var rt = (RectTransform)go.transform;
            rt.SetParent(parent, false);
            rt.anchorMin = aMin; rt.anchorMax = aMax; rt.sizeDelta = sizeDelta; rt.anchoredPosition = pos;
            return go;
        }

        static GameObject MakeTMP(string name, Transform parent, string text, float size, Color color,
                                  TextAlignmentOptions align, Vector2 aMin, Vector2 aMax, Vector2 sizeDelta, Vector2 pos)
        {
            var go = MakeRect(name, parent, aMin, aMax, sizeDelta, pos);
            var t = go.AddComponent<TextMeshProUGUI>();
            t.text = text; t.fontSize = size; t.color = color; t.alignment = align; t.raycastTarget = false;
            return go;
        }

        static GameObject BuildInputField(Transform parent, Color textColor, Color caret, out TMP_InputField field)
        {
            var go = MakeRect("Input", parent, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var bg = go.AddComponent<Image>(); bg.color = new Color(1f, 1f, 1f, 0.06f);
            field = go.AddComponent<TMP_InputField>();

            var area = MakeRect("Text Area", go.transform, Vector2.zero, Vector2.one, new Vector2(-24, -8), Vector2.zero);
            area.AddComponent<RectMask2D>();
            var ph = MakeTMP("Placeholder", area.transform, "Type to Anya…", 26, new Color(0.82f, 0.79f, 0.72f, 0.5f),
                             TextAlignmentOptions.MidlineLeft, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            ph.GetComponent<TMP_Text>().fontStyle = FontStyles.Italic;
            var txt = MakeTMP("Text", area.transform, "", 26, textColor, TextAlignmentOptions.MidlineLeft,
                              Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);

            field.textViewport = (RectTransform)area.transform;
            field.textComponent = txt.GetComponent<TMP_Text>();
            field.placeholder = ph.GetComponent<TMP_Text>();
            field.caretColor = caret; field.customCaretColor = true; field.caretWidth = 3;
            field.lineType = TMP_InputField.LineType.SingleLine;
            field.onFocusSelectAll = false;
            return go;
        }

        static Button BuildButton(Transform parent, string label, Color color, float width)
        {
            var go = MakeRect("Btn_" + label, parent, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var img = go.AddComponent<Image>(); img.color = new Color(1f, 1f, 1f, 0.10f);
            var btn = go.AddComponent<Button>();
            var le = go.AddComponent<LayoutElement>(); le.minWidth = width; le.preferredWidth = width;
            MakeTMP("Label", go.transform, label, 24, color, TextAlignmentOptions.Center,
                    Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            return btn;
        }

        // SerializedObject setters — reach inherited (base-class) serialized fields and dirty properly
        static SerializedProperty Prop(Object c, string p, out SerializedObject so)
        {
            so = new SerializedObject(c);
            var sp = so.FindProperty(p);
            if (sp == null) Debug.LogWarning($"[Anya] field '{p}' not found on {c.GetType().Name}");
            return sp;
        }
        static void SetObj(Object c, string p, Object v) { var sp = Prop(c, p, out var so); if (sp != null) { sp.objectReferenceValue = v; so.ApplyModifiedPropertiesWithoutUndo(); } }
        static void SetStr(Object c, string p, string v) { var sp = Prop(c, p, out var so); if (sp != null) { sp.stringValue = v; so.ApplyModifiedPropertiesWithoutUndo(); } }
        static void SetEnum(Object c, string p, int v) { var sp = Prop(c, p, out var so); if (sp != null) { sp.enumValueIndex = v; so.ApplyModifiedPropertiesWithoutUndo(); } }
        static void SetI(Object c, string p, int v) { var sp = Prop(c, p, out var so); if (sp != null) { sp.intValue = v; so.ApplyModifiedPropertiesWithoutUndo(); } }
        static void SetB(Object c, string p, bool v) { var sp = Prop(c, p, out var so); if (sp != null) { sp.boolValue = v; so.ApplyModifiedPropertiesWithoutUndo(); } }
    }
}
#endif

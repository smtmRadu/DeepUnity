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
            var anya = BuildVisual(out _);          // STATIC body (user reverted the body-animator round)
            BuildChatAndNpc(anya);
            AttachIdle(anya);                       // face/head behaviour stack (+ tiny procedural breathing)
            anya.AddComponent<FaceSync>();          // audio-driven mouth (base face-sync), layered on top of the idle
            Directory.CreateDirectory(ROOT);
            EditorSceneManager.SaveScene(scene, SCENE);
            AssetDatabase.SaveAssets();
            Debug.Log($"[Anya] CHAT scene built (Qwen3.5-0.8B int8 + pocket-tts + idle + lip-sync). Saved to {SCENE}");
        }

        [MenuItem("DeepUnity/Anya/Build Face-Preview Scene (no LLM)")]
        public static void BuildFacePreview()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var anya = BuildVisual(out _);
            AttachIdle(anya);
            Directory.CreateDirectory(ROOT);
            EditorSceneManager.SaveScene(scene, PREVIEW);
            AssetDatabase.SaveAssets();
            Debug.Log($"[Anya] FACE-PREVIEW scene built (idle only, NO LLM/TTS). Open {PREVIEW} and press Play.");
        }

        // Attach the facial IDLE: the generalized behaviour system (AnyaBehaviourIdle) — camera-
        // anchored gaze with rare look-aways, hard camera-lock while speaking, damped head motion,
        // and blinks/smiles/brows/rest-mouth as modular AnyaBehaviour units. Replaces the mocap
        // replay (AnyaMocapIdle), which drove eye-look + head every frame and could not honor the
        // "always look at the camera / lock while talking" gaze model. Shared by the chat scene
        // and the face-preview scene. AnyaLipSync (order 100) still blends the mouth on top.
        static void AttachIdle(GameObject anya)
        {
            anya.AddComponent<AnyaBehaviourIdle>();
        }

        // ================================================================================
        // KNOWN ISSUE (accepted for now, 2026-07-22): "dark flickers / flashy dark tones"
        // on her collar/shirt on the GTX 1650 laptop (Pavilion). If they reappear or need
        // fixing, DO NOT re-test what is already ruled out:
        //   - NOT shadow acne: key switched directional->spot, biases tuned, then cast
        //     shadows turned fully OFF -> flicker unchanged.
        //   - NOT post-processing: AO disabled, DoF disabled -> unchanged.
        //   - NOT shading aliasing: body material flattened (normal+spec maps stripped,
        //     matte) -> unchanged.
        //   - NOT z-fighting/depth precision: near 0.05->0.3, far clamped to 10 -> unchanged.
        //   - VRAM relief attempted: MSAA off (SMAA covers AA), Anya KV 8192->4096 -> still there.
        // KEY OBSERVATION (user): the first seconds of play are CLEAN — the flicker starts
        // exactly when the LLM+TTS finish loading onto the GPU. Leading hypothesis: VRAM
        // overcommit on the 4 GB card (Qwen int8 + KV + pocket-tts + editor + render targets)
        // -> WDDM evicts/pages graphics resources mid-frame. It may also be compute-vs-render
        // queue contention from the per-frame decode/synthesis dispatches.
        // NEXT STEPS if revisited: (1) reproduce in a STANDALONE BUILD (editor holds extra
        // VRAM; play mode is the worst case); (2) watch Task Manager -> GPU -> Dedicated
        // memory during the transition; (3) compress the Rocketbox TGAs / cap texture sizes;
        // (4) the planned HDRP-project port replaces this whole render stack anyway — verify
        // there before investing more here.
        // ================================================================================
        // shared visual: dark studio, the Rocketbox character (scale-normalized + skinned), a portrait
        // "webcam" camera, a dark backdrop and 3-point lighting. Returns the character root.
        // BODY IS STATIC (user reverted the body-animator round: Humanoid retarget clenched her
        // fists + offset her direction, and the clip stance broke the calibrated framing): Generic
        // rig, LowerArms rest pose, ORIGINAL bounds-based camera framing. The only body motion is
        // the tiny PROCEDURAL breathing applied by AnyaBreathingBehaviour in LateUpdate.
        static GameObject BuildVisual(out float faceY)
        {
            faceY = 1.46f;
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.16f, 0.16f, 0.19f);
            RenderSettings.fog = false;

            EnsureGenericRig();   // revert the Humanoid experiment BEFORE instantiating
            AssetDatabase.DeleteAsset(ROOT + "/AnyaBodyAnimator.controller");   // stale body-round assets
            AssetDatabase.DeleteAsset(ROOT + "/AnyaIdleStatic.anim");
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
            faceY = Mathf.Lerp(b.center.y, b.max.y, 0.72f);   // eyes/nose height (the calibrated framing)

            ApplySkin(smr);
            AnyaBodyPose.LowerArms(anya.transform);   // FBX imports in A-pose; drop the arms to a natural rest

            var camGO = new GameObject("Main Camera", typeof(Camera), typeof(AudioListener));
            camGO.tag = "MainCamera";
            var cam = camGO.GetComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.05f, 0.055f, 0.07f);
            cam.fieldOfView = 34f;
            // near 0.3, far tight: a 0.05 near plane wrecked depth precision scene-wide and let the
            // collar's two nearly-coplanar cloth layers z-fight (dark flashes as she moved). Nothing
            // sits closer than ~0.5 m to this fixed portrait camera, and the backdrop ends at ~1.7 m.
            cam.nearClipPlane = 0.3f;
            cam.farClipPlane = 10f;
            // MSAA OFF (VRAM-pressure fix, 2026-07-22): the collar "dark flicker" starts exactly
            // when the LLM+TTS go GPU-resident — on the 4 GB 1650 the card overcommits and WDDM
            // evicts render targets mid-frame (post toggles never affected it, model load timing
            // does). MSAA multiplies every render target; SMAA in the post stack already does the
            // anti-aliasing, so this is pure VRAM savings with no visual cost.
            cam.allowMSAA = false;
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

            // intensities tuned for LINEAR color space + ACES tonemapping.
            // Key is a SPOT, not a directional: a sun-type key shares one low-precision cascade
            // over the whole shadow distance and produced crawling acne on the shirt as she moved;
            // a spot's own ~4 m shadow map at portrait range is artifact-free (real studio key).
            var keyGO = new GameObject("Key", typeof(Light));
            keyGO.transform.position = new Vector3(0.85f, faceY + 0.75f, 1.15f);   // camera-left, high (same side the old key shone from)
            keyGO.transform.LookAt(new Vector3(0f, faceY - 0.1f, 0f));
            var key = keyGO.GetComponent<Light>();
            key.type = LightType.Spot;
            key.range = 4f;
            key.spotAngle = 65f;
            key.color = new Color(1.0f, 0.96f, 0.9f);
            key.intensity = 3.2f;              // spot falloff at ~1.6 m ≈ the old directional 1.45
            // NO cast shadows: the collar is thin double-layer cloth in a tight crease — the one
            // geometry class shadow maps can't handle (self-shadow acne no bias fixes without
            // breaking elsewhere; it read as dark flickers on the collar as she breathed). In a
            // head-and-shoulders portrait cast shadows add ~nothing: the fake-SSS skin shading
            // carries the form and the (tamed) AO pass below carries contact darkening.
            key.shadows = LightShadows.None;
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
            // MATTE shirt (collar-flicker fix, 2026-07-22): normal + spec maps on the thin collar
            // folds shimmered at grazing angles as she moved (shading aliasing survives MSAA and
            // every post toggle). Albedo only — at portrait framing the cloth reads fine flat.
            body.SetTexture("_BumpMap", null);
            body.DisableKeyword("_NORMALMAP");
            body.SetTexture("_SpecGlossMap", null);
            body.DisableKeyword("_SPECGLOSSMAP");
            body.SetFloat("_Glossiness", 0.12f);
            body.SetColor("_SpecColor", new Color(0.04f, 0.04f, 0.04f));

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

        // Import the voice-clone reference so the Mimi encoder can read raw samples: decompress on
        // load, mono, preloaded. Returns the AudioClip (null if the file isn't there).
        static AudioClip LoadVoiceRef(string path)
        {
            if (AssetImporter.GetAtPath(path) is AudioImporter imp)
            {
                var s = imp.defaultSampleSettings;
                s.loadType = AudioClipLoadType.DecompressOnLoad;
                s.preloadAudioData = true;   // per-platform setting since 2020; not the obsolete importer flag
                imp.defaultSampleSettings = s;
                imp.forceToMono = true;
                imp.SaveAndReimport();
            }
            return AssetDatabase.LoadAssetAtPath<AudioClip>(path);
        }

        // ---------------------------------------------------------------- body (REVERTED to static)
        // The body-animator round is fully reverted (user): the Humanoid retarget clenched her
        // fists (UAL clips carry no finger curves -> muscle-default curl) and offset her facing
        // (avatar root normalization), and the Idle_Loop stance broke the calibrated framing.
        // The FBX goes back to its ORIGINAL Generic rig; the body is the static LowerArms pose;
        // the only body motion is the tiny PROCEDURAL breathing (AnyaBreathingBehaviour).
        // AnyaBodyGestures.cs is kept on disk (not added to the scene) for a future attempt.
        static void EnsureGenericRig()
        {
            var imp = AssetImporter.GetAtPath(FBX) as ModelImporter;
            if (imp == null) return;
            if (imp.animationType != ModelImporterAnimationType.Generic)
            {
                imp.animationType = ModelImporterAnimationType.Generic;
                imp.SaveAndReimport();
                Debug.Log("[Anya] FBX reverted to Generic rig (pre-body-round import settings)");
            }
        }

        // ---- probes: framing + procedural-breathing extremes. The breathing behaviours are plain
        // deterministic C# — evaluate them directly in edit mode (no play mode, no animator). They
        // nudge the clavicle/chest bones; re-run BuildScene afterwards for a pristine save.
        public static void ProbeFraming() { Shot("ProbeLogs/anya_framing.png"); }
        public static void ProbeBreathA() { BreathePose(1f); Shot("ProbeLogs/anya_breath_a.png"); }   // inhale peak (sin=+1 @ 0.25 Hz)
        public static void ProbeBreathB() { BreathePose(3f); Shot("ProbeLogs/anya_breath_b.png"); }   // exhale trough (sin=-1)

        static void BreathePose(float t)
        {
            var smr = Object.FindObjectOfType<SkinnedMeshRenderer>();
            if (smr == null) { Debug.LogError("[Anya] breath probe: no SkinnedMeshRenderer in scene"); return; }
            var rig = new AnyaFaceRig();
            rig.Init(smr);
            var br = new AnyaBreathingBehaviour { Amount = 0.03f };   // the shipped default
            br.Init(rig);
            var f = new AnyaIdleFrame { t = t };
            br.Evaluate(rig, in f);
            Debug.Log($"[Anya] procedural breath posed at t={t:F1}s (amount 0.03)");
        }

        static void Shot(string path)
        {
            var cam = Camera.main;
            if (cam == null) { Debug.LogError("[Anya] probe: no main camera"); return; }
            var rt = new RenderTexture(1200, 900, 24);
            cam.targetTexture = rt; cam.Render(); cam.targetTexture = null;
            RenderTexture.active = rt;
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0); tex.Apply();
            RenderTexture.active = null;
            Directory.CreateDirectory("ProbeLogs");
            File.WriteAllBytes(path, tex.EncodeToPNG());
            Object.DestroyImmediate(rt); Object.DestroyImmediate(tex);
            Debug.Log($"[Anya] probe saved to {path}");
        }

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

            // AO DISABLED (collar-flicker bisection, 2026-07-22): shadows were already ruled out;
            // AO and DoF are the remaining post suspects for the dark flicker on the collar.
            // Re-add (intensity ~0.3) only if the flicker proves to be DoF and the contact
            // shading is missed:
            // var ao = profile.AddSettings<AmbientOcclusion>();
            // ao.mode.Override(AmbientOcclusionMode.MultiScaleVolumetricObscurance);
            // ao.intensity.Override(0.35f);

            // DoF DISABLED (collar-flicker bisection, 2026-07-22): the collar sits centimeters
            // behind the face focus plane, right in the sharp->blur transition band — a dark
            // high-contrast edge crossing it while she breathes reads as flashy dark smears.
            // If the flicker is gone with this off, re-add with a slower aperture (f/8+) and
            // focus pushed slightly back, or leave off (backdrop is featureless anyway):
            // var dof = profile.AddSettings<DepthOfField>();
            // dof.focusDistance.Override(focusDist);
            // dof.aperture.Override(5f);
            // dof.focalLength.Override(55f);

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
            // bold black outline + soft drop shadow so the floating white text stays legible over her
            // brightly lit face and pink shirt (no dark backing panel — keeps the clean floating look)
            var lineMat = new Material(lineTmp.fontSharedMaterial);
            lineMat.EnableKeyword(ShaderUtilities.Keyword_Outline);
            lineMat.SetColor(ShaderUtilities.ID_OutlineColor, Color.black);
            lineMat.SetFloat(ShaderUtilities.ID_OutlineWidth, 0.3f);
            lineMat.EnableKeyword(ShaderUtilities.Keyword_Underlay);
            lineMat.SetColor(ShaderUtilities.ID_UnderlayColor, new Color(0f, 0f, 0f, 0.85f));
            lineMat.SetFloat(ShaderUtilities.ID_UnderlayOffsetX, 0.6f);
            lineMat.SetFloat(ShaderUtilities.ID_UnderlayOffsetY, -0.6f);
            lineMat.SetFloat(ShaderUtilities.ID_UnderlaySoftness, 0.35f);
            lineTmp.fontSharedMaterial = lineMat;

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

            var npc = anya.AddComponent<NPCInteractorAnya>();
            SetObj(npc, "chatWindow", win);
            SetStr(npc, "NpcName", "Anya");
            SetStr(npc, "system_prompt", PERSONA);
            SetStr(npc, "model", "Qwen3.5-0.8B");
            SetEnum(npc, "quantization", (int)LLMQuant.INT8);
            SetEnum(npc, "conversationMode", (int)NPCChatBase.ConversationMode.LlmPlusTts);
            SetEnum(npc, "historyMode", (int)NPCChatBase.HistoryMode.ResetEveryTime);
            SetEnum(npc, "ttsModel", (int)NPCChatBase.TtsModel.PocketTTS);
            SetStr(npc, "ttsVoice", "jean");   // baked fallback; the cloned clip below overrides it
            var voiceRef = LoadVoiceRef(ROOT + "/Art/VoiceRefs/female-voice-2.wav");
            if (voiceRef == null) voiceRef = LoadVoiceRef(ROOT + "/Art/VoiceRefs/anya_voice_ref.mp3");   // fallback reference
            if (voiceRef != null) { SetObj(npc, "clonedVoiceClip", voiceRef); Debug.Log($"[Anya] voice-clone reference plugged in ('{voiceRef.name}', overrides 'jean')."); }
            else Debug.LogWarning("[Anya] no voice-clone reference under Art/VoiceRefs/ — using baked 'jean'.");
            SetI(npc, "maxContextLength", 4096);   // halved KV (VRAM headroom on 4 GB cards) — plenty for ResetEveryTime chats
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

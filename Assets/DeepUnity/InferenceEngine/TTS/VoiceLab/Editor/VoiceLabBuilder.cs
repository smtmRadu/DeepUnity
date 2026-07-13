using TMPro;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace DeepUnity
{
    /// <summary>
    /// Builds the VoiceLab TTS-audition scene (Assets/DeepUnity/InferenceEngine/TTS/VoiceLab/VoiceLab.unity):
    /// camera + EventSystem + a dark screen-space canvas with engine/voice dropdowns,
    /// pitch/speed sliders, a multiline text input with sample-line buttons, SPEAK/STOP,
    /// a status readout and a SAVE PRESET row — all wired into a VoiceLab component
    /// (runtime listeners live in VoiceLab.Awake; the builder only assigns references).
    /// UI-construction idioms follow ChatDemo2DBuilder. The scene is built ADDITIVELY and
    /// closed after saving, so the scenes currently open in the editor are not disturbed.
    /// </summary>
    public static class VoiceLabBuilder
    {
        const string ROOT = "Assets/DeepUnity/InferenceEngine/TTS/VoiceLab";
        const string SCENE_PATH = ROOT + "/VoiceLab.unity";

        // dark neutral palette
        static readonly Color BG        = new Color(0.086f, 0.086f, 0.098f);
        static readonly Color PANEL     = new Color(0.125f, 0.129f, 0.145f);
        static readonly Color PANEL2    = new Color(0.100f, 0.104f, 0.118f);
        static readonly Color FIELD     = new Color(0.170f, 0.175f, 0.195f);
        static readonly Color TEXT      = new Color(0.88f, 0.89f, 0.90f);
        static readonly Color TEXT_DIM  = new Color(0.55f, 0.57f, 0.61f);
        static readonly Color ACCENT    = new Color(0.28f, 0.51f, 0.90f);
        static readonly Color STOP_RED  = new Color(0.62f, 0.27f, 0.25f);
        static readonly Color HANDLE    = new Color(0.82f, 0.83f, 0.87f);

        static readonly Vector2 TL = new Vector2(0f, 1f);   // top-left anchor/pivot shorthand

        [MenuItem("DeepUnity/TTS/Build VoiceLab Scene")]
        public static void BuildScene()
        {
            // rebuilding while VoiceLab.unity is open alongside other scenes: close our stale copy
            Scene existing = SceneManager.GetSceneByPath(SCENE_PATH);
            if (existing.IsValid() && existing.isLoaded && SceneManager.sceneCount > 1)
                EditorSceneManager.CloseScene(existing, true);

            Scene prev = SceneManager.GetActiveScene();
            if (prev.path == SCENE_PATH)
            {
                // the lab is the only open scene: rebuild it in place
                Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                BuildContents();
                EditorSceneManager.SaveScene(scene, SCENE_PATH);
            }
            else
            {
                // normal path: build additively, save, close — the user's scene setup is untouched
                Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive);
                SceneManager.SetActiveScene(scene);
                try
                {
                    BuildContents();
                    EditorSceneManager.SaveScene(scene, SCENE_PATH);
                }
                finally
                {
                    if (prev.IsValid()) SceneManager.SetActiveScene(prev);
                    EditorSceneManager.CloseScene(scene, true);
                }
            }
            AssetDatabase.SaveAssets();
            Debug.Log("[VoiceLabBuilder] Scene saved at " + SCENE_PATH);
        }

        // ---------------------------------------------------------------- scene contents

        static void BuildContents()
        {
            // camera (also hosts the AudioListener the voices play through)
            var camGO = new GameObject("Main Camera", typeof(Camera), typeof(AudioListener));
            camGO.tag = "MainCamera";
            var cam = camGO.GetComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = BG;
            cam.orthographic = true;
            camGO.transform.position = new Vector3(0f, 0f, -10f);

            new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));

            // canvas
            var canvasGO = new GameObject("UI", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            var canvas = canvasGO.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGO.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
            scaler.matchWidthOrHeight = 0.5f;

            var bg = MakeRect("Background", canvasGO.transform, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var bgImg = bg.AddComponent<Image>();
            bgImg.color = BG;
            bgImg.raycastTarget = false;

            MakeTMP("Title", canvasGO.transform, "VOICE LAB", 34, TEXT, TextAlignmentOptions.Left,
                    TL, TL, new Vector2(600, 44), new Vector2(40, -26), TL);
            MakeTMP("Subtitle", canvasGO.transform,
                    "Audition baked TTS voices — pick an engine and voice, tweak, speak, save the preset.",
                    15, TEXT_DIM, TextAlignmentOptions.Left,
                    TL, TL, new Vector2(1200, 24), new Vector2(40, -72), TL);

            // the VoiceLab driver GO (voice components are added to it at runtime)
            var labGO = new GameObject("VoiceLab");
            var src = labGO.AddComponent<AudioSource>();
            src.playOnAwake = false;
            src.spatialBlend = 0f;
            var lab = labGO.AddComponent<VoiceLab>();

            BuildLeftPanel(canvasGO.transform, lab);
            BuildRightPanel(canvasGO.transform, lab);
        }

        // ---------------------------------------------------------------- left panel (controls)

        static void BuildLeftPanel(Transform canvas, VoiceLab lab)
        {
            var panel = MakePanel("ControlsPanel", canvas, new Vector2(430, 830), new Vector2(40, -110));

            MakeTMP("EngineLabel", panel.transform, "ENGINE", 14, TEXT_DIM, TextAlignmentOptions.Left,
                    TL, TL, new Vector2(382, 20), new Vector2(24, -22), TL);
            var engineDd = BuildDropdown(panel.transform, "EngineDropdown", new Vector2(24, -46), new Vector2(382, 46));

            MakeTMP("VoiceLabel", panel.transform, "VOICE", 14, TEXT_DIM, TextAlignmentOptions.Left,
                    TL, TL, new Vector2(382, 20), new Vector2(24, -112), TL);
            var voiceDd = BuildDropdown(panel.transform, "VoiceDropdown", new Vector2(24, -136), new Vector2(382, 46));

            MakeTMP("PitchLabel", panel.transform, "PITCH", 14, TEXT_DIM, TextAlignmentOptions.Left,
                    TL, TL, new Vector2(200, 20), new Vector2(24, -202), TL);
            var pitchVal = MakeTMP("PitchValue", panel.transform, "1.00", 15, TEXT, TextAlignmentOptions.Right,
                                   TL, TL, new Vector2(382, 20), new Vector2(24, -202), TL);
            var pitchSlider = BuildSlider(panel.transform, "PitchSlider", new Vector2(24, -228), new Vector2(382, 26));

            // speed row wrapped in a CanvasGroup so VoiceLab can grey it for non-Kokoro engines
            var speedRowGO = MakeRect("SpeedRow", panel.transform, TL, TL, new Vector2(430, 66), new Vector2(0, -282), TL);
            var speedGroup = speedRowGO.AddComponent<CanvasGroup>();
            MakeTMP("SpeedLabel", speedRowGO.transform, "SPEED  (Kokoro only)", 14, TEXT_DIM, TextAlignmentOptions.Left,
                    TL, TL, new Vector2(260, 20), new Vector2(24, 0), TL);
            var speedVal = MakeTMP("SpeedValue", speedRowGO.transform, "1.00", 15, TEXT, TextAlignmentOptions.Right,
                                   TL, TL, new Vector2(382, 20), new Vector2(24, 0), TL);
            var speedSlider = BuildSlider(speedRowGO.transform, "SpeedSlider", new Vector2(24, -26), new Vector2(382, 26));

            var divider = MakeRect("Divider", panel.transform, TL, TL, new Vector2(382, 1), new Vector2(24, -366), TL);
            var divImg = divider.AddComponent<Image>();
            divImg.color = FIELD;
            divImg.raycastTarget = false;

            MakeTMP("PresetLabel", panel.transform, "PRESET NAME", 14, TEXT_DIM, TextAlignmentOptions.Left,
                    TL, TL, new Vector2(382, 20), new Vector2(24, -386), TL);
            var presetField = BuildInput(panel.transform, "PresetNameInput", new Vector2(24, -410), new Vector2(382, 46),
                                         false, "e.g. elder_hobb", 18);
            var saveBtn = BuildButton(panel.transform, "SaveButton", "SAVE PRESET", new Vector2(24, -472),
                                      new Vector2(382, 52), new Color(0.20f, 0.42f, 0.30f), TEXT, 19);

            MakeTMP("Hint", panel.transform,
                    "Saving appends/updates Assets/DeepUnity/InferenceEngine/TTS/VoiceLab/voice_presets.json and logs the " +
                    "NPCChatBase inspector values (ttsModel / ttsVoice / voicePitch) to the Console.\n\n" +
                    "Voice switching: Kokoro reloads ~150 MB (seconds); CosyVoice3 and Chatterbox re-stream " +
                    "their full weights on every voice change (GB-scale — watch the Load line).",
                    12.5f, TEXT_DIM, TextAlignmentOptions.TopLeft,
                    TL, TL, new Vector2(382, 250), new Vector2(24, -544), TL);

            SetRef(lab, "engineDropdown", engineDd);
            SetRef(lab, "voiceDropdown", voiceDd);
            SetRef(lab, "pitchSlider", pitchSlider);
            SetRef(lab, "speedSlider", speedSlider);
            SetRef(lab, "pitchValue", pitchVal.GetComponent<TMP_Text>());
            SetRef(lab, "speedValue", speedVal.GetComponent<TMP_Text>());
            SetRef(lab, "speedRow", speedGroup);
            SetRef(lab, "presetNameInput", presetField);
            SetRef(lab, "saveButton", saveBtn);
        }

        // ---------------------------------------------------------------- right panel (text + status)

        static void BuildRightPanel(Transform canvas, VoiceLab lab)
        {
            var panel = MakePanel("AuditionPanel", canvas, new Vector2(1380, 830), new Vector2(500, -110));

            MakeTMP("TextLabel", panel.transform, "TEXT", 14, TEXT_DIM, TextAlignmentOptions.Left,
                    TL, TL, new Vector2(400, 20), new Vector2(24, -22), TL);
            var textField = BuildInput(panel.transform, "TextInput", new Vector2(24, -46), new Vector2(1332, 330),
                                       true, "Type something to speak...", 20);
            SetString(textField, "m_Text", VoiceLab.SAMPLE_MEDIUM);   // prefill the medium sample line

            MakeTMP("SamplesLabel", panel.transform, "SAMPLES", 14, TEXT_DIM, TextAlignmentOptions.Left,
                    TL, TL, new Vector2(100, 20), new Vector2(24, -400), TL);
            var shortBtn = BuildButton(panel.transform, "SampleShortButton", "SHORT", new Vector2(130, -392),
                                       new Vector2(150, 40), FIELD, TEXT, 15);
            var mediumBtn = BuildButton(panel.transform, "SampleMediumButton", "MEDIUM", new Vector2(292, -392),
                                        new Vector2(150, 40), FIELD, TEXT, 15);
            var longBtn = BuildButton(panel.transform, "SampleLongButton", "LONG", new Vector2(454, -392),
                                      new Vector2(150, 40), FIELD, TEXT, 15);

            var speakBtn = BuildButton(panel.transform, "SpeakButton", "SPEAK", new Vector2(24, -452),
                                       new Vector2(240, 58), ACCENT, Color.white, 22);
            var stopBtn = BuildButton(panel.transform, "StopButton", "STOP", new Vector2(280, -452),
                                      new Vector2(150, 58), STOP_RED, TEXT, 20);

            var statusPanel = MakeRect("StatusPanel", panel.transform, TL, TL, new Vector2(1332, 282), new Vector2(24, -524), TL);
            var statusBg = statusPanel.AddComponent<Image>();
            statusBg.color = PANEL2;
            statusBg.raycastTarget = false;
            var statusGO = MakeTMP("StatusText", statusPanel.transform, "Status appears here in play mode.",
                                   17, TEXT, TextAlignmentOptions.TopLeft,
                                   TL, TL, new Vector2(1300, 254), new Vector2(16, -14), TL);

            SetRef(lab, "textInput", textField);
            SetRef(lab, "speakButton", speakBtn);
            SetRef(lab, "stopButton", stopBtn);
            SetRef(lab, "sampleShortButton", shortBtn);
            SetRef(lab, "sampleMediumButton", mediumBtn);
            SetRef(lab, "sampleLongButton", longBtn);
            SetRef(lab, "statusLabel", statusGO.GetComponent<TMP_Text>());
        }

        // ---------------------------------------------------------------- widget builders

        static GameObject MakePanel(string name, Transform parent, Vector2 size, Vector2 pos)
        {
            var go = MakeRect(name, parent, TL, TL, size, pos, TL);
            var img = go.AddComponent<Image>();
            img.color = PANEL;
            img.raycastTarget = false;
            return go;
        }

        static TMP_Dropdown BuildDropdown(Transform parent, string name, Vector2 pos, Vector2 size)
        {
            var go = MakeRect(name, parent, TL, TL, size, pos, TL);
            var bg = go.AddComponent<Image>();
            bg.color = FIELD;
            var dd = go.AddComponent<TMP_Dropdown>();
            dd.targetGraphic = bg;
            SetTint(dd);

            // caption: stretch with 14px left / 32px right padding
            var captionGO = MakeTMP("Label", go.transform, "-", 19, TEXT, TextAlignmentOptions.Left,
                                    Vector2.zero, Vector2.one, new Vector2(-46, 0), new Vector2(-9, 0));
            MakeTMP("Arrow", go.transform, "v", 15, TEXT_DIM, TextAlignmentOptions.Center,
                    new Vector2(1, 0.5f), new Vector2(1, 0.5f), new Vector2(24, 24), new Vector2(-20, 0));

            // template (inactive; TMP_Dropdown clones it as the popup list)
            var template = MakeRect("Template", go.transform, new Vector2(0, 0), new Vector2(1, 0),
                                    new Vector2(0, 340), new Vector2(0, -2), new Vector2(0.5f, 1f));
            var tImg = template.AddComponent<Image>();
            tImg.color = PANEL2;
            var scroll = template.AddComponent<ScrollRect>();

            var viewport = MakeRect("Viewport", template.transform, Vector2.zero, Vector2.one,
                                    Vector2.zero, Vector2.zero, TL);
            viewport.AddComponent<RectMask2D>();

            var content = MakeRect("Content", viewport.transform, new Vector2(0, 1), new Vector2(1, 1),
                                   new Vector2(0, 34), Vector2.zero, new Vector2(0.5f, 1f));

            var item = MakeRect("Item", content.transform, new Vector2(0, 0.5f), new Vector2(1, 0.5f),
                                new Vector2(0, 34), Vector2.zero);
            var toggle = item.AddComponent<Toggle>();
            var itemBgGO = MakeRect("Item Background", item.transform, Vector2.zero, Vector2.one,
                                    Vector2.zero, Vector2.zero);
            var itemBg = itemBgGO.AddComponent<Image>();
            itemBg.color = PANEL2;
            var checkGO = MakeRect("Item Checkmark", item.transform, new Vector2(0, 0.5f), new Vector2(0, 0.5f),
                                   new Vector2(12, 12), new Vector2(18, 0));
            var check = checkGO.AddComponent<Image>();
            check.color = ACCENT;
            check.raycastTarget = false;
            var itemLabelGO = MakeTMP("Item Label", item.transform, "-", 17, TEXT, TextAlignmentOptions.Left,
                                      Vector2.zero, Vector2.one, new Vector2(-46, 0), new Vector2(9, 0));

            toggle.targetGraphic = itemBg;
            toggle.graphic = check;
            toggle.isOn = true;
            SetTint(toggle);

            scroll.content = (RectTransform)content.transform;
            scroll.viewport = (RectTransform)viewport.transform;
            scroll.horizontal = false;
            scroll.vertical = true;
            scroll.movementType = ScrollRect.MovementType.Clamped;
            scroll.scrollSensitivity = 30f;

            dd.template = (RectTransform)template.transform;
            dd.captionText = captionGO.GetComponent<TMP_Text>();
            dd.itemText = itemLabelGO.GetComponent<TMP_Text>();
            template.SetActive(false);
            return dd;
        }

        static Slider BuildSlider(Transform parent, string name, Vector2 pos, Vector2 size)
        {
            var go = MakeRect(name, parent, TL, TL, size, pos, TL);
            var slider = go.AddComponent<Slider>();

            var bgGO = MakeRect("Background", go.transform, new Vector2(0, 0.5f), new Vector2(1, 0.5f),
                                new Vector2(0, 6), Vector2.zero);
            var bgImg = bgGO.AddComponent<Image>();
            bgImg.color = FIELD;
            bgImg.raycastTarget = false;

            var fillArea = MakeRect("Fill Area", go.transform, new Vector2(0, 0.5f), new Vector2(1, 0.5f),
                                    new Vector2(-18, 6), Vector2.zero);
            var fillGO = MakeRect("Fill", fillArea.transform, Vector2.zero, new Vector2(0.5f, 1f),
                                  new Vector2(9, 0), Vector2.zero);
            var fillImg = fillGO.AddComponent<Image>();
            fillImg.color = ACCENT;
            fillImg.raycastTarget = false;

            var handleArea = MakeRect("Handle Slide Area", go.transform, Vector2.zero, Vector2.one,
                                      new Vector2(-18, 0), Vector2.zero);
            var handleGO = MakeRect("Handle", handleArea.transform, Vector2.zero, new Vector2(0, 1),
                                    new Vector2(18, 0), Vector2.zero);
            var handleImg = handleGO.AddComponent<Image>();
            handleImg.color = HANDLE;

            slider.fillRect = (RectTransform)fillGO.transform;
            slider.handleRect = (RectTransform)handleGO.transform;
            slider.targetGraphic = handleImg;
            slider.minValue = 0.5f;
            slider.maxValue = 1.5f;
            slider.value = 1f;
            SetTint(slider);
            return slider;
        }

        static Button BuildButton(Transform parent, string name, string label, Vector2 pos, Vector2 size,
                                  Color bgColor, Color textColor, float fontSize)
        {
            var go = MakeRect(name, parent, TL, TL, size, pos, TL);
            var img = go.AddComponent<Image>();
            img.color = bgColor;
            var btn = go.AddComponent<Button>();
            btn.targetGraphic = img;
            SetTint(btn);
            MakeTMP("Label", go.transform, label, fontSize, textColor, TextAlignmentOptions.Center,
                    Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            return btn;
        }

        static TMP_InputField BuildInput(Transform parent, string name, Vector2 pos, Vector2 size,
                                         bool multiline, string placeholder, float fontSize)
        {
            var go = MakeRect(name, parent, TL, TL, size, pos, TL);
            var bg = go.AddComponent<Image>();
            bg.color = FIELD;
            var field = go.AddComponent<TMP_InputField>();
            field.targetGraphic = bg;
            SetTint(field);

            var areaGO = MakeRect("Text Area", go.transform, Vector2.zero, Vector2.one,
                                  new Vector2(-28, -18), Vector2.zero);
            areaGO.AddComponent<RectMask2D>();

            var align = multiline ? TextAlignmentOptions.TopLeft : TextAlignmentOptions.Left;
            var phGO = MakeTMP("Placeholder", areaGO.transform, placeholder, fontSize, TEXT_DIM, align,
                               Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            phGO.GetComponent<TMP_Text>().fontStyle = FontStyles.Italic;
            var txtGO = MakeTMP("Text", areaGO.transform, "", fontSize, TEXT, align,
                                Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);

            field.textViewport = (RectTransform)areaGO.transform;
            field.textComponent = txtGO.GetComponent<TMP_Text>();
            field.placeholder = phGO.GetComponent<TMP_Text>();
            field.lineType = multiline ? TMP_InputField.LineType.MultiLineNewline
                                       : TMP_InputField.LineType.SingleLine;
            field.richText = false;
            field.caretColor = ACCENT;
            field.customCaretColor = true;
            field.caretWidth = 2;
            field.selectionColor = new Color(ACCENT.r, ACCENT.g, ACCENT.b, 0.45f);
            return field;
        }

        static void SetTint(Selectable s)
        {
            var c = s.colors;
            c.normalColor = Color.white;
            c.highlightedColor = new Color(1.18f, 1.18f, 1.18f, 1f);
            c.pressedColor = new Color(0.82f, 0.82f, 0.82f, 1f);
            c.selectedColor = new Color(1.10f, 1.10f, 1.10f, 1f);
            c.disabledColor = new Color(0.55f, 0.55f, 0.55f, 0.5f);
            s.colors = c;
        }

        // ---------------------------------------------------------------- shared helpers

        static GameObject MakeRect(string name, Transform parent, Vector2 anchorMin, Vector2 anchorMax,
                                   Vector2 sizeDelta, Vector2 anchoredPos, Vector2? pivot = null)
        {
            var go = new GameObject(name, typeof(RectTransform));
            var rt = (RectTransform)go.transform;
            rt.SetParent(parent, false);
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            if (pivot.HasValue) rt.pivot = pivot.Value;
            rt.sizeDelta = sizeDelta;
            rt.anchoredPosition = anchoredPos;
            return go;
        }

        static GameObject MakeTMP(string name, Transform parent, string text, float size, Color color,
                                  TextAlignmentOptions align, Vector2 anchorMin, Vector2 anchorMax,
                                  Vector2 sizeDelta, Vector2 anchoredPos, Vector2? pivot = null)
        {
            var go = MakeRect(name, parent, anchorMin, anchorMax, sizeDelta, anchoredPos, pivot);
            var tmp = go.AddComponent<TextMeshProUGUI>();   // default TMP font
            tmp.text = text;
            tmp.fontSize = size;
            tmp.color = color;
            tmp.alignment = align;
            tmp.raycastTarget = false;
            return go;
        }

        static void SetRef(Component c, string field, UnityEngine.Object value)
        {
            var so = new SerializedObject(c);
            var prop = so.FindProperty(field);
            if (prop == null) throw new System.Exception($"No serialized field '{field}' on {c.GetType().Name}");
            prop.objectReferenceValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        static void SetString(Component c, string field, string value)
        {
            var so = new SerializedObject(c);
            var prop = so.FindProperty(field);
            if (prop == null) throw new System.Exception($"No serialized field '{field}' on {c.GetType().Name}");
            prop.stringValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }
    }
}

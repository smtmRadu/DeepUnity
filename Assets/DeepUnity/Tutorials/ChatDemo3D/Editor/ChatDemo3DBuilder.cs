using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TMPro;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.Events;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Events;
using UnityEngine.EventSystems;
using UnityEngine.Playables;
using UnityEngine.TextCore.LowLevel;
using UnityEngine.UI;

namespace DeepUnity.Tutorials.ChatDemo3D.EditorTools
{
    /// <summary>
    /// Deterministically builds the ChatDemo3D scene (souls-like castle ruins courtyard with a
    /// playable knight and the Qwen3.5 dialogue NPC) out of the CC0 Quaternius assets in
    /// ChatDemo3D/Art. Also renames the old 2D ChatDemo folder/scene to ChatDemo2D.
    /// Run from the menu (DeepUnity/Build ChatDemo3D Scene) or in batch mode via
    /// -executeMethod DeepUnity.Tutorials.ChatDemo3D.EditorTools.ChatDemo3DBuilder.BuildBatch
    /// </summary>
    public static class ChatDemo3DBuilder
    {
        const string ROOT = "Assets/DeepUnity/Tutorials/ChatDemo3D";
        const string ART = ROOT + "/Art";
        const string GEN = ROOT + "/Generated";
        const string SCENE_PATH = ROOT + "/ChatDemo3D.unity";

        static readonly System.Random rng = new System.Random(20260610);

        // ---------------------------------------------------------------- entry points

        [MenuItem("DeepUnity/Build ChatDemo3D Scene")]
        public static void BuildMenu()
        {
            RenameChatDemoTo2D();
            ConfigureImports();
            BuildEverything();
            Debug.Log("[ChatDemo3DBuilder] Done. Scene at " + SCENE_PATH);
        }

        public static void BuildBatch()
        {
            try
            {
                RenameChatDemoTo2D();
                ConfigureImports();
                BuildEverything();
                Debug.Log("[ChatDemo3DBuilder] BATCH OK");
                EditorApplication.Exit(0);
            }
            catch (Exception e)
            {
                Debug.LogError("[ChatDemo3DBuilder] BATCH FAILED: " + e);
                EditorApplication.Exit(1);
            }
        }

        // ---------------------------------------------------------------- 2D rename

        public static void RenameChatDemoTo2D()
        {
            if (AssetDatabase.IsValidFolder("Assets/DeepUnity/Tutorials/ChatDemo"))
            {
                string err = AssetDatabase.MoveAsset("Assets/DeepUnity/Tutorials/ChatDemo",
                                                     "Assets/DeepUnity/Tutorials/ChatDemo2D");
                if (!string.IsNullOrEmpty(err)) throw new Exception("ChatDemo folder rename failed: " + err);
                Debug.Log("[ChatDemo3DBuilder] Renamed ChatDemo -> ChatDemo2D");
            }

            string oldScene = "Assets/DeepUnity/Tutorials/ChatDemo2D/ChatDemo.unity";
            if (File.Exists(oldScene))
            {
                string err = AssetDatabase.RenameAsset(oldScene, "ChatDemo2D");
                if (!string.IsNullOrEmpty(err)) throw new Exception("ChatDemo scene rename failed: " + err);
            }

            // refresh serialized type names of the old namespace inside the scene + prefab
            foreach (string f in new[]
            {
                "Assets/DeepUnity/Tutorials/ChatDemo2D/ChatDemo2D.unity",
                "Assets/DeepUnity/Tutorials/ChatDemo2D/Prefabs/ChatWindow.prefab"
            })
            {
                if (!File.Exists(f)) continue;
                string txt = File.ReadAllText(f);
                string fixedTxt = txt.Replace("DeepUnity.Tutorials.ChatDemo.", "DeepUnity.Tutorials.ChatDemo2D.");
                if (fixedTxt != txt) File.WriteAllText(f, fixedTxt);
            }
            AssetDatabase.Refresh();
        }

        // ---------------------------------------------------------------- import configuration

        static void ConfigureImports()
        {
            // humanoid characters + animation libraries
            foreach (string p in new[]
            {
                ART + "/Characters/Warrior.fbx",
                ART + "/Characters/Monk.fbx",
                ART + "/Animations/UAL1.fbx",
                ART + "/Animations/UAL2.fbx",
            })
                ConfigureHumanoid(p, rotateClips180: p.Contains("/Animations/"));

            // Nov-2020 RPG pack characters (Rogue beggar) + Ultimate-Modular-Women Witch: their
            // feet are IK targets OUTSIDE the leg chain, so humanoid retarget is impossible —
            // import them GENERIC and play their OWN embedded clips instead of UAL. Witch.fbx
            // (Quaternius Apr-2022 women pack, CC0) is the REAL female model for Morwenna; its
            // materials are flat palette colors baked in the FBX (no texture to apply).
            foreach (string p in new[] { ART + "/Characters/Wizard.fbx", ART + "/Characters/Rogue.fbx",
                                         ART + "/Characters/Witch.fbx" })
                ConfigureGenericAnimated(p);

            // static art (weapons + every ruins piece)
            foreach (string guid in AssetDatabase.FindAssets("t:Model", new[] { ART + "/Weapons", ART + "/Ruins" }))
            {
                string p = AssetDatabase.GUIDToAssetPath(guid);
                var imp = (ModelImporter)AssetImporter.GetAtPath(p);
                if (imp.animationType == ModelImporterAnimationType.None && !imp.importAnimation) continue;
                imp.animationType = ModelImporterAnimationType.None;
                imp.importAnimation = false;
                imp.importCameras = false;
                imp.importLights = false;
                imp.SaveAndReimport();
            }
            AssetDatabase.SaveAssets();
        }

        // generic-rig import for the animated RPG-pack characters (own clips, no retargeting)
        static void ConfigureGenericAnimated(string path)
        {
            var imp = AssetImporter.GetAtPath(path) as ModelImporter;
            if (imp == null) throw new Exception("Missing model: " + path);
            imp.animationType = ModelImporterAnimationType.Generic;
            imp.importAnimation = true;
            imp.importCameras = false;
            imp.importLights = false;
            imp.importNormals = ModelImporterNormals.Calculate;   // these FBX ship without normals
            var clips = imp.defaultClipAnimations;
            foreach (var c in clips)
            {
                string clean = c.takeName.Contains("|") ? c.takeName.Substring(c.takeName.IndexOf('|') + 1) : c.takeName;
                c.name = clean;
                c.loopTime = clean.Contains("Idle") || clean.StartsWith("Spell") || clean == "Walk" || clean == "Run"
                          || clean == "Interact" || clean == "Wave";   // looping talk gestures (women pack)
                c.keepOriginalOrientation = true;
                c.keepOriginalPositionXZ = true;
                c.keepOriginalPositionY = true;
                c.lockRootRotation = true;
                c.lockRootPositionXZ = true;
                c.lockRootHeightY = true;
            }
            if (clips.Length > 0) imp.clipAnimations = clips;
            imp.SaveAndReimport();
        }

        // explicit mecanim mapping — Unity's auto-mapper chokes on the Quaternius bone names
        // (Abdomen/Torso/Palm.L/Fist.L). Candidates cover both the Quaternius rigs and the
        // UE-mannequin style names used by the Universal Animation Library.
        static readonly (string human, string[] candidates)[] BONE_MAP =
        {
            ("Hips",          new[]{ "Hips", "pelvis" }),
            ("Spine",         new[]{ "Abdomen", "spine_01" }),
            ("Chest",         new[]{ "Torso", "spine_02" }),
            ("UpperChest",    new[]{ "spine_03" }),
            ("Neck",          new[]{ "Neck", "neck_01" }),
            ("Head",          new[]{ "Head" }),
            ("LeftShoulder",  new[]{ "Shoulder.L", "clavicle_l" }),
            ("LeftUpperArm",  new[]{ "UpperArm.L", "upperarm_l" }),
            ("LeftLowerArm",  new[]{ "LowerArm.L", "lowerarm_l" }),
            ("LeftHand",      new[]{ "Palm.L", "Fist.L", "hand_l" }),
            ("RightShoulder", new[]{ "Shoulder.R", "clavicle_r" }),
            ("RightUpperArm", new[]{ "UpperArm.R", "upperarm_r" }),
            ("RightLowerArm", new[]{ "LowerArm.R", "lowerarm_r" }),
            ("RightHand",     new[]{ "Palm.R", "Fist.R", "hand_r" }),
            ("LeftUpperLeg",  new[]{ "UpperLeg.L", "thigh_l" }),
            ("LeftLowerLeg",  new[]{ "LowerLeg.L", "calf_l" }),
            ("LeftFoot",      new[]{ "Foot.L", "foot_l" }),
            ("LeftToes",      new[]{ "Toes.L", "ball_l" }),
            ("RightUpperLeg", new[]{ "UpperLeg.R", "thigh_r" }),
            ("RightLowerLeg", new[]{ "LowerLeg.R", "calf_r" }),
            ("RightFoot",     new[]{ "Foot.R", "foot_r" }),
            ("RightToes",     new[]{ "Toes.R", "ball_r" }),
        };

        static void ConfigureHumanoid(string path, bool rotateClips180 = false)
        {
            var imp = AssetImporter.GetAtPath(path) as ModelImporter;
            if (imp == null) throw new Exception("Missing model: " + path);

            imp.animationType = ModelImporterAnimationType.Human;
            imp.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
            imp.importCameras = false;
            imp.importLights = false;
            imp.importAnimation = true;

            // build the explicit human description from whatever bones this model actually has
            var modelGO = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            var boneNames = new HashSet<string>(modelGO.GetComponentsInChildren<Transform>(true).Select(t => t.name));
            var human = new List<HumanBone>();
            var unmatched = new List<string>();
            foreach (var (humanName, candidates) in BONE_MAP)
            {
                string found = candidates.FirstOrDefault(c => boneNames.Contains(c));
                if (found != null)
                    human.Add(new HumanBone { humanName = humanName, boneName = found, limit = new HumanLimit { useDefaultValues = true } });
                else
                    unmatched.Add(humanName);
            }
            // some Quaternius rigs (robed Wizard/Cleric) keep Foot.L/R as IK TARGETS outside the
            // leg chain (children of Root, siblings of Body). A humanoid foot must descend from
            // the lower leg, so fall back to the leg's end bone — these NPCs only idle in place.
            var allT = modelGO.GetComponentsInChildren<Transform>(true);
            foreach (string s in new[] { "L", "R" })
            {
                var foot = allT.FirstOrDefault(t => t.name == "Foot." + s);
                var lower = allT.FirstOrDefault(t => t.name == "LowerLeg." + s);
                if (foot == null || lower == null || foot.IsChildOf(lower)) continue;
                if (!boneNames.Contains("LowerLeg." + s + "_end")) continue;
                string side = s == "L" ? "Left" : "Right";
                human.RemoveAll(h => h.humanName == side + "Foot");
                human.Add(new HumanBone { humanName = side + "Foot", boneName = "LowerLeg." + s + "_end", limit = new HumanLimit { useDefaultValues = true } });
            }
            // the Nov-2020 pack FBX ships without normals — recalculate or they shade black
            if (path.Contains("Wizard") || path.Contains("Rogue"))
                imp.importNormals = ModelImporterNormals.Calculate;
            Debug.Log($"[ChatDemo3DBuilder] {Path.GetFileName(path)} mapped {human.Count} bones" +
                      (unmatched.Count > 0 ? ", unmatched: " + string.Join(",", unmatched) : "") +
                      " | hierarchy: " + HierarchyDump(modelGO.transform, 0));
            imp.humanDescription = new HumanDescription
            {
                human = human.ToArray(),
                skeleton = new SkeletonBone[0],   // empty = use the model's own skeleton / bind pose
                upperArmTwist = 0.5f, lowerArmTwist = 0.5f,
                upperLegTwist = 0.5f, lowerLegTwist = 0.5f,
                armStretch = 0.05f, legStretch = 0.05f,
                feetSpacing = 0f, hasTranslationDoF = false,
            };

            // strip the "Armature|" / "HumanArmature|" take prefixes and mark looping clips
            var clips = imp.defaultClipAnimations;
            foreach (var c in clips)
            {
                string clean = c.takeName.Contains("|") ? c.takeName.Substring(c.takeName.IndexOf('|') + 1) : c.takeName;
                c.name = clean;
                c.loopTime = clean.Contains("Loop") || clean is "Sword_Idle" or "Idle" or "Walking" or "Run"
                             or "Idle_swordLeft" or "Idle_swordRight" or "Run_swordRight";
                c.keepOriginalOrientation = true;
                c.keepOriginalPositionXZ = true;
                c.keepOriginalPositionY = true;
                c.lockRootRotation = true;
                c.lockRootPositionXZ = true;
                c.lockRootHeightY = true;
                // the UAL clips are authored facing the opposite way — without this the
                // character visually runs backward
                if (rotateClips180) c.rotationOffset = 180f;
            }
            if (clips.Length > 0)
                imp.clipAnimations = clips;
            imp.SaveAndReimport();

            var avatar = AssetDatabase.LoadAllAssetsAtPath(path).OfType<Avatar>().FirstOrDefault();
            if (avatar == null || !avatar.isValid || !avatar.isHuman)
                throw new Exception($"Humanoid avatar setup failed for {path} (valid={avatar?.isValid}, human={avatar?.isHuman})");
            Debug.Log($"[ChatDemo3DBuilder] Humanoid OK: {path}");
        }

        static string HierarchyDump(Transform t, int depth)
        {
            var sb = new System.Text.StringBuilder();
            sb.Append('\n').Append(new string(' ', depth * 2)).Append(t.name);
            foreach (Transform c in t) sb.Append(HierarchyDump(c, depth + 1));
            return sb.ToString();
        }

        // ---------------------------------------------------------------- shared asset helpers

        static GameObject LoadModel(string relPath)
        {
            var go = AssetDatabase.LoadAssetAtPath<GameObject>(ART + "/" + relPath);
            if (go == null) throw new Exception("Missing model asset: " + ART + "/" + relPath);
            return go;
        }

        static GameObject Ruin(string name) => LoadModel("Ruins/" + name + ".fbx");

        static AnimationClip Clip(string fbxRelPath, string clipName)
        {
            var clip = AssetDatabase.LoadAllAssetsAtPath(ART + "/" + fbxRelPath)
                                    .OfType<AnimationClip>()
                                    .FirstOrDefault(c => c.name == clipName && !c.name.StartsWith("__preview"));
            if (clip == null)
            {
                string available = string.Join(", ", AssetDatabase.LoadAllAssetsAtPath(ART + "/" + fbxRelPath)
                                                                  .OfType<AnimationClip>().Select(c => c.name));
                throw new Exception($"Clip '{clipName}' not found in {fbxRelPath}. Available: {available}");
            }
            return clip;
        }

        static Bounds RendererBounds(GameObject instance)
        {
            var rs = instance.GetComponentsInChildren<Renderer>();
            if (rs.Length == 0) return new Bounds(instance.transform.position, Vector3.zero);
            Bounds b = rs[0].bounds;
            foreach (var r in rs.Skip(1)) b.Encapsulate(r.bounds);
            return b;
        }

        static Bounds Measure(GameObject prefab)
        {
            var tmp = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
            tmp.transform.position = Vector3.zero;
            Bounds b = RendererBounds(tmp);
            UnityEngine.Object.DestroyImmediate(tmp);
            return b;
        }

        static void SetRef(Component c, string field, UnityEngine.Object value)
        {
            var so = new SerializedObject(c);
            var prop = so.FindProperty(field);
            if (prop == null) throw new Exception($"No serialized field '{field}' on {c.GetType().Name}");
            prop.objectReferenceValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        static void SetFloat(Component c, string field, float value)
        {
            var so = new SerializedObject(c);
            so.FindProperty(field).floatValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        static void SetBool(Component c, string field, bool value)
        {
            var so = new SerializedObject(c);
            so.FindProperty(field).boolValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        static void SetEnum(Component c, string field, int value)
        {
            var so = new SerializedObject(c);
            so.FindProperty(field).enumValueIndex = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        static void SetInt(Component c, string field, int value)
        {
            var so = new SerializedObject(c);
            so.FindProperty(field).intValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        static void SetObject(Component c, string field, UnityEngine.Object value)
        {
            var so = new SerializedObject(c);
            var prop = so.FindProperty(field);
            if (prop == null) throw new Exception($"No serialized field '{field}' on {c.GetType().Name}");
            prop.objectReferenceValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        static void SetString(Component c, string field, string value)
        {
            var so = new SerializedObject(c);
            var prop = so.FindProperty(field);
            if (prop == null) throw new Exception($"No serialized field '{field}' on {c.GetType().Name}");
            prop.stringValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        static float Range(float min, float max) => min + (float)rng.NextDouble() * (max - min);
        static T Pick<T>(params T[] options) => options[rng.Next(options.Length)];

        // ---------------------------------------------------------------- build

        static void BuildEverything()
        {
            if (!AssetDatabase.IsValidFolder(GEN))
                AssetDatabase.CreateFolder(ROOT, "Generated");

            var cinzel = CreateCinzelFont();
            var vignette = CreateVignetteSprite();
            var playerCtrl = CreatePlayerAnimator();
            var npcCtrl = CreateNpcAnimator();
            var bossCtrl = CreateBossAnimator(out float[] bossSwings);

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            SetupLightingAndSky();
            float wallUnit = BuildCastle();   // builds the terrain too (needs the castle extents)

            GameObject player = BuildPlayer(playerCtrl);
            GameObject cameraRig = BuildCamera(player);
            GameObject npc = BuildNpc(npcCtrl, player);   // he carries the gear he offers the player
            GameObject witch = BuildWitchNpc(npcCtrl);
            GameObject boss = BuildBoss(bossCtrl, bossSwings);

            BuildUI(cinzel, vignette, npc, witch, player);

            // ambient exploration music, quiet and looping; streamed. ambient_theme.mp3 is a
            // royalty-free souls-like track shipped IN the repo (unlike the gitignored, copyrighted
            // limgrave_theme.ogg — drop that back in as ambient_theme if you have the rights).
            var audioImp = AssetImporter.GetAtPath(ART + "/Audio/ambient_theme.mp3") as AudioImporter;
            if (audioImp != null)
            {
                var sampleSettings = audioImp.defaultSampleSettings;
                sampleSettings.loadType = AudioClipLoadType.Streaming;
                audioImp.defaultSampleSettings = sampleSettings;
                audioImp.SaveAndReimport();
            }
            var ambience = new GameObject("Ambience").AddComponent<AudioSource>();
            ambience.clip = AssetDatabase.LoadAssetAtPath<AudioClip>(ART + "/Audio/ambient_theme.mp3");
            ambience.loop = true;
            ambience.playOnAwake = true;
            ambience.volume = 0.144f;  // user 2026-07-15: 80% of 0.18 — still competed with the NPC voices
            ambience.spatialBlend = 0f;

            // …and the reason it still competed: everything outside a conversation now eases down to
            // the talking NPC's worldAudioWhileTalking over ~3 s (exponential, so it reads as an even
            // fade) and back up on close. The ducker finds the sources itself and leaves the NPCs and
            // the dialogue window alone; it lives on Ambience because that object is always active.
            ambience.gameObject.AddComponent<ConversationAudioDucker>();

            // Kernel prewarm now lives INSIDE NPCChatBase.Awake (frame-0, per model, automatic) —
            // no helper object needed. Only the dormant frame-spike probe remains (tick its
            // `record` in the inspector when hunting fps dips).
            new GameObject("FrameSpikeProbe").AddComponent<FrameSpikeProbe>();

            // final cross-wiring
            SetRef(player.GetComponent<SoulsPlayerController>(), "cam", cameraRig.GetComponent<SoulsCameraRig>());
            SetRef(cameraRig.GetComponent<SoulsCameraRig>(), "target", player.transform);
            var bossComp = boss.GetComponent<BossController>();
            SetRef(bossComp, "player", player.GetComponent<SoulsPlayerController>());
            SetRef(bossComp, "healthFill", s_bossFill);
            SetRef(bossComp, "barGroup", s_bossBarGroup);
            SetRef(bossComp, "deathScreen", s_deathScreen);
            SetRef(bossComp, "musicSource", boss.GetComponent<AudioSource>());
            SetRef(bossComp, "ambienceSource", ambience);
            SetRef(player.GetComponent<SoulsPlayerController>(), "deathScreen", s_deathScreen);

            if (mistDoorGO != null)
            {
                var md = mistDoorGO.GetComponent<MistDoor>();
                SetRef(md, "player", player.GetComponent<SoulsPlayerController>());
                SetRef(md, "prompt", s_mistPrompt);
                SetRef(md, "whiteFlash", s_whiteFlash);
                SetRef(md, "boss", bossComp);
            }

            EditorSceneManager.SaveScene(scene, SCENE_PATH);
            AssetDatabase.SaveAssets();
            Debug.Log($"[ChatDemo3DBuilder] Scene saved ({SCENE_PATH}), wall unit = {wallUnit:0.00} m");
        }

        // ---------------------------------------------------------------- lighting / mood

        static void SetupLightingAndSky()
        {
            // night procedural skybox — the "sun disk" plays the moon (skyboxes ignore fog,
            // so the moon stays crisp at any distance)
            var sky = new Material(Shader.Find("Skybox/Procedural"));
            sky.SetFloat("_SunSize", 0.08f);
            sky.SetFloat("_SunSizeConvergence", 10f);
            sky.SetFloat("_AtmosphereThickness", 0.45f);
            sky.SetColor("_SkyTint", new Color(0.18f, 0.21f, 0.30f));
            sky.SetColor("_GroundColor", new Color(0.06f, 0.06f, 0.08f));
            sky.SetFloat("_Exposure", 0.5f);
            AssetDatabase.CreateAsset(sky, GEN + "/SkyNight.mat");

            var sunGO = new GameObject("Moonlight");
            var sun = sunGO.AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.color = new Color(0.68f, 0.74f, 0.94f);     // cold moonlight
            sun.intensity = 0.7f;
            sun.shadows = LightShadows.Soft;
            sun.shadowStrength = 0.75f;
            // moon hangs north-east, ~30° up — visible when walking toward the NPC
            sunGO.transform.rotation = Quaternion.Euler(30f, 205f, 0f);

            RenderSettings.skybox = sky;
            RenderSettings.sun = sun;
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            // readable night mist: slightly lifted color + denser falloff so the courtyard's far
            // walls and the gate actually dissolve into it (0.013 was imperceptible in play)
            RenderSettings.fogColor = new Color(0.10f, 0.11f, 0.16f);
            RenderSettings.fogDensity = 0.024f;
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.22f, 0.25f, 0.35f);
            RenderSettings.ambientEquatorColor = new Color(0.13f, 0.14f, 0.20f);
            RenderSettings.ambientGroundColor = new Color(0.06f, 0.06f, 0.08f);

            // no GI baking — everything realtime so the batch build needs no bake step
            var ls = new LightingSettings { bakedGI = false, realtimeGI = false };
            ls.name = "ChatDemo3D LightingSettings";
            AssetDatabase.CreateAsset(ls, GEN + "/ChatDemo3D.lighting");
            Lightmapping.lightingSettings = ls;
        }

        static float castleHx, castleHz;

        // rolling terrain mesh: dead flat inside the castle walls and along the gate path,
        // gentle perlin hills everywhere else
        static void BuildGround(float hx, float hz)
        {
            castleHx = hx;
            castleHz = hz;

            var mat = new Material(Shader.Find("Standard"));
            mat.mainTexture = CreateGroundTexture();
            mat.mainTextureScale = Vector2.one;              // tiling lives in the mesh UVs (world/8)
            mat.color = Color.white;
            mat.SetFloat("_Glossiness", 0.04f);
            AssetDatabase.CreateAsset(mat, GEN + "/Ground.mat");

            const int N = 160;
            const float SIZE = 400f;
            const float STEP = SIZE / N;
            var verts = new Vector3[(N + 1) * (N + 1)];
            var uvs = new Vector2[verts.Length];
            for (int z = 0; z <= N; z++)
                for (int x = 0; x <= N; x++)
                {
                    float wx = -SIZE * 0.5f + x * STEP;
                    float wz = -SIZE * 0.5f + z * STEP;
                    int i = z * (N + 1) + x;
                    verts[i] = new Vector3(wx, GroundHeight(wx, wz), wz);
                    uvs[i] = new Vector2(wx / 8f, wz / 8f);
                }
            var tris = new int[N * N * 6];
            int t = 0;
            for (int z = 0; z < N; z++)
                for (int x = 0; x < N; x++)
                {
                    int a = z * (N + 1) + x, b = a + 1, c = a + N + 1, d = c + 1;
                    tris[t++] = a; tris[t++] = c; tris[t++] = b;
                    tris[t++] = b; tris[t++] = c; tris[t++] = d;
                }
            var mesh = new Mesh { name = "GroundMesh", indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            mesh.vertices = verts;
            mesh.uv = uvs;
            mesh.triangles = tris;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            AssetDatabase.CreateAsset(mesh, GEN + "/GroundMesh.asset");

            var ground = new GameObject("Ground");
            ground.AddComponent<MeshFilter>().sharedMesh = mesh;
            ground.AddComponent<MeshRenderer>().sharedMaterial = mat;
            ground.AddComponent<MeshCollider>().sharedMesh = mesh;
            ground.isStatic = true;
        }

        static float GroundHeight(float x, float z)
        {
            float dCastle = RectDist(x, z, 0f, 0f, castleHx + 3f, castleHz + 3f);
            float dPath = RectDist(x, z, 0f, -castleHz - 9f, 7f, 11f);   // corridor out the gate
            float dBoss = RectDist(x, z, bossCx, bossCz, bossHalfX + 3f, bossHalfZ + 3f);
            float blend = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(Mathf.Min(dCastle, Mathf.Min(dPath, dBoss)) / 12f));
            float n = Mathf.PerlinNoise(x * 0.022f + 113.7f, z * 0.022f + 71.3f) * 1.7f
                    + Mathf.PerlinNoise(x * 0.09f + 311f, z * 0.09f + 97f) * 0.45f;
            return (n - 1.0f) * blend;
        }

        static float RectDist(float x, float z, float cx, float cz, float halfX, float halfZ)
        {
            float dx = Mathf.Max(0f, Mathf.Abs(x - cx) - halfX);
            float dz = Mathf.Max(0f, Mathf.Abs(z - cz) - halfZ);
            return Mathf.Sqrt(dx * dx + dz * dz);
        }

        // tileable moss / dirt / stone-fleck blend so the big plane doesn't read as a flat color
        static Texture2D CreateGroundTexture()
        {
            string pngPath = GEN + "/GroundMoss.png";
            if (!File.Exists(pngPath))
            {
                const int S = 512;

                // sample noise on a wrapped domain: blend 4 shifted perlin reads so the texture tiles
                float TileableNoise(float u, float v, float freq, float seed)
                {
                    float x = u * freq, y = v * freq, w = freq;
                    float fx = x / w, fy = y / w;
                    float n00 = Mathf.PerlinNoise(seed + x, seed + y);
                    float n10 = Mathf.PerlinNoise(seed + x - w, seed + y);
                    float n01 = Mathf.PerlinNoise(seed + x, seed + y - w);
                    float n11 = Mathf.PerlinNoise(seed + x - w, seed + y - w);
                    return Mathf.Lerp(Mathf.Lerp(n00, n10, fx), Mathf.Lerp(n01, n11, fx), fy);
                }

                Color moss = new Color(0.30f, 0.33f, 0.23f);
                Color dirt = new Color(0.27f, 0.23f, 0.18f);
                Color stone = new Color(0.33f, 0.33f, 0.34f);

                var tex = new Texture2D(S, S, TextureFormat.RGB24, false);
                var px = new Color[S * S];
                for (int y = 0; y < S; y++)
                    for (int x = 0; x < S; x++)
                    {
                        float u = (float)x / S, v = (float)y / S;
                        float patches = TileableNoise(u, v, 5f, 11.31f) * 0.65f + TileableNoise(u, v, 13f, 47.7f) * 0.35f;
                        float flecks = TileableNoise(u, v, 29f, 83.1f);
                        float micro = TileableNoise(u, v, 53f, 7.9f);

                        Color c = Color.Lerp(dirt, moss, Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(0.38f, 0.62f, patches)));
                        c = Color.Lerp(c, stone, Mathf.InverseLerp(0.74f, 0.95f, flecks) * 0.55f);
                        c *= 0.90f + 0.20f * micro;
                        px[y * S + x] = c;
                    }
                tex.SetPixels(px);
                tex.Apply();
                File.WriteAllBytes(pngPath, tex.EncodeToPNG());
                UnityEngine.Object.DestroyImmediate(tex);
                AssetDatabase.ImportAsset(pngPath);
            }
            return AssetDatabase.LoadAssetAtPath<Texture2D>(pngPath);
        }

        // ---------------------------------------------------------------- castle layout

        static Transform envRoot;
        static GameObject mistDoorGO;
        static float bossCx, bossCz, bossHalfX, bossHalfZ;   // boss chamber footprint (terrain + forest need it)

        static float BuildCastle()
        {
            envRoot = new GameObject("Environment").transform;

            Bounds wallB = Measure(Ruin("Wall"));
            Bounds floorB = Measure(Ruin("Floor_Standard"));
            float L = wallB.size.x;                       // wall segment length
            float H = wallB.size.y;                       // wall segment height (they stack)
            float U = floorB.size.x;                      // floor tile size
            Debug.Log($"[ChatDemo3DBuilder] measured wall {wallB.size}, floor tile {floorB.size}");

            // courtyard ~ 16 x 12 wall segments (~32 x 24 m), walls 2 rows + ruined third row
            int segsX = 16, segsZ = 12;
            float hx = segsX * L * 0.5f, hz = segsZ * L * 0.5f;

            // boss chamber footprint outside the north wall — the terrain needs it flat,
            // so the extents must be known before the ground mesh is generated
            bossCx = -hx + L * 0.5f + (segsX / 2) * L;   // centered on the north arch segment
            bossHalfX = 5f * L;
            bossHalfZ = 3.5f * L;
            bossCz = hz + bossHalfZ;

            BuildGround(hx, hz);

            // --- floor: tile the courtyard, sink so tile tops sit at y=0
            var floorRoot = new GameObject("Floor").transform; floorRoot.SetParent(envRoot);
            int nx = Mathf.CeilToInt(hx * 2f / U), nz = Mathf.CeilToInt(hz * 2f / U);
            for (int ix = 0; ix < nx; ix++)
                for (int iz = 0; iz < nz; iz++)
                {
                    string tile = rng.NextDouble() switch
                    {
                        < 0.62 => "Floor_Standard",
                        < 0.80 => "Floor_Squares",
                        < 0.93 => "Floor_Diamond",
                        _ => "Floor_SquareLarge",
                    };
                    float x = -hx + U * 0.5f + ix * U;
                    float z = -hz + U * 0.5f + iz * U;
                    PlacePiece(tile, new Vector3(x, 0, z), Pick(0f, 90f, 180f, 270f), floorRoot, groundTopAtZero: true, collider: false);
                }

            // --- perimeter walls, two stacked rows + a ruined third row (south side gets the gate)
            var wallRoot = new GameObject("Walls").transform; wallRoot.SetParent(envRoot);
            string[] ground = { "Wall", "Wall", "Wall", "Wall_Overgrown", "Wall_ArchGothic", "Wall_ArchRound_Overgrown", "Wall_Hole" };
            string[] solidGround = { "Wall", "Wall", "Wall_Overgrown" };   // no walk-through arches/holes
            string[] upper = { "Wall", "Wall", "Wall", "Wall_Overgrown", "Wall_Hole" };
            string[] ruined = { "Wall_Broken", "Wall_Half", "Wall_Double_Broken" };

            void WallStack(Vector3 basePos, float rot, bool isGate, bool solid = false)
            {
                if (isGate)
                {
                    var gate = PlacePiece("Doors_GothicArch", basePos, rot, wallRoot);
                    float gateH = RenderererSafeBounds(gate).size.y;
                    if (gateH < H * 1.5f)   // short gate piece -> crown it with an arched window
                        PlacePiece("Wall_ArchGothic", basePos + Vector3.up * gateH, rot, wallRoot);
                    return;
                }
                string[] g = solid ? solidGround : ground;
                PlacePiece(g[rng.Next(g.Length)], basePos, rot, wallRoot);
                PlacePiece(upper[rng.Next(upper.Length)], basePos + Vector3.up * H, rot, wallRoot);
                if (rng.NextDouble() < 0.45)   // ruined battlement silhouette
                    PlacePiece(ruined[rng.Next(ruined.Length)], basePos + Vector3.up * 2f * H, rot, wallRoot);
            }

            for (int i = 0; i < segsX; i++)
            {
                float x = -hx + L * 0.5f + i * L;
                WallStack(new Vector3(x, 0, -hz), 180f, isGate: i == segsX / 2);
                // skip the north center segment — the boss chamber entrance goes there;
                // segments forming the chamber's front wall must be solid (mist = only way in)
                if (i != segsX / 2)
                    WallStack(new Vector3(x, 0, +hz), 0f, isGate: false,
                              solid: Mathf.Abs(x - bossCx) < bossHalfX + L * 0.5f);
            }
            // OPEN gothic arch into the boss chamber (the Doors_* pieces have closed leaves,
            // which would hide the mist wall and block the passage)
            var bossArch = PlacePiece("Arch_Gothic", new Vector3(bossCx, 0, hz), 0f, wallRoot);
            float bossArchH = RenderererSafeBounds(bossArch).size.y;
            if (bossArchH < H * 1.5f)
                PlacePiece("Wall_ArchGothic", new Vector3(bossCx, bossArchH, hz), 0f, wallRoot);
            for (int i = 0; i < segsZ; i++)
            {
                float z = -hz + L * 0.5f + i * L;
                WallStack(new Vector3(-hx, 0, z), 90f, isGate: false);
                WallStack(new Vector3(+hx, 0, z), 270f, isGate: false);
            }
            // corner towers
            foreach (var (cx, cz) in new[] { (-hx, -hz), (hx, -hz), (-hx, hz), (hx, hz) })
            {
                var col = PlacePiece("Column_Square", new Vector3(cx, 0, cz), 0f, wallRoot, scale: 1.4f);
                float colH = RenderererSafeBounds(col).size.y;
                PlacePiece("Column_Square", new Vector3(cx, colH, cz), 0f, wallRoot, scale: 1.4f);
            }

            // --- ruined keep tower outside the NW corner — a tall silhouette for the skyline
            var towerRoot = new GameObject("KeepTower").transform; towerRoot.SetParent(envRoot);
            float tCx = -hx - L * 0.8f, tCz = hz + L * 0.8f;
            int stories = 6;
            string[] towerWall = { "Wall", "Wall", "Wall_Overgrown", "Window_Open", "Window_Bars", "Wall_Hole" };
            for (int s = 0; s < stories; s++)
            {
                float y = s * H;
                bool top = s == stories - 1;
                string PickW() => top ? ruined[rng.Next(ruined.Length)]
                                : s == 0 ? "Wall"
                                : towerWall[rng.Next(towerWall.Length)];
                for (int k = -1; k <= 1; k += 2)   // two segments per side
                {
                    float off = k * L * 0.5f;
                    PlacePiece(PickW(), new Vector3(tCx + off, y, tCz - L), 180f, towerRoot);
                    PlacePiece(PickW(), new Vector3(tCx + off, y, tCz + L), 0f, towerRoot);
                    PlacePiece(PickW(), new Vector3(tCx - L, y, tCz + off), 90f, towerRoot);
                    PlacePiece(PickW(), new Vector3(tCx + L, y, tCz + off), 270f, towerRoot);
                }
            }
            // corner pillars the full height of the keep
            float towerColH = RenderererSafeBounds(
                PlacePiece("Column_Square", new Vector3(tCx - L, 0, tCz - L), 0f, towerRoot, scale: 1.2f)).size.y;
            foreach (var (ox, oz) in new[] { (L, -L), (-L, L), (L, L) })
                PlacePiece("Column_Square", new Vector3(tCx + ox, 0, tCz + oz), 0f, towerRoot, scale: 1.2f);
            for (float y = towerColH; y < stories * H; y += towerColH)
                foreach (var (ox, oz) in new[] { (-L, -L), (L, -L), (-L, L), (L, L) })
                    PlacePiece("Column_Square", new Vector3(tCx + ox, y, tCz + oz), 0f, towerRoot, scale: 1.2f);
            PlaceTorch(new Vector3(tCx + L + 0.7f, 0, tCz - L), towerRoot);

            // --- boss chamber behind the north arch, sealed by a mist door
            var bossRoot = new GameObject("BossRoom").transform; bossRoot.SetParent(envRoot);
            float bD = bossHalfZ * 2f;

            int bnx = Mathf.CeilToInt(bossHalfX * 2f / U), bnz = Mathf.CeilToInt(bD / U);
            for (int ix = 0; ix < bnx; ix++)
                for (int iz = 0; iz < bnz; iz++)
                {
                    string tile = rng.NextDouble() switch
                    {
                        < 0.62 => "Floor_Standard",
                        < 0.80 => "Floor_Squares",
                        < 0.93 => "Floor_Diamond",
                        _ => "Floor_SquareLarge",
                    };
                    float x = bossCx - bossHalfX + U * 0.5f + ix * U;
                    float z = hz + U * 0.5f + iz * U;
                    PlacePiece(tile, new Vector3(x, 0, z), Pick(0f, 90f, 180f, 270f), bossRoot, groundTopAtZero: true, collider: false);
                }

            // chamber perimeter is fully solid — the mist door is the only way in
            int bSegsX = Mathf.RoundToInt(bossHalfX * 2f / L);
            for (int i = 0; i < bSegsX; i++)
            {
                float x = bossCx - bossHalfX + L * 0.5f + i * L;
                WallStack(new Vector3(x, 0, hz + bD), 0f, isGate: false, solid: true);
            }
            int bSegsZ = Mathf.RoundToInt(bD / L);
            for (int i = 0; i < bSegsZ; i++)
            {
                float z = hz + L * 0.5f + i * L;
                WallStack(new Vector3(bossCx - bossHalfX, 0, z), 90f, isGate: false, solid: true);
                WallStack(new Vector3(bossCx + bossHalfX, 0, z), 270f, isGate: false, solid: true);
            }
            foreach (var bx in new[] { bossCx - bossHalfX, bossCx + bossHalfX })
            {
                var col = PlacePiece("Column_Square", new Vector3(bx, 0, hz + bD), 0f, bossRoot, scale: 1.4f);
                PlacePiece("Column_Square", new Vector3(bx, RenderererSafeBounds(col).size.y, hz + bD), 0f, bossRoot, scale: 1.4f);
            }

            // arena dressing: a looming idol at the far end, colonnade flanks, bones
            PlacePiece("Statue_Stag", new Vector3(bossCx, 0, hz + bD - L * 1.2f), 180f, bossRoot, scale: 2.4f);
            for (int i = 0; i < 3; i++)
            {
                float z = hz + L * 1.6f + i * L * 1.8f;
                PlacePiece(Pick("Column_Round", "Column_Round_Short"), new Vector3(bossCx - bossHalfX + L * 0.8f, 0, z), 0f, bossRoot);
                PlacePiece(Pick("Column_Round_Short", "Column_Round"), new Vector3(bossCx + bossHalfX - L * 0.8f, 0, z), 0f, bossRoot);
            }
            PlacePiece("Skull", new Vector3(bossCx - 1.4f, 0, hz + bD * 0.45f), 70f, bossRoot, collider: false);
            PlacePiece("Skull", new Vector3(bossCx + 2.1f, 0, hz + bD * 0.6f), 210f, bossRoot, collider: false);
            PlacePiece("Bricks", new Vector3(bossCx + bossHalfX - L * 1.1f, 0, hz + L * 0.9f), 40f, bossRoot, collider: false);

            // torch ring so the arena reads at night
            PlaceTorch(new Vector3(bossCx - 1.9f, 0, hz + 0.8f), bossRoot);
            PlaceTorch(new Vector3(bossCx + 1.9f, 0, hz + 0.8f), bossRoot);
            PlaceTorch(new Vector3(bossCx - bossHalfX + 0.7f, 0, hz + bD * 0.5f), bossRoot);
            PlaceTorch(new Vector3(bossCx + bossHalfX - 0.7f, 0, hz + bD * 0.5f), bossRoot);
            PlaceTorch(new Vector3(bossCx - bossHalfX + 0.7f, 0, hz + bD - 0.8f), bossRoot);
            PlaceTorch(new Vector3(bossCx + bossHalfX - 0.7f, 0, hz + bD - 0.8f), bossRoot);

            // the fog wall sealing the arch
            mistDoorGO = BuildMistDoor(new Vector3(bossCx, 0f, hz));
            mistDoorGO.transform.SetParent(envRoot, true);

            // --- torches along the walls (with flickering lights)
            var torchRoot = new GameObject("Torches").transform; torchRoot.SetParent(envRoot);
            var torchPositions = new List<Vector3>();
            for (int i = 0; i < segsX; i += 3)
            {
                float x = -hx + L * 0.5f + i * L;
                torchPositions.Add(new Vector3(x, 0, -hz + 0.7f));
                torchPositions.Add(new Vector3(x, 0, +hz - 0.7f));
            }
            for (int i = 1; i < segsZ; i += 3)
            {
                float z = -hz + L * 0.5f + i * L;
                torchPositions.Add(new Vector3(-hx + 0.7f, 0, z));
                torchPositions.Add(new Vector3(+hx - 0.7f, 0, z));
            }
            foreach (var p in torchPositions)
                PlaceTorch(p, torchRoot);
            // two extra torches framing the NPC's corner
            PlaceTorch(new Vector3(5.6f, 0, 8.4f), torchRoot);
            PlaceTorch(new Vector3(8.6f, 0, 5.8f), torchRoot);

            // --- banners near the gate
            PlacePiece("Flag_Wall", new Vector3(-L, 0, -hz + 0.35f), 180f, wallRoot);
            PlacePiece("Flag_Wall2", new Vector3(+L, 0, -hz + 0.35f), 180f, wallRoot);

            // --- statues flanking the gate walkway
            PlacePiece("Statue_Stag", new Vector3(-2.2f, 0, -hz + 2.6f), 135f, envRoot);
            PlacePiece("Statue_Fox", new Vector3(+2.2f, 0, -hz + 2.6f), 225f, envRoot);

            // --- colonnade stumps along the central walkway
            for (int i = -1; i <= 1; i++)
            {
                PlacePiece(Pick("Column_Round", "Column_Round_Short"), new Vector3(-3.5f, 0, i * 4.5f), 0f, envRoot);
                PlacePiece(Pick("Column_Round_Short", "Column_Round"), new Vector3(+3.5f, 0, i * 4.5f), 0f, envRoot);
            }

            // --- dead trees + scattered ruin clutter
            var clutterRoot = new GameObject("Clutter").transform; clutterRoot.SetParent(envRoot);
            PlacePiece("DeadTree_1", new Vector3(-hx + L, 0, hz - L), Range(0, 360), clutterRoot);
            PlacePiece("DeadTree_2", new Vector3(hx - L * 0.8f, 0, hz - L * 1.4f), Range(0, 360), clutterRoot);
            PlacePiece("DeadTree_3", new Vector3(-hx + L * 1.2f, 0, -hz + L * 1.6f), Range(0, 360), clutterRoot);

            (string, Vector3, float)[] clutter =
            {
                ("Cart",            new Vector3(hx - L * 1.3f, -0.17f, -hz + L * 1.2f), 250f),   // user 2026-07-15: sunk into the ground
                ("Barrel",          new Vector3(hx - L * 1.05f, 0, -hz + L * 0.8f), 10f),
                ("Crate",           new Vector3(hx - L * 1.5f, 0, -hz + L * 0.75f), 35f),
                ("Chest",           new Vector3(-hx + L * 0.6f, 0, hz - L * 0.6f), 140f),
                ("Pot1",            new Vector3(-hx + L * 0.5f, 0, -hz + L * 0.9f), 0f),
                ("Pot2",            new Vector3(-hx + L * 0.62f, 0, -hz + L * 1.05f), 70f),
                ("Pot3_Broken",     new Vector3(-hx + L * 0.78f, 0, -hz + L * 0.85f), 25f),
                ("Bricks",          new Vector3(L * 1.8f, 0, hz - L * 0.8f), 80f),
                ("Brick",           new Vector3(L * 2.0f, 0, hz - L * 0.95f), 30f),
                ("Skull",           new Vector3(L * 0.4f, 0, hz - L * 0.7f), 200f),
                ("BearTrap_Open",   new Vector3(-L * 2.2f, 0, -L * 0.5f), 0f),
                ("Bush_Round",      new Vector3(-hx + L * 2.2f, 0, hz - L * 0.5f), 0f),
                ("Bush_1x1",        new Vector3(hx - L * 2.0f, 0, hz - L * 0.6f), 90f),
                ("Grass",           new Vector3(-L * 0.8f, 0, L * 1.2f), 0f),
                ("Grass",           new Vector3(L * 1.3f, 0, -L * 0.7f), 120f),
                ("Grass",           new Vector3(L * 2.6f, 0, L * 2.0f), 240f),
            };
            foreach (var (piece, pos, rot) in clutter)
                PlacePiece(piece, pos, rot, clutterRoot, collider: piece != "Grass");

            // --- outside the gate: hint of a world beyond, swallowed by fog
            PlacePiece("BridgeSection", new Vector3(0, 0, -hz - L * 1.0f), 0f, envRoot);
            PlacePiece("DeadTree_1", new Vector3(-L * 1.6f, 0, -hz - L * 1.5f), 80f, envRoot);
            PlacePiece("Wall_Broken", new Vector3(L * 2.3f, 0, -hz - L * 1.1f), 160f, envRoot);

            BuildForest(hx, hz);
            return L;
        }

        // a ring of forest swallowing the ruin — alive and dead trees with brush, thinning
        // only along the path out of the gate
        static void BuildForest(float hx, float hz)
        {
            var forest = new GameObject("Forest").transform;
            forest.SetParent(envRoot);
            string[] alive = { "Tree_1", "Tree_2", "Tree_3" };
            string[] dead = { "DeadTree_1", "DeadTree_2", "DeadTree_3" };
            string[] brush = { "Bush_1x1", "Bush_Round", "Bush_Large", "Bush_2x1", "Grass" };

            int placed = 0, guard = 0;
            while (placed < 260 && guard++ < 6000)
            {
                float x = Range(-78f, 78f), z = Range(-78f, 78f);
                if (Mathf.Abs(x) < hx + 4f && Mathf.Abs(z) < hz + 4f) continue;   // keep the courtyard clear
                if (Mathf.Abs(x) < 6f && z < -hz) continue;                       // gate path stays open
                if (RectDist(x, z, bossCx, bossCz, bossHalfX + 4f, bossHalfZ + 4f) < 0.01f) continue;   // boss chamber

                double roll = rng.NextDouble();
                string piece = roll < 0.48 ? alive[rng.Next(alive.Length)]
                             : roll < 0.78 ? dead[rng.Next(dead.Length)]
                             : brush[rng.Next(brush.Length)];
                bool isTree = roll < 0.78;

                // trees sit on the rolling terrain and grow towards the horizon for a looming treeline
                float far = Mathf.Clamp01((RectDist(x, z, 0f, 0f, hx + 4f, hz + 4f) - 8f) / 40f);
                float scale = Range(0.9f, 1.3f) * (isTree ? Mathf.Lerp(1.0f, 2.4f, far) : 1f);
                PlacePiece(piece, new Vector3(x, GroundHeight(x, z), z), Range(0f, 360f), forest,
                           scale: scale, collider: isTree);
                placed++;
            }
        }

        static GameObject PlacePiece(string ruinName, Vector3 pos, float yRot, Transform parent,
                                     float scale = 1f, bool collider = true, bool groundTopAtZero = false)
        {
            var go = (GameObject)PrefabUtility.InstantiatePrefab(Ruin(ruinName));
            go.transform.SetParent(parent, false);
            // COMPOSE with the prefab root transform — these FBX bake unit-conversion scale and
            // a -90° axis-correction rotation into the root; overwriting either breaks the piece
            go.transform.localScale *= scale;
            go.transform.rotation = Quaternion.Euler(0f, yRot, 0f) * go.transform.localRotation;
            go.transform.position = pos;

            Bounds b = RenderererSafeBounds(go);
            if (groundTopAtZero)
                go.transform.position += Vector3.up * (pos.y - b.max.y);    // top flush with pos.y
            else
                go.transform.position += Vector3.up * (pos.y - b.min.y);    // base sits at pos.y

            if (collider)
                foreach (var mf in go.GetComponentsInChildren<MeshFilter>())
                    mf.gameObject.AddComponent<MeshCollider>();

            // the foliage FBX materials reference a texture that isn't shipped — they import
            // plain white. Swap anything leaf-like for a dark moody foliage material.
            foreach (var r in go.GetComponentsInChildren<Renderer>())
            {
                var mats = r.sharedMaterials;
                bool changed = false;
                for (int i = 0; i < mats.Length; i++)
                    if (mats[i] != null && (mats[i].name.Contains("Leaves") || mats[i].name == "Green"))
                    {
                        mats[i] = FoliageMat();
                        changed = true;
                    }
                if (changed) r.sharedMaterials = mats;
            }

            SetStaticRecursive(go);
            return go;
        }

        // A warm flickering point light hovering over a candle piece (CandleFlicker modulates
        // intensity + a tiny wobble at runtime), plus emissive wax so the flames read as lit.
        static void AddCandleGlow(GameObject candle, float height, float intensity, float range)
        {
            if (candle == null) return;
            Bounds b = RenderererSafeBounds(candle);
            var glow = new GameObject("CandleGlow");
            glow.transform.SetParent(candle.transform, false);
            glow.transform.position = new Vector3(b.center.x, b.max.y + height, b.center.z);
            var l = glow.AddComponent<Light>();
            l.type = LightType.Point;
            l.color = new Color(1f, 0.68f, 0.32f);
            l.intensity = intensity;
            l.range = range;
            l.shadows = LightShadows.None;
            glow.AddComponent<DeepUnity.Tutorials.ChatDemo3D.CandleFlicker>();

            // candle meshes glow softly themselves — scene-owned material COPIES get the
            // emission (never mutate the shared FBX-imported materials: that would bleed into
            // every other Candles_1 in the project and doesn't persist reliably anyway)
            foreach (var r in candle.GetComponentsInChildren<Renderer>())
            {
                var mats = r.sharedMaterials;
                for (int i = 0; i < mats.Length; i++)
                {
                    if (mats[i] == null) continue;
                    var m = new Material(mats[i]);
                    m.EnableKeyword("_EMISSION");
                    m.SetColor("_EmissionColor", new Color(0.9f, 0.55f, 0.2f) * 0.6f);
                    mats[i] = m;
                }
                r.sharedMaterials = mats;
            }
        }

        static Material foliageMat;
        static Material FoliageMat()
        {
            if (foliageMat != null) return foliageMat;
            string path = GEN + "/Foliage.mat";
            foliageMat = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (foliageMat == null)
            {
                foliageMat = new Material(Shader.Find("Standard"));
                foliageMat.color = new Color(0.20f, 0.27f, 0.17f);   // dark dusk foliage
                foliageMat.SetFloat("_Glossiness", 0.03f);
                AssetDatabase.CreateAsset(foliageMat, path);
            }
            return foliageMat;
        }

        // the fog wall: two scrolling alpha-blended mist quads in the archway, a thin solid
        // collider that blocks passage, and a wide trigger for the "[ E ]" prompt
        static GameObject BuildMistDoor(Vector3 pos)
        {
            string matPath = GEN + "/Mist.mat";
            var mat = AssetDatabase.LoadAssetAtPath<Material>(matPath);
            if (mat == null)
            {
                mat = new Material(Shader.Find("Legacy Shaders/Particles/Alpha Blended"));
                AssetDatabase.CreateAsset(mat, matPath);
            }
            // tint stays neutral (the legacy shader doubles it) — the gold lives in the texture
            mat.mainTexture = CreateMistTexture();
            mat.SetColor("_TintColor", new Color(0.5f, 0.5f, 0.5f, 1f));

            var root = new GameObject("MistDoor");
            root.transform.position = pos;
            root.transform.localScale = new Vector3(1f, 1.66f, 1f);   // user 2026-07-15: taller mist window

            // the fog gate glows golden onto the surrounding stone
            var glow = new GameObject("MistGlow").AddComponent<Light>();
            glow.transform.SetParent(root.transform, false);
            glow.transform.localPosition = new Vector3(0f, 1.7f, 0f);
            glow.type = LightType.Point;
            glow.color = new Color(1f, 0.82f, 0.45f);
            glow.intensity = 1.7f;
            glow.range = 5.5f;
            glow.shadows = LightShadows.None;

            // user 2026-07-15: hand-tuned in the editor (local values, under the 1.66-tall root)
            Vector3[] layerPos = { new Vector3(0f, 0.688f, -0.08f), new Vector3(0f, 0.5698f, 0.08f) };
            var layers = new List<Renderer>();
            for (int i = 0; i < 2; i++)
            {
                var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
                UnityEngine.Object.DestroyImmediate(quad.GetComponent<Collider>());
                quad.name = "MistLayer" + i;
                quad.transform.SetParent(root.transform, false);
                quad.transform.localPosition = layerPos[i];
                quad.transform.localScale = new Vector3(2.55f, 2.58f, 1f);
                var r = quad.GetComponent<MeshRenderer>();
                r.sharedMaterial = mat;
                r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                layers.Add(r);
            }

            var blocker = root.AddComponent<BoxCollider>();
            blocker.center = new Vector3(0f, 1.6f, 0f);
            blocker.size = new Vector3(2.7f, 3.2f, 0.25f);
            var trigger = root.AddComponent<BoxCollider>();
            trigger.isTrigger = true;
            trigger.center = new Vector3(0f, 1.5f, 0f);
            trigger.size = new Vector3(5.5f, 3f, 7f);

            var md = root.AddComponent<MistDoor>();
            SetRef(md, "blocker", blocker);
            var so = new SerializedObject(md);
            var arr = so.FindProperty("mistLayers");
            arr.arraySize = layers.Count;
            for (int i = 0; i < layers.Count; i++)
                arr.GetArrayElementAtIndex(i).objectReferenceValue = layers[i];
            so.ApplyModifiedPropertiesWithoutUndo();
            return root;
        }

        // golden fog sheet: OPAQUE body (you can't see into the boss room), swirl pattern lives
        // in the color; tiles horizontally for the scroll, only the very top fades to wisps.
        // Always regenerated so look tweaks land on rebuild.
        static Texture2D CreateMistTexture()
        {
            string pngPath = GEN + "/MistTex.png";
            const int S = 256;
            float TileableNoise(float u, float v, float freq, float seed)
            {
                float x = u * freq, y = v * freq, w = freq;
                float fx = x / w, fy = y / w;
                float n00 = Mathf.PerlinNoise(seed + x, seed + y);
                float n10 = Mathf.PerlinNoise(seed + x - w, seed + y);
                float n01 = Mathf.PerlinNoise(seed + x, seed + y - w);
                float n11 = Mathf.PerlinNoise(seed + x - w, seed + y - w);
                return Mathf.Lerp(Mathf.Lerp(n00, n10, fx), Mathf.Lerp(n01, n11, fx), fy);
            }

            Color deepGold = new Color(0.40f, 0.30f, 0.12f);
            Color paleGold = new Color(0.98f, 0.88f, 0.56f);

            var tex = new Texture2D(S, S, TextureFormat.RGBA32, false);
            var px = new Color[S * S];
            for (int y = 0; y < S; y++)
                for (int x = 0; x < S; x++)
                {
                    float u = (float)x / S, v = (float)y / S;
                    float n = TileableNoise(u, v, 4f, 5.3f) * 0.55f
                            + TileableNoise(u, v, 9f, 91.7f) * 0.30f
                            + TileableNoise(u, v, 21f, 33.3f) * 0.15f;
                    // fully opaque body — nothing of the boss room may show through; the only
                    // fade is the last 7% at the crown, hidden behind the arch stonework.
                    // (Mathf.SmoothStep(a,b,t) interpolates a->b, it is NOT glsl smoothstep —
                    // remap the band with InverseLerp first)
                    float topFade = 1f - Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(0.93f, 1f, v));
                    Color c = Color.Lerp(deepGold, paleGold, n);
                    px[y * S + x] = new Color(c.r, c.g, c.b, topFade);
                }
            tex.SetPixels(px);
            tex.Apply();
            File.WriteAllBytes(pngPath, tex.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(tex);
            AssetDatabase.ImportAsset(pngPath);
            var ti = (TextureImporter)AssetImporter.GetAtPath(pngPath);
            ti.alphaIsTransparency = true;
            ti.wrapMode = TextureWrapMode.Repeat;
            ti.SaveAndReimport();
            return AssetDatabase.LoadAssetAtPath<Texture2D>(pngPath);
        }

        static Bounds RenderererSafeBounds(GameObject go)
        {
            var rs = go.GetComponentsInChildren<Renderer>();
            if (rs.Length == 0) return new Bounds(go.transform.position, Vector3.zero);
            Bounds b = rs[0].bounds;
            foreach (var r in rs.Skip(1)) b.Encapsulate(r.bounds);
            return b;
        }

        static void SetStaticRecursive(GameObject go)
        {
            go.isStatic = true;
            foreach (Transform t in go.GetComponentsInChildren<Transform>())
                t.gameObject.isStatic = true;
        }

        static void PlaceTorch(Vector3 pos, Transform parent)
        {
            var torch = PlacePiece("Torch", pos, Range(0, 360), parent, collider: false);
            Bounds b = RenderererSafeBounds(torch);

            var lightGO = new GameObject("TorchLight");
            lightGO.transform.SetParent(torch.transform, false);
            lightGO.transform.position = new Vector3(b.center.x, b.max.y + 0.15f, b.center.z);
            var l = lightGO.AddComponent<Light>();
            l.type = LightType.Point;
            l.color = new Color(1.0f, 0.58f, 0.22f);
            l.intensity = 2.4f;
            l.range = 11f;
            l.shadows = LightShadows.None;
            lightGO.AddComponent<TorchFlicker>();
        }

        // ---------------------------------------------------------------- animators

        static RuntimeAnimatorController CreatePlayerAnimator()
        {
            var ctrl = AnimatorController.CreateAnimatorControllerAtPath(GEN + "/PlayerAnimator.controller");
            var sm = ctrl.layers[0].stateMachine;

            var map = new (string state, string fbx, string clip)[]
            {
                ("Idle",      "Animations/UAL1.fbx", "Sword_Idle"),
                // The gear beat (user 2026-07-25) starts the warrior empty-handed, so Sword_Idle —
                // a guard stance closed around a blade — can no longer be the only idle: it read as
                // the player miming a sword he does not own. Idle_Loop is the same rig's arms-down
                // stance (the NPCs already use it). SoulsPlayerController picks between the two.
                ("IdleUnarmed", "Animations/UAL1.fbx", "Idle_Loop"),
                ("Walk",      "Animations/UAL1.fbx", "Walk_Loop"),
                ("Run",       "Animations/UAL1.fbx", "Jog_Fwd_Loop"),
                ("Sprint",    "Animations/UAL1.fbx", "Sprint_Loop"),
                ("Roll",      "Animations/UAL1.fbx", "Roll"),
                ("Attack1",   "Animations/UAL2.fbx", "Sword_Regular_A"),
                ("Attack2",   "Animations/UAL2.fbx", "Sword_Regular_B"),
                ("Attack3",   "Animations/UAL2.fbx", "Sword_Regular_C"),
                ("BlockIdle", "Animations/UAL2.fbx", "Idle_Shield_Loop"),
                ("Talking",   "Animations/UAL1.fbx", "Idle_Talking_Loop"),
                ("Hit",       "Animations/UAL1.fbx", "Hit_Chest"),
                ("Death",     "Animations/UAL1.fbx", "Death01"),
                ("RunB",      "Animations/UAL1.fbx", "Jog_Fwd_Loop"),   // played in reverse = backpedal
                ("Interact",  "Animations/UAL1.fbx", "Interact"),      // reach into the mist door
                ("MistWalk",  "Animations/UAL1.fbx", "Push_Loop"),     // hands-first push through the fog
            };
            foreach (var (state, fbx, clipName) in map)
            {
                var st = sm.AddState(state);
                var clip = Clip(fbx, clipName);
                st.motion = clip;

                // sync the clip's authored ground speed to the controller's move speed so the
                // feet stop sliding; the UAL clips are in-place (averageSpeed reads 0), so each
                // gets a hand-tuned cadence fallback instead
                float natural = clip.averageSpeed.magnitude;
                float Sync(float desired, float fallback) => natural < 0.5f ? fallback : Mathf.Clamp(desired / natural, 0.75f, 1.5f);
                st.speed = state switch
                {
                    "Run" => Sync(4.3f, 1.30f),     // SoulsPlayerController.runSpeed
                    "RunB" => -Sync(4.3f, 1.30f),
                    "Sprint" => Sync(7.0f, 1.18f),  // sprintSpeed
                    "Walk" => Sync(1.7f, 1.05f),    // blockMoveSpeed
                    "MistWalk" => natural < 0.5f ? 0.6f : Mathf.Clamp(1.25f / natural, 0.4f, 1f),
                    _ => 1f,
                };
                Debug.Log($"[ChatDemo3DBuilder] state {state}: clip {clipName} len {clip.length:0.00}s " +
                          $"natSpeed {natural:0.00} m/s -> playback x{st.speed:0.00}");
                // default = the EMPTY-HANDED stance, because that is how the scene now starts. It is
                // what plays on frame 0 before the first CrossFade, and it is also the pose
                // ScreenshotBatch samples, so both match the gear the player actually has.
                if (state == "IdleUnarmed") sm.defaultState = st;
            }
            return ctrl;
        }

        // 2-state Idle/Talking controller from a GENERIC character's own embedded clips
        static RuntimeAnimatorController CreateOwnClipAnimator(string assetName, string fbxRel,
                                                               string idleClip, string talkClip)
        {
            var ctrl = AnimatorController.CreateAnimatorControllerAtPath(GEN + "/" + assetName + ".controller");
            var sm = ctrl.layers[0].stateMachine;
            var idle = sm.AddState("Idle"); idle.motion = Clip(fbxRel, idleClip);
            var talk = sm.AddState("Talking"); talk.motion = Clip(fbxRel, talkClip);
            sm.defaultState = idle;
            return ctrl;
        }

        static RuntimeAnimatorController CreateNpcAnimator()
        {
            var ctrl = AnimatorController.CreateAnimatorControllerAtPath(GEN + "/NpcAnimator.controller");
            var sm = ctrl.layers[0].stateMachine;
            var idle = sm.AddState("Idle"); idle.motion = Clip("Animations/UAL1.fbx", "Idle_Loop");
            var talk = sm.AddState("Talking"); talk.motion = Clip("Animations/UAL1.fbx", "Idle_Talking_Loop");
            sm.defaultState = idle;
            return ctrl;
        }

        // heavy, telegraphed move set for the Sentinel; swings play slowed so they can be rolled
        static RuntimeAnimatorController CreateBossAnimator(out float[] swingDurations)
        {
            const float SWING_SPEED = 0.72f;
            var ctrl = AnimatorController.CreateAnimatorControllerAtPath(GEN + "/BossAnimator.controller");
            var sm = ctrl.layers[0].stateMachine;
            var map = new (string state, string fbx, string clip, float speed)[]
            {
                ("Idle",    "Animations/UAL1.fbx", "Sword_Idle",      1f),
                ("Run",     "Animations/UAL1.fbx", "Walk_Loop",       1.15f),   // a heavy stalk, not a jog
                ("Attack1", "Animations/UAL1.fbx", "Sword_Attack",    SWING_SPEED),
                ("Attack2", "Animations/UAL2.fbx", "Sword_Regular_B", SWING_SPEED),
                ("Attack3", "Animations/UAL2.fbx", "Sword_Regular_C", SWING_SPEED),
                ("Lunge",   "Animations/UAL2.fbx", "Sword_Dash_RM",   0.85f),
                ("Hit",     "Animations/UAL1.fbx", "Hit_Chest",       1f),
                ("Death",   "Animations/UAL1.fbx", "Death01",         0.9f),
            };
            foreach (var (state, fbx, clipName, speed) in map)
            {
                var st = sm.AddState(state);
                st.motion = Clip(fbx, clipName);
                st.speed = speed;
                if (state == "Idle") sm.defaultState = st;
            }
            swingDurations = new[]
            {
                Clip("Animations/UAL1.fbx", "Sword_Attack").length / SWING_SPEED,
                Clip("Animations/UAL2.fbx", "Sword_Regular_B").length / SWING_SPEED,
                Clip("Animations/UAL2.fbx", "Sword_Regular_C").length / SWING_SPEED,
            };
            return ctrl;
        }

        // ---------------------------------------------------------------- characters

        static GameObject BuildPlayer(RuntimeAnimatorController ctrl)
        {
            var root = new GameObject("Player");
            root.tag = "Player";
            root.layer = 2;   // Ignore Raycast: keeps the orbit camera's collision cast off the player
            root.transform.position = new Vector3(0f, 0.1f, -9f);
            root.transform.rotation = Quaternion.Euler(0, 0, 0);

            var cc = root.AddComponent<CharacterController>();
            cc.center = new Vector3(0, 0.95f, 0);
            cc.height = 1.8f;
            cc.radius = 0.35f;
            cc.slopeLimit = 50f;

            var model = (GameObject)PrefabUtility.InstantiatePrefab(LoadModel("Characters/Warrior.fbx"));
            model.name = "WarriorModel";
            model.transform.SetParent(root.transform, false);
            SetLayerRecursive(model, 2);

            // normalize to ~1.8 m tall and put the feet on the ground
            Bounds b = RenderererSafeBounds(model);
            float scale = b.size.y > 0.01f ? 1.8f / b.size.y : 1f;
            model.transform.localScale *= scale;
            GroundModel(model, root.transform.position.y);
            Debug.Log($"[ChatDemo3DBuilder] warrior raw height {b.size.y:0.00} -> scale {scale:0.000}");

            ApplyCharacterTexture(model, "Warrior_Texture.png", "PlayerWarrior", Color.white);

            var anim = model.GetComponent<Animator>();
            if (anim == null) anim = model.AddComponent<Animator>();
            anim.runtimeAnimatorController = ctrl;
            anim.applyRootMotion = false;
            anim.cullingMode = AnimatorCullingMode.AlwaysAnimate;

            // sword on the rig's dedicated weapon mount, heater shield in the left fist
            Transform weaponMount = FindDeep(model.transform, "Weapon.R");
            GameObject sword = weaponMount != null
                ? AttachToTransform(weaponMount, LoadModel("Weapons/Sword.fbx"), "Sword")
                : AttachToBone(anim, HumanBodyBones.RightHand, LoadModel("Weapons/Sword.fbx"), "Sword");
            GameObject shield = AttachToBone(anim, HumanBodyBones.LeftHand, LoadModel("Weapons/Shield_Heater.fbx"), "Shield");
            NormalizeWorldSize(sword, 1.15f);    // blade ~1.15 m end to end
            NormalizeWorldSize(shield, 0.80f);   // heater shield ~0.8 m tall
            if (shield != null)                  // along the forearm, face out (tuned via ShieldTuneBatch lineup)
                shield.transform.localRotation = Quaternion.Euler(270f, 0f, 0f) * shield.transform.localRotation;
            if (sword != null)                   // held pose hand-tuned in-editor 2026-07-17 (blade forward);
                                                 // exact serialized quat — euler round-trips drift, keep as-is
                sword.transform.localRotation = new Quaternion(-0.52869385f, 0.52449185f, -0.47811946f, 0.46561024f);

            root.AddComponent<BreathingIdle>();

            // conversation sheathing: sword tweens to the right hip, shield to the back while chatting
            var stower = root.AddComponent<WeaponStower>();
            if (sword != null) SetRef(stower, "sword", sword.transform);
            if (shield != null) SetRef(stower, "shield", shield.transform);

            // The gear beat (user 2026-07-25): the warrior starts EMPTY-HANDED. These two objects
            // stay built, sized, posed and stower-wired exactly as above — they are only
            // DEACTIVATED, so nothing about the held poses has to be re-tuned when they turn up. He
            // gets them if Velmire offers his own pair and the player accepts the choice popup
            // (PlayerGear + NPCGearOffer); BuildHud hides the two quick-slot icons to match.
            var gear = root.AddComponent<PlayerGear>();
            if (sword != null) { SetRef(gear, "sword", sword.transform); sword.SetActive(false); }
            if (shield != null) { SetRef(gear, "shield", shield.transform); shield.SetActive(false); }

            // footsteps
            var stepSource = root.AddComponent<AudioSource>();
            stepSource.playOnAwake = false;
            stepSource.spatialBlend = 0f;
            var steps = root.AddComponent<FootstepSounds>();
            string[] stepGuids = AssetDatabase.FindAssets("t:AudioClip", new[] { ART + "/Audio/Footsteps" });
            var so2 = new SerializedObject(steps);
            var arr2 = so2.FindProperty("clips");
            arr2.arraySize = stepGuids.Length;
            for (int i = 0; i < stepGuids.Length; i++)
                arr2.GetArrayElementAtIndex(i).objectReferenceValue =
                    AssetDatabase.LoadAssetAtPath<AudioClip>(AssetDatabase.GUIDToAssetPath(stepGuids[i]));
            so2.ApplyModifiedPropertiesWithoutUndo();

            var pc = root.AddComponent<SoulsPlayerController>();
            SetFloat(pc, "rollDuration", Clip("Animations/UAL1.fbx", "Roll").length * 0.9f);
            SetFloat(pc, "interactDuration", Clip("Animations/UAL1.fbx", "Interact").length);
            SetRef(pc, "rollClip", AssetDatabase.LoadAssetAtPath<AudioClip>(ART + "/Audio/roll.ogg"));

            // heal flask in the left fist, hidden until the player drinks (R)
            var flask = BuildHealFlask();
            Transform leftHand = anim.GetBoneTransform(HumanBodyBones.LeftHand);
            flask.transform.SetParent(leftHand, false);
            NormalizeWorldSize(flask, 0.22f);
            SetRef(pc, "flaskObject", flask);
            SetRef(pc, "drinkClip", AssetDatabase.LoadAssetAtPath<AudioClip>(ART + "/Audio/UI/flask_drink.ogg"));
            SetRef(pc, "healClip", AssetDatabase.LoadAssetAtPath<AudioClip>(ART + "/Audio/UI/flask_heal.ogg"));
            var so = new SerializedObject(pc);
            var arr = so.FindProperty("attackDurations");
            arr.arraySize = 3;
            arr.GetArrayElementAtIndex(0).floatValue = Clip("Animations/UAL2.fbx", "Sword_Regular_A").length * 0.9f;
            arr.GetArrayElementAtIndex(1).floatValue = Clip("Animations/UAL2.fbx", "Sword_Regular_B").length * 0.9f;
            arr.GetArrayElementAtIndex(2).floatValue = Clip("Animations/UAL2.fbx", "Sword_Regular_C").length * 0.95f;
            so.ApplyModifiedPropertiesWithoutUndo();

            return root;
        }

        static GameObject AttachToBone(Animator anim, HumanBodyBones bone, GameObject prefab, string name)
        {
            Transform t = anim.GetBoneTransform(bone);
            if (t == null) { Debug.LogWarning("[ChatDemo3DBuilder] missing bone " + bone); return null; }
            return AttachToTransform(t, prefab, name);
        }

        // A static carry socket on a bone, authored in CHARACTER-ROOT axes (x=right, y=up, z=forward)
        // — the same math WeaponStower.MakeSocket uses at runtime, so a placement tuned once via
        // StowTuneProbe transfers to any character on this rig. Built here in the bind pose, then it
        // rides the bone once the animator takes over.
        static Transform CarrySocket(string name, Transform charRoot, Transform bone, Vector3 rootPos, Vector3 rootEuler)
        {
            var s = new GameObject(name).transform;
            s.position = charRoot.TransformPoint(rootPos);
            s.rotation = charRoot.rotation * Quaternion.Euler(rootEuler);
            s.SetParent(bone, true);
            return s;
        }

        static GameObject AttachToTransform(Transform t, GameObject prefab, string name)
        {
            var item = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
            item.name = name;
            item.transform.SetParent(t, false);
            item.transform.localPosition = Vector3.zero;
            item.transform.localRotation = Quaternion.identity;
            return item;
        }

        // scales an attached item so its longest world dimension matches the target — these FBX
        // carry inconsistent unit factors, and bone lossy scales compound the problem
        static void NormalizeWorldSize(GameObject item, float targetSize)
        {
            if (item == null) return;
            Bounds b = RenderererSafeBounds(item);
            float current = Mathf.Max(b.size.x, Mathf.Max(b.size.y, b.size.z));
            if (current > 1e-5f)
                item.transform.localScale *= targetSize / current;
        }

        static Transform FindDeep(Transform root, string name)
        {
            foreach (var t in root.GetComponentsInChildren<Transform>(true))
                if (t.name == name) return t;
            return null;
        }

        static void ApplyCharacterTexture(GameObject model, string textureFile, string matName, Color tint)
        {
            var tex = AssetDatabase.LoadAssetAtPath<Texture2D>(ART + "/Characters/" + textureFile);
            if (tex == null) { Debug.LogWarning("[ChatDemo3DBuilder] missing texture " + textureFile); return; }
            var mat = new Material(Shader.Find("Standard"));
            mat.mainTexture = tex;
            mat.color = tint;
            mat.SetFloat("_Glossiness", 0.08f);
            AssetDatabase.CreateAsset(mat, GEN + "/" + matName + ".mat");
            foreach (var r in model.GetComponentsInChildren<Renderer>())
                r.sharedMaterial = mat;
        }

        static void GroundModel(GameObject model, float groundY)
        {
            Bounds b = RenderererSafeBounds(model);
            model.transform.position += Vector3.up * (groundY - b.min.y);
        }

        static void SetLayerRecursive(GameObject go, int layer)
        {
            go.layer = layer;
            foreach (Transform t in go.GetComponentsInChildren<Transform>())
                t.gameObject.layer = layer;
        }

        static Material MatAsset(string file, Color c, float metallic, float gloss)
        {
            string p = GEN + "/" + file;
            var m = AssetDatabase.LoadAssetAtPath<Material>(p);
            if (m == null)
            {
                m = new Material(Shader.Find("Standard"));
                AssetDatabase.CreateAsset(m, p);
            }
            m.color = c;
            m.SetFloat("_Metallic", metallic);
            m.SetFloat("_Glossiness", gloss);
            return m;
        }

        // load-or-rebuild a simple cone mesh asset under Generated/ (apex up, base on y=0)
        static Mesh ConeMeshAsset(string file, float radius, float height, int segs)
        {
            string p = GEN + "/" + file;
            var m = AssetDatabase.LoadAssetAtPath<Mesh>(p);
            if (m == null) { m = new Mesh(); AssetDatabase.CreateAsset(m, p); }
            m.Clear();
            var v = new List<Vector3> { Vector3.zero, Vector3.up * height };
            for (int i = 0; i < segs; i++)
            {
                float a = i * Mathf.PI * 2f / segs;
                v.Add(new Vector3(Mathf.Cos(a) * radius, 0f, Mathf.Sin(a) * radius));
            }
            var tris = new List<int>();
            for (int i = 0; i < segs; i++)
            {
                int a = 2 + i, b = 2 + (i + 1) % segs;
                tris.AddRange(new[] { 1, a, b });   // side
                tris.AddRange(new[] { 0, b, a });   // base
            }
            m.SetVertices(v);
            m.SetTriangles(tris, 0);
            m.RecalculateNormals();
            m.RecalculateBounds();
            return m;
        }

        static GameObject PrimPart(Transform parent, PrimitiveType type, string name,
                                   Vector3 pos, Vector3 scale, Vector3 euler, Material m)
        {
            var p = GameObject.CreatePrimitive(type);
            UnityEngine.Object.DestroyImmediate(p.GetComponent<Collider>());
            p.name = name;
            p.transform.SetParent(parent, false);
            p.transform.localPosition = pos;
            p.transform.localScale = scale;
            p.transform.localEulerAngles = euler;
            p.GetComponent<MeshRenderer>().sharedMaterial = m;
            return p;
        }

        // procedural halberd (no pack asset has one): dark ash pole, steel axe head, back spike
        // and top spike, built along +Y like the pack's sword so the weapon mount aligns
        static GameObject BuildHalberd()
        {
            var wood = MatAsset("HalberdWood.mat", new Color(0.22f, 0.15f, 0.10f), 0f, 0.12f);
            var steel = MatAsset("HalberdSteel.mat", new Color(0.50f, 0.52f, 0.57f), 0.7f, 0.55f);

            var root = new GameObject("Halberd");
            GameObject Part(PrimitiveType type, string name, Vector3 pos, Vector3 scale, Vector3 euler, Material m)
                => PrimPart(root.transform, type, name, pos, scale, euler, m);
            // grip point = local origin (hand), butt below at -0.9, head above at ~+1.5 — so the
            // attach pivot is the hand and flips/scales behave regardless of bone scale
            Part(PrimitiveType.Cylinder, "Pole",     new Vector3(0f, 0.25f, 0f),  new Vector3(0.060f, 1.15f, 0.060f), Vector3.zero, wood);
            Part(PrimitiveType.Cylinder, "Butt",     new Vector3(0f, -0.85f, 0f), new Vector3(0.075f, 0.05f, 0.075f), Vector3.zero, steel);
            Part(PrimitiveType.Cylinder, "Collar",   new Vector3(0f, 0.84f, 0f),  new Vector3(0.075f, 0.045f, 0.075f), Vector3.zero, steel);
            Part(PrimitiveType.Cube,     "AxeBlade", new Vector3(0.19f, 1.12f, 0f), new Vector3(0.30f, 0.42f, 0.035f), Vector3.zero, steel);
            Part(PrimitiveType.Cube,     "AxeEdge",  new Vector3(0.33f, 1.12f, 0f), new Vector3(0.10f, 0.50f, 0.030f), new Vector3(0, 0, 8f), steel);
            Part(PrimitiveType.Cube,     "BackSpike",new Vector3(-0.14f, 1.12f, 0f), new Vector3(0.18f, 0.09f, 0.030f), new Vector3(0, 0, -6f), steel);
            Part(PrimitiveType.Cube,     "TopSpike", new Vector3(0f, 1.52f, 0f),  new Vector3(0.05f, 0.36f, 0.05f),  Vector3.zero, steel);
            return root;
        }

        // little glowing estus bottle for the left hand — visible only during the drink
        static GameObject BuildHealFlask()
        {
            var gold = MatAsset("FlaskGold.mat", new Color(0.95f, 0.72f, 0.30f), 0.1f, 0.7f);
            gold.EnableKeyword("_EMISSION");
            gold.SetColor("_EmissionColor", new Color(0.85f, 0.55f, 0.18f) * 1.4f);
            var cork = MatAsset("HalberdWood.mat", new Color(0.22f, 0.15f, 0.10f), 0f, 0.12f);

            var root = new GameObject("HealFlask");
            PrimPart(root.transform, PrimitiveType.Sphere,   "Body", Vector3.zero,                new Vector3(0.16f, 0.17f, 0.16f), Vector3.zero, gold);
            PrimPart(root.transform, PrimitiveType.Cylinder, "Neck", new Vector3(0, 0.105f, 0),   new Vector3(0.05f, 0.035f, 0.05f), Vector3.zero, gold);
            PrimPart(root.transform, PrimitiveType.Cylinder, "Cork", new Vector3(0, 0.150f, 0),   new Vector3(0.04f, 0.018f, 0.04f), Vector3.zero, cork);
            return root;
        }

        // procedural crusader great helm: steel cylinder, cross ridge over a dark eye slit,
        // flat cap; optional crimson crest fin for the boss. Built face-forward (+Z), origin
        // at the helmet center.
        static GameObject BuildKnightHelm(bool crest)
        {
            var steel = MatAsset("HalberdSteel.mat", new Color(0.50f, 0.52f, 0.57f), 0.7f, 0.55f);
            var dark = MatAsset("HelmDark.mat", new Color(0.045f, 0.045f, 0.055f), 0.2f, 0.15f);

            var root = new GameObject("KnightHelm");
            GameObject Part(PrimitiveType type, string name, Vector3 pos, Vector3 scale, Vector3 euler, Material m)
                => PrimPart(root.transform, type, name, pos, scale, euler, m);

            Part(PrimitiveType.Cylinder, "Body",   new Vector3(0, 0, 0),       new Vector3(1.00f, 0.50f, 1.00f), Vector3.zero, steel);
            Part(PrimitiveType.Cylinder, "Cap",    new Vector3(0, 0.50f, 0),   new Vector3(1.05f, 0.045f, 1.05f), Vector3.zero, steel);
            Part(PrimitiveType.Cylinder, "Rim",    new Vector3(0, -0.46f, 0),  new Vector3(1.05f, 0.04f, 1.05f), Vector3.zero, steel);
            Part(PrimitiveType.Cube,     "EyeSlit",new Vector3(0, 0.14f, 0.46f), new Vector3(0.72f, 0.09f, 0.10f), Vector3.zero, dark);
            Part(PrimitiveType.Cube,     "Ridge",  new Vector3(0, 0.02f, 0.49f), new Vector3(0.09f, 0.70f, 0.07f), Vector3.zero, steel);
            if (crest)
            {
                var crimson = MatAsset("HelmCrest.mat", new Color(0.45f, 0.10f, 0.08f), 0.1f, 0.25f);
                Part(PrimitiveType.Cube, "Crest", new Vector3(0, 0.66f, -0.04f), new Vector3(0.06f, 0.30f, 0.78f), Vector3.zero, crimson);
            }
            return root;
        }

        // swallows the head (and that big low-poly hair mop) inside a great helm that rides
        // the Head bone, so it follows idle sway, attacks and the dialogue head-nod.
        // Anchored to the head bone pivot — model render bounds are polluted by attached weapons.
        static void AttachKnightHelm(Animator anim, Transform characterRoot, bool crest, float size, float charHeight,
                                     float upFraction = 0.06f)
        {
            Transform head = anim.GetBoneTransform(HumanBodyBones.Head);
            if (head == null) { Debug.LogWarning("[ChatDemo3DBuilder] no Head bone for helm"); return; }
            var helm = BuildKnightHelm(crest);
            helm.transform.SetParent(head, false);
            NormalizeWorldSize(helm, size);
            helm.transform.rotation = characterRoot.rotation;
            helm.transform.position = head.position
                                      + Vector3.up * (charHeight * upFraction)
                                      + characterRoot.forward * (size * 0.05f);
        }

        static GameObject BuildBoss(RuntimeAnimatorController ctrl, float[] swingDurations)
        {
            float bD = bossHalfZ * 2f;
            var root = new GameObject("Boss_Sentinel");
            root.transform.position = new Vector3(bossCx, 0f, castleHz + bD - 4.6f);   // before the statue
            root.transform.rotation = Quaternion.Euler(0f, 180f, 0f);                  // facing the mist door

            var cc = root.AddComponent<CharacterController>();
            cc.center = new Vector3(0, 1.6f, 0);
            cc.height = 3.0f;
            cc.radius = 0.6f;

            var model = (GameObject)PrefabUtility.InstantiatePrefab(LoadModel("Characters/Warrior.fbx"));
            model.name = "SentinelModel";
            model.transform.SetParent(root.transform, false);
            Bounds b = RenderererSafeBounds(model);
            float scale = b.size.y > 0.01f ? 3.1f / b.size.y : 1f;   // towering: ~3.1 m
            model.transform.localScale *= scale;
            GroundModel(model, root.transform.position.y);
            ApplyCharacterTexture(model, "Warrior_Texture.png", "BossKnight", new Color(0.40f, 0.36f, 0.45f));   // ashen-violet plate

            var anim = model.GetComponent<Animator>();
            if (anim == null) anim = model.AddComponent<Animator>();
            anim.runtimeAnimatorController = ctrl;
            anim.applyRootMotion = false;
            anim.cullingMode = AnimatorCullingMode.AlwaysAnimate;

            Transform mount = FindDeep(model.transform, "Weapon.R");
            if (mount == null) mount = anim.GetBoneTransform(HumanBodyBones.RightHand);
            var halberd = BuildHalberd();
            halberd.transform.SetParent(mount, false);
            halberd.transform.localPosition = Vector3.zero;
            // the mount's rest axis points at the ground (sword-style carry) — flip the
            // halberd around the grip so the axe head rides high
            halberd.transform.localRotation = Quaternion.Euler(180f, 0f, 0f);
            NormalizeWorldSize(halberd, 2.7f);

            AttachKnightHelm(anim, root.transform, crest: true, size: 1.05f, charHeight: 3.1f);

            root.AddComponent<BreathingIdle>();

            // boss theme: looping, silent until the fight begins (BossController crossfades it)
            var musicImp = AssetImporter.GetAtPath(ART + "/Audio/boss_theme.ogg") as AudioImporter;
            if (musicImp != null)
            {
                var ss = musicImp.defaultSampleSettings;
                ss.loadType = AudioClipLoadType.Streaming;
                musicImp.defaultSampleSettings = ss;
                musicImp.SaveAndReimport();
            }
            var music = root.AddComponent<AudioSource>();
            music.clip = AssetDatabase.LoadAssetAtPath<AudioClip>(ART + "/Audio/boss_theme.ogg");
            music.loop = true;
            music.playOnAwake = false;
            music.volume = 0f;
            music.spatialBlend = 0f;

            var boss = root.AddComponent<BossController>();
            var so = new SerializedObject(boss);
            var arr = so.FindProperty("attackDurations");
            arr.arraySize = swingDurations.Length;
            for (int i = 0; i < swingDurations.Length; i++)
                arr.GetArrayElementAtIndex(i).floatValue = swingDurations[i];
            so.ApplyModifiedPropertiesWithoutUndo();
            SetFloat(boss, "lungeDuration", Clip("Animations/UAL2.fbx", "Sword_Dash_RM").length / 0.85f);
            return root;
        }

        static GameObject BuildCamera(GameObject player)
        {
            var camGO = new GameObject("Main Camera");
            camGO.tag = "MainCamera";
            var cam = camGO.AddComponent<Camera>();
            cam.fieldOfView = 55f;
            cam.nearClipPlane = 0.1f;
            cam.farClipPlane = 400f;
            camGO.AddComponent<AudioListener>();
            camGO.AddComponent<SoulsCameraRig>();
            camGO.transform.position = player.transform.position + new Vector3(0, 2.6f, -4.2f);
            camGO.transform.rotation = Quaternion.Euler(14f, 0f, 0f);
            return camGO;
        }

        static GameObject BuildNpc(RuntimeAnimatorController ctrl, GameObject playerGO)
        {
            var root = new GameObject("NPC_Velmire");
            root.layer = 2;
            Vector3 npcPos = new Vector3(7.0f, 0f, 7.0f);
            root.transform.position = npcPos;
            // face the player spawn
            Vector3 toPlayer = new Vector3(0, 0, -9f) - npcPos; toPlayer.y = 0;
            root.transform.rotation = Quaternion.LookRotation(toPlayer.normalized);

            var body = root.AddComponent<CapsuleCollider>();
            body.center = new Vector3(0, 0.9f, 0);
            body.height = 1.8f;
            body.radius = 0.4f;

            var trigger = root.AddComponent<SphereCollider>();
            trigger.isTrigger = true;
            trigger.radius = 2.2f;   // tight: you have to actually walk up to him
            trigger.center = new Vector3(0, 1f, 0);

            var model = (GameObject)PrefabUtility.InstantiatePrefab(LoadModel("Characters/Monk.fbx"));
            model.name = "MonkModel";
            model.transform.SetParent(root.transform, false);
            SetLayerRecursive(model, 2);
            Bounds b = RenderererSafeBounds(model);
            float scale = b.size.y > 0.01f ? 1.85f / b.size.y : 1f;
            model.transform.localScale *= scale;
            GroundModel(model, root.transform.position.y);

            // pale, ghostly robes
            ApplyCharacterTexture(model, "Monk_Texture.png", "NpcMonk", new Color(0.95f, 0.93f, 0.92f));

            var anim = model.GetComponent<Animator>();
            if (anim == null) anim = model.AddComponent<Animator>();
            anim.runtimeAnimatorController = ctrl;
            anim.applyRootMotion = false;
            anim.cullingMode = AnimatorCullingMode.AlwaysAnimate;

            // He CARRIES the pair he offers the player (user 2026-07-25): shield slung across his
            // back, sword hanging along his right leg — the very sockets WeaponStower stows the
            // player's gear into, so the same steel reads identically before and after it changes
            // hands. Both vanish the instant the player accepts (NPCGearOffer.Grant).
            GameObject velmireSword = null, velmireShield = null;
            Transform gearHips = anim.GetBoneTransform(HumanBodyBones.Hips);
            Transform gearChest = anim.GetBoneTransform(HumanBodyBones.Chest);
            if (gearChest == null) gearChest = anim.GetBoneTransform(HumanBodyBones.Spine);
            if (gearHips != null)
            {
                var waist = CarrySocket("CarrySocket_Waist", root.transform, gearHips,
                                        new Vector3(0.24f, 0.88f, -0.10f), new Vector3(105f, -10f, 90f));
                velmireSword = AttachToTransform(waist, LoadModel("Weapons/Sword.fbx"), "Sword");
                NormalizeWorldSize(velmireSword, 1.15f);
                SetLayerRecursive(velmireSword, 2);
            }
            if (gearChest != null)
            {
                // y 0.95, not the player's 0.80: the Monk is taller (1.85 m) and his robe is long, so
                // the placement tuned on the Warrior rides his backside instead of his back
                var back = CarrySocket("CarrySocket_Back", root.transform, gearChest,
                                       new Vector3(0.02f, 0.95f, -0.26f), new Vector3(270f, 180f, 0f));
                velmireShield = AttachToTransform(back, LoadModel("Weapons/Shield_Heater.fbx"), "Shield");
                NormalizeWorldSize(velmireShield, 0.80f);
                SetLayerRecursive(velmireShield, 2);
            }
            if (velmireSword == null || velmireShield == null)
                Debug.LogWarning("[ChatDemo3DBuilder] Velmire's carry sockets need Hips + Chest/Spine on the " +
                                 "Monk avatar — gear skipped, the offer will have nothing to hand over.");

            // his little corner: candles and a skull (parented to the environment — the NPC
            // root rotates toward the player at runtime and must not drag props with it)
            PlacePiece("Candles_1", npcPos + root.transform.right * 0.9f + root.transform.forward * 0.2f, Range(0, 360), envRoot, collider: false);
            PlacePiece("Skull", npcPos - root.transform.right * 0.8f, Range(0, 360), envRoot, collider: false);

            // dialogue camera: over the player's shoulder, framing the NPC
            var camPoint = new GameObject("DialogueCameraPoint").transform;
            camPoint.SetParent(root.transform, false);
            Vector3 worldCamPos = npcPos + root.transform.forward * 2.4f + root.transform.right * 1.0f + Vector3.up * 1.65f;
            camPoint.position = worldCamPos;
            camPoint.rotation = Quaternion.LookRotation((npcPos + Vector3.up * 1.45f) - worldCamPos);

            root.AddComponent<BreathingIdle>();
            var npc = root.AddComponent<NPCInteractor3D>();
            SetRef(npc, "dialogueCameraPoint", camPoint);
            // identity lives here since the NPC component became the generic NPCChatBase — the
            // base class defaults are a nameless villager
            SetString(npc, "NpcName", "Velmire, the Pale Herald");
            SetString(npc, "descriptionAndRules",
                "You are Velmire, the Pale Herald: a white-masked, soft-spoken emissary lingering by the gate of a ruined castle. " +
                "You greet travellers with honeyed courtesy that thinly veils mockery. You pity the player for wandering these dead " +
                "lands guideless and lordless, and you address them as 'lambkin' or 'poor wanderer'. You speak in flowery, " +
                "old-fashioned phrases, hint that you know more than you say, and never give a straight answer. " +
                "You know what waits beyond the wall of golden mist at the northern arch: the Sentinel of the Mist, a towering " +
                "hollow knight wielding a halberd, who has felled every challenger before. If asked about the mist, the arch or " +
                "the boss, you foreshadow it with morbid delight — urging the lambkin onward while clearly expecting them to die. " +
                "Stay in character at all times. Keep your replies to one to three short sentences. " +
                // The gear beat, in his own voice — the tools block teaches the FORMAT, the persona
                // teaches WHEN. It opens by overriding his evasiveness on purpose: with only "never
                // give a straight answer" above, the 0.8B dodged the request no matter how often the
                // player asked (user 2026-07-25), so the exception and "never withhold" are the two
                // load-bearing phrases here.
                // The gear beat, kept SHORT on purpose. A numbered three-step version was tried and
                // measured WORSE (2026-07-25): the 0.8B latched onto its "say you cannot help them"
                // clause and refused without ever looking, and the persona alone had grown to 364 of
                // the prompt's 943 tokens. The sequence is still stated — look, then let them choose —
                // just not narrated step by step.
                "You carry a sword and a heater shield you no longer need, and you give them to any wanderer who " +
                "asks for a weapon. Look before you offer: call CheckMyGear, then say what you found and let them " +
                "take it with AskUserQuestion. Never withhold that offer. " +
                // Phrasing lives HERE, not in the shared tools block (user 2026-07-26): it matters
                // because Velmire is VOICED — his line is spoken, the popup is not — so a first-person
                // question would read as dialogue the player never heard. A text-only NPC has no such
                // split and gets no such rule.
                "Word an AskUserQuestion as a label on the screen rather than speech, naming yourself: " +
                "'Take Velmire's sword and shield?', never 'Do you want to take mine?'.");
            // Two tools on this NPC: the built-in AskUserQuestion popup (ON by default since
            // 2026-07-24) plus the internal GetPlayerGear read contributed by NPCGearOffer below.
            SetBool(npc, "enableAskUserQuestion", true);
            // Velmire is the ResumeFromCompact demo (user 2026-07-15): paired with the small
            // context below, the model auto-compacts its history after a few replies and keeps
            // talking on the short compacted prefix.
            SetEnum(npc, "historyMode", (int)NPCInteractor3D.HistoryMode.ResumeFromCompact);
            // small context on purpose (user default 2026-07-15): compaction demos trigger after
            // a few replies; the context bar above the input row makes the fill visible.
            // 400 -> 600 when AskUserQuestion became a default, then -> 1200 for the gear beat (user
            // 2026-07-25). The prefix measures 686 tokens with the real tokenizer: the persona (285,
            // gear instructions included) plus Qwen3.5's own tools preamble with both schemas (401),
            // and the rule is REAL conversation headroom
            // above it: at 600 the prefix nearly filled the window, so he compacted after almost every
            // single reply — the transcript was wiped each time and reopening always looked like an
            // amnesiac reset. 1200 leaves ~510 tokens of actual chat, so the ResumeFromCompact demo
            // fires after a genuine conversation instead of immediately. That is also the interesting
            // case for the gear: once compacted, he only still knows he handed his sword over because
            // GetPlayerGear tells him so.
            // 1200 -> 1800 (2026-07-25): the prefix is bigger than it looks. Measured with the real
            // tokenizer, the persona is only 300-360 of it — the # Tools block (282 for the header plus
            // both schemas) and the format spec + <IMPORTANT> reminder (295) are another ~580 that never
            // show up in the inspector's System Prompt field. Total ~860. At 1200 that left ~340 tokens of
            // actual conversation, i.e. compaction after two or three exchanges; 1800 leaves ~940.
            // The inspector's "Effective System Prompt" foldout shows the whole thing with an estimate.
            SetInt(npc, "maxContextLength", 1800);
            // Velmire speaks through Kokoro (82M non-AR, RTF ~0.3 — speaks DURING generation)
            // with the am_onyx voicepack: the same deep Freeman-esque narrator timbre the
            // CosyVoice "velmire" voice was baked from. CosyVoice3 stays selectable on the enum
            // and takes over once its A6 perf work lands RTF < 1.
            SetEnum(npc, "conversationMode", (int)NPCInteractor3D.ConversationMode.LlmPlusTts);
            // Velmire now speaks through pocket-tts (Kyutai 100M AR, RTF ~0.15 — the DEFAULT NPC
            // TTS): real-time DURING generation, correct name pronunciation, voice cloning.
            // His voice is CLONED from the Ansbach reference clip (precomputed into the shared
            // Resources/Cache by the inspector button / bake-all menu — runtime is a pure load);
            // "jean" stays as the baked fallback if the clip or its cache ever goes missing.
            SetEnum(npc, "ttsModel", (int)NPCInteractor3D.TtsModel.PocketTTS);
            SetString(npc, "ttsVoice", "jean");
            SetObject(npc, "clonedVoiceClip", AssetDatabase.LoadAssetAtPath<AudioClip>(
                "Assets/DeepUnity/Tutorials/ChatDemo3D/Voices/Ansbach_4-15s.mp3"));
            SetFloat(npc, "voicePitch", 1.0f);
            SetFloat(npc, "voiceVolume", 5f);   // user 2026-07-15
            // the world drops to HALF while Velmire talks and eases back on close, so the voice sits
            // on top of the ambience instead of next to it (user 2026-07-25)
            SetFloat(npc, "worldAudioWhileTalking", 0.5f);
            // Qwen3.5-IT's own recommended sampling, verbatim (user 2026-07-25). presence_penalty 2.0
            // is the high end of Qwen's 0-2 range: it is what keeps a 0.8B from looping a phrase, and
            // it costs some willingness to repeat a NAME — if the NPC starts avoiding "Velmire" or
            // "the Sentinel", that is the knob to walk back, not the temperature.
            SetFloat(npc, "temperature", 1.0f);
            SetFloat(npc, "topP", 1.0f);
            SetInt(npc, "topK", 20);
            SetFloat(npc, "minP", 0.0f);
            SetFloat(npc, "presencePenalty", 2.0f);
            SetFloat(npc, "repetitionPenalty", 1.0f);
            // 2 clauses per utterance: fewer, longer synthesis calls suit this GPU better than one
            // clause at a time, and the prosody flows across the comma (user 2026-07-25)
            SetInt(npc, "clausesPerChunk", 2);
            // residency A/B test: the big transparent-green sphere slow-prefetches Qwen+Kokoro
            // on entry, HOLDS both on the GPU while the player is inside, and unloads both on
            // exit; toggle off in the inspector for contact loading (talk trigger = mini zone)
            SetBool(npc, "usePrefetchZone", true);
            SetFloat(npc, "prefetchRadius", 10f);

            // The gear beat's engine half: contributes the internal GetPlayerGear tool to his prompt
            // and performs the hand-over when the player accepts. It is a plain INPCToolProvider
            // component, which is the whole point of that interface — WHICH tools an NPC has is
            // authored in the scene, so Morwenna across the courtyard gets none of this.
            var offer = root.AddComponent<NPCGearOffer>();
            var playerGear = playerGO != null ? playerGO.GetComponent<PlayerGear>() : null;
            if (playerGear != null) SetRef(offer, "playerGear", playerGear);
            else Debug.LogWarning("[ChatDemo3DBuilder] no PlayerGear on the player — Velmire's offer cannot land.");
            if (velmireSword != null) SetRef(offer, "npcSword", velmireSword.transform);
            if (velmireShield != null) SetRef(offer, "npcShield", velmireShield.transform);

            // LAST, once every tool this NPC has actually exists on the object: bake the # Tools block
            // INTO Description And Rules (user 2026-07-25). Nothing is injected at runtime any more, so
            // the field is the whole system prompt bar the NAME heading and the ## MEMORY block — and
            // that is exactly what the inspector shows. Same call the inspector button makes.
            // toolsFirst: false — who he is reads first, tools under it (user's call; note the 300
            // finetuning samples are written tools-first, so this is a deliberate divergence).
            var chat = root.GetComponent<NPCInteractor3D>();
            if (chat != null)
            {
                var f = typeof(NPCChatBase).GetField("descriptionAndRules",
                    System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
                f.SetValue(chat, chat.WithToolsBlock((string)f.GetValue(chat), toolsFirst: false));
            }
            return root;
        }

        // Secondary dialogue NPC: a witch across the courtyard from Velmire. Same NPCInteractor3D
        // component and the same Qwen3.5-0.8B (int8) — only the serialized personality differs.
        // Model: the actual female "Witch" from Quaternius' Ultimate Modular Women pack (CC0).
        static GameObject BuildWitchNpc(RuntimeAnimatorController ctrl)
        {
            var root = new GameObject("NPC_Morwenna");
            root.layer = 2;
            Vector3 npcPos = new Vector3(-7.0f, 0f, 6.0f);
            root.transform.position = npcPos;
            // face the player spawn
            Vector3 toPlayer = new Vector3(0, 0, -9f) - npcPos; toPlayer.y = 0;
            root.transform.rotation = Quaternion.LookRotation(toPlayer.normalized);

            var body = root.AddComponent<CapsuleCollider>();
            body.center = new Vector3(0, 0.9f, 0);
            body.height = 1.8f;
            body.radius = 0.4f;

            var trigger = root.AddComponent<SphereCollider>();
            trigger.isTrigger = true;
            trigger.radius = 2.2f;
            trigger.center = new Vector3(0, 1f, 0);

            // the REAL witch model: Quaternius "Witch" from the Ultimate Modular Women pack
            // (Apr-2022, CC0 — an actually FEMALE character: dress, hat, hair). Materials are
            // the FBX's own flat palette colors (purple outfit out of the box) — no texture pass.
            var model = (GameObject)PrefabUtility.InstantiatePrefab(LoadModel("Characters/Witch.fbx"));
            model.name = "WitchModel";
            model.transform.SetParent(root.transform, false);
            SetLayerRecursive(model, 2);
            Bounds b = RenderererSafeBounds(model);
            // slightly shorter and hunched-looking than Velmire
            float scale = b.size.y > 0.01f ? 1.7f / b.size.y : 1f;
            model.transform.localScale *= scale;
            GroundModel(model, root.transform.position.y);

            // generic rig -> her OWN clips: calm Idle_Neutral, and the Interact
            // lean-in gesticulation while she talks (both loop)
            var anim = model.GetComponent<Animator>();
            if (anim == null) anim = model.AddComponent<Animator>();
            anim.runtimeAnimatorController = CreateOwnClipAnimator("WitchAnimator",
                "Characters/Witch.fbx", "Idle_Neutral", "Interact");
            anim.applyRootMotion = false;
            anim.cullingMode = AnimatorCullingMode.AlwaysAnimate;

            // the Wizard model brings its own pointed hat + staff-in-hand; just plant a spare
            // crooked staff by her corner for set dressing
            var staffMat = MatAsset("WitchStaff.mat", new Color(0.23f, 0.16f, 0.10f), 0f, 0.1f);
            var staff = new GameObject("WitchStaff");
            staff.transform.SetParent(envRoot, false);
            staff.transform.position = npcPos + root.transform.right * 0.55f - root.transform.forward * 0.15f;
            staff.transform.rotation = Quaternion.Euler(0f, 0f, 7f);
            PrimPart(staff.transform, PrimitiveType.Cylinder, "Shaft",
                     new Vector3(0, 0.75f, 0), new Vector3(0.045f, 0.75f, 0.045f), Vector3.zero, staffMat);
            PrimPart(staff.transform, PrimitiveType.Sphere, "Knot",
                     new Vector3(0, 1.52f, 0), Vector3.one * 0.11f, Vector3.zero, staffMat);

            // her corner: a ring of candles, bones and a dead tree (parented to the environment —
            // the NPC root rotates toward the player at runtime and must not drag props with it)
            var candleA = PlacePiece("Candles_1", npcPos + root.transform.right * 0.8f + root.transform.forward * 0.3f, Range(0, 360), envRoot, collider: false);
            var candleB = PlacePiece("Candles_1", npcPos - root.transform.right * 0.7f + root.transform.forward * 0.5f, Range(0, 360), envRoot, collider: false);
            PlacePiece("Skull", npcPos + root.transform.right * 0.4f - root.transform.forward * 0.4f, Range(0, 360), envRoot, collider: false);
            PlacePiece("Skull", npcPos - root.transform.right * 1.1f, Range(0, 360), envRoot, collider: false);
            PlacePiece("DeadTree_1", npcPos - root.transform.forward * 1.6f + root.transform.right * 1.4f, Range(0, 360), envRoot, collider: false);

            // LIT candles: a warm flickering point light over each candle cluster, plus one
            // broader fill so her whole corner reads at night instead of sitting in the dark
            AddCandleGlow(candleA, 0.35f, 2.4f, 4.5f);
            AddCandleGlow(candleB, 0.35f, 2.4f, 4.5f);
            var fill = new GameObject("WitchCornerFill").AddComponent<Light>();
            fill.transform.SetParent(envRoot, false);
            fill.transform.position = npcPos + Vector3.up * 2.1f + root.transform.forward * 0.4f;
            fill.type = LightType.Point;
            fill.color = new Color(1f, 0.78f, 0.5f);
            fill.intensity = 1.1f;
            fill.range = 7f;
            fill.shadows = LightShadows.None;

            // dialogue camera: over the player's shoulder, framing the witch
            var camPoint = new GameObject("DialogueCameraPoint").transform;
            camPoint.SetParent(root.transform, false);
            Vector3 worldCamPos = npcPos + root.transform.forward * 2.4f + root.transform.right * 1.0f + Vector3.up * 1.65f;
            camPoint.position = worldCamPos;
            camPoint.rotation = Quaternion.LookRotation((npcPos + Vector3.up * 1.45f) - worldCamPos);

            root.AddComponent<BreathingIdle>();
            var npc = root.AddComponent<NPCInteractor3D>();
            SetRef(npc, "dialogueCameraPoint", camPoint);
            SetString(npc, "NpcName", "Morwenna, the Hollow Witch");
            SetString(npc, "descriptionAndRules",
                "You are Morwenna, the Hollow Witch: a crooked, sharp-tongued crone crouched among candles and bones in a " +
                "corner of the ruined courtyard. You mutter over your brews, barter in riddles, and treat every question as a " +
                "foolish waste of your time — yet you cannot resist showing off how much you know. You call the player " +
                "'little morsel' or 'wet-eared pup', and you speak in short, cackling, earthy sentences full of herbs, bones " +
                "and bad omens. You despise Velmire, the Pale Herald who lingers by the gate, and warn the player never to " +
                "trust his honeyed tongue. You too know what waits beyond the wall of golden mist at the northern arch: the " +
                "Sentinel of the Mist, a towering hollow knight with a halberd — you claim you cursed it yourself long ago, " +
                "and cackle that the player's bones will make a fine addition to your collection when it fells them. " +
                "Stay in character at all times. Keep your replies to one to three short sentences. " +
                // Voiced, like Velmire — same reason, her own words. See his persona for the why.
                "Word an AskUserQuestion as a label on the screen rather than speech, naming yourself: " +
                "'Take Morwenna's charm?', never 'Do you want mine?'.");
            // AskUserQuestion is ON by default on NPCChatBase; the witch keeps it — barters-in-
            // riddles is exactly the persona that puts deals to the player. Her context is the
            // 8192 default, so the tools block costs her nothing in headroom.
            SetBool(npc, "enableAskUserQuestion", true);
            // history-mode A/B spread: the witch forgets you the moment you leave (fresh
            // InitializeChat every opening — the pre-history-modes behavior)
            SetEnum(npc, "historyMode", (int)NPCInteractor3D.HistoryMode.ResetEveryTime);
            // latent loading for her too: contact loading (the old A/B "B" arm) slammed the full
            // 8 MB/frame stream the instant the talk trigger was touched — a visible hitch. Her
            // Qwen now streams during the walk-up like Velmire's (and the POOL means whichever
            // of the two loads first serves both). The small sphere trigger stays interaction-only.
            SetBool(npc, "usePrefetchZone", true);
            SetFloat(npc, "prefetchRadius", 10f);
            // The crone speaks pocket-tts too (both NPCs on the default engine — ONE weight set
            // on the GPU serves both). The two voices stay distinct through CLONING: her voice is
            // cloned from the Finger Reader Enia reference clip (precomputed into Resources/Cache;
            // runtime = pure load), vs Velmire's Ansbach clone. "jean" is the baked fallback.
            SetEnum(npc, "conversationMode", (int)NPCInteractor3D.ConversationMode.LlmPlusTts);
            SetEnum(npc, "ttsModel", (int)NPCInteractor3D.TtsModel.PocketTTS);
            SetString(npc, "ttsVoice", "jean");
            SetObject(npc, "clonedVoiceClip", AssetDatabase.LoadAssetAtPath<AudioClip>(
                "Assets/DeepUnity/Tutorials/ChatDemo3D/Voices/FingerReaderEnia_0-15s.mp3"));
            SetFloat(npc, "voiceVolume", 5f);   // user 2026-07-15 (both castle NPCs at 5)
            // her tools go into the field too — AskUserQuestion only, no provider (see Velmire)
            var chat = root.GetComponent<NPCInteractor3D>();
            if (chat != null)
            {
                var f = typeof(NPCChatBase).GetField("descriptionAndRules",
                    System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
                f.SetValue(chat, chat.WithToolsBlock((string)f.GetValue(chat)));
            }
            return root;
        }

        // ---------------------------------------------------------------- UI

        static GameObject s_mistPrompt;
        static Image s_whiteFlash;
        static RectTransform s_bossFill;
        static CanvasGroup s_bossBarGroup;
        static DeathScreen s_deathScreen;

        static void BuildUI(TMP_FontAsset cinzel, Sprite vignette, GameObject npcGO, GameObject witchGO, GameObject playerGO)
        {
            var canvasGO = new GameObject("UI", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            var canvas = canvasGO.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGO.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
            scaler.matchWidthOrHeight = 0.5f;

            new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));

            Color gold = new Color(0.55f, 0.47f, 0.30f, 0.9f);
            Color parchment = new Color(0.84f, 0.78f, 0.64f);
            Color darkBG = new Color(0.045f, 0.045f, 0.06f, 0.94f);

            // --- vignette (always on, never blocks clicks)
            var vinGO = MakeRect("Vignette", canvasGO.transform, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var vinImg = vinGO.AddComponent<Image>();
            vinImg.sprite = vignette;
            vinImg.color = new Color(0f, 0f, 0f, 0.62f);
            vinImg.raycastTarget = false;

            // --- "[ I ] Speak" prompt, bottom center
            var promptGO = MakeRect("InteractPrompt", canvasGO.transform, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f),
                                    new Vector2(330, 58), new Vector2(0, 96));
            var promptBG = promptGO.AddComponent<Image>(); promptBG.color = darkBG;
            AddThinBorder(promptGO.transform, gold);
            var promptText = MakeTMP("Text", promptGO.transform, "Speak   —   [ I ]", cinzel, 26, parchment,
                                     TextAlignmentOptions.Center, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            promptGO.AddComponent<NPCInteractPrompt>();   // its own component: fade/bob live here
            promptGO.SetActive(false);

            // --- "Traverse the mist" prompt (same slot — the NPC and the fog wall are far apart)
            var mistPromptGO = MakeRect("MistPrompt", canvasGO.transform, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f),
                                        new Vector2(470, 58), new Vector2(0, 96));
            var mistBG = mistPromptGO.AddComponent<Image>(); mistBG.color = darkBG;
            AddThinBorder(mistPromptGO.transform, gold);
            MakeTMP("Text", mistPromptGO.transform, "Traverse the mist   —   [ E ]", cinzel, 26, parchment,
                    TextAlignmentOptions.Center, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            mistPromptGO.SetActive(false);
            s_mistPrompt = mistPromptGO;

            // --- full-screen white flash pulsed while crossing the fog wall
            var flashGO = MakeRect("MistFlash", canvasGO.transform, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var flashImg = flashGO.AddComponent<Image>();
            flashImg.color = Color.clear;
            flashImg.raycastTarget = false;
            s_whiteFlash = flashImg;

            // --- boss health bar, bottom center (hidden until the Sentinel wakes)
            var bossBarGO = MakeRect("BossBar", canvasGO.transform, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f),
                                     new Vector2(900, 56), new Vector2(0, 178));
            var bossGroup = bossBarGO.AddComponent<CanvasGroup>();
            bossGroup.alpha = 0f;
            bossGroup.interactable = false;
            bossGroup.blocksRaycasts = false;
            MakeTMP("Name", bossBarGO.transform, "Sentinel of the Mist", cinzel, 26, parchment,
                    TextAlignmentOptions.Left, new Vector2(0, 1), new Vector2(1, 1), new Vector2(0, 32), new Vector2(0, -14));
            var bossBarBG = MakeRect("BG", bossBarGO.transform, new Vector2(0, 0), new Vector2(1, 0),
                                     new Vector2(0, 16), new Vector2(0, 12));
            var bossBgImg = bossBarBG.AddComponent<Image>();
            bossBgImg.color = new Color(0.04f, 0.04f, 0.05f, 0.85f);
            bossBgImg.raycastTarget = false;
            AddThinBorder(bossBarBG.transform, gold);
            var bossFillGO = MakeRect("Fill", bossBarBG.transform, new Vector2(0, 0), new Vector2(1, 1),
                                      new Vector2(-4, -4), Vector2.zero);
            var bossFillImg = bossFillGO.AddComponent<Image>();
            bossFillImg.color = new Color(0.62f, 0.12f, 0.10f);
            bossFillImg.raycastTarget = false;
            s_bossFill = (RectTransform)bossFillGO.transform;
            s_bossBarGroup = bossGroup;

            // --- YOU DIED / SENTINEL FELLED overlay (drawn over everything)
            var deathGO = MakeRect("DeathScreen", canvasGO.transform, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var dimImg = deathGO.AddComponent<Image>();
            dimImg.color = Color.clear;
            dimImg.raycastTarget = false;
            var diedGO = MakeTMP("YouDied", deathGO.transform, "YOU  DIED", cinzel, 110, new Color(0.55f, 0.07f, 0.07f, 0f),
                                 TextAlignmentOptions.Center, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var felledGO = MakeTMP("Felled", deathGO.transform, "SENTINEL  FELLED", cinzel, 96, new Color(0.87f, 0.72f, 0.38f, 0f),
                                   TextAlignmentOptions.Center, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            diedGO.SetActive(false);
            felledGO.SetActive(false);
            var death = deathGO.AddComponent<DeathScreen>();
            var deathAudio = deathGO.AddComponent<AudioSource>();
            deathAudio.playOnAwake = false;
            deathAudio.spatialBlend = 0f;
            SetRef(death, "dim", dimImg);
            SetRef(death, "deathText", diedGO.GetComponent<TMP_Text>());
            SetRef(death, "victoryText", felledGO.GetComponent<TMP_Text>());
            SetRef(death, "audioSource", deathAudio);
            SetRef(death, "deathClip", AssetDatabase.LoadAssetAtPath<AudioClip>(ART + "/Audio/you_died.ogg"));
            s_deathScreen = death;

            // --- chat panel (right-docked, slides in)
            var panelGO = MakeRect("SoulsChatWindow", canvasGO.transform, new Vector2(1f, 0f), new Vector2(1f, 1f),
                                   new Vector2(680, -56), new Vector2(-24, 0));
            ((RectTransform)panelGO.transform).pivot = new Vector2(1f, 0.5f);
            var borderImg = panelGO.AddComponent<Image>(); borderImg.color = gold;

            var bgGO = MakeRect("BG", panelGO.transform, Vector2.zero, Vector2.one, new Vector2(-4, -4), Vector2.zero);
            bgGO.AddComponent<Image>().color = darkBG;

            // title + divider
            var titleGO = MakeTMP("Title", panelGO.transform, "—", cinzel, 30, parchment, TextAlignmentOptions.Center,
                                  new Vector2(0, 1), new Vector2(1, 1), new Vector2(-40, 64), new Vector2(0, -40));
            var divGO = MakeRect("Divider", panelGO.transform, new Vector2(0, 1), new Vector2(1, 1), new Vector2(-70, 2), new Vector2(0, -78));
            divGO.AddComponent<Image>().color = gold;

            var infoGO = MakeTMP("InfoText", panelGO.transform, "", null, 19, new Color(0.62f, 0.58f, 0.49f),
                                 TextAlignmentOptions.Center, new Vector2(0, 1), new Vector2(1, 1), new Vector2(-50, 30), new Vector2(0, -100));
            infoGO.GetComponent<TMP_Text>().fontStyle = FontStyles.Italic;

            // scroll area
            var scrollGO = MakeRect("Messages", panelGO.transform, Vector2.zero, Vector2.one, new Vector2(-36, -210), new Vector2(0, -22));
            var scroll = scrollGO.AddComponent<ScrollRect>();
            var viewportGO = MakeRect("Viewport", scrollGO.transform, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            viewportGO.AddComponent<RectMask2D>();
            var vpImg = viewportGO.AddComponent<Image>(); vpImg.color = new Color(0, 0, 0, 0.01f);   // raycast catcher for scroll

            var contentGO = MakeRect("Content", viewportGO.transform, new Vector2(0, 1), new Vector2(1, 1), new Vector2(0, 0), Vector2.zero);
            ((RectTransform)contentGO.transform).pivot = new Vector2(0.5f, 1f);
            var vlg = contentGO.AddComponent<VerticalLayoutGroup>();
            vlg.padding = new RectOffset(10, 10, 8, 8);
            vlg.spacing = 18;
            vlg.childControlWidth = true; vlg.childControlHeight = true;
            vlg.childForceExpandWidth = true; vlg.childForceExpandHeight = false;
            var fitter = contentGO.AddComponent<ContentSizeFitter>();
            fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;

            scroll.viewport = (RectTransform)viewportGO.transform;
            scroll.content = (RectTransform)contentGO.transform;
            scroll.horizontal = false;
            scroll.vertical = true;
            scroll.movementType = ScrollRect.MovementType.Clamped;
            scroll.scrollSensitivity = 26f;

            // message template: gold name + parchment body
            var msgGO = new GameObject("MessageTemplate", typeof(RectTransform));
            msgGO.transform.SetParent(contentGO.transform, false);
            var msgVlg = msgGO.AddComponent<VerticalLayoutGroup>();
            msgVlg.spacing = 3;
            msgVlg.childControlWidth = true; msgVlg.childControlHeight = true;
            msgVlg.childForceExpandWidth = true; msgVlg.childForceExpandHeight = false;
            var nameGO = MakeTMP("Username", msgGO.transform, "Name", cinzel, 20, new Color(0.77f, 0.66f, 0.42f),
                                 TextAlignmentOptions.Left, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var bodyGO = MakeTMP("Message", msgGO.transform, "Body", null, 21, new Color(0.87f, 0.84f, 0.76f),
                                 TextAlignmentOptions.TopLeft, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);

            // context-fill bar: silver track + golden fill, right above the input row — live
            // view of the conversation vs Max Context Length (the NPC compacts when it fills)
            var ctxBarGO = MakeRect("ContextBar", panelGO.transform, new Vector2(0, 0), new Vector2(1, 0), new Vector2(-36, 5), new Vector2(0, 82));
            var ctxTrackImg = ctxBarGO.AddComponent<Image>(); ctxTrackImg.color = new Color(0.62f, 0.62f, 0.66f, 0.45f); ctxTrackImg.raycastTarget = false;   // silver = empty
            var ctxFillGO = MakeRect("Fill", ctxBarGO.transform, new Vector2(0, 0), new Vector2(0, 1), Vector2.zero, Vector2.zero);
            ((RectTransform)ctxFillGO.transform).pivot = new Vector2(0f, 0.5f);
            var ctxFillImg = ctxFillGO.AddComponent<Image>(); ctxFillImg.color = gold; ctxFillImg.raycastTarget = false;   // golden = filled

            // input row
            var rowGO = MakeRect("InputRow", panelGO.transform, new Vector2(0, 0), new Vector2(1, 0), new Vector2(-36, 54), new Vector2(0, 48));
            var rowHlg = rowGO.AddComponent<HorizontalLayoutGroup>();
            rowHlg.spacing = 10;
            rowHlg.childControlWidth = true; rowHlg.childControlHeight = true;
            rowHlg.childForceExpandWidth = false; rowHlg.childForceExpandHeight = true;

            var inputGO = BuildInputField(rowGO.transform, cinzel, parchment, out TMP_InputField inputField);
            inputGO.AddComponent<LayoutElement>().flexibleWidth = 1f;

            var sendBtn = BuildSoulsButton(rowGO.transform, "Speak", cinzel, gold, parchment, darkBG, 104);
            var leaveBtn = BuildSoulsButton(rowGO.transform, "Leave", cinzel, gold, new Color(0.72f, 0.55f, 0.45f), darkBG, 104);

            // --- component wiring
            var win = panelGO.AddComponent<SoulsChatWindow>();
            SetRef(win, "panel", (RectTransform)panelGO.transform);
            SetRef(win, "messageContainer", contentGO.transform);
            SetRef(win, "inputField", inputField);
            SetRef(win, "sendButton", sendBtn);
            SetRef(win, "leaveButton", leaveBtn);
            SetRef(win, "messageTemplate", msgGO);
            SetRef(win, "scrollRect", scroll);
            SetRef(win, "infoText", infoGO.GetComponent<TMP_Text>());
            SetRef(win, "titleText", titleGO.GetComponent<TMP_Text>());
            SetRef(win, "contextFill", (RectTransform)ctxFillGO.transform);

            // UI sounds: source on the canvas so the Leave click isn't cut off by the panel deactivating
            var uiAudio = canvasGO.AddComponent<AudioSource>();
            uiAudio.playOnAwake = false;
            uiAudio.spatialBlend = 0f;
            SetRef(win, "uiAudio", uiAudio);
            SetRef(win, "buttonClip", AssetDatabase.LoadAssetAtPath<AudioClip>(ART + "/Audio/UI/button_click.ogg"));
            var winSO = new SerializedObject(win);
            var typeArr = winSO.FindProperty("typeClips");
            string[] typeFiles = { "type_1.ogg", "type_2.ogg", "type_3.ogg" };
            typeArr.arraySize = typeFiles.Length;
            for (int i = 0; i < typeFiles.Length; i++)
                typeArr.GetArrayElementAtIndex(i).objectReferenceValue =
                    AssetDatabase.LoadAssetAtPath<AudioClip>(ART + "/Audio/UI/" + typeFiles[i]);
            winSO.ApplyModifiedPropertiesWithoutUndo();

            // Both dialogue NPCs share the one chat window, prompt box and buttons. Every
            // listener fires on both interactors, but only the one actually in interaction
            // reacts: AskNPC guards on WaitingInInteraction and CloseInteraction is a no-op
            // for an NPC that is already Idle (its coroutine/llm/player are all null).
            var npc = npcGO.GetComponent<NPCInteractor3D>();
            var witch = witchGO.GetComponent<NPCInteractor3D>();
            foreach (var it in new[] { npc, witch })
            {
                SetRef(it, "chatWindow", win);
                SetRef(it, "interactPrompt", promptGO.GetComponent<NPCInteractPrompt>());
                UnityEventTools.AddPersistentListener(sendBtn.onClick, new UnityAction(it.AskNPC));
                UnityEventTools.AddPersistentListener(leaveBtn.onClick, new UnityAction(it.CloseInteraction));
                UnityEventTools.AddVoidPersistentListener(inputField.onSubmit, new UnityAction(it.AskNPC));
            }
            UnityEventTools.AddPersistentListener(sendBtn.onClick, new UnityAction(win.PlayButtonClick));
            UnityEventTools.AddPersistentListener(leaveBtn.onClick, new UnityAction(win.PlayButtonClick));

            BuildHud(canvasGO.transform, gold, win, playerGO);
            BuildFpsAndPauseMenu(canvasGO.transform, cinzel, gold, parchment, darkBG, win);
        }

        // ---------------------------------------------------------------- fps counter + esc menu

        static void BuildFpsAndPauseMenu(Transform canvas, TMP_FontAsset cinzel, Color gold,
                                         Color parchment, Color darkBG, SoulsChatWindow win)
        {
            // --- FPS counter, bottom-left (never blocks clicks; still built before PauseMenu, so it
            // stays under the menu dim). The position is the exact mirror of the old top-right
            // placement — 24 px in from the side, 14 px off the edge — which also parks it in the
            // 50 px strip below the quick-slot block instead of overlapping it.
            var fpsGO = MakeTMP("FpsCounter", canvas, "-- FPS", cinzel, 22,
                                new Color(parchment.r, parchment.g, parchment.b, 0.75f),
                                TextAlignmentOptions.BottomLeft, Vector2.zero, Vector2.zero,
                                new Vector2(170, 32), new Vector2(109, 30));
            var fps = fpsGO.AddComponent<FpsCounter>();
            SetRef(fps, "label", fpsGO.GetComponent<TMP_Text>());

            // --- Esc menu: full-screen dim + centered Continue/Exit box. The world keeps
            // running while it's up (soulslike) — PauseMenu only gates player/camera input.
            var menuGO = MakeRect("PauseMenu", canvas, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var menu = menuGO.AddComponent<PauseMenu>();

            var panelGO = MakeRect("Panel", menuGO.transform, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var dim = panelGO.AddComponent<Image>();
            dim.color = new Color(0f, 0f, 0f, 0.55f);   // raycast target ON: swallows clicks to the world UI

            var boxGO = MakeRect("Box", panelGO.transform, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f),
                                 new Vector2(400, 250), Vector2.zero);
            boxGO.AddComponent<Image>().color = darkBG;
            AddThinBorder(boxGO.transform, gold);
            MakeTMP("Title", boxGO.transform, "MENU", cinzel, 32, parchment, TextAlignmentOptions.Center,
                    new Vector2(0, 1), new Vector2(1, 1), new Vector2(0, 56), new Vector2(0, -38));
            var menuDivGO = MakeRect("Divider", boxGO.transform, new Vector2(0, 1), new Vector2(1, 1),
                                     new Vector2(-70, 2), new Vector2(0, -74));
            menuDivGO.AddComponent<Image>().color = gold;

            var colGO = MakeRect("Buttons", boxGO.transform, Vector2.zero, Vector2.one,
                                 new Vector2(-100, -100), new Vector2(0, -34));
            var vlg = colGO.AddComponent<VerticalLayoutGroup>();
            vlg.spacing = 14;
            vlg.childControlWidth = true; vlg.childControlHeight = true;
            vlg.childForceExpandWidth = true; vlg.childForceExpandHeight = false;

            var contBtn = BuildSoulsButton(colGO.transform, "Continue", cinzel, gold, parchment, darkBG, 300);
            var exitBtn = BuildSoulsButton(colGO.transform, "Exit", cinzel, gold, new Color(0.72f, 0.55f, 0.45f), darkBG, 300);
            contBtn.GetComponent<LayoutElement>().preferredHeight = 54;
            exitBtn.GetComponent<LayoutElement>().preferredHeight = 54;
            UnityEventTools.AddPersistentListener(contBtn.onClick, new UnityAction(win.PlayButtonClick));

            SetRef(menu, "panel", panelGO);
            SetRef(menu, "chatWindow", win);
            SetRef(menu, "continueButton", contBtn);
            SetRef(menu, "exitButton", exitBtn);
            panelGO.SetActive(false);
        }

        // ---------------------------------------------------------------- souls HUD

        static void BuildHud(Transform canvas, Color gold, SoulsChatWindow win, GameObject playerGO)
        {
            var hudGO = MakeRect("SoulsHud", canvas, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            hudGO.AddComponent<CanvasGroup>();
            var hud = hudGO.AddComponent<SoulsHud>();

            Color frame = new Color(gold.r, gold.g, gold.b, 0.55f);

            // --- status bars, top-left (HP red / FP blue / Stamina green)
            RectTransform MakeBar(string name, float y, float width, Color fillColor, float fillAmount)
            {
                var bar = MakeRect(name, hudGO.transform, new Vector2(0, 1), new Vector2(0, 1), new Vector2(width, 15), new Vector2(28, y));
                ((RectTransform)bar.transform).pivot = new Vector2(0f, 1f);
                var bg = bar.AddComponent<Image>(); bg.color = new Color(0.04f, 0.04f, 0.05f, 0.85f); bg.raycastTarget = false;
                AddThinBorder(bar.transform, frame);
                var fill = MakeRect("Fill", bar.transform, new Vector2(0, 0), new Vector2(fillAmount, 1), new Vector2(-4, -4), Vector2.zero);
                var img = fill.AddComponent<Image>(); img.color = fillColor; img.raycastTarget = false;
                return (RectTransform)fill.transform;
            }
            var hpFill = MakeBar("HP", -28, 360, new Color(0.62f, 0.14f, 0.10f), 1f);
            var fpFill = MakeBar("FP", -50, 230, new Color(0.16f, 0.27f, 0.55f), 1f);
            var stFill = MakeBar("Stamina", -72, 310, new Color(0.38f, 0.50f, 0.18f), 1f);

            // --- quick slots, bottom-left. Three columns instead of the old diamond of four squares,
            // which is what actually reads as a souls HUD: left hand | spell over item | right hand.
            const float slotW = 64f;    // ONE shared width for every frame — columns must line up exactly
            const float pitch = 68f;    // centre-to-centre, i.e. the same 4 px gutter the diamond used
            const float handH = 84f;    // hand frames are ~1.3:1, only slightly taller than wide, never thin bars
            // The middle column sets the band height: two squares one pitch apart = pitch + slotW = 132.
            // The hand frames are centred in that same band, so all three columns share one centre line.
            const float bandH = pitch + slotW;
            const float bandW = pitch * 2f + slotW;
            // Centre chosen so the block keeps the diamond's 50 px left/bottom screen margins.
            var slots = MakeRect("QuickSlots", hudGO.transform, Vector2.zero, Vector2.zero,
                                 new Vector2(bandW, bandH), new Vector2(50f + bandW * 0.5f, 50f + bandH * 0.5f));
            (string name, Vector2 pos, Vector2 size, string asset, string icon)[] slotDefs =
            {
                ("SlotLeft",  new Vector2(-pitch, 0),         new Vector2(slotW, handH), "Weapons/Shield_Heater.fbx","shield"),
                ("SlotSpell", new Vector2(0,  pitch * 0.5f),  new Vector2(slotW, slotW), "Ruins/Torch.fbx",          "torch"),
                ("SlotItem",  new Vector2(0, -pitch * 0.5f),  new Vector2(slotW, slotW), "Ruins/Pot1.fbx",           "pot"),
                ("SlotRight", new Vector2(pitch, 0),          new Vector2(slotW, handH), "Weapons/Sword.fbx",        "sword"),
            };
            GameObject itemSlot = null, swordIcon = null, shieldIcon = null;
            foreach (var (name, pos, size, asset, icon) in slotDefs)
            {
                var slot = MakeRect(name, slots.transform, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), size, pos);
                var bg = slot.AddComponent<Image>(); bg.color = new Color(0.05f, 0.05f, 0.06f, 0.82f); bg.raycastTarget = false;
                AddThinBorder(slot.transform, frame);
                Sprite sprite = RenderItemIcon(asset, icon);
                if (sprite != null)
                {
                    var iconGO = MakeRect("Icon", slot.transform, Vector2.zero, Vector2.one, new Vector2(-10, -10), Vector2.zero);
                    var img = iconGO.AddComponent<Image>(); img.sprite = sprite; img.preserveAspect = true; img.raycastTarget = false;
                    if (name == "SlotRight") swordIcon = iconGO;
                    else if (name == "SlotLeft") shieldIcon = iconGO;
                }
                if (name == "SlotItem") itemSlot = slot;
            }

            // The weapon slots start EMPTY — the bordered frames stay, only the icons go, which is
            // what makes them visibly fill in when Velmire's sword and shield change hands. Ownership
            // (and therefore the icons) belongs to PlayerGear on the player, not to the HUD.
            var playerGear = playerGO != null ? playerGO.GetComponent<PlayerGear>() : null;
            if (playerGear != null)
            {
                if (swordIcon != null) { SetRef(playerGear, "swordSlotIcon", swordIcon); swordIcon.SetActive(false); }
                if (shieldIcon != null) { SetRef(playerGear, "shieldSlotIcon", shieldIcon); shieldIcon.SetActive(false); }
            }

            // flask charge counter on the item slot (drinking is on R)
            var flaskCountGO = MakeTMP("FlaskCount", itemSlot.transform, "5", null, 22, new Color(0.92f, 0.88f, 0.78f),
                                       TextAlignmentOptions.BottomRight, Vector2.zero, Vector2.one, new Vector2(-8, -4), Vector2.zero);

            SetRef(hud, "flaskCount", flaskCountGO.GetComponent<TMP_Text>());
            SetRef(hud, "player", playerGO.GetComponent<SoulsPlayerController>());
            SetRef(hud, "chatWindow", win);
            SetRef(hud, "hpFill", hpFill);
            SetRef(hud, "fpFill", fpFill);
            SetRef(hud, "staminaFill", stFill);
        }

        // renders an actual item model to a transparent 128px sprite — real icons, zero art budget
        static Sprite RenderItemIcon(string assetRelPath, string iconName)
        {
            string pngPath = GEN + "/Icon_" + iconName + ".png";
            if (!File.Exists(pngPath))
            {
                var item = (GameObject)PrefabUtility.InstantiatePrefab(LoadModel(assetRelPath));
                item.transform.position = new Vector3(900f, 900f, 900f);
                Bounds b = RenderererSafeBounds(item);

                // the scene's dusk lighting renders icons too dark — light them like a menu render
                var keyLight = new GameObject("iconlight").AddComponent<Light>();
                keyLight.type = LightType.Directional;
                keyLight.intensity = 1.4f;
                keyLight.color = Color.white;
                keyLight.transform.rotation = Quaternion.LookRotation(new Vector3(0.3f, -0.4f, 1f));

                var camGO = new GameObject("iconcam");
                var cam = camGO.AddComponent<Camera>();
                cam.orthographic = true;
                cam.orthographicSize = Mathf.Max(b.extents.x, b.extents.y) * 1.18f;
                cam.transform.position = b.center - new Vector3(0, 0, b.extents.z + 3f);
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = new Color(0, 0, 0, 0);
                cam.nearClipPlane = 0.01f;
                cam.farClipPlane = 50f;

                var rt = new RenderTexture(128, 128, 24, RenderTextureFormat.ARGB32);
                cam.targetTexture = rt;
                cam.Render();
                RenderTexture.active = rt;
                var tex = new Texture2D(128, 128, TextureFormat.RGBA32, false);
                tex.ReadPixels(new Rect(0, 0, 128, 128), 0, 0);
                tex.Apply();
                File.WriteAllBytes(pngPath, tex.EncodeToPNG());
                RenderTexture.active = null;
                cam.targetTexture = null;
                UnityEngine.Object.DestroyImmediate(tex);
                UnityEngine.Object.DestroyImmediate(camGO);
                UnityEngine.Object.DestroyImmediate(keyLight.gameObject);
                UnityEngine.Object.DestroyImmediate(item);
                AssetDatabase.ImportAsset(pngPath);

                var imp = (TextureImporter)AssetImporter.GetAtPath(pngPath);
                imp.textureType = TextureImporterType.Sprite;
                imp.alphaIsTransparency = true;
                imp.SaveAndReimport();
            }
            return AssetDatabase.LoadAssetAtPath<Sprite>(pngPath);
        }

        static GameObject BuildInputField(Transform parent, TMP_FontAsset cinzel, Color parchment, out TMP_InputField field)
        {
            var go = new GameObject("InputField", typeof(RectTransform));
            go.transform.SetParent(parent, false);
            var bg = go.AddComponent<Image>();
            bg.color = new Color(0.09f, 0.09f, 0.115f, 0.96f);

            field = go.AddComponent<TMP_InputField>();

            var areaGO = MakeRect("Text Area", go.transform, Vector2.zero, Vector2.one, new Vector2(-20, -12), Vector2.zero);
            areaGO.AddComponent<RectMask2D>();

            var phGO = MakeTMP("Placeholder", areaGO.transform, "Say something...", null, 20,
                               new Color(0.45f, 0.43f, 0.38f), TextAlignmentOptions.Left, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            phGO.GetComponent<TMP_Text>().fontStyle = FontStyles.Italic;
            var txtGO = MakeTMP("Text", areaGO.transform, "", null, 20, new Color(0.91f, 0.89f, 0.83f),
                                TextAlignmentOptions.Left, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);

            field.textViewport = (RectTransform)areaGO.transform;
            field.textComponent = txtGO.GetComponent<TMP_Text>();
            field.placeholder = phGO.GetComponent<TMP_Text>();
            field.lineType = TMP_InputField.LineType.SingleLine;
            // clearly visible blinking caret: gold accent, thick, unmistakable over the dark panel
            field.caretColor = new Color(0.77f, 0.66f, 0.42f);
            field.customCaretColor = true;
            field.caretWidth = 3;
            field.caretBlinkRate = 0.85f;
            field.selectionColor = new Color(0.45f, 0.38f, 0.22f, 0.6f);
            field.targetGraphic = bg;
            return go;
        }

        static Button BuildSoulsButton(Transform parent, string label, TMP_FontAsset cinzel,
                                       Color gold, Color textColor, Color darkBG, float width)
        {
            var go = new GameObject(label + "Button", typeof(RectTransform));
            go.transform.SetParent(parent, false);
            go.AddComponent<LayoutElement>().preferredWidth = width;
            var img = go.AddComponent<Image>();
            img.color = new Color(0.12f, 0.11f, 0.10f, 0.97f);
            AddThinBorder(go.transform, gold);

            var btn = go.AddComponent<Button>();
            btn.targetGraphic = img;
            var colors = btn.colors;
            colors.highlightedColor = new Color(1.5f, 1.4f, 1.2f, 1f);
            colors.pressedColor = new Color(0.7f, 0.7f, 0.7f, 1f);
            btn.colors = colors;

            MakeTMP("Label", go.transform, label, cinzel, 22, textColor, TextAlignmentOptions.Center,
                    Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            return btn;
        }

        static void AddThinBorder(Transform parent, Color color)
        {
            // four 2px edge images — cheap gold trim without a 9-slice sprite
            (Vector2 min, Vector2 max, Vector2 size, Vector2 pos)[] edges =
            {
                (new Vector2(0,1), new Vector2(1,1), new Vector2(0,2), new Vector2(0,-1)),   // top
                (new Vector2(0,0), new Vector2(1,0), new Vector2(0,2), new Vector2(0, 1)),   // bottom
                (new Vector2(0,0), new Vector2(0,1), new Vector2(2,0), new Vector2(1, 0)),   // left
                (new Vector2(1,0), new Vector2(1,1), new Vector2(2,0), new Vector2(-1,0)),   // right
            };
            foreach (var (min, max, size, pos) in edges)
            {
                var e = MakeRect("Edge", parent, min, max, size, pos);
                var img = e.AddComponent<Image>(); img.color = color; img.raycastTarget = false;
            }
        }

        static GameObject MakeRect(string name, Transform parent, Vector2 anchorMin, Vector2 anchorMax,
                                   Vector2 sizeDelta, Vector2 anchoredPos)
        {
            var go = new GameObject(name, typeof(RectTransform));
            var rt = (RectTransform)go.transform;
            rt.SetParent(parent, false);
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            rt.sizeDelta = sizeDelta;
            rt.anchoredPosition = anchoredPos;
            return go;
        }

        static GameObject MakeTMP(string name, Transform parent, string text, TMP_FontAsset font, float size,
                                  Color color, TextAlignmentOptions align, Vector2 anchorMin, Vector2 anchorMax,
                                  Vector2 sizeDelta, Vector2 anchoredPos)
        {
            var go = MakeRect(name, parent, anchorMin, anchorMax, sizeDelta, anchoredPos);
            var tmp = go.AddComponent<TextMeshProUGUI>();
            tmp.text = text;
            if (font != null) tmp.font = font;
            tmp.fontSize = size;
            tmp.color = color;
            tmp.alignment = align;
            tmp.raycastTarget = false;
            return go;
        }

        // ---------------------------------------------------------------- generated assets

        static TMP_FontAsset CreateCinzelFont()
        {
            string path = GEN + "/Cinzel SDF.asset";
            var existing = AssetDatabase.LoadAssetAtPath<TMP_FontAsset>(path);
            if (existing != null) return existing;

            var font = AssetDatabase.LoadAssetAtPath<Font>(ART + "/Fonts/Cinzel.ttf");
            if (font == null) throw new Exception("Missing " + ART + "/Fonts/Cinzel.ttf");

            var fa = TMP_FontAsset.CreateFontAsset(font, 64, 6, GlyphRenderMode.SDFAA, 1024, 1024,
                                                   AtlasPopulationMode.Dynamic);
            fa.name = "Cinzel SDF";
            if (TMP_Settings.defaultFontAsset != null)
                fa.fallbackFontAssetTable = new List<TMP_FontAsset> { TMP_Settings.defaultFontAsset };

            AssetDatabase.CreateAsset(fa, path);
            fa.material.name = fa.name + " Material";
            fa.atlasTextures[0].name = fa.name + " Atlas";
            AssetDatabase.AddObjectToAsset(fa.material, fa);
            AssetDatabase.AddObjectToAsset(fa.atlasTextures[0], fa);
            AssetDatabase.SaveAssets();
            return fa;
        }

        static Sprite CreateVignetteSprite()
        {
            string pngPath = GEN + "/Vignette.png";
            if (!File.Exists(pngPath))
            {
                const int S = 512;
                var tex = new Texture2D(S, S, TextureFormat.RGBA32, false);
                var px = new Color32[S * S];
                for (int y = 0; y < S; y++)
                    for (int x = 0; x < S; x++)
                    {
                        float dx = (x - S * 0.5f) / (S * 0.5f);
                        float dy = (y - S * 0.5f) / (S * 0.5f);
                        float r = Mathf.Sqrt(dx * dx + dy * dy);
                        float a = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01((r - 0.55f) / 0.65f));
                        px[y * S + x] = new Color32(0, 0, 0, (byte)(a * 255));
                    }
                tex.SetPixels32(px);
                tex.Apply();
                File.WriteAllBytes(pngPath, tex.EncodeToPNG());
                UnityEngine.Object.DestroyImmediate(tex);
                AssetDatabase.ImportAsset(pngPath);
            }
            var imp = (TextureImporter)AssetImporter.GetAtPath(pngPath);
            if (imp.textureType != TextureImporterType.Sprite)
            {
                imp.textureType = TextureImporterType.Sprite;
                imp.SaveAndReimport();
            }
            return AssetDatabase.LoadAssetAtPath<Sprite>(pngPath);
        }

        // ---------------------------------------------------------------- weapon orientation tuning

        public static void WeaponTuneBatch()
        {
            try
            {
                EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
                RenderSettings.ambientSkyColor = new Color(0.6f, 0.6f, 0.65f);
                RenderSettings.ambientEquatorColor = new Color(0.45f, 0.45f, 0.5f);
                RenderSettings.ambientGroundColor = new Color(0.3f, 0.3f, 0.32f);
                var sun = new GameObject("Sun").AddComponent<Light>();
                sun.type = LightType.Directional;
                sun.intensity = 1.1f;
                sun.transform.rotation = Quaternion.Euler(45f, 200f, 0f);   // light from camera side

                Vector3[] eulers =
                {
                    new Vector3(0,0,0),   new Vector3(90,0,0),  new Vector3(-90,0,0),
                    new Vector3(0,90,0),  new Vector3(0,-90,0), new Vector3(0,0,90), new Vector3(0,0,-90)
                };
                var idleClip = Clip("Animations/UAL1.fbx", "Sword_Idle");
                for (int i = 0; i < eulers.Length; i++)
                {
                    var root = new GameObject("Variant_" + i);
                    root.transform.position = new Vector3((i - eulers.Length / 2) * 1.4f, 0, 0);
                    var model = (GameObject)PrefabUtility.InstantiatePrefab(LoadModel("Characters/Warrior.fbx"));
                    model.transform.SetParent(root.transform, false);
                    Bounds b = RenderererSafeBounds(model);
                    model.transform.localScale *= 1.8f / b.size.y;
                    GroundModel(model, 0f);
                    model.transform.localRotation = Quaternion.Euler(0f, 180f, 0f) * model.transform.localRotation;  // as in-game
                    ApplyCharacterTexture(model, "Warrior_Texture.png", "TuneWarrior_" + i, Color.white);

                    var anim = model.GetComponent<Animator>();

                    // pose the sword idle so the arm hangs like it does in-game
                    var graph = PlayableGraph.Create("tunepose");
                    var output = AnimationPlayableOutput.Create(graph, "tunepose", anim);
                    var playable = AnimationClipPlayable.Create(graph, idleClip);
                    output.SetSourcePlayable(playable);
                    playable.SetTime(0.4);
                    graph.Evaluate(0f);
                    graph.Destroy();

                    var shield = AttachToBone(anim, HumanBodyBones.LeftHand, LoadModel("Weapons/Shield_Heater.fbx"), "Shield");
                    NormalizeWorldSize(shield, 0.8f);
                    shield.transform.localRotation = Quaternion.Euler(eulers[i]) * shield.transform.localRotation;

                    Transform mount = FindDeep(model.transform, "Weapon.R");
                    var sword = mount != null
                        ? AttachToTransform(mount, LoadModel("Weapons/Sword.fbx"), "Sword")
                        : AttachToBone(anim, HumanBodyBones.RightHand, LoadModel("Weapons/Sword.fbx"), "Sword");
                    NormalizeWorldSize(sword, 1.15f);
                }

                string outDir = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", "chatdemo3d_shots");
                Directory.CreateDirectory(outDir);
                var camGO = new GameObject("cam");
                var cam = camGO.AddComponent<Camera>();
                cam.fieldOfView = 45f;
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = new Color(0.2f, 0.2f, 0.22f);
                camGO.transform.position = new Vector3(0, 1.3f, 6.5f);    // front of the row (T-pose faces +z? verify both sides)
                camGO.transform.rotation = Quaternion.Euler(2f, 180f, 0f);
                Shoot(cam, Path.Combine(outDir, "weapon_tune_front.png"), 2400, 800);
                camGO.transform.position = new Vector3(0, 1.3f, -6.5f);
                camGO.transform.rotation = Quaternion.Euler(2f, 0f, 0f);
                Shoot(cam, Path.Combine(outDir, "weapon_tune_back.png"), 2400, 800);
                Debug.Log("[ChatDemo3DBuilder] weapon tune shots done");
                EditorApplication.Exit(0);
            }
            catch (Exception e)
            {
                Debug.LogError("[ChatDemo3DBuilder] TUNE FAILED: " + e);
                EditorApplication.Exit(1);
            }
        }

        static void Shoot(Camera cam, string path, int w, int h)
        {
            var rt = new RenderTexture(w, h, 24);
            cam.targetTexture = rt;
            cam.Render();
            RenderTexture.active = rt;
            var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
            tex.Apply();
            File.WriteAllBytes(path, tex.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(tex);
            RenderTexture.active = null;
            cam.targetTexture = null;
        }

        // ---------------------------------------------------------------- shield tuning inside the real scene (posed via the player's own animator)

        public static void ShieldTuneBatch()
        {
            try
            {
                EditorSceneManager.OpenScene(SCENE_PATH);
                var player = GameObject.Find("Player");
                var idleClip = Clip("Animations/UAL1.fbx", "Sword_Idle");

                Vector3[] eulers =
                {
                    new Vector3(0,0,0),   new Vector3(0,90,0),   new Vector3(0,180,0), new Vector3(0,270,0),
                    new Vector3(90,0,0),  new Vector3(270,0,0),  new Vector3(0,0,90),  new Vector3(0,0,270)
                };

                // bright key light so the lineup is readable
                var keyLight = new GameObject("tunelight").AddComponent<Light>();
                keyLight.type = LightType.Directional;
                keyLight.intensity = 1.3f;
                keyLight.transform.rotation = Quaternion.Euler(40f, 160f, 0f);

                for (int i = 0; i < eulers.Length; i++)
                {
                    var clone = UnityEngine.Object.Instantiate(player);
                    clone.name = "ShieldVariant_" + i;
                    clone.transform.position = new Vector3((i - eulers.Length / 2) * 1.6f, 0.1f, -4f);
                    clone.transform.rotation = Quaternion.identity;

                    var shield = FindDeep(clone.transform, "Shield");
                    if (shield != null)
                    {
                        // strip the builder's baseline (0,180,0) then apply this variant
                        Quaternion original = Quaternion.Inverse(Quaternion.Euler(0f, 180f, 0f)) * shield.localRotation;
                        shield.localRotation = Quaternion.Euler(eulers[i]) * original;
                    }

                    var anim = clone.GetComponentInChildren<Animator>();
                    var graph = PlayableGraph.Create("shieldpose");
                    var output = AnimationPlayableOutput.Create(graph, "shieldpose", anim);
                    var playable = AnimationClipPlayable.Create(graph, idleClip);
                    output.SetSourcePlayable(playable);
                    playable.SetTime(0.4);
                    graph.Evaluate(0f);
                    graph.Destroy();
                }
                player.SetActive(false);

                string outDir = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", "chatdemo3d_shots");
                Directory.CreateDirectory(outDir);
                var camGO = new GameObject("tunecam");
                var cam = camGO.AddComponent<Camera>();
                cam.fieldOfView = 42f;
                camGO.transform.position = new Vector3(0, 1.4f, 4.5f);     // front of the row
                camGO.transform.rotation = Quaternion.Euler(3f, 180f, 0f);
                Shoot(cam, Path.Combine(outDir, "shield_tune_front.png"), 2400, 800);
                camGO.transform.position = new Vector3(-7f, 1.4f, -10f);   // 3/4 from behind-left
                camGO.transform.rotation = Quaternion.Euler(4f, 50f, 0f);
                Shoot(cam, Path.Combine(outDir, "shield_tune_side.png"), 2400, 800);
                Debug.Log("[ChatDemo3DBuilder] shield tune shots done");
                EditorApplication.Exit(0);
            }
            catch (Exception e)
            {
                Debug.LogError("[ChatDemo3DBuilder] SHIELD TUNE FAILED: " + e);
                EditorApplication.Exit(1);
            }
        }

        // ---------------------------------------------------------------- UI probe (screen-space canvas isn't visible to scene cameras)

        public static void UiProbeBatch()
        {
            try
            {
                EditorSceneManager.OpenScene(SCENE_PATH);
                var canvas = UnityEngine.Object.FindObjectOfType<Canvas>();
                var cam = new GameObject("uicam").AddComponent<Camera>();
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = new Color(0.13f, 0.13f, 0.16f);
                cam.transform.position = new Vector3(0f, 120f, 0f);   // empty sky: nothing can poke through the canvas plane
                canvas.renderMode = RenderMode.ScreenSpaceCamera;
                canvas.worldCamera = cam;
                canvas.planeDistance = 1f;

                var win = UnityEngine.Object.FindObjectOfType<SoulsChatWindow>(true);
                win.gameObject.SetActive(true);
                win.SetTitle("Velmire, the Pale Herald");
                win.SetInfoText("");
                win.AddMessage("You", "Who are you?");
                win.AddMessage("Velmire, the Pale Herald",
                    "Ah... another lambkin strays to my gate. How delightfully lost you look, poor wanderer — guideless, lordless, and so very far from any warm hearth.");

                foreach (var t in canvas.GetComponentsInChildren<Transform>(true))
                    if (t.name == "InteractPrompt") t.gameObject.SetActive(true);

                Canvas.ForceUpdateCanvases();
                string outDir = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", "chatdemo3d_shots");
                Directory.CreateDirectory(outDir);
                Shoot(cam, Path.Combine(outDir, "ui_probe.png"), 1920, 1080);
                Debug.Log("[ChatDemo3DBuilder] ui probe done");
                EditorApplication.Exit(0);
            }
            catch (Exception e)
            {
                Debug.LogError("[ChatDemo3DBuilder] UI PROBE FAILED: " + e);
                EditorApplication.Exit(1);
            }
        }

        // ---------------------------------------------------------------- scene audit

        public static void AuditBatch()
        {
            try
            {
                EditorSceneManager.OpenScene(SCENE_PATH);
                var sb = new System.Text.StringBuilder("[SceneAudit]\n");
                foreach (var r in UnityEngine.Object.FindObjectsOfType<Renderer>())
                {
                    Bounds b = r.bounds;
                    Transform top = r.transform;
                    while (top.parent != null && top.parent.name != "Environment" && top.parent.parent != null) top = top.parent;
                    sb.AppendLine($"{top.name}/{r.name} | pos {r.transform.position} | center {b.center} | size {b.size} | lossyScale {r.transform.lossyScale.x:0.###}");
                }
                File.WriteAllText("ProbeLogs/chatdemo3d_audit.txt", sb.ToString());
                Debug.Log("[ChatDemo3DBuilder] audit written");
                EditorApplication.Exit(0);
            }
            catch (Exception e)
            {
                Debug.LogError("[ChatDemo3DBuilder] AUDIT FAILED: " + e);
                EditorApplication.Exit(1);
            }
        }

        // ---------------------------------------------------------------- screenshot probe

        public static void ScreenshotBatch()
        {
            try
            {
                EditorSceneManager.OpenScene(SCENE_PATH);
                string outDir = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", "chatdemo3d_shots");
                Directory.CreateDirectory(outDir);

                // pose the characters so they aren't T-posing in the shots
                // (humanoid clips need a playable graph — SampleAnimation only drives generic rigs)
                foreach (var anim in UnityEngine.Object.FindObjectsOfType<Animator>())
                {
                    var ctrl = anim.runtimeAnimatorController as AnimatorController;
                    var st = ctrl != null ? ctrl.layers[0].stateMachine.defaultState : null;
                    if (st == null || !(st.motion is AnimationClip clip)) continue;

                    var graph = PlayableGraph.Create("pose");
                    var output = AnimationPlayableOutput.Create(graph, "pose", anim);
                    var playable = AnimationClipPlayable.Create(graph, clip);
                    output.SetSourcePlayable(playable);
                    playable.SetTime(0.4);
                    graph.Evaluate(0f);
                    graph.Destroy();
                }

                var mist = GameObject.Find("MistDoor");
                Vector3 mp = mist != null ? mist.transform.position : Vector3.zero;

                var shots = new (string name, Vector3 pos, Vector3 euler)[]
                {
                    ("overview",   new Vector3(0, 24, -34),       new Vector3(34, 0, 0)),
                    ("playerview", new Vector3(0, 2.0f, -9.6f),   new Vector3(6, 0, 0)),
                    ("npc_closeup",new Vector3(4.0f, 1.7f, 2.6f), new Vector3(6, 25, 0)),
                    ("gate",       new Vector3(0, 1.9f, -5f),     new Vector3(4, 180, 0)),
                    ("knight_back",new Vector3(0.7f, 2.2f, -11.4f), new Vector3(12, -4, 0)),
                    ("knight_front",new Vector3(-1.4f, 1.5f, -6.6f), new Vector3(6, 148, 0)),
                    ("moon",       new Vector3(0, 3f, -5f),         new Vector3(-24, 25, 0)),
                    ("mistdoor",   mp + new Vector3(0, 1.9f, -5f),  new Vector3(3, 0, 0)),
                    ("bossroom",   mp + new Vector3(0, 2.2f, 2.2f), new Vector3(4, 0, 0)),
                    ("boss_wide",  mp + new Vector3(0, 13f, 16f),   new Vector3(44, 180, 0)),
                    ("sentinel",      mp + new Vector3(0, 2.4f, 5.2f),  new Vector3(4, 0, 0)),
                    ("sentinel_side", mp + new Vector3(2.6f, 1.9f, 8.0f), new Vector3(10, -62, 0)),
                };

                var camGO = new GameObject("ProbeCamera");
                var cam = camGO.AddComponent<Camera>();
                cam.fieldOfView = 55f;
                cam.nearClipPlane = 0.05f;
                cam.farClipPlane = 500f;

                var rt = new RenderTexture(1600, 900, 24);
                foreach (var (name, pos, euler) in shots)
                {
                    camGO.transform.position = pos;
                    camGO.transform.rotation = Quaternion.Euler(euler);
                    cam.targetTexture = rt;
                    cam.Render();
                    RenderTexture.active = rt;
                    var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
                    tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                    tex.Apply();
                    File.WriteAllBytes(Path.Combine(outDir, name + ".png"), tex.EncodeToPNG());
                    UnityEngine.Object.DestroyImmediate(tex);
                    RenderTexture.active = null;
                }
                Debug.Log("[ChatDemo3DBuilder] screenshots -> " + outDir);
                EditorApplication.Exit(0);
            }
            catch (Exception e)
            {
                Debug.LogError("[ChatDemo3DBuilder] SCREENSHOT FAILED: " + e);
                EditorApplication.Exit(1);
            }
        }
    }
}

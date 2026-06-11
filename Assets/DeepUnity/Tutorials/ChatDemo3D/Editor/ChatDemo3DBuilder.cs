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
            GameObject npc = BuildNpc(npcCtrl);
            GameObject boss = BuildBoss(bossCtrl, bossSwings);

            BuildUI(cinzel, vignette, npc, player);

            // ambient exploration music, quiet and looping; streamed — it's a 6-minute track
            var audioImp = AssetImporter.GetAtPath(ART + "/Audio/limgrave_theme.ogg") as AudioImporter;
            if (audioImp != null)
            {
                var sampleSettings = audioImp.defaultSampleSettings;
                sampleSettings.loadType = AudioClipLoadType.Streaming;
                audioImp.defaultSampleSettings = sampleSettings;
                audioImp.SaveAndReimport();
            }
            var ambience = new GameObject("Ambience").AddComponent<AudioSource>();
            ambience.clip = AssetDatabase.LoadAssetAtPath<AudioClip>(ART + "/Audio/limgrave_theme.ogg");
            ambience.loop = true;
            ambience.playOnAwake = true;
            ambience.volume = 0.3f;
            ambience.spatialBlend = 0f;

            // LLM frame-pacing helpers: compile the compute kernels at scene start (one visible
            // hitch here instead of mid-game on first chat open) + record any slow frame with the
            // LLM phase active at the time (ProbeLogs/frame_spikes.csv) so dips get attributed.
            var llmHelper = new GameObject("LLMBootHelper");
            llmHelper.AddComponent<LLMPrewarm>();
            llmHelper.AddComponent<FrameSpikeProbe>();

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
            RenderSettings.fogColor = new Color(0.07f, 0.08f, 0.12f);
            RenderSettings.fogDensity = 0.013f;
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
                const float REGION = 31.7f;   // noise-space size of the tile

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
                ("Cart",            new Vector3(hx - L * 1.3f, 0, -hz + L * 1.2f), 250f),
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

            // the fog gate glows golden onto the surrounding stone
            var glow = new GameObject("MistGlow").AddComponent<Light>();
            glow.transform.SetParent(root.transform, false);
            glow.transform.localPosition = new Vector3(0f, 1.7f, 0f);
            glow.type = LightType.Point;
            glow.color = new Color(1f, 0.82f, 0.45f);
            glow.intensity = 1.7f;
            glow.range = 5.5f;
            glow.shadows = LightShadows.None;

            var layers = new List<Renderer>();
            for (int i = 0; i < 2; i++)
            {
                var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
                UnityEngine.Object.DestroyImmediate(quad.GetComponent<Collider>());
                quad.name = "MistLayer" + i;
                quad.transform.SetParent(root.transform, false);
                quad.transform.localPosition = new Vector3(0f, 1.55f, (i - 0.5f) * 0.16f);
                quad.transform.localScale = new Vector3(2.7f, 3.1f, 1f);
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
                if (state == "Idle") sm.defaultState = st;
            }
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

            root.AddComponent<BreathingIdle>();

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

        static GameObject BuildNpc(RuntimeAnimatorController ctrl)
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
            return root;
        }

        // ---------------------------------------------------------------- UI

        static GameObject s_mistPrompt;
        static Image s_whiteFlash;
        static RectTransform s_bossFill;
        static CanvasGroup s_bossBarGroup;
        static DeathScreen s_deathScreen;

        static void BuildUI(TMP_FontAsset cinzel, Sprite vignette, GameObject npcGO, GameObject playerGO)
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

            var npc = npcGO.GetComponent<NPCInteractor3D>();
            SetRef(npc, "chatWindow", win);
            SetRef(npc, "interactPrompt", promptGO);

            UnityEventTools.AddPersistentListener(sendBtn.onClick, new UnityAction(npc.AskNPC));
            UnityEventTools.AddPersistentListener(leaveBtn.onClick, new UnityAction(npc.CloseInteraction));
            UnityEventTools.AddPersistentListener(sendBtn.onClick, new UnityAction(win.PlayButtonClick));
            UnityEventTools.AddPersistentListener(leaveBtn.onClick, new UnityAction(win.PlayButtonClick));
            UnityEventTools.AddVoidPersistentListener(inputField.onSubmit, new UnityAction(npc.AskNPC));

            BuildHud(canvasGO.transform, gold, win, playerGO);
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

            // --- quick slots, bottom-left (spell / left hand / right hand / item)
            var slots = MakeRect("QuickSlots", hudGO.transform, Vector2.zero, Vector2.zero, new Vector2(240, 240), new Vector2(150, 150));
            (string name, Vector2 pos, string asset, string icon)[] slotDefs =
            {
                ("SlotSpell", new Vector2(0,  68), "Ruins/Torch.fbx",          "torch"),
                ("SlotLeft",  new Vector2(-68, 0), "Weapons/Shield_Heater.fbx","shield"),
                ("SlotRight", new Vector2(68,  0), "Weapons/Sword.fbx",        "sword"),
                ("SlotItem",  new Vector2(0, -68), "Ruins/Pot1.fbx",           "pot"),
            };
            GameObject itemSlot = null;
            foreach (var (name, pos, asset, icon) in slotDefs)
            {
                var slot = MakeRect(name, slots.transform, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(64, 64), pos);
                var bg = slot.AddComponent<Image>(); bg.color = new Color(0.05f, 0.05f, 0.06f, 0.82f); bg.raycastTarget = false;
                AddThinBorder(slot.transform, frame);
                Sprite sprite = RenderItemIcon(asset, icon);
                if (sprite != null)
                {
                    var iconGO = MakeRect("Icon", slot.transform, Vector2.zero, Vector2.one, new Vector2(-10, -10), Vector2.zero);
                    var img = iconGO.AddComponent<Image>(); img.sprite = sprite; img.preserveAspect = true; img.raycastTarget = false;
                }
                if (name == "SlotItem") itemSlot = slot;
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
            field.caretColor = new Color(0.77f, 0.66f, 0.42f);
            field.customCaretColor = true;
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

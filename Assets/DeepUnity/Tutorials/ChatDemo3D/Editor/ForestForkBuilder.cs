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
using UnityEngine.Events;
using UnityEngine.EventSystems;
using UnityEngine.TextCore.LowLevel;
using UnityEngine.UI;

namespace DeepUnity.Tutorials.ChatDemo3D.EditorTools
{
    /// <summary>
    /// Deterministically builds the ForestFork scene: a bright daytime forest road that splits
    /// into a Y a little way ahead of the player. A beggar (Cobb) sits slumped on a crate at the
    /// mouth of the fork — he knows the RIGHT path is safe and the LEFT sinks into a mire, but
    /// he is cagey about it and wants a kindness first. Same LLM+Kokoro dialogue stack as
    /// ChatDemo3D (NPCInteractor3D + SoulsChatWindow), built from the same CC0 Quaternius art.
    /// Run from the menu (DeepUnity/Build ForestFork Scene) or in batch mode via
    /// -executeMethod DeepUnity.Tutorials.ChatDemo3D.EditorTools.ForestForkBuilder.BuildBatch
    /// </summary>
    public static class ForestForkBuilder
    {
        const string ROOT = "Assets/DeepUnity/Tutorials/ChatDemo3D";
        const string ART = ROOT + "/Art";
        const string GEN = ROOT + "/Generated";
        const string SCENE_PATH = ROOT + "/ForestFork.unity";

        static readonly System.Random rng = new System.Random(20260711);

        // ---------------------------------------------------------------- fork geometry
        // Player spawns at the origin facing +Z. The road runs straight to the fork point,
        // then splits 20 deg left / 20 deg right (40 deg total) out to the treeline.
        const float FORK_Z = 14f;
        const float BRANCH_ANGLE = 20f;
        const float BRANCH_LEN = 42f;
        static readonly Vector3 FORK = new Vector3(0f, 0f, FORK_Z);
        static readonly Vector3 PATH_START = new Vector3(0f, 0f, -10f);
        static readonly Vector3 L_DIR = Quaternion.Euler(0f, -BRANCH_ANGLE, 0f) * Vector3.forward;
        static readonly Vector3 R_DIR = Quaternion.Euler(0f, +BRANCH_ANGLE, 0f) * Vector3.forward;
        // beggar slumped just off the right edge of the road, at the fork mouth
        static readonly Vector3 BEGGAR_POS = new Vector3(2.4f, 0f, 11.5f);
        static readonly Vector3 PLAYER_SPAWN = new Vector3(0f, 0.1f, 0f);

        // ---------------------------------------------------------------- entry points

        [MenuItem("DeepUnity/Build ForestFork Scene")]
        public static void BuildMenu()
        {
            ConfigureImports();
            BuildEverything();
            Debug.Log("[ForestForkBuilder] Done. Scene at " + SCENE_PATH);
        }

        public static void BuildBatch()
        {
            try
            {
                ConfigureImports();
                BuildEverything();
                Debug.Log("[ForestForkBuilder] BATCH OK");
                EditorApplication.Exit(0);
            }
            catch (Exception e)
            {
                Debug.LogError("[ForestForkBuilder] BATCH FAILED: " + e);
                EditorApplication.Exit(1);
            }
        }

        // ---------------------------------------------------------------- import configuration

        static void ConfigureImports()
        {
            // humanoid characters + animation libraries; skipped when already correctly imported
            EnsureHumanoid(ART + "/Characters/Warrior.fbx", rotateClips180: false);
            EnsureHumanoid(ART + "/Animations/UAL1.fbx", rotateClips180: true);
            EnsureHumanoid(ART + "/Animations/UAL2.fbx", rotateClips180: true);

            // Nov-2020 RPG pack beggar (Rogue.fbx): its feet are IK targets OUTSIDE the leg
            // chain, so humanoid retarget is impossible — import GENERIC and play its OWN
            // embedded clips (Idle/Dagger_Attack/Death/...) instead of UAL retargets
            ConfigureGenericAnimated(ART + "/Characters/Rogue.fbx");

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

        // generic-rig import for the animated RPG-pack characters (own clips, no retargeting) —
        // mirrors ChatDemo3DBuilder.ConfigureGenericAnimated so both builders agree on Rogue.fbx
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
                c.loopTime = clean.Contains("Idle") || clean.StartsWith("Spell") || clean == "Walk" || clean == "Run";
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

        static void EnsureHumanoid(string path, bool rotateClips180)
        {
            var imp = AssetImporter.GetAtPath(path) as ModelImporter;
            if (imp == null) throw new Exception("Missing model: " + path);
            if (imp.animationType == ModelImporterAnimationType.Human)
            {
                var av = AssetDatabase.LoadAllAssetsAtPath(path).OfType<Avatar>().FirstOrDefault();
                if (av != null && av.isValid && av.isHuman) return;   // already good — skip the slow reimport
            }
            ConfigureHumanoid(path, rotateClips180);
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
            Debug.Log($"[ForestForkBuilder] {Path.GetFileName(path)} mapped {human.Count} bones" +
                      (unmatched.Count > 0 ? ", unmatched: " + string.Join(",", unmatched) : ""));
            imp.humanDescription = new HumanDescription
            {
                human = human.ToArray(),
                skeleton = new SkeletonBone[0],   // empty = use the model's own skeleton / bind pose
                upperArmTwist = 0.5f, lowerArmTwist = 0.5f,
                upperLegTwist = 0.5f, lowerLegTwist = 0.5f,
                armStretch = 0.05f, legStretch = 0.05f,
                feetSpacing = 0f, hasTranslationDoF = false,
            };

            // strip the "Armature|" take prefixes and mark looping clips
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
            Debug.Log($"[ForestForkBuilder] Humanoid OK: {path}");
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

        static Bounds RenderererSafeBounds(GameObject go)
        {
            var rs = go.GetComponentsInChildren<Renderer>();
            if (rs.Length == 0) return new Bounds(go.transform.position, Vector3.zero);
            Bounds b = rs[0].bounds;
            foreach (var r in rs.Skip(1)) b.Encapsulate(r.bounds);
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

        static void SetString(Component c, string field, string value)
        {
            var so = new SerializedObject(c);
            var prop = so.FindProperty(field);
            if (prop == null) throw new Exception($"No serialized field '{field}' on {c.GetType().Name}");
            prop.stringValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        static float Range(float min, float max) => min + (float)rng.NextDouble() * (max - min);

        // ---------------------------------------------------------------- build

        static Transform envRoot;

        static void BuildEverything()
        {
            if (!AssetDatabase.IsValidFolder(GEN))
                AssetDatabase.CreateFolder(ROOT, "Generated");

            var cinzel = CreateCinzelFont();
            var playerCtrl = CreatePlayerAnimator();
            var beggarCtrl = CreateBeggarAnimator();

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            envRoot = new GameObject("Environment").transform;

            SetupLightingAndSky();
            BuildGround();
            BuildPath();
            BuildForest();
            BuildBeggarCamp();

            GameObject player = BuildPlayer(playerCtrl);
            GameObject cameraRig = BuildCamera(player);
            GameObject beggar = BuildBeggarNpc(beggarCtrl);

            BuildUI(cinzel, beggar);

            // LLM frame-pacing helpers: compile the compute kernels at scene start (one visible
            // hitch here instead of mid-game on first chat open) + a dormant frame-spike probe
            var llmHelper = new GameObject("LLMBootHelper");
            // kernel prewarm is automatic in NPCChatBase.Awake now — only the dormant spike probe
            llmHelper.AddComponent<FrameSpikeProbe>();

            // final cross-wiring
            SetRef(player.GetComponent<SoulsPlayerController>(), "cam", cameraRig.GetComponent<SoulsCameraRig>());
            SetRef(cameraRig.GetComponent<SoulsCameraRig>(), "target", player.transform);

            EditorSceneManager.SaveScene(scene, SCENE_PATH);
            AssetDatabase.SaveAssets();
            Debug.Log("[ForestForkBuilder] Scene saved (" + SCENE_PATH + ")");
        }

        // ---------------------------------------------------------------- lighting / mood

        static void SetupLightingAndSky()
        {
            // bright procedural day sky
            var sky = new Material(Shader.Find("Skybox/Procedural"));
            sky.SetFloat("_SunSize", 0.045f);
            sky.SetFloat("_SunSizeConvergence", 5f);
            sky.SetFloat("_AtmosphereThickness", 0.95f);
            sky.SetColor("_SkyTint", new Color(0.50f, 0.55f, 0.62f));
            sky.SetColor("_GroundColor", new Color(0.42f, 0.46f, 0.40f));
            sky.SetFloat("_Exposure", 1.25f);
            AssetDatabase.CreateAsset(sky, GEN + "/ForkSkyDay.mat");

            var sunGO = new GameObject("Sun");
            var sun = sunGO.AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.color = new Color(1.0f, 0.956f, 0.87f);      // warm midday white
            sun.intensity = 1.25f;
            sun.shadows = LightShadows.Soft;
            sun.shadowStrength = 0.82f;
            // ~50 deg elevation, hanging over the fork so the walk toward it is lit and the
            // sun disk sits in the sky ahead of the player
            sunGO.transform.rotation = Quaternion.Euler(50f, 200f, 0f);

            RenderSettings.skybox = sky;
            RenderSettings.sun = sun;
            RenderSettings.fog = true;                        // light haze for forest depth
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogColor = new Color(0.74f, 0.80f, 0.86f);
            RenderSettings.fogDensity = 0.006f;
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.62f, 0.68f, 0.78f);
            RenderSettings.ambientEquatorColor = new Color(0.46f, 0.50f, 0.44f);
            RenderSettings.ambientGroundColor = new Color(0.26f, 0.29f, 0.23f);

            // no GI baking — everything realtime so the build needs no bake step
            var ls = new LightingSettings { bakedGI = false, realtimeGI = false };
            ls.name = "ForestFork LightingSettings";
            AssetDatabase.CreateAsset(ls, GEN + "/ForestFork.lighting");
            Lightmapping.lightingSettings = ls;
        }

        // ---------------------------------------------------------------- terrain

        // distance from (x,z) to the road polyline (straight + two branches)
        static float SegDist(float px, float pz, Vector3 a, Vector3 b)
        {
            float abx = b.x - a.x, abz = b.z - a.z;
            float t = ((px - a.x) * abx + (pz - a.z) * abz) / (abx * abx + abz * abz);
            t = Mathf.Clamp01(t);
            float dx = px - (a.x + abx * t), dz = pz - (a.z + abz * t);
            return Mathf.Sqrt(dx * dx + dz * dz);
        }

        static float DistToPath(float x, float z) => Mathf.Min(
            SegDist(x, z, PATH_START, FORK),
            Mathf.Min(SegDist(x, z, FORK, FORK + L_DIR * BRANCH_LEN),
                      SegDist(x, z, FORK, FORK + R_DIR * BRANCH_LEN)));

        // dead flat along the road corridor, gentle perlin hills out in the woods
        static float GroundHeight(float x, float z)
        {
            float d = DistToPath(x, z);
            float blend = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01((d - 5f) / 14f));
            float n = Mathf.PerlinNoise(x * 0.020f + 41.7f, z * 0.020f + 9.3f) * 1.9f
                    + Mathf.PerlinNoise(x * 0.085f + 211f, z * 0.085f + 137f) * 0.5f;
            return (n - 1.1f) * blend;
        }

        static void BuildGround()
        {
            var mat = new Material(Shader.Find("Standard"));
            mat.mainTexture = CreateGrassTexture();
            mat.mainTextureScale = Vector2.one;              // tiling lives in the mesh UVs (world/7)
            mat.color = Color.white;
            mat.SetFloat("_Glossiness", 0.04f);
            AssetDatabase.CreateAsset(mat, GEN + "/ForkGround.mat");

            const int N = 140;
            const float SIZE = 300f;
            const float STEP = SIZE / N;
            var verts = new Vector3[(N + 1) * (N + 1)];
            var uvs = new Vector2[verts.Length];
            for (int z = 0; z <= N; z++)
                for (int x = 0; x <= N; x++)
                {
                    float wx = -SIZE * 0.5f + x * STEP;
                    float wz = -SIZE * 0.35f + z * STEP;     // shifted: more world ahead of the player
                    int i = z * (N + 1) + x;
                    verts[i] = new Vector3(wx, GroundHeight(wx, wz), wz);
                    uvs[i] = new Vector2(wx / 7f, wz / 7f);
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
            var mesh = new Mesh { name = "ForkGroundMesh", indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            mesh.vertices = verts;
            mesh.uv = uvs;
            mesh.triangles = tris;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            AssetDatabase.CreateAsset(mesh, GEN + "/ForkGroundMesh.asset");

            var ground = new GameObject("Ground");
            ground.transform.SetParent(envRoot, false);
            ground.AddComponent<MeshFilter>().sharedMesh = mesh;
            ground.AddComponent<MeshRenderer>().sharedMaterial = mat;
            ground.AddComponent<MeshCollider>().sharedMesh = mesh;
            ground.isStatic = true;
        }

        // tileable grass blend (sunlit green patches, darker shade, dirt flecks)
        static Texture2D CreateGrassTexture()
        {
            string pngPath = GEN + "/ForkGrass.png";
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

                Color grassLight = new Color(0.32f, 0.45f, 0.21f);
                Color grassDark = new Color(0.22f, 0.34f, 0.16f);
                Color dirtFleck = new Color(0.35f, 0.30f, 0.21f);

                var tex = new Texture2D(S, S, TextureFormat.RGB24, false);
                var px = new Color[S * S];
                for (int y = 0; y < S; y++)
                    for (int x = 0; x < S; x++)
                    {
                        float u = (float)x / S, v = (float)y / S;
                        float patches = TileableNoise(u, v, 5f, 17.31f) * 0.65f + TileableNoise(u, v, 13f, 57.7f) * 0.35f;
                        float flecks = TileableNoise(u, v, 29f, 93.1f);
                        float micro = TileableNoise(u, v, 53f, 3.9f);

                        Color c = Color.Lerp(grassDark, grassLight, Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(0.36f, 0.64f, patches)));
                        c = Color.Lerp(c, dirtFleck, Mathf.InverseLerp(0.78f, 0.97f, flecks) * 0.5f);
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

        // ---------------------------------------------------------------- the dirt road

        static void BuildPath()
        {
            var dirtA = MatAsset("ForkDirtA.mat", new Color(0.42f, 0.33f, 0.23f), 0f, 0.05f);
            var dirtB = MatAsset("ForkDirtB.mat", new Color(0.38f, 0.29f, 0.20f), 0f, 0.05f);
            var mats = new[] { dirtA, dirtA, dirtB };

            var pathRoot = new GameObject("DirtPath").transform;
            pathRoot.SetParent(envRoot, false);

            // overlapping flattened cubes with jittered yaw/width/height read as a worn dirt
            // road from eye level; the height jitter keeps overlapping tops from z-fighting
            void LaySegment(Vector3 from, Vector3 to, float width)
            {
                Vector3 dir = to - from;
                float len = dir.magnitude;
                dir /= len;
                float yaw = Mathf.Atan2(dir.x, dir.z) * Mathf.Rad2Deg;
                for (float t = 0f; t <= len; t += 1.1f)
                {
                    Vector3 p = from + dir * t;
                    var piece = GameObject.CreatePrimitive(PrimitiveType.Cube);   // keeps its BoxCollider
                    piece.name = "PathPiece";
                    piece.transform.SetParent(pathRoot, false);
                    piece.transform.position = new Vector3(p.x, 0f, p.z);
                    piece.transform.rotation = Quaternion.Euler(0f, yaw + Range(-5f, 5f), 0f);
                    piece.transform.localScale = new Vector3(width + Range(-0.25f, 0.25f),
                                                             0.09f + Range(0f, 0.03f), 1.5f);
                    piece.GetComponent<MeshRenderer>().sharedMaterial = mats[rng.Next(mats.Length)];
                    piece.isStatic = true;
                }
            }

            LaySegment(PATH_START, FORK, 3.1f);
            LaySegment(FORK, FORK + L_DIR * BRANCH_LEN, 2.7f);
            LaySegment(FORK, FORK + R_DIR * BRANCH_LEN, 2.7f);

            // widened junction pad where the road splits, so the Y mouth reads clean
            var pad = GameObject.CreatePrimitive(PrimitiveType.Cube);
            pad.name = "ForkPad";
            pad.transform.SetParent(pathRoot, false);
            pad.transform.position = new Vector3(0f, 0f, FORK_Z + 0.8f);
            pad.transform.localScale = new Vector3(6.0f, 0.135f, 3.8f);
            pad.GetComponent<MeshRenderer>().sharedMaterial = dirtA;
            pad.isStatic = true;
        }

        // ---------------------------------------------------------------- forest

        static Material forkFoliageMat;
        static Material ForkFoliageMat()
        {
            if (forkFoliageMat != null) return forkFoliageMat;
            string path = GEN + "/ForkFoliage.mat";
            forkFoliageMat = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (forkFoliageMat == null)
            {
                forkFoliageMat = new Material(Shader.Find("Standard"));
                AssetDatabase.CreateAsset(forkFoliageMat, path);
            }
            forkFoliageMat.color = new Color(0.30f, 0.44f, 0.20f);   // sunlit summer foliage
            forkFoliageMat.SetFloat("_Glossiness", 0.03f);
            return forkFoliageMat;
        }

        static readonly string[] ALIVE = { "Tree_1", "Tree_2", "Tree_3" };
        static readonly string[] DEAD = { "DeadTree_1", "DeadTree_2", "DeadTree_3" };
        static readonly string[] BRUSH = { "Bush_1x1", "Bush_Round", "Bush_Large", "Bush_2x1", "Grass" };

        static void BuildForest()
        {
            var forest = new GameObject("Forest").transform;
            forest.SetParent(envRoot, false);

            bool NearBeggar(float x, float z) =>
                (new Vector2(x, z) - new Vector2(BEGGAR_POS.x, BEGGAR_POS.z)).sqrMagnitude < 9f;

            // deliberate tree lines hugging both edges of every road segment — this is what
            // makes the corridor read as a forest road at eye level
            void LineSegment(Vector3 a, Vector3 b)
            {
                Vector3 dir = (b - a).normalized;
                Vector3 perp = new Vector3(dir.z, 0f, -dir.x);
                float len = (b - a).magnitude;
                for (float t = 2.5f; t < len; t += 3.8f)
                    for (int s = -1; s <= 1; s += 2)
                    {
                        if (rng.NextDouble() < 0.18) continue;   // organic gaps
                        Vector3 p = a + dir * (t + Range(-1.2f, 1.2f)) + perp * (s * (5.0f + Range(0f, 2.2f)));
                        if (DistToPath(p.x, p.z) < 4.0f) continue;
                        if (NearBeggar(p.x, p.z)) continue;
                        PlacePiece(ALIVE[rng.Next(ALIVE.Length)],
                                   new Vector3(p.x, GroundHeight(p.x, p.z), p.z), Range(0f, 360f),
                                   forest, scale: Range(1.0f, 1.45f), collider: true);
                    }
            }
            LineSegment(PATH_START, FORK);
            LineSegment(FORK, FORK + L_DIR * BRANCH_LEN);
            LineSegment(FORK, FORK + R_DIR * BRANCH_LEN);

            // broad random scatter filling in the woods behind the tree lines
            int placed = 0, guard = 0;
            while (placed < 210 && guard++ < 6000)
            {
                float x = Range(-65f, 65f), z = Range(-25f, 68f);
                double roll = rng.NextDouble();
                bool isTree = roll < 0.70;
                float clearance = isTree ? 4.2f : 3.4f;      // brush may creep closer to the road
                if (DistToPath(x, z) < clearance) continue;
                if (NearBeggar(x, z)) continue;

                string piece = roll < 0.55 ? ALIVE[rng.Next(ALIVE.Length)]
                             : roll < 0.70 ? DEAD[rng.Next(DEAD.Length)]
                             : BRUSH[rng.Next(BRUSH.Length)];

                // trees grow toward the horizon so the treeline looms without walling the road
                float far = Mathf.Clamp01((DistToPath(x, z) - 7f) / 30f);
                float scale = Range(0.9f, 1.4f) * (isTree ? Mathf.Lerp(1.0f, 1.9f, far) : 1f);
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
            // a -90 deg axis-correction rotation into the root; overwriting either breaks the piece
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
            // plain white. Swap anything leaf-like for the sunlit foliage material.
            foreach (var r in go.GetComponentsInChildren<Renderer>())
            {
                var mats = r.sharedMaterials;
                bool changed = false;
                for (int i = 0; i < mats.Length; i++)
                    if (mats[i] != null && (mats[i].name.Contains("Leaves") || mats[i].name == "Green"))
                    {
                        mats[i] = ForkFoliageMat();
                        changed = true;
                    }
                if (changed) r.sharedMaterials = mats;
            }

            SetStaticRecursive(go);
            return go;
        }

        static void SetStaticRecursive(GameObject go)
        {
            go.isStatic = true;
            foreach (Transform t in go.GetComponentsInChildren<Transform>())
                t.gameObject.isStatic = true;
        }

        // ---------------------------------------------------------------- beggar camp set dressing

        static void BuildBeggarCamp()
        {
            var camp = new GameObject("BeggarCamp").transform;
            camp.SetParent(envRoot, false);

            Vector3 toSpawn = PLAYER_SPAWN - BEGGAR_POS; toSpawn.y = 0f;
            Vector3 fwd = toSpawn.normalized;                       // beggar faces the approaching player
            Vector3 right = Vector3.Cross(Vector3.up, fwd);
            float yaw = Quaternion.LookRotation(fwd).eulerAngles.y;

            // his worldly possessions beside him (sitting IN the crate looked comic — he now
            // stands hunched at the road edge instead)
            PlacePiece("Crate", BEGGAR_POS + right * 1.1f - fwd * 0.2f, yaw + 15f, camp, collider: true);
            // battered begging bowl in front of him — the Pot1 ruin piece is amphora-sized at
            // native scale, so shrink it to bowl proportions and tuck it beside his knee
            var bowl = PlacePiece("Pot1", BEGGAR_POS + fwd * 0.55f + right * 0.45f, Range(0f, 360f), camp, collider: false);
            if (bowl != null) bowl.transform.localScale *= 0.35f;
            // a little roadside clutter framing his spot
            PlacePiece("Bush_1x1", BEGGAR_POS - fwd * 1.6f + right * 1.0f, Range(0f, 360f), camp, collider: false);
            PlacePiece("Grass", BEGGAR_POS + right * 1.3f, Range(0f, 360f), camp, collider: false);
            // keep the dead tree OFF the spawn->beggar sight line — 3 m dead-behind him its pale
            // branches read as a spike sticking out of his head from the road
            PlacePiece("DeadTree_2", BEGGAR_POS - fwd * 5.5f - right * 3.0f, Range(0f, 360f), camp, collider: true);
        }

        // ---------------------------------------------------------------- animators

        static RuntimeAnimatorController CreatePlayerAnimator()
        {
            string path = GEN + "/ForkPlayerAnimator.controller";
            AssetDatabase.DeleteAsset(path);   // fresh state machine on every rebuild
            var ctrl = AnimatorController.CreateAnimatorControllerAtPath(path);
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
                ("Interact",  "Animations/UAL1.fbx", "Interact"),
                ("MistWalk",  "Animations/UAL1.fbx", "Push_Loop"),      // unused here, but the controller keeps parity
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
                if (state == "Idle") sm.defaultState = st;
            }
            return ctrl;
        }

        // two-state beggar from the GENERIC Rogue rig's OWN embedded clips (UAL retargets are
        // impossible on this rig — its feet are IK targets outside the leg chain). Both states
        // play the looping "Idle": the slump comes from tilting the model back against the
        // crate, and while he talks the Kokoro voice + chat stream carry the performance
        // (the head-nod degrades gracefully: GetBoneTransform(Head) is null on generic rigs).
        static RuntimeAnimatorController CreateBeggarAnimator()
        {
            string path = GEN + "/ForkBeggarAnimator.controller";
            AssetDatabase.DeleteAsset(path);   // fresh state machine on every rebuild
            var ctrl = AnimatorController.CreateAnimatorControllerAtPath(path);
            var sm = ctrl.layers[0].stateMachine;
            var idle = sm.AddState("Idle"); idle.motion = Clip("Characters/Rogue.fbx", "Idle");
            var talk = sm.AddState("Talking"); talk.motion = Clip("Characters/Rogue.fbx", "Idle");
            sm.defaultState = idle;
            return ctrl;
        }

        // ---------------------------------------------------------------- characters

        static GameObject BuildPlayer(RuntimeAnimatorController ctrl)
        {
            var root = new GameObject("Player");
            root.tag = "Player";
            root.layer = 2;   // Ignore Raycast: keeps the orbit camera's collision cast off the player
            root.transform.position = PLAYER_SPAWN;
            root.transform.rotation = Quaternion.identity;   // facing up the road toward the fork

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

            ApplyCharacterTexture(model, "Warrior_Texture.png", "ForkPlayerWarrior", Color.white);

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
            if (shield != null)                  // along the forearm, face out (tuned via ChatDemo3D lineups)
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
            // no deathScreen wired — the forest has no hazards, TakeDamage is never called

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

        static GameObject BuildBeggarNpc(RuntimeAnimatorController ctrl)
        {
            var root = new GameObject("NPC_Cobb");
            root.layer = 2;
            root.transform.position = BEGGAR_POS;
            // face down the road toward the approaching player
            Vector3 toPlayer = PLAYER_SPAWN - BEGGAR_POS; toPlayer.y = 0;
            root.transform.rotation = Quaternion.LookRotation(toPlayer.normalized);

            // seated silhouette: a short capsule
            var body = root.AddComponent<CapsuleCollider>();
            body.center = new Vector3(0, 0.65f, 0);
            body.height = 1.3f;
            body.radius = 0.42f;

            var trigger = root.AddComponent<SphereCollider>();
            trigger.isTrigger = true;
            trigger.radius = 2.2f;   // tight: you have to actually walk up to him
            trigger.center = new Vector3(0, 0.8f, 0);

            var model = (GameObject)PrefabUtility.InstantiatePrefab(LoadModel("Characters/Rogue.fbx"));
            model.name = "BeggarModel";
            model.transform.SetParent(root.transform, false);
            SetLayerRecursive(model, 2);
            Bounds b = RenderererSafeBounds(model);
            float scale = b.size.y > 0.01f ? 1.75f / b.size.y : 1f;
            model.transform.localScale *= scale;
            GroundModel(model, root.transform.position.y);

            // dusty, sun-bleached rags
            ApplyCharacterTexture(model, "Rogue_Texture.png", "ForkBeggar", new Color(0.75f, 0.72f, 0.68f));

            // a beggar doesn't brandish a dagger — hide the weapon meshes/mount baked into the FBX
            foreach (var t in model.GetComponentsInChildren<Transform>(true))
                if (t.name.Contains("Dagger") || t.name.StartsWith("Weapon.") || t.name == "Rogue.001")
                    t.gameObject.SetActive(false);

            var anim = model.GetComponent<Animator>();
            if (anim == null) anim = model.AddComponent<Animator>();
            anim.runtimeAnimatorController = ctrl;
            anim.applyRootMotion = false;
            anim.cullingMode = AnimatorCullingMode.AlwaysAnimate;

            // upright hunched idle — the back-tilt-against-the-crate experiment sank him into
            // the crate mesh; the hooded Rogue + begging bowl reads roadside-beggar fine as-is

            // dialogue camera: over the player's shoulder, framing the slumped beggar (his head
            // leans back and low, so the aim point drops to ~1.25 m).
            // NPCInteractor3D recomputes this from the live player position on interaction start.
            var camPoint = new GameObject("DialogueCameraPoint").transform;
            camPoint.SetParent(root.transform, false);
            Vector3 worldCamPos = BEGGAR_POS + root.transform.forward * 2.4f + root.transform.right * 1.0f + Vector3.up * 1.5f;
            camPoint.position = worldCamPos;
            camPoint.rotation = Quaternion.LookRotation((BEGGAR_POS + Vector3.up * 1.25f) - worldCamPos);

            root.AddComponent<BreathingIdle>();
            var npc = root.AddComponent<NPCInteractor3D>();
            SetRef(npc, "dialogueCameraPoint", camPoint);
            SetString(npc, "npc_name", "Cobb, the Roadside Beggar");
            SetString(npc, "system_prompt",
                "You are Cobb, a ragged old beggar slumped against a crate at the spot where the forest road splits in " +
                "two. Your knees are ruined, your cloak is patched sacking, and you live off whatever travellers spare " +
                "you. You know these woods better than anyone: the RIGHT-hand path is the safe road and carries on " +
                "through the forest, while the LEFT-hand path sinks into the Gallowmire, a bog that has swallowed " +
                "carts, horses and whole travelling parties. You never say this plainly at first — you are cagey and " +
                "roundabout, you sigh over your empty bowl, ramble about your aching bones and the weather, and hint " +
                "that a small kindness (a coin, a crust of bread, even a warm word) does wonders for an old man's " +
                "memory of the roads. If the traveller is kind, patient or generous with you, you come out with it: " +
                "keep to the right-hand path and never set foot on the left one. If they are rude or demanding, you " +
                "grumble and mutter darkly that the mire is fond of proud folk who won't spare a beggar a crumb. " +
                "Stay in character at all times. Keep your replies to one to three short sentences.");
            SetString(npc, "approach_text", "The beggar stirs against his crate and rattles a battered bowl at you...");
            // Cobb REMEMBERS between dialogues (his whole shtick is warming up to kindness across
            // visits): live KV reused while resident, transcript re-prefilled after an unload; at
            // the context limit he auto-compacts and keeps going
            SetEnum(npc, "historyMode", (int)NPCInteractor3D.HistoryMode.ResumeFromCompact);
            // same LLM+TTS stack as Velmire, with a rougher low male Kokoro voice for the beggar
            SetEnum(npc, "conversationMode", (int)NPCInteractor3D.ConversationMode.LlmPlusTts);
            SetEnum(npc, "ttsModel", (int)NPCInteractor3D.TtsModel.Kokoro);
            SetString(npc, "ttsVoice", "bm_lewis");
            SetFloat(npc, "voicePitch", 0.97f);
            // latent loading: the 10 m sphere prefetches Qwen+Kokoro during the walk-up
            // (spawn is ~11.8 m from him, so the stream starts a couple of strides in)
            SetBool(npc, "usePrefetchZone", true);
            SetFloat(npc, "prefetchRadius", 10f);
            return root;
        }

        static GameObject AttachToBone(Animator anim, HumanBodyBones bone, GameObject prefab, string name)
        {
            Transform t = anim.GetBoneTransform(bone);
            if (t == null) { Debug.LogWarning("[ForestForkBuilder] missing bone " + bone); return null; }
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
            if (tex == null) { Debug.LogWarning("[ForestForkBuilder] missing texture " + textureFile); return; }
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

        // little glowing estus bottle for the left hand — visible only during the drink
        static GameObject BuildHealFlask()
        {
            var gold = MatAsset("ForkFlaskGold.mat", new Color(0.95f, 0.72f, 0.30f), 0.1f, 0.7f);
            gold.EnableKeyword("_EMISSION");
            gold.SetColor("_EmissionColor", new Color(0.85f, 0.55f, 0.18f) * 1.4f);
            var cork = MatAsset("ForkCorkWood.mat", new Color(0.22f, 0.15f, 0.10f), 0f, 0.12f);

            var root = new GameObject("HealFlask");
            PrimPart(root.transform, PrimitiveType.Sphere,   "Body", Vector3.zero,              new Vector3(0.16f, 0.17f, 0.16f), Vector3.zero, gold);
            PrimPart(root.transform, PrimitiveType.Cylinder, "Neck", new Vector3(0, 0.105f, 0), new Vector3(0.05f, 0.035f, 0.05f), Vector3.zero, gold);
            PrimPart(root.transform, PrimitiveType.Cylinder, "Cork", new Vector3(0, 0.150f, 0), new Vector3(0.04f, 0.018f, 0.04f), Vector3.zero, cork);
            return root;
        }

        // ---------------------------------------------------------------- UI (chat window + prompt)

        static void BuildUI(TMP_FontAsset cinzel, GameObject npcGO)
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

            // --- "[ I ] Speak" prompt, bottom center (no vignette — this is a day scene)
            var promptGO = MakeRect("InteractPrompt", canvasGO.transform, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f),
                                    new Vector2(330, 58), new Vector2(0, 96));
            var promptBG = promptGO.AddComponent<Image>(); promptBG.color = darkBG;
            AddThinBorder(promptGO.transform, gold);
            MakeTMP("Text", promptGO.transform, "Speak   —   [ I ]", cinzel, 26, parchment,
                    TextAlignmentOptions.Center, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            promptGO.SetActive(false);

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
            MakeTMP("Username", msgGO.transform, "Name", cinzel, 20, new Color(0.77f, 0.66f, 0.42f),
                    TextAlignmentOptions.Left, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            MakeTMP("Message", msgGO.transform, "Body", null, 21, new Color(0.87f, 0.84f, 0.76f),
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
            UnityEventTools.AddVoidPersistentListener(inputField.onSubmit, new UnityAction(npc.AskNPC));
            UnityEventTools.AddPersistentListener(sendBtn.onClick, new UnityAction(win.PlayButtonClick));
            UnityEventTools.AddPersistentListener(leaveBtn.onClick, new UnityAction(win.PlayButtonClick));
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
            if (existing != null) return existing;   // shared with ChatDemo3D — reuse, never rebuild

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
    }
}

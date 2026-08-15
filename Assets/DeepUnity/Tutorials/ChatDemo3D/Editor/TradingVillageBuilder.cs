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
    /// Deterministically builds the TradingVillage scene — the third chat demo: a warm, living
    /// market village (golden-hour daylight, Witcher-village mood, not souls). Two villagers
    /// stroll a loop through the square GOSSIPING OUT LOUD — real pocket-tts voices, the lines
    /// typing themselves into bubbles above their heads in sync with the audio — with a cow
    /// tagging along on their flank. Press E beside them: the pair stops, both turn to face
    /// you, and a real LLM conversation opens with the one you pressed. A fishmonger works a
    /// stall on the square (GiveItem: he actually sells his trout), and a pig shuffles around
    /// a pen. Same dialogue stack as ChatDemo3D/ForestFork (NPCChatBase → VillageInteractor,
    /// SoulsChatWindow → VillageChatWindow), same CC0 Quaternius art, cottages/stalls/animals
    /// built from primitives.
    /// Run from the menu (DeepUnity/Build TradingVillage Scene) or in batch mode via
    /// -executeMethod DeepUnity.Tutorials.ChatDemo3D.EditorTools.TradingVillageBuilder.BuildBatch
    /// </summary>
    public static class TradingVillageBuilder
    {
        const string ROOT = "Assets/DeepUnity/Tutorials/ChatDemo3D";
        const string ART = ROOT + "/Art";
        const string GEN = ROOT + "/Generated";
        const string SCENE_PATH = ROOT + "/TradingVillage.unity";

        static readonly System.Random rng = new System.Random(20260809);

        // ---------------------------------------------------------------- village geometry
        // Player spawns at the south end of the main street facing +Z; the square opens up
        // around z 9..21 with the fish stall on its east side. The village core is dead flat;
        // perlin meadows and a loose tree ring take over outside it.
        static readonly Vector3 PLAYER_SPAWN = new Vector3(0f, 0.1f, -12f);
        const float VILLAGE_CX = 0f, VILLAGE_CZ = 8f, VILLAGE_HX = 15f, VILLAGE_HZ = 26f;

        static readonly Vector3 FISH_STALL_POS = new Vector3(5.6f, 0f, 15.2f);
        static readonly Vector3 BRAM_POS = new Vector3(6.6f, 0f, 15.2f);
        static readonly Vector3 PEN_CENTER = new Vector3(11f, 0f, 9f);

        // The road THROUGH the village (user 2026-08-10: it dead-ended, and the world was flat):
        // in from the south, up the main street, then out the north where the terrain climbs a
        // ridge and the road forks at a wayside shrine. Every segment listed here is kept clear
        // of tree scatter and gets rock edging.
        static readonly Vector3[][] ROADS =
        {
            new[] { new Vector3(0f, 0f, -34f), new Vector3(0f, 0f, 8f) },                     // south road + main street
            new[] { new Vector3(0f, 0f, 22f), new Vector3(0.8f, 0f, 38f),
                    new Vector3(-0.5f, 0f, 56f), new Vector3(0f, 0f, 74f) },                  // the climb out
            new[] { new Vector3(0f, 0f, 74f), new Vector3(13f, 0f, 97f) },                    // fork, right-hand path
            new[] { new Vector3(0f, 0f, 74f), new Vector3(-12f, 0f, 95f) },                   // fork, left-hand path
        };
        static readonly Vector3 FORK = new Vector3(0f, 0f, 74f);

        static float SegDist(float px, float pz, Vector3 a, Vector3 b)
        {
            float abx = b.x - a.x, abz = b.z - a.z;
            float t = ((px - a.x) * abx + (pz - a.z) * abz) / Mathf.Max(1e-5f, abx * abx + abz * abz);
            t = Mathf.Clamp01(t);
            float dx = px - (a.x + abx * t), dz = pz - (a.z + abz * t);
            return Mathf.Sqrt(dx * dx + dz * dz);
        }

        static float DistToRoads(float x, float z)
        {
            float d = float.MaxValue;
            foreach (var road in ROADS)
                for (int i = 0; i < road.Length - 1; i++)
                    d = Mathf.Min(d, SegDist(x, z, road[i], road[i + 1]));
            return d;
        }

        // the strolling loop: south leg down the main street toward the spawn (that's the leg
        // where the pair — cow on the player's left — comes at you head-on), then up the west
        // side of the square, across the top and back down the east side
        static readonly Vector3[] STROLL_LOOP =
        {
            new Vector3( 0.0f, 0f, -8.0f),
            new Vector3(-0.6f, 0f,  4.0f),
            new Vector3(-2.4f, 0f, 10.5f),
            new Vector3(-2.8f, 0f, 15.5f),
            new Vector3(-1.4f, 0f, 20.5f),
            new Vector3( 1.2f, 0f, 22.3f),
            new Vector3( 3.6f, 0f, 19.8f),
            new Vector3( 3.9f, 0f, 13.5f),
            new Vector3( 2.2f, 0f,  8.5f),
            new Vector3( 0.6f, 0f,  1.5f),
        };

        // ---------------------------------------------------------------- entry points

        [MenuItem("DeepUnity/Build TradingVillage Scene")]
        public static void BuildMenu()
        {
            ConfigureImports();
            BuildEverything();
            Debug.Log("[TradingVillageBuilder] Done. Scene at " + SCENE_PATH);
        }

        public static void BuildBatch()
        {
            try
            {
                ConfigureImports();
                BuildEverything();
                Debug.Log("[TradingVillageBuilder] BATCH OK");
                EditorApplication.Exit(0);
            }
            catch (Exception e)
            {
                Debug.LogError("[TradingVillageBuilder] BATCH FAILED: " + e);
                EditorApplication.Exit(1);
            }
        }

        // ---------------------------------------------------------------- import configuration

        static void ConfigureImports()
        {
            EnsureHumanoid(ART + "/Characters/Warrior.fbx", rotateClips180: false);   // player
            EnsureHumanoid(ART + "/Characters/Monk.fbx", rotateClips180: false);      // fishmonger
            EnsureHumanoid(ART + "/Animations/UAL1.fbx", rotateClips180: true);
            EnsureHumanoid(ART + "/Animations/UAL2.fbx", rotateClips180: true);

            // Odo is a GENERIC rig playing his own embedded clips (RPG-pack feet are IK targets
            // outside the leg chain — humanoid retarget is impossible); the pack ships a real
            // Walk cycle, verified via ClipListBatch 2026-08-09. Rogue is the same pack — he
            // provides the two background villagers. (Fenn walks on the HUMANOID Monk instead:
            // the Witch model read as costume fantasy, not village — user 2026-08-10.)
            ConfigureGenericAnimated(ART + "/Characters/Wizard.fbx");
            ConfigureGenericAnimated(ART + "/Characters/Rogue.fbx");

            // static art import pass (idempotent)
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

            // Odo's clone reference must be READABLE for the Mimi encoder (clone-from-clip)
            var vImp = AssetImporter.GetAtPath(ROOT + "/Voices/Moore.mp3") as AudioImporter;
            if (vImp != null)
            {
                var ss = vImp.defaultSampleSettings;
                if (ss.loadType != AudioClipLoadType.DecompressOnLoad)
                {
                    ss.loadType = AudioClipLoadType.DecompressOnLoad;
                    vImp.defaultSampleSettings = ss;
                    vImp.SaveAndReimport();
                }
            }
            AssetDatabase.SaveAssets();
        }

        // generic-rig import (own clips, no retargeting) — mirrors the other two builders, plus
        // looping for the gesture clips this scene actually uses as Talking states
        static void ConfigureGenericAnimated(string path)
        {
            var imp = AssetImporter.GetAtPath(path) as ModelImporter;
            if (imp == null) throw new Exception("Missing model: " + path);
            imp.animationType = ModelImporterAnimationType.Generic;
            imp.importAnimation = true;
            imp.importCameras = false;
            imp.importLights = false;
            imp.importNormals = ModelImporterNormals.Calculate;
            var clips = imp.defaultClipAnimations;
            foreach (var c in clips)
            {
                string clean = c.takeName.Contains("|") ? c.takeName.Substring(c.takeName.IndexOf('|') + 1) : c.takeName;
                c.name = clean;
                c.loopTime = clean.Contains("Idle") || clean.StartsWith("Spell")
                          || clean == "Walk" || clean == "Run" || clean == "Interact" || clean == "Wave";
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
                if (av != null && av.isValid && av.isHuman) return;
            }
            ConfigureHumanoid(path, rotateClips180);
        }

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

            var modelGO = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            var boneNames = new HashSet<string>(modelGO.GetComponentsInChildren<Transform>(true).Select(t => t.name));
            var human = new List<HumanBone>();
            foreach (var (humanName, candidates) in BONE_MAP)
            {
                string found = candidates.FirstOrDefault(c => boneNames.Contains(c));
                if (found != null)
                    human.Add(new HumanBone { humanName = humanName, boneName = found, limit = new HumanLimit { useDefaultValues = true } });
            }
            imp.humanDescription = new HumanDescription
            {
                human = human.ToArray(),
                skeleton = new SkeletonBone[0],
                upperArmTwist = 0.5f, lowerArmTwist = 0.5f,
                upperLegTwist = 0.5f, lowerLegTwist = 0.5f,
                armStretch = 0.05f, legStretch = 0.05f,
                feetSpacing = 0f, hasTranslationDoF = false,
            };

            var clips = imp.defaultClipAnimations;
            foreach (var c in clips)
            {
                string clean = c.takeName.Contains("|") ? c.takeName.Substring(c.takeName.IndexOf('|') + 1) : c.takeName;
                c.name = clean;
                c.loopTime = clean.Contains("Loop") || clean is "Sword_Idle" or "Idle" or "Walking" or "Run";
                c.keepOriginalOrientation = true;
                c.keepOriginalPositionXZ = true;
                c.keepOriginalPositionY = true;
                c.lockRootRotation = true;
                c.lockRootPositionXZ = true;
                c.lockRootHeightY = true;
                if (rotateClips180) c.rotationOffset = 180f;
            }
            if (clips.Length > 0) imp.clipAnimations = clips;
            imp.SaveAndReimport();

            var avatar = AssetDatabase.LoadAllAssetsAtPath(path).OfType<Avatar>().FirstOrDefault();
            if (avatar == null || !avatar.isValid || !avatar.isHuman)
                throw new Exception($"Humanoid avatar setup failed for {path}");
        }

        // ---------------------------------------------------------------- shared helpers

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

        static void SetInt(Component c, string field, int value)
        {
            var so = new SerializedObject(c);
            so.FindProperty(field).intValue = value;
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

        static void SetObject(Component c, string field, UnityEngine.Object value) => SetRef(c, field, value);

        static float Range(float min, float max) => min + (float)rng.NextDouble() * (max - min);

        static Material MatAsset(string file, Color c, float metallic, float gloss)
        {
            string path = GEN + "/" + file;
            var m = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (m == null)
            {
                m = new Material(Shader.Find("Standard"));
                AssetDatabase.CreateAsset(m, path);
            }
            m.color = c;
            m.SetFloat("_Metallic", metallic);
            m.SetFloat("_Glossiness", gloss);
            return m;
        }

        static GameObject PrimPart(Transform parent, PrimitiveType type, string name,
                                   Vector3 pos, Vector3 scale, Vector3 euler, Material mat, bool collider = false)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = pos;
            go.transform.localScale = scale;
            go.transform.localEulerAngles = euler;
            go.GetComponent<MeshRenderer>().sharedMaterial = mat;
            if (!collider) UnityEngine.Object.DestroyImmediate(go.GetComponent<Collider>());
            return go;
        }

        static void SetLayerRecursive(GameObject go, int layer)
        {
            go.layer = layer;
            foreach (var t in go.GetComponentsInChildren<Transform>(true))
                t.gameObject.layer = layer;
        }

        static void SetStaticRecursive(GameObject go)
        {
            go.isStatic = true;
            foreach (Transform t in go.GetComponentsInChildren<Transform>())
                t.gameObject.isStatic = true;
        }

        static void GroundModel(GameObject model, float groundY)
        {
            Bounds b = RenderererSafeBounds(model);
            model.transform.position += Vector3.up * (groundY - b.min.y);
        }

        static void ApplyCharacterTexture(GameObject model, string textureFile, string matName, Color tint)
        {
            string texPath = ART + "/Characters/" + textureFile;
            if (!File.Exists(texPath)) return;   // some rigs bake flat palette colors instead
            var tex = AssetDatabase.LoadAssetAtPath<Texture2D>(texPath);
            var mat = MatAsset(matName + ".mat", tint, 0f, 0.08f);
            mat.mainTexture = tex;
            foreach (var r in model.GetComponentsInChildren<Renderer>())
                r.sharedMaterials = r.sharedMaterials.Select(_ => mat).ToArray();
        }

        static void HideParts(GameObject model, params string[] nameFragments)
        {
            foreach (var t in model.GetComponentsInChildren<Transform>(true))
                if (nameFragments.Any(f => t.name.IndexOf(f, StringComparison.OrdinalIgnoreCase) >= 0))
                    t.gameObject.SetActive(false);
        }

        static GameObject PlacePiece(string ruinName, Vector3 pos, float yRot, Transform parent,
                                     float scale = 1f, bool collider = true)
        {
            var go = (GameObject)PrefabUtility.InstantiatePrefab(Ruin(ruinName));
            go.transform.SetParent(parent, false);
            // COMPOSE with the prefab root transform — these FBX bake unit-conversion scale and
            // a -90 deg axis-correction rotation into the root
            go.transform.localScale *= scale;
            go.transform.rotation = Quaternion.Euler(0f, yRot, 0f) * go.transform.localRotation;
            go.transform.position = pos;

            Bounds b = RenderererSafeBounds(go);
            go.transform.position += Vector3.up * (pos.y - b.min.y);

            if (collider)
                foreach (var mf in go.GetComponentsInChildren<MeshFilter>())
                    mf.gameObject.AddComponent<MeshCollider>();

            Material fol = null;   // one shade per PIECE, picked lazily
            foreach (var r in go.GetComponentsInChildren<Renderer>())
            {
                var mats = r.sharedMaterials;
                bool changed = false;
                for (int i = 0; i < mats.Length; i++)
                    if (mats[i] != null && (mats[i].name.Contains("Leaves") || mats[i].name == "Green"))
                    {
                        if (fol == null) fol = FoliageMat();
                        mats[i] = fol;
                        changed = true;
                    }
                if (changed) r.sharedMaterials = mats;
            }

            SetStaticRecursive(go);
            return go;
        }

        // three summer shades, assigned per placed piece — a one-color forest is what read
        // as "bland" from any distance
        static readonly Color[] FOLIAGE_SHADES =
        {
            new Color(0.33f, 0.47f, 0.21f),
            new Color(0.28f, 0.42f, 0.19f),
            new Color(0.40f, 0.51f, 0.24f),
        };

        static Material FoliageMat() => FoliageMat(rng.Next(FOLIAGE_SHADES.Length));

        static Material FoliageMat(int shade)
        {
            shade = Mathf.Clamp(shade, 0, FOLIAGE_SHADES.Length - 1);
            return MatAsset("VillageFoliage" + shade + ".mat", FOLIAGE_SHADES[shade], 0f, 0.03f);
        }

        // ---------------------------------------------------------------- build

        static Transform envRoot;

        static void BuildEverything()
        {
            if (!AssetDatabase.IsValidFolder(GEN))
                AssetDatabase.CreateFolder(ROOT, "Generated");

            var cinzel = CreateCinzelFont();
            var playerCtrl = CreatePlayerAnimator();
            var monkCtrl = CreateMonkNpcAnimator();
            // the Wizard pack has no true talking clip, so Odo's Talking borrows his loopable
            // Spell1 arm-gesture (reads as animated gesturing while he talks to the player);
            // Fenn (humanoid Monk) gets the real UAL talking loop
            var odoCtrl = CreateVillagerAnimator("VillageOdoAnimator", "Characters/Wizard.fbx", "Idle", "Walk", "Spell1");
            var fennCtrl = CreateMonkStrollerAnimator();
            var extraCtrl = CreateExtraAnimator("VillageExtraAnimator", "Characters/Rogue.fbx", "Idle");

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            envRoot = new GameObject("Environment").transform;

            SetupLightingAndSky();
            BuildGround();
            BuildStreets();
            BuildCottages();
            BuildSquareDressing();
            BuildPen();
            ScatterGreenery();

            GameObject player = BuildPlayer(playerCtrl);
            GameObject cameraRig = BuildCamera(player);

            GameObject bram = BuildFishmonger(monkCtrl);
            // start positions sit on the loop where it leaves the square heading south — the
            // pair's first appearance is them coming OUT of the market toward the player
            // (startDistance below is what makes runtime agree with this)
            GameObject odo = BuildStroller("NPC_Odo", "Odo, the Peddler", OdoPersona(),
                "Characters/Wizard.fbx", odoCtrl, 1.78f, "VillageOdo", new Color(0.80f, 0.74f, 0.62f),
                new[] { "Staff", "Weapon", "Sword", "Knife" },
                ROOT + "/Voices/Moore.mp3", 0.98f, new Vector3(2.6f, 0f, 8.0f));
            GameObject fenn = BuildStroller("NPC_Fenn", "Fenn, the Herbalist", FennPersona(),
                "Characters/Monk.fbx", fennCtrl, 1.76f, "VillageFenn", new Color(0.84f, 0.80f, 0.70f),
                new string[0],
                ROOT + "/Voices/Ansbach_4-15s.mp3", 1.06f, new Vector3(1.5f, 0f, 7.8f));

            GameObject cow = BuildCow(new Vector3(2.6f, 0f, 9.7f), out QuadrupedGait cowGait);
            BuildStrollDirector(cinzel, odo, fenn, cow, cowGait);

            BuildVillageLife(extraCtrl);

            BuildUI(cinzel, new[] { bram, odo, fenn });

            // quiet looping ambience + the conversation ducker that eases it down while any
            // dialogue is open (worldAudioWhileInteracting on the NPC in interaction)
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
            ambience.volume = 0.09f;   // barely-there underscore; the village's life IS the audio here
            ambience.spatialBlend = 0f;
            ambience.gameObject.AddComponent<DeepUnity.ConversationAudioDucker>();

            new GameObject("FrameSpikeProbe").AddComponent<FrameSpikeProbe>();

            SetRef(player.GetComponent<SoulsPlayerController>(), "cam", cameraRig.GetComponent<SoulsCameraRig>());
            SetRef(cameraRig.GetComponent<SoulsCameraRig>(), "target", player.transform);

            EditorSceneManager.SaveScene(scene, SCENE_PATH);
            AssetDatabase.SaveAssets();
            Debug.Log("[TradingVillageBuilder] Scene saved (" + SCENE_PATH + ")");
        }

        // ---------------------------------------------------------------- lighting / mood

        static void SetupLightingAndSky()
        {
            // golden-hour day: low warm sun, soft haze — market village an hour before supper
            var sky = new Material(Shader.Find("Skybox/Procedural"));
            sky.SetFloat("_SunSize", 0.05f);
            sky.SetFloat("_SunSizeConvergence", 5f);
            sky.SetFloat("_AtmosphereThickness", 1.05f);
            sky.SetColor("_SkyTint", new Color(0.52f, 0.54f, 0.60f));
            sky.SetColor("_GroundColor", new Color(0.44f, 0.42f, 0.34f));
            sky.SetFloat("_Exposure", 1.18f);
            AssetDatabase.CreateAsset(sky, GEN + "/VillageSkyDay.mat");

            var sunGO = new GameObject("Sun");
            var sun = sunGO.AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.color = new Color(1.0f, 0.91f, 0.76f);
            sun.intensity = 1.18f;
            sun.shadows = LightShadows.Soft;
            sun.shadowStrength = 0.8f;
            // ~33 deg up, hanging west-southwest: long warm shadows across the square
            sunGO.transform.rotation = Quaternion.Euler(33f, 230f, 0f);

            RenderSettings.skybox = sky;
            RenderSettings.sun = sun;
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogColor = new Color(0.78f, 0.76f, 0.72f);
            RenderSettings.fogDensity = 0.005f;
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.60f, 0.62f, 0.70f);
            RenderSettings.ambientEquatorColor = new Color(0.50f, 0.47f, 0.40f);
            RenderSettings.ambientGroundColor = new Color(0.28f, 0.26f, 0.21f);

            var ls = new LightingSettings { bakedGI = false, realtimeGI = false };
            ls.name = "TradingVillage LightingSettings";
            AssetDatabase.CreateAsset(ls, GEN + "/TradingVillage.lighting");
            Lightmapping.lightingSettings = ls;
        }

        // ---------------------------------------------------------------- terrain

        static float RectDist(float x, float z, float cx, float cz, float halfX, float halfZ)
        {
            float dx = Mathf.Max(0f, Mathf.Abs(x - cx) - halfX);
            float dz = Mathf.Max(0f, Mathf.Abs(z - cz) - halfZ);
            return Mathf.Sqrt(dx * dx + dz * dz);
        }

        // Dead flat inside the village bounds, gentle perlin meadows beyond — and a broad ridge
        // rising across the whole north, so the road out of the village genuinely climbs
        // (~11 m over ~55 m, a 9-10 degree walk-up) and the fork at the top looks back down on
        // the rooftops.
        static float GroundHeight(float x, float z)
        {
            float d = RectDist(x, z, VILLAGE_CX, VILLAGE_CZ, VILLAGE_HX, VILLAGE_HZ);
            float blend = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01((d - 3f) / 14f));
            float n = Mathf.PerlinNoise(x * 0.021f + 77.7f, z * 0.021f + 19.3f) * 1.8f
                    + Mathf.PerlinNoise(x * 0.08f + 311f, z * 0.08f + 47f) * 0.45f;
            float ridge = 11f * Mathf.SmoothStep(0f, 1f, Mathf.Clamp01((z - 38f) / 57f))
                          * (1f + 0.06f * Mathf.PerlinNoise(x * 0.03f + 5.1f, z * 0.03f + 9.7f));
            return (n - 1.05f) * blend + ridge;
        }

        static void BuildGround()
        {
            var mat = new Material(Shader.Find("Standard"));
            mat.mainTexture = CreateGrassTexture();
            mat.mainTextureScale = Vector2.one;
            mat.color = Color.white;
            mat.SetFloat("_Glossiness", 0.04f);
            AssetDatabase.CreateAsset(mat, GEN + "/VillageGround.mat");

            const int N = 130;
            const float SIZE = 300f;
            const float STEP = SIZE / N;
            var verts = new Vector3[(N + 1) * (N + 1)];
            var uvs = new Vector2[verts.Length];
            for (int z = 0; z <= N; z++)
                for (int x = 0; x <= N; x++)
                {
                    float wx = -SIZE * 0.5f + x * STEP;
                    float wz = -SIZE * 0.35f + z * STEP;   // shifted: more world past the square
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
            var mesh = new Mesh { name = "VillageGroundMesh", indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            mesh.vertices = verts;
            mesh.uv = uvs;
            mesh.triangles = tris;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            AssetDatabase.CreateAsset(mesh, GEN + "/VillageGroundMesh.asset");

            var ground = new GameObject("Ground");
            ground.transform.SetParent(envRoot, false);
            ground.AddComponent<MeshFilter>().sharedMesh = mesh;
            ground.AddComponent<MeshRenderer>().sharedMaterial = mat;
            ground.AddComponent<MeshCollider>().sharedMesh = mesh;
            ground.isStatic = true;
        }

        // warm summer grass blend (sunnier than ForestFork's)
        static Texture2D CreateGrassTexture()
        {
            string pngPath = GEN + "/VillageGrass.png";
            if (!File.Exists(pngPath))
            {
                const int S = 512;
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

                Color grassLight = new Color(0.38f, 0.47f, 0.22f);
                Color grassDark = new Color(0.27f, 0.36f, 0.17f);
                Color dirtFleck = new Color(0.40f, 0.33f, 0.22f);

                var tex = new Texture2D(S, S, TextureFormat.RGB24, false);
                var px = new Color[S * S];
                for (int y = 0; y < S; y++)
                    for (int x = 0; x < S; x++)
                    {
                        float u = (float)x / S, v = (float)y / S;
                        float patches = TileableNoise(u, v, 5f, 27.31f) * 0.65f + TileableNoise(u, v, 13f, 87.7f) * 0.35f;
                        float flecks = TileableNoise(u, v, 29f, 53.1f);
                        float micro = TileableNoise(u, v, 53f, 7.9f);

                        Color c = Color.Lerp(grassDark, grassLight, Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(0.36f, 0.64f, patches)));
                        c = Color.Lerp(c, dirtFleck, Mathf.InverseLerp(0.76f, 0.97f, flecks) * 0.55f);
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

        // ---------------------------------------------------------------- streets

        static void BuildStreets()
        {
            var dirtA = MatAsset("VillageDirtA.mat", new Color(0.45f, 0.36f, 0.25f), 0f, 0.05f);
            var dirtB = MatAsset("VillageDirtB.mat", new Color(0.41f, 0.32f, 0.22f), 0f, 0.05f);
            var rock = MatAsset("VillageRock.mat", new Color(0.55f, 0.53f, 0.49f), 0f, 0.08f);
            var mats = new[] { dirtA, dirtA, dirtB };

            var pathRoot = new GameObject("Streets").transform;
            pathRoot.SetParent(envRoot, false);

            // slope-aware: each piece sits on the terrain and pitches with it, so the same
            // code lays the flat main street and the climb over the north ridge
            void LaySegment(Vector3 from, Vector3 to, float width, bool edging = false)
            {
                Vector3 dir = to - from;
                dir.y = 0f;
                float len = dir.magnitude;
                dir /= len;
                for (float t = 0f; t <= len; t += 1.1f)
                {
                    Vector3 p = from + dir * t;
                    float hBack = GroundHeight(p.x - dir.x * 0.75f, p.z - dir.z * 0.75f);
                    float hFwd = GroundHeight(p.x + dir.x * 0.75f, p.z + dir.z * 0.75f);
                    Vector3 slopeDir = (dir * 1.5f + Vector3.up * (hFwd - hBack)).normalized;

                    var piece = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    piece.name = "StreetPiece";
                    piece.transform.SetParent(pathRoot, false);
                    piece.transform.position = new Vector3(p.x, GroundHeight(p.x, p.z), p.z);
                    piece.transform.rotation = Quaternion.LookRotation(slopeDir) * Quaternion.Euler(0f, Range(-5f, 5f), 0f);
                    piece.transform.localScale = new Vector3(width + Range(-0.25f, 0.25f),
                                                             0.09f + Range(0f, 0.03f), 1.5f);
                    piece.GetComponent<MeshRenderer>().sharedMaterial = mats[rng.Next(mats.Length)];
                    piece.isStatic = true;

                    // wayside stones every few meters — cheap detail that breaks the bare edges
                    if (edging && rng.NextDouble() < 0.42)
                    {
                        int s = rng.NextDouble() < 0.5 ? -1 : 1;
                        Vector3 perp = new Vector3(dir.z, 0f, -dir.x) * s * (width * 0.5f + Range(0.35f, 0.8f));
                        Vector3 rp = p + perp;
                        PrimPart(pathRoot, PrimitiveType.Sphere, "Stone",
                                 new Vector3(rp.x, GroundHeight(rp.x, rp.z) + 0.05f, rp.z),
                                 new Vector3(Range(0.18f, 0.42f), Range(0.10f, 0.2f), Range(0.18f, 0.38f)),
                                 new Vector3(0, Range(0, 360f), 0), rock);
                    }
                }
            }

            void LayRoad(Vector3[] pts, float width, bool edging)
            {
                for (int i = 0; i < pts.Length - 1; i++)
                    LaySegment(pts[i], pts[i + 1], width, edging);
            }

            LayRoad(ROADS[0], 3.6f, edging: true);                                     // south road + main street
            LayRoad(ROADS[1], 3.2f, edging: true);                                     // the climb out
            LayRoad(ROADS[2], 2.6f, edging: true);                                     // fork branches
            LayRoad(ROADS[3], 2.6f, edging: true);
            LaySegment(new Vector3(2f, 0f, 9f), new Vector3(9.5f, 0f, 9f), 2.2f);      // lane to the pen

            // the market square: one broad worn pad
            var pad = GameObject.CreatePrimitive(PrimitiveType.Cube);
            pad.name = "SquarePad";
            pad.transform.SetParent(pathRoot, false);
            pad.transform.position = new Vector3(0f, 0f, 15.2f);
            pad.transform.localScale = new Vector3(15.5f, 0.12f, 13.5f);
            pad.GetComponent<MeshRenderer>().sharedMaterial = dirtA;
            pad.isStatic = true;

            // fence runs flanking the south entrance — the road narrows INTO somewhere
            var plank = MatAsset("VillageFencePlank.mat", new Color(0.48f, 0.36f, 0.23f), 0f, 0.06f);
            var timber = MatAsset("VillageTimber.mat", new Color(0.27f, 0.20f, 0.13f), 0f, 0.08f);
            void FenceRun(Vector3 from, Vector3 to)
            {
                Vector3 d = to - from; d.y = 0;
                float len = d.magnitude; d /= len;
                float yaw = Mathf.Atan2(d.x, d.z) * Mathf.Rad2Deg;
                int posts = Mathf.Max(2, Mathf.CeilToInt(len / 1.9f) + 1);
                for (int i = 0; i < posts; i++)
                {
                    Vector3 p = from + d * (len * i / (posts - 1));
                    PrimPart(pathRoot, PrimitiveType.Cube, "FencePost",
                             new Vector3(p.x, GroundHeight(p.x, p.z) + 0.5f, p.z),
                             new Vector3(0.12f, 1.0f, 0.12f), new Vector3(0, yaw, 0), timber, collider: true);
                }
                Vector3 mid = from + d * (len * 0.5f);
                foreach (float py in new[] { 0.42f, 0.78f })
                    PrimPart(pathRoot, PrimitiveType.Cube, "FenceRail",
                             new Vector3(mid.x, GroundHeight(mid.x, mid.z) + py, mid.z),
                             new Vector3(0.07f, 0.09f, len), new Vector3(0, yaw, 0), plank, collider: true);
            }
            FenceRun(new Vector3(-2.6f, 0f, -13.5f), new Vector3(-2.6f, 0f, -7.5f));
            FenceRun(new Vector3(2.6f, 0f, -13.0f), new Vector3(2.6f, 0f, -8.0f));

            // signposts: the village's own at the entrance, and the two-armed one at the fork
            void Signpost(Vector3 at, params (float yaw, float h)[] boards)
            {
                float gy = GroundHeight(at.x, at.z);
                PrimPart(pathRoot, PrimitiveType.Cube, "SignPost",
                         new Vector3(at.x, gy + 1.05f, at.z), new Vector3(0.13f, 2.1f, 0.13f), Vector3.zero, timber, collider: true);
                foreach (var (yaw, h) in boards)
                    PrimPart(pathRoot, PrimitiveType.Cube, "SignBoard",
                             new Vector3(at.x, gy + h, at.z) + Quaternion.Euler(0, yaw, 0) * new Vector3(0.42f, 0f, 0f),
                             new Vector3(0.85f, 0.28f, 0.06f), new Vector3(0, yaw + 90f, 0), plank);
            }
            Signpost(new Vector3(3.4f, 0f, -12.2f), (0f, 1.75f));                       // "Birchbrook"
            Signpost(FORK + new Vector3(-1.8f, 0f, 1.2f), (35f, 1.85f), (-45f, 1.55f)); // the two ways on

            // wayside shrine at the fork — someone has to watch over the road
            PrimPart(pathRoot, PrimitiveType.Cube, "ShrineBase",
                     new Vector3(2.4f, GroundHeight(2.4f, 76.5f) + 0.25f, 76.5f),
                     new Vector3(1.1f, 0.5f, 1.1f), new Vector3(0, 15f, 0), rock, collider: true);
            PlacePiece("Statue_Fox", new Vector3(2.4f, GroundHeight(2.4f, 76.5f) + 0.5f, 76.5f), 195f, pathRoot,
                       scale: 0.45f, collider: false);
            PlacePiece("Candles_1", new Vector3(1.75f, GroundHeight(1.75f, 76.0f), 76.0f), Range(0, 360), pathRoot, collider: false);
        }

        // ---------------------------------------------------------------- cottages

        static void BuildCottages()
        {
            var root = new GameObject("Cottages").transform;
            root.SetParent(envRoot, false);

            // (pos, yaw so the door faces the street/square, variant 0..3 — plaster hue + roof)
            (Vector3 pos, float yaw, int v)[] lots =
            {
                (new Vector3(-8.6f, 0f,  3.0f),  90f, 0),
                (new Vector3(-9.0f, 0f, 15.0f),  90f, 1),
                (new Vector3(-8.4f, 0f, 24.0f),  75f, 2),
                (new Vector3( 8.8f, 0f,  1.5f), -90f, 3),
                (new Vector3( 9.2f, 0f, 22.5f), -95f, 0),
                (new Vector3(-2.5f, 0f, 30.5f), 175f, 1),
                (new Vector3( 4.5f, 0f, 30.0f), 190f, 2),
                // two farmsteads flanking the road out, where the climb begins — the village
                // reads bigger and the road has a reason to keep going (user 2026-08-10)
                (new Vector3(-6.8f, 0f, 36.0f), 115f, 3),
                (new Vector3( 7.2f, 0f, 39.5f), 250f, 1),
            };
            foreach (var (pos, yaw, v) in lots)
                BuildCottage(root, new Vector3(pos.x, GroundHeight(pos.x, pos.z), pos.z), yaw, v);
        }

        static readonly Color[] PLASTER_HUES =
        {
            new Color(0.84f, 0.78f, 0.66f),   // warm cream
            new Color(0.78f, 0.70f, 0.58f),   // tan
            new Color(0.76f, 0.74f, 0.66f),   // grey-green
            new Color(0.82f, 0.72f, 0.58f),   // ochre
        };

        static void BuildCottage(Transform parent, Vector3 pos, float yaw, int variant)
        {
            var plaster = MatAsset("VillagePlaster" + variant + ".mat", PLASTER_HUES[variant % 4], 0f, 0.06f);
            plaster.mainTexture = CreatePlasterTexture();
            plaster.mainTextureScale = new Vector2(2.2f, 1.4f);
            var timber = MatAsset("VillageTimber.mat", new Color(0.27f, 0.20f, 0.13f), 0f, 0.08f);
            var thatch = MatAsset(variant % 2 == 0 ? "VillageThatchA.mat" : "VillageThatchB.mat",
                                  variant % 2 == 0 ? new Color(0.68f, 0.56f, 0.32f) : new Color(0.62f, 0.49f, 0.28f), 0f, 0.04f);
            thatch.mainTexture = CreateThatchTexture();
            thatch.mainTextureScale = new Vector2(2.6f, 1.3f);
            var dark = MatAsset("VillageDark.mat", new Color(0.10f, 0.09f, 0.10f), 0f, 0.05f);
            var stone = MatAsset("VillageStone.mat", new Color(0.52f, 0.51f, 0.48f), 0f, 0.05f);
            var shutter = MatAsset("VillageShutter.mat", new Color(0.38f, 0.27f, 0.17f), 0f, 0.07f);

            var c = new GameObject("Cottage").transform;
            c.SetParent(parent, false);
            c.position = pos;
            c.rotation = Quaternion.Euler(0f, yaw, 0f);

            // timber-framed body: plaster box on a stone footing, dark posts, beams and braces
            PrimPart(c, PrimitiveType.Cube, "Body", new Vector3(0, 1.35f, 0), new Vector3(4.6f, 2.7f, 3.9f), Vector3.zero, plaster, collider: true);
            PrimPart(c, PrimitiveType.Cube, "Footing", new Vector3(0, 0.14f, 0), new Vector3(4.75f, 0.28f, 4.05f), Vector3.zero, stone);
            foreach (float sx in new[] { -2.24f, 2.24f })
                foreach (float sz in new[] { -1.88f, 1.88f })
                    PrimPart(c, PrimitiveType.Cube, "Post", new Vector3(sx, 1.35f, sz), new Vector3(0.18f, 2.72f, 0.18f), Vector3.zero, timber);
            foreach (float sy in new[] { 0.35f, 2.60f })
                foreach (float sz in new[] { -1.90f, 1.90f })
                    PrimPart(c, PrimitiveType.Cube, "Beam", new Vector3(0, sy, sz), new Vector3(4.7f, 0.14f, 0.14f), Vector3.zero, timber);
            // the diagonal braces that make Fachwerk read as Fachwerk (front + back faces)
            foreach (float sz in new[] { -1.92f, 1.92f })
                foreach (float sx in new[] { -1.85f, 1.85f })
                    PrimPart(c, PrimitiveType.Cube, "Brace", new Vector3(sx, 1.05f, sz),
                             new Vector3(0.11f, 1.35f, 0.08f), new Vector3(0, 0, sx > 0 ? -34f : 34f), timber);

            // door with a lintel + two cross-framed, shuttered windows on the street face (+Z)
            PrimPart(c, PrimitiveType.Cube, "Door", new Vector3(0.0f, 0.90f, 1.97f), new Vector3(0.95f, 1.8f, 0.10f), Vector3.zero, dark);
            PrimPart(c, PrimitiveType.Cube, "Lintel", new Vector3(0.0f, 1.88f, 2.0f), new Vector3(1.2f, 0.14f, 0.12f), Vector3.zero, timber);
            foreach (float wx in new[] { -1.45f, 1.45f })
            {
                PrimPart(c, PrimitiveType.Cube, "Window", new Vector3(wx, 1.62f, 1.97f), new Vector3(0.72f, 0.72f, 0.08f), Vector3.zero, dark);
                PrimPart(c, PrimitiveType.Cube, "WinBarV", new Vector3(wx, 1.62f, 2.00f), new Vector3(0.07f, 0.74f, 0.05f), Vector3.zero, timber);
                PrimPart(c, PrimitiveType.Cube, "WinBarH", new Vector3(wx, 1.62f, 2.00f), new Vector3(0.74f, 0.07f, 0.05f), Vector3.zero, timber);
                PrimPart(c, PrimitiveType.Cube, "Sill", new Vector3(wx, 1.22f, 2.0f), new Vector3(0.85f, 0.09f, 0.14f), Vector3.zero, timber);
                foreach (int s in new[] { -1, 1 })
                    PrimPart(c, PrimitiveType.Cube, "Shutter", new Vector3(wx + s * 0.55f, 1.62f, 1.99f),
                             new Vector3(0.30f, 0.72f, 0.05f), Vector3.zero, shutter);
            }
            // one shuttered window on each gable side too — the backs stopped being blank walls
            foreach (float gx in new[] { -2.33f, 2.33f })
            {
                PrimPart(c, PrimitiveType.Cube, "SideWindow", new Vector3(gx, 1.62f, -0.4f), new Vector3(0.08f, 0.66f, 0.66f), Vector3.zero, dark);
                PrimPart(c, PrimitiveType.Cube, "SideBarH", new Vector3(gx > 0 ? gx + 0.02f : gx - 0.02f, 1.62f, -0.4f), new Vector3(0.05f, 0.07f, 0.68f), Vector3.zero, timber);
            }

            // pitched thatch roof (ridge along local X) + ridge beam + stepped gables + chimney
            PrimPart(c, PrimitiveType.Cube, "RoofL", new Vector3(0, 3.36f, -1.10f), new Vector3(5.3f, 0.15f, 2.72f), new Vector3(-33f, 0f, 0f), thatch, collider: true);
            PrimPart(c, PrimitiveType.Cube, "RoofR", new Vector3(0, 3.36f, 1.10f), new Vector3(5.3f, 0.15f, 2.72f), new Vector3(33f, 0f, 0f), thatch, collider: true);
            PrimPart(c, PrimitiveType.Cube, "RidgeBeam", new Vector3(0, 4.12f, 0), new Vector3(5.4f, 0.16f, 0.22f), Vector3.zero, timber);
            foreach (float gx in new[] { -2.24f, 2.24f })
            {
                PrimPart(c, PrimitiveType.Cube, "Gable1", new Vector3(gx, 3.05f, 0), new Vector3(0.14f, 0.75f, 2.9f), Vector3.zero, plaster);
                PrimPart(c, PrimitiveType.Cube, "Gable2", new Vector3(gx, 3.65f, 0), new Vector3(0.14f, 0.55f, 1.5f), Vector3.zero, plaster);
            }
            if (variant % 2 == 1)
                PrimPart(c, PrimitiveType.Cube, "Chimney", new Vector3(1.35f, 3.9f, 0.55f), new Vector3(0.5f, 1.5f, 0.5f), Vector3.zero, stone);

            SetStaticRecursive(c.gameObject);
        }

        // subtle mottled plaster so the cottage walls stop reading as flat vinyl
        static Texture2D CreatePlasterTexture()
        {
            string pngPath = GEN + "/VillagePlaster.png";
            if (!File.Exists(pngPath))
            {
                const int S = 256;
                var tex = new Texture2D(S, S, TextureFormat.RGB24, false);
                var px = new Color[S * S];
                for (int y = 0; y < S; y++)
                    for (int x = 0; x < S; x++)
                    {
                        float u = (float)x / S, v = (float)y / S;
                        float n = Mathf.PerlinNoise(u * 9f + 3.7f, v * 9f + 8.1f) * 0.6f
                                + Mathf.PerlinNoise(u * 31f + 77f, v * 31f + 13f) * 0.4f;
                        float shade = 0.88f + 0.16f * n;
                        px[y * S + x] = new Color(shade, shade * 0.995f, shade * 0.97f);
                    }
                tex.SetPixels(px);
                tex.Apply();
                File.WriteAllBytes(pngPath, tex.EncodeToPNG());
                UnityEngine.Object.DestroyImmediate(tex);
                AssetDatabase.ImportAsset(pngPath);
            }
            return AssetDatabase.LoadAssetAtPath<Texture2D>(pngPath);
        }

        // horizontal straw striations for the roofs
        static Texture2D CreateThatchTexture()
        {
            string pngPath = GEN + "/VillageThatch.png";
            if (!File.Exists(pngPath))
            {
                const int S = 256;
                var tex = new Texture2D(S, S, TextureFormat.RGB24, false);
                var px = new Color[S * S];
                for (int y = 0; y < S; y++)
                    for (int x = 0; x < S; x++)
                    {
                        float u = (float)x / S, v = (float)y / S;
                        float rows = Mathf.Abs(Mathf.Sin(v * Mathf.PI * 14f));            // layered bundles
                        float straw = Mathf.PerlinNoise(u * 60f, v * 6f + 31f);           // along-the-straw streaks
                        float shade = 0.72f + 0.22f * rows * 0.5f + 0.20f * straw;
                        px[y * S + x] = new Color(shade, shade * 0.94f, shade * 0.78f);
                    }
                tex.SetPixels(px);
                tex.Apply();
                File.WriteAllBytes(pngPath, tex.EncodeToPNG());
                UnityEngine.Object.DestroyImmediate(tex);
                AssetDatabase.ImportAsset(pngPath);
            }
            return AssetDatabase.LoadAssetAtPath<Texture2D>(pngPath);
        }

        // ---------------------------------------------------------------- square dressing

        static void BuildSquareDressing()
        {
            var root = new GameObject("Square").transform;
            root.SetParent(envRoot, false);

            var wood = MatAsset("VillageStallWood.mat", new Color(0.42f, 0.31f, 0.20f), 0f, 0.08f);
            var canvasA = MatAsset("VillageCanvasRed.mat", new Color(0.62f, 0.26f, 0.22f), 0f, 0.05f);
            var canvasB = MatAsset("VillageCanvasCream.mat", new Color(0.84f, 0.79f, 0.66f), 0f, 0.05f);
            var canvasC = MatAsset("VillageCanvasGreen.mat", new Color(0.30f, 0.44f, 0.28f), 0f, 0.05f);

            // Bram's fish stall (east side, counter facing the square) + three dressed stalls —
            // a market, not a clearing (density pass, user 2026-08-10)
            var fish = BuildStall(root, "FishStall", FISH_STALL_POS, -90f, wood, canvasA, canvasB);
            AddFishDisplay(fish.transform, wood);
            BuildStall(root, "ProduceStall", new Vector3(-5.9f, 0f, 12.0f), 90f, wood, canvasC, canvasB, dressProduce: true);
            BuildStall(root, "ClothStall", new Vector3(-5.9f, 0f, 18.6f), 90f, wood, canvasB, canvasA, dressCloth: true);
            var pottery = BuildStall(root, "PotteryStall", new Vector3(0.3f, 0f, 9.4f), 0f, wood, canvasB, canvasC);
            PlacePiece("Pot1", new Vector3(-0.2f, 0.90f, 9.85f), Range(0, 360), root, scale: 0.35f, collider: false);
            PlacePiece("Pot3", new Vector3(0.75f, 0.90f, 9.75f), Range(0, 360), root, scale: 0.35f, collider: false);
            PlacePiece("Pot2", new Vector3(1.6f, 0f, 8.6f), Range(0, 360), root, collider: false);

            // the old well, MIDDLE of the square — the stroll loop circles it
            BuildWell(root, new Vector3(0.6f, 0f, 16.8f));

            // clutter that says "market day": cart, barrels, crates, pots, sacks, hay.
            // The cart is primitive-built: the ruins-pack Cart FBX rests on its axis-correction
            // pose and read as FLOATING in the 2026-08-10 shots — this one has its wheels on the
            // ground by construction.
            BuildHandCart(root, new Vector3(5.8f, 0f, 6.2f), 205f);
            PlacePiece("Barrel", FISH_STALL_POS + new Vector3(0.4f, 0f, 1.9f), Range(0, 360), root);
            PlacePiece("Barrel", FISH_STALL_POS + new Vector3(0.7f, 0f, -1.8f), Range(0, 360), root);
            PlacePiece("Crate", FISH_STALL_POS + new Vector3(1.3f, 0f, 1.6f), 40f, root);
            PlacePiece("Crate", new Vector3(-6.6f, 0f, 14.4f), Range(0, 360), root);
            PlacePiece("Crate", new Vector3(-6.3f, 0f, 10.1f), 25f, root);
            PlacePiece("Crate", new Vector3(-6.9f, 0f, 20.9f), 70f, root);
            PlacePiece("Barrel", new Vector3(-6.9f, 0f, 16.8f), Range(0, 360), root);
            PlacePiece("Barrel", new Vector3(4.9f, 0f, 21.6f), Range(0, 360), root);
            PlacePiece("Crate", new Vector3(5.5f, 0f, 21.1f), 15f, root);
            PlacePiece("Pot1", new Vector3(-7.8f, 0f, 3.8f), Range(0, 360), root, collider: false);
            PlacePiece("Pot2", new Vector3(7.4f, 0f, 2.6f), Range(0, 360), root, collider: false);
            PlacePiece("Pot3", new Vector3(-7.2f, 0f, 24.9f), Range(0, 360), root, collider: false);

            var sack = MatAsset("VillageSack.mat", new Color(0.62f, 0.54f, 0.40f), 0f, 0.04f);
            void SackPile(Vector3 at)
            {
                PrimPart(root, PrimitiveType.Sphere, "Sack", at + new Vector3(0, 0.22f, 0), new Vector3(0.55f, 0.44f, 0.55f), new Vector3(0, Range(0, 360), 0), sack);
                PrimPart(root, PrimitiveType.Sphere, "Sack", at + new Vector3(0.45f, 0.19f, 0.15f), new Vector3(0.48f, 0.38f, 0.48f), new Vector3(0, Range(0, 360), 0), sack);
                PrimPart(root, PrimitiveType.Sphere, "Sack", at + new Vector3(0.2f, 0.55f, 0.05f), new Vector3(0.46f, 0.36f, 0.46f), new Vector3(0, Range(0, 360), 0), sack);
            }
            SackPile(new Vector3(-4.9f, 0f, 13.4f));
            SackPile(new Vector3(6.1f, 0f, 17.4f));

            var hay = MatAsset("VillageHay.mat", new Color(0.78f, 0.66f, 0.32f), 0f, 0.05f);
            PrimPart(root, PrimitiveType.Cube, "HayBale", new Vector3(8.6f, 0.26f, 11.9f), new Vector3(0.9f, 0.52f, 0.55f), new Vector3(0, 20f, 0), hay, collider: true);
            PrimPart(root, PrimitiveType.Cube, "HayBale", new Vector3(8.75f, 0.75f, 11.85f), new Vector3(0.8f, 0.46f, 0.5f), new Vector3(0, 55f, 0), hay, collider: true);
        }

        static GameObject BuildStall(Transform parent, string name, Vector3 pos, float yaw,
                                     Material wood, Material stripeA, Material stripeB,
                                     bool dressProduce = false, bool dressCloth = false)
        {
            var s = new GameObject(name).transform;
            s.SetParent(parent, false);
            s.position = pos;
            s.rotation = Quaternion.Euler(0f, yaw, 0f);

            foreach (float px in new[] { -1.15f, 1.15f })
                foreach (float pz in new[] { -0.55f, 0.72f })
                    PrimPart(s, PrimitiveType.Cylinder, "Post", new Vector3(px, 1.12f, pz), new Vector3(0.11f, 1.12f, 0.11f), Vector3.zero, wood);

            PrimPart(s, PrimitiveType.Cube, "Counter", new Vector3(0, 0.44f, 0.42f), new Vector3(2.5f, 0.88f, 0.9f), Vector3.zero, wood, collider: true);
            PrimPart(s, PrimitiveType.Cube, "BackShelf", new Vector3(0, 1.15f, -0.42f), new Vector3(2.5f, 0.10f, 0.5f), Vector3.zero, wood);

            // striped canvas canopy, tilted down toward the customers
            for (int i = 0; i < 7; i++)
            {
                float x = -1.11f + i * 0.37f;
                PrimPart(s, PrimitiveType.Cube, "Canopy", new Vector3(x, 2.34f, 0.18f),
                         new Vector3(0.37f, 0.05f, 2.05f), new Vector3(-14f, 0f, 0f),
                         i % 2 == 0 ? stripeA : stripeB);
            }

            if (dressProduce)
            {
                var sack = MatAsset("VillageSack.mat", new Color(0.62f, 0.54f, 0.40f), 0f, 0.04f);
                var bread = MatAsset("VillageBread.mat", new Color(0.68f, 0.49f, 0.26f), 0f, 0.10f);
                PrimPart(s, PrimitiveType.Sphere, "Sack1", new Vector3(-0.7f, 1.02f, 0.35f), new Vector3(0.42f, 0.30f, 0.42f), Vector3.zero, sack);
                PrimPart(s, PrimitiveType.Sphere, "Sack2", new Vector3(-0.2f, 1.00f, 0.5f), new Vector3(0.36f, 0.26f, 0.36f), Vector3.zero, sack);
                for (int i = 0; i < 3; i++)
                    PrimPart(s, PrimitiveType.Sphere, "Loaf", new Vector3(0.45f + i * 0.32f, 0.97f, 0.3f), new Vector3(0.26f, 0.16f, 0.20f), new Vector3(0, Range(0, 90), 0), bread);
            }
            if (dressCloth)
            {
                var bolt1 = MatAsset("VillageBolt1.mat", new Color(0.48f, 0.28f, 0.42f), 0f, 0.08f);
                var bolt2 = MatAsset("VillageBolt2.mat", new Color(0.26f, 0.36f, 0.52f), 0f, 0.08f);
                var bolt3 = MatAsset("VillageBolt3.mat", new Color(0.70f, 0.58f, 0.30f), 0f, 0.08f);
                Material[] bolts = { bolt1, bolt2, bolt3 };
                for (int i = 0; i < 3; i++)
                    PrimPart(s, PrimitiveType.Cylinder, "Bolt", new Vector3(-0.55f + i * 0.55f, 1.02f, 0.38f),
                             new Vector3(0.24f, 0.55f, 0.24f), new Vector3(0f, Range(-10, 10), 90f), bolts[i]);
            }

            SetStaticRecursive(s.gameObject);
            return s.gameObject;
        }

        static void AddFishDisplay(Transform stall, Material wood)
        {
            var silver = MatAsset("VillageFishSilver.mat", new Color(0.62f, 0.68f, 0.72f), 0.25f, 0.55f);
            var dark = MatAsset("VillageFishDark.mat", new Color(0.34f, 0.42f, 0.46f), 0.2f, 0.45f);

            // wet-grass display tray with the morning's catch laid out in a row
            PrimPart(stall, PrimitiveType.Cube, "Tray", new Vector3(0, 0.93f, 0.42f), new Vector3(2.2f, 0.10f, 0.72f), Vector3.zero, FoliageMat());
            for (int i = 0; i < 4; i++)
            {
                var mat = i % 2 == 0 ? silver : dark;
                var fish = new GameObject("Fish").transform;
                fish.SetParent(stall, false);
                fish.localPosition = new Vector3(-0.78f + i * 0.52f, 1.02f, 0.42f);
                fish.localRotation = Quaternion.Euler(0f, Range(-14f, 14f), 0f);
                PrimPart(fish, PrimitiveType.Sphere, "Body", Vector3.zero, new Vector3(0.46f, 0.13f, 0.16f), Vector3.zero, mat);
                PrimPart(fish, PrimitiveType.Cube, "Tail", new Vector3(-0.27f, 0f, 0f), new Vector3(0.10f, 0.02f, 0.13f), new Vector3(0, 0, 18f), mat);
            }
            // hanging scale: little arm + pan
            PrimPart(stall, PrimitiveType.Cube, "ScaleArm", new Vector3(0.95f, 1.75f, -0.30f), new Vector3(0.05f, 0.55f, 0.05f), Vector3.zero, wood);
            PrimPart(stall, PrimitiveType.Cylinder, "ScalePan", new Vector3(0.95f, 1.44f, -0.30f), new Vector3(0.30f, 0.02f, 0.30f), Vector3.zero, wood);
        }

        // Two-wheeled wooden handcart resting on its handles — wheels touch the ground by
        // construction (r 0.38 on axle at y 0.38), which the ruins-pack Cart FBX never managed.
        static void BuildHandCart(Transform parent, Vector3 pos, float yaw)
        {
            var wood = MatAsset("VillageStallWood.mat", new Color(0.42f, 0.31f, 0.20f), 0f, 0.08f);
            var timber = MatAsset("VillageTimber.mat", new Color(0.27f, 0.20f, 0.13f), 0f, 0.08f);
            var sack = MatAsset("VillageSack.mat", new Color(0.62f, 0.54f, 0.40f), 0f, 0.04f);

            var c = new GameObject("HandCart").transform;
            c.SetParent(parent, false);
            c.position = pos;
            c.rotation = Quaternion.Euler(0f, yaw, 0f);

            const float tilt = 13f;   // bed slopes down toward the grounded handles (+Z is the front)
            PrimPart(c, PrimitiveType.Cube, "Bed", new Vector3(0, 0.33f, 0.05f), new Vector3(1.05f, 0.09f, 1.75f), new Vector3(tilt, 0, 0), wood, collider: true);
            foreach (float sx in new[] { -0.55f, 0.55f })
                PrimPart(c, PrimitiveType.Cube, "Rail", new Vector3(sx, 0.53f, 0.05f), new Vector3(0.06f, 0.30f, 1.75f), new Vector3(tilt, 0, 0), wood);
            PrimPart(c, PrimitiveType.Cube, "Front", new Vector3(0, 0.50f, -0.78f), new Vector3(1.05f, 0.28f, 0.06f), new Vector3(tilt, 0, 0), wood);

            PrimPart(c, PrimitiveType.Cylinder, "Axle", new Vector3(0, 0.38f, -0.55f), new Vector3(0.05f, 0.66f, 0.05f), new Vector3(0, 0, 90f), timber);
            foreach (float sx in new[] { -0.64f, 0.64f })
                PrimPart(c, PrimitiveType.Cylinder, "Wheel", new Vector3(sx, 0.38f, -0.55f), new Vector3(0.76f, 0.05f, 0.76f), new Vector3(0, 0, 90f), timber, collider: true);

            // handle poles running from the bed front down to the ground
            foreach (float sx in new[] { -0.40f, 0.40f })
                PrimPart(c, PrimitiveType.Cylinder, "Handle", new Vector3(sx, 0.22f, 0.92f), new Vector3(0.05f, 0.42f, 0.05f), new Vector3(72f, 0, 0), timber);

            PrimPart(c, PrimitiveType.Sphere, "Sack1", new Vector3(-0.2f, 0.56f, -0.25f), new Vector3(0.46f, 0.34f, 0.46f), new Vector3(tilt, 40f, 0), sack);
            PrimPart(c, PrimitiveType.Sphere, "Sack2", new Vector3(0.25f, 0.52f, 0.25f), new Vector3(0.40f, 0.30f, 0.40f), new Vector3(tilt, 200f, 0), sack);

            SetStaticRecursive(c.gameObject);
        }

        static void BuildWell(Transform parent, Vector3 pos)
        {
            var stone = MatAsset("VillageStone.mat", new Color(0.52f, 0.51f, 0.48f), 0f, 0.05f);
            var timber = MatAsset("VillageTimber.mat", new Color(0.27f, 0.20f, 0.13f), 0f, 0.08f);
            var thatch = MatAsset("VillageThatchA.mat", new Color(0.62f, 0.50f, 0.28f), 0f, 0.04f);
            var water = MatAsset("VillageWater.mat", new Color(0.13f, 0.22f, 0.26f), 0f, 0.75f);

            var w = new GameObject("Well").transform;
            w.SetParent(parent, false);
            w.position = pos;

            PrimPart(w, PrimitiveType.Cylinder, "Ring", new Vector3(0, 0.45f, 0), new Vector3(1.5f, 0.45f, 1.5f), Vector3.zero, stone, collider: true);
            PrimPart(w, PrimitiveType.Cylinder, "Water", new Vector3(0, 0.905f, 0), new Vector3(1.18f, 0.01f, 1.18f), Vector3.zero, water);
            foreach (float px in new[] { -0.62f, 0.62f })
                PrimPart(w, PrimitiveType.Cube, "Post", new Vector3(px, 1.5f, 0), new Vector3(0.12f, 1.4f, 0.12f), Vector3.zero, timber);
            PrimPart(w, PrimitiveType.Cylinder, "Crossbar", new Vector3(0, 2.08f, 0), new Vector3(0.07f, 0.72f, 0.07f), new Vector3(0, 0, 90f), timber);
            PrimPart(w, PrimitiveType.Cube, "RoofL", new Vector3(0, 2.5f, -0.42f), new Vector3(1.9f, 0.07f, 1.05f), new Vector3(-32f, 0f, 0f), thatch);
            PrimPart(w, PrimitiveType.Cube, "RoofR", new Vector3(0, 2.5f, 0.42f), new Vector3(1.9f, 0.07f, 1.05f), new Vector3(32f, 0f, 0f), thatch);
            PrimPart(w, PrimitiveType.Cube, "Bucket", new Vector3(0.2f, 1.55f, 0), new Vector3(0.24f, 0.20f, 0.24f), Vector3.zero, timber);

            SetStaticRecursive(w.gameObject);
        }

        // ---------------------------------------------------------------- pig pen

        static void BuildPen()
        {
            var root = new GameObject("PigPen").transform;
            root.SetParent(envRoot, false);

            // Wooden fence from primitives. The ruins pack's Rail pieces are dark STONE
            // balustrades — placed around a pen they read as a heap of coal blocks (seen in the
            // 2026-08-09 layout shots), so the pen gets honest timber posts and planks instead.
            var timber = MatAsset("VillageTimber.mat", new Color(0.27f, 0.20f, 0.13f), 0f, 0.08f);
            var plankMat = MatAsset("VillageFencePlank.mat", new Color(0.48f, 0.36f, 0.23f), 0f, 0.06f);

            const float half = 1.9f;
            var fence = new GameObject("Fence").transform;
            fence.SetParent(root, false);
            fence.position = PEN_CENTER;

            for (int side = 0; side < 4; side++)
            {
                Quaternion rot = Quaternion.Euler(0f, side * 90f, 0f);
                // corner + midpoint posts
                foreach (float along in new[] { -half, 0f, half })
                    PrimPart(fence, PrimitiveType.Cube, "Post", rot * new Vector3(along, 0.5f, half),
                             new Vector3(0.13f, 1.0f, 0.13f), rot.eulerAngles, timber, collider: true);
                // two planks per side; the west side keeps a gate gap toward the lane
                foreach (float py in new[] { 0.42f, 0.78f })
                {
                    if (side == 3)
                    {
                        PrimPart(fence, PrimitiveType.Cube, "Plank", rot * new Vector3(-half * 0.62f, py, half),
                                 new Vector3(half * 0.76f, 0.09f, 0.06f), rot.eulerAngles, plankMat, collider: true);
                        PrimPart(fence, PrimitiveType.Cube, "Plank", rot * new Vector3(half * 0.62f, py, half),
                                 new Vector3(half * 0.76f, 0.09f, 0.06f), rot.eulerAngles, plankMat, collider: true);
                    }
                    else
                    {
                        PrimPart(fence, PrimitiveType.Cube, "Plank", rot * new Vector3(0f, py, half),
                                 new Vector3(half * 2f + 0.1f, 0.09f, 0.06f), rot.eulerAngles, plankMat, collider: true);
                    }
                }
            }

            // trough + mud patch
            var mud = MatAsset("VillageMud.mat", new Color(0.33f, 0.26f, 0.19f), 0f, 0.25f);
            PrimPart(root, PrimitiveType.Cube, "Trough", PEN_CENTER + new Vector3(1.2f, 0.18f, 0.9f), new Vector3(0.9f, 0.3f, 0.4f), Vector3.zero, timber, collider: true);
            PrimPart(root, PrimitiveType.Cube, "Mud", PEN_CENTER + new Vector3(-0.4f, 0.065f, -0.5f), new Vector3(2.2f, 0.13f, 1.8f), new Vector3(0, 15f, 0), mud);

            BuildPig(PEN_CENTER + new Vector3(0.3f, 0f, 0.2f));
        }

        // ---------------------------------------------------------------- village life
        // The density pass (user 2026-08-10: "quite empty"): gardens, firewood, laundry,
        // flowers, pecking chickens and two background villagers, so the square reads lived-in
        // even before the talkable NPCs say a word.

        static void BuildVillageLife(RuntimeAnimatorController extraCtrl)
        {
            var root = new GameObject("VillageLife").transform;
            root.SetParent(envRoot, false);

            var timber = MatAsset("VillageTimber.mat", new Color(0.27f, 0.20f, 0.13f), 0f, 0.08f);
            var plank = MatAsset("VillageFencePlank.mat", new Color(0.48f, 0.36f, 0.23f), 0f, 0.06f);
            var soil = MatAsset("VillageSoil.mat", new Color(0.28f, 0.21f, 0.15f), 0f, 0.1f);
            var cabbage = MatAsset("VillageCabbage.mat", new Color(0.45f, 0.58f, 0.30f), 0f, 0.15f);

            // cabbage gardens (Odo's soup has to come from somewhere)
            void Garden(Vector3 at, float yaw)
            {
                var g = new GameObject("Garden").transform;
                g.SetParent(root, false);
                g.position = at;
                g.rotation = Quaternion.Euler(0f, yaw, 0f);
                PrimPart(g, PrimitiveType.Cube, "Soil", new Vector3(0, 0.07f, 0), new Vector3(2.7f, 0.14f, 1.8f), Vector3.zero, soil);
                for (int r = 0; r < 2; r++)
                    for (int i = 0; i < 4; i++)
                        PrimPart(g, PrimitiveType.Sphere, "Cabbage",
                                 new Vector3(-0.95f + i * 0.62f, 0.20f, -0.42f + r * 0.84f),
                                 Vector3.one * Range(0.24f, 0.32f), new Vector3(0, Range(0, 360), 0), cabbage);
                foreach (float sx in new[] { -1.38f, 1.38f })
                    PrimPart(g, PrimitiveType.Cube, "Rail", new Vector3(sx, 0.3f, 0), new Vector3(0.07f, 0.07f, 1.9f), Vector3.zero, plank);
                PrimPart(g, PrimitiveType.Cube, "Rail", new Vector3(0, 0.3f, -0.95f), new Vector3(2.85f, 0.07f, 0.07f), Vector3.zero, plank);
            }
            Garden(new Vector3(-8.0f, 0f, 7.2f), 4f);
            Garden(new Vector3(8.7f, 0f, 18.6f), -8f);

            // firewood stacked against two cottage walls
            void Firewood(Vector3 at, float yaw)
            {
                var f = new GameObject("Firewood").transform;
                f.SetParent(root, false);
                f.position = at;
                f.rotation = Quaternion.Euler(0f, yaw, 0f);
                for (int layer = 0; layer < 2; layer++)
                    for (int i = 0; i < 4 - layer; i++)
                        PrimPart(f, PrimitiveType.Cylinder, "Log",
                                 new Vector3(-0.27f + i * 0.18f + layer * 0.09f, 0.09f + layer * 0.17f, 0),
                                 new Vector3(0.17f, 0.30f, 0.17f), new Vector3(90f, 0, 0), timber);
            }
            Firewood(new Vector3(-8.5f, 0f, 0.6f), 90f);
            Firewood(new Vector3(9.0f, 0f, 24.9f), -95f);

            // laundry line beside the west cottages
            var rope = MatAsset("VillageRope.mat", new Color(0.42f, 0.38f, 0.30f), 0f, 0.05f);
            Color[] clothColors = { new Color(0.84f, 0.79f, 0.66f), new Color(0.36f, 0.45f, 0.58f), new Color(0.62f, 0.36f, 0.28f) };
            var laundry = new GameObject("Laundry").transform;
            laundry.SetParent(root, false);
            laundry.position = new Vector3(-6.9f, 0f, 14.8f);
            foreach (float lz in new[] { -1.4f, 1.4f })
                PrimPart(laundry, PrimitiveType.Cube, "Post", new Vector3(0, 0.95f, lz), new Vector3(0.09f, 1.9f, 0.09f), Vector3.zero, timber, collider: true);
            PrimPart(laundry, PrimitiveType.Cylinder, "Line", new Vector3(0, 1.82f, 0), new Vector3(0.025f, 1.4f, 0.025f), new Vector3(90f, 0, 0), rope);
            for (int i = 0; i < 3; i++)
            {
                var mat = MatAsset("VillageCloth" + i + ".mat", clothColors[i], 0f, 0.05f);
                PrimPart(laundry, PrimitiveType.Cube, "Cloth", new Vector3(0, 1.52f, -0.85f + i * 0.85f),
                         new Vector3(0.03f, 0.58f, 0.55f), new Vector3(0, Range(-6f, 6f), 0), mat);
            }

            // little flower patches around the cottages and the well
            Color[] petals = { new Color(0.92f, 0.90f, 0.82f), new Color(0.88f, 0.74f, 0.28f), new Color(0.74f, 0.32f, 0.30f) };
            void Flowers(Vector3 at)
            {
                for (int i = 0; i < 5; i++)
                {
                    var m = MatAsset("VillagePetal" + (i % 3) + ".mat", petals[i % 3], 0f, 0.2f);
                    PrimPart(root, PrimitiveType.Sphere, "Flower",
                             at + new Vector3(Range(-0.5f, 0.5f), 0.08f, Range(-0.5f, 0.5f)),
                             Vector3.one * Range(0.06f, 0.10f), Vector3.zero, m);
                }
            }
            Flowers(new Vector3(-6.6f, 0f, 5.2f));
            Flowers(new Vector3(6.8f, 0f, 4.0f));
            Flowers(new Vector3(-0.8f, 0.12f, 18.3f));
            Flowers(new Vector3(-6.2f, 0f, 25.8f));
            Flowers(new Vector3(6.3f, 0f, 27.2f));

            // chickens loose on the square
            BuildChicken(new Vector3(-4.0f, 0.15f, 13.5f), new Vector2(2.2f, 2.0f));
            BuildChicken(new Vector3(4.3f, 0.15f, 17.8f), new Vector2(1.8f, 1.8f));
            BuildChicken(new Vector3(-3.3f, 0.15f, 20.3f), new Vector2(1.8f, 1.5f));

            // two background villagers — no dialogue, just people existing in the town
            BuildExtra(extraCtrl, "Extra_Shopper", new Vector3(-4.7f, 0f, 11.3f), new Vector3(-5.9f, 0f, 12.0f),
                       new Color(0.72f, 0.62f, 0.50f));
            BuildExtra(extraCtrl, "Extra_Idler", new Vector3(2.1f, 0f, 18.1f), new Vector3(0.6f, 0f, 16.8f),
                       new Color(0.52f, 0.58f, 0.48f));
        }

        static void BuildChicken(Vector3 pos, Vector2 wanderHalfExtents)
        {
            var white = MatAsset("VillageChickenWhite.mat", new Color(0.93f, 0.91f, 0.86f), 0f, 0.15f);
            var red = MatAsset("VillageChickenRed.mat", new Color(0.78f, 0.22f, 0.18f), 0f, 0.2f);
            var beakMat = MatAsset("VillageChickenBeak.mat", new Color(0.85f, 0.60f, 0.20f), 0f, 0.2f);
            var legMat = MatAsset("VillageChickenBeak.mat", new Color(0.85f, 0.60f, 0.20f), 0f, 0.2f);

            var chicken = new GameObject("Chicken");
            chicken.layer = 2;
            chicken.transform.position = pos;
            chicken.transform.rotation = Quaternion.Euler(0f, Range(0f, 360f), 0f);

            var body = new GameObject("BodyPivot").transform;
            body.SetParent(chicken.transform, false);
            PrimPart(body, PrimitiveType.Cube, "Body", new Vector3(0, 0.19f, 0), new Vector3(0.20f, 0.18f, 0.28f), Vector3.zero, white);
            PrimPart(body, PrimitiveType.Cube, "Tail", new Vector3(0, 0.28f, -0.16f), new Vector3(0.10f, 0.12f, 0.08f), new Vector3(-25f, 0, 0), white);
            foreach (float lx in new[] { -0.05f, 0.05f })
                PrimPart(body, PrimitiveType.Cube, "Leg", new Vector3(lx, 0.05f, 0.02f), new Vector3(0.025f, 0.10f, 0.025f), Vector3.zero, legMat);

            var head = new GameObject("HeadPivot").transform;
            head.SetParent(body, false);
            head.localPosition = new Vector3(0, 0.30f, 0.12f);
            PrimPart(head, PrimitiveType.Cube, "Head", new Vector3(0, 0.05f, 0.02f), new Vector3(0.11f, 0.13f, 0.11f), Vector3.zero, white);
            PrimPart(head, PrimitiveType.Cube, "Comb", new Vector3(0, 0.14f, 0.02f), new Vector3(0.03f, 0.06f, 0.08f), Vector3.zero, red);
            PrimPart(head, PrimitiveType.Cube, "Beak", new Vector3(0, 0.04f, 0.10f), new Vector3(0.03f, 0.03f, 0.06f), Vector3.zero, beakMat);

            var wander = chicken.AddComponent<PenWanderer>();
            var so = new SerializedObject(wander);
            so.FindProperty("penCenter").vector3Value = pos;
            so.FindProperty("penHalfExtents").vector2Value = wanderHalfExtents;
            so.FindProperty("speed").floatValue = 0.5f;
            so.FindProperty("turnDegPerSec").floatValue = 320f;
            so.ApplyModifiedPropertiesWithoutUndo();

            var peck = chicken.AddComponent<ChickenPeck>();
            SetRef(peck, "head", head);
            SetRef(peck, "body", body);
            SetRef(peck, "wanderer", wander);
        }

        static GameObject BuildExtra(RuntimeAnimatorController ctrl, string name, Vector3 pos, Vector3 lookAt, Color tint)
        {
            var root = new GameObject(name);
            root.layer = 2;
            root.transform.position = pos;
            Vector3 to = lookAt - pos; to.y = 0f;
            if (to.sqrMagnitude > 1e-4f) root.transform.rotation = Quaternion.LookRotation(to.normalized);

            var body = root.AddComponent<CapsuleCollider>();
            body.center = new Vector3(0, 0.85f, 0);
            body.height = 1.7f;
            body.radius = 0.35f;

            var model = (GameObject)PrefabUtility.InstantiatePrefab(LoadModel("Characters/Rogue.fbx"));
            model.name = name + "Model";
            model.transform.SetParent(root.transform, false);
            SetLayerRecursive(model, 2);
            Bounds b = RenderererSafeBounds(model);
            float scale = b.size.y > 0.01f ? 1.72f / b.size.y : 1f;
            model.transform.localScale *= scale;
            GroundModel(model, root.transform.position.y);
            ApplyCharacterTexture(model, "Rogue_Texture.png", "Village" + name, tint);
            // a villager, not a cutthroat — hide the pack's baked-in dagger and weapon mount
            HideParts(model, "Dagger", "Weapon", "Knife");

            var anim = model.GetComponent<Animator>();
            if (anim == null) anim = model.AddComponent<Animator>();
            anim.runtimeAnimatorController = ctrl;
            anim.applyRootMotion = false;
            anim.cullingMode = AnimatorCullingMode.CullUpdateTransforms;

            root.AddComponent<BreathingIdle>();
            return root;
        }

        // ---------------------------------------------------------------- animals (primitive-built)

        static GameObject BuildCow(Vector3 pos, out QuadrupedGait gait)
        {
            var hide = MatAsset("VillageCowHide.mat", new Color(0.93f, 0.90f, 0.84f), 0f, 0.12f);
            var patch = MatAsset("VillageCowPatch.mat", new Color(0.30f, 0.23f, 0.18f), 0f, 0.12f);
            var pink = MatAsset("VillageCowPink.mat", new Color(0.86f, 0.62f, 0.58f), 0f, 0.15f);
            var hoof = MatAsset("VillageHoof.mat", new Color(0.16f, 0.13f, 0.11f), 0f, 0.2f);
            var horn = MatAsset("VillageHorn.mat", new Color(0.82f, 0.78f, 0.68f), 0f, 0.25f);

            var root = new GameObject("Cow");
            root.layer = 2;   // keep the orbit camera's collision cast from jolting off her
            root.transform.position = pos;

            var col = root.AddComponent<CapsuleCollider>();
            col.direction = 2;   // along Z — she is longer than tall
            col.center = new Vector3(0, 0.8f, 0);
            col.height = 1.9f;
            col.radius = 0.42f;

            PrimPart(root.transform, PrimitiveType.Cube, "Body", new Vector3(0, 0.95f, 0), new Vector3(0.72f, 0.72f, 1.55f), Vector3.zero, hide);
            PrimPart(root.transform, PrimitiveType.Cube, "PatchL", new Vector3(0.33f, 1.02f, 0.28f), new Vector3(0.10f, 0.42f, 0.55f), Vector3.zero, patch);
            PrimPart(root.transform, PrimitiveType.Cube, "PatchR", new Vector3(-0.33f, 0.92f, -0.30f), new Vector3(0.10f, 0.38f, 0.48f), Vector3.zero, patch);
            PrimPart(root.transform, PrimitiveType.Cube, "PatchBack", new Vector3(0.05f, 1.28f, -0.55f), new Vector3(0.45f, 0.12f, 0.40f), Vector3.zero, patch);
            PrimPart(root.transform, PrimitiveType.Sphere, "Udder", new Vector3(0, 0.52f, -0.32f), new Vector3(0.34f, 0.24f, 0.30f), Vector3.zero, pink);

            // head on its own pivot so the gait can bob and graze it
            var headPivot = new GameObject("HeadPivot").transform;
            headPivot.SetParent(root.transform, false);
            headPivot.localPosition = new Vector3(0, 1.22f, 0.82f);
            PrimPart(headPivot, PrimitiveType.Cube, "Head", new Vector3(0, 0.02f, 0.18f), new Vector3(0.38f, 0.40f, 0.42f), Vector3.zero, hide);
            PrimPart(headPivot, PrimitiveType.Cube, "Snout", new Vector3(0, -0.10f, 0.42f), new Vector3(0.30f, 0.20f, 0.16f), Vector3.zero, pink);
            PrimPart(headPivot, PrimitiveType.Cube, "EarL", new Vector3(0.26f, 0.14f, 0.10f), new Vector3(0.16f, 0.07f, 0.10f), new Vector3(0, 0, -12f), hide);
            PrimPart(headPivot, PrimitiveType.Cube, "EarR", new Vector3(-0.26f, 0.14f, 0.10f), new Vector3(0.16f, 0.07f, 0.10f), new Vector3(0, 0, 12f), hide);
            PrimPart(headPivot, PrimitiveType.Cube, "HornL", new Vector3(0.14f, 0.26f, 0.12f), new Vector3(0.07f, 0.16f, 0.07f), new Vector3(0, 0, -18f), horn);
            PrimPart(headPivot, PrimitiveType.Cube, "HornR", new Vector3(-0.14f, 0.26f, 0.12f), new Vector3(0.07f, 0.16f, 0.07f), new Vector3(0, 0, 18f), horn);

            var tailPivot = new GameObject("TailPivot").transform;
            tailPivot.SetParent(root.transform, false);
            tailPivot.localPosition = new Vector3(0, 1.25f, -0.78f);
            PrimPart(tailPivot, PrimitiveType.Cube, "Tail", new Vector3(0, -0.30f, -0.03f), new Vector3(0.07f, 0.60f, 0.07f), Vector3.zero, hide);
            PrimPart(tailPivot, PrimitiveType.Cube, "Tuft", new Vector3(0, -0.63f, -0.03f), new Vector3(0.10f, 0.12f, 0.10f), Vector3.zero, patch);

            // rope halter + the anchor the leash attaches to (the rope itself is a LineRenderer
            // strung from Odo's hand at runtime — see BuildStrollDirector)
            var ropeMat = MatAsset("VillageRope.mat", new Color(0.42f, 0.38f, 0.30f), 0f, 0.05f);
            var neck = new GameObject("NeckAnchor").transform;
            neck.SetParent(root.transform, false);
            neck.localPosition = new Vector3(0f, 1.18f, 0.72f);
            PrimPart(neck, PrimitiveType.Cylinder, "Halter", Vector3.zero, new Vector3(0.46f, 0.03f, 0.50f), new Vector3(12f, 0, 0), ropeMat);

            Transform Leg(string name, float x, float z)
            {
                var pivot = new GameObject(name).transform;
                pivot.SetParent(root.transform, false);
                pivot.localPosition = new Vector3(x, 0.68f, z);
                PrimPart(pivot, PrimitiveType.Cube, "Bone", new Vector3(0, -0.30f, 0), new Vector3(0.16f, 0.62f, 0.16f), Vector3.zero, hide);
                PrimPart(pivot, PrimitiveType.Cube, "Hoof", new Vector3(0, -0.585f, 0), new Vector3(0.17f, 0.09f, 0.17f), Vector3.zero, hoof);
                return pivot;
            }
            var fl = Leg("LegFL", 0.26f, 0.58f);
            var fr = Leg("LegFR", -0.26f, 0.58f);
            var rl = Leg("LegRL", 0.26f, -0.58f);
            var rr = Leg("LegRR", -0.26f, -0.58f);

            gait = root.AddComponent<QuadrupedGait>();
            SetRef(gait, "legFL", fl); SetRef(gait, "legFR", fr);
            SetRef(gait, "legRL", rl); SetRef(gait, "legRR", rr);
            SetRef(gait, "head", headPivot); SetRef(gait, "tail", tailPivot);
            return root;
        }

        static void BuildPig(Vector3 pos)
        {
            var pinkHide = MatAsset("VillagePigHide.mat", new Color(0.88f, 0.66f, 0.60f), 0f, 0.18f);
            var pinkDark = MatAsset("VillagePigDark.mat", new Color(0.78f, 0.54f, 0.50f), 0f, 0.18f);

            var root = new GameObject("Pig");
            root.layer = 2;
            root.transform.position = pos;
            root.transform.rotation = Quaternion.Euler(0f, Range(0f, 360f), 0f);

            PrimPart(root.transform, PrimitiveType.Cube, "Body", new Vector3(0, 0.42f, 0), new Vector3(0.46f, 0.42f, 0.85f), Vector3.zero, pinkHide);

            var headPivot = new GameObject("HeadPivot").transform;
            headPivot.SetParent(root.transform, false);
            headPivot.localPosition = new Vector3(0, 0.52f, 0.45f);
            PrimPart(headPivot, PrimitiveType.Cube, "Head", new Vector3(0, 0f, 0.10f), new Vector3(0.30f, 0.28f, 0.26f), Vector3.zero, pinkHide);
            PrimPart(headPivot, PrimitiveType.Cube, "Snout", new Vector3(0, -0.04f, 0.26f), new Vector3(0.14f, 0.12f, 0.07f), Vector3.zero, pinkDark);
            PrimPart(headPivot, PrimitiveType.Cube, "EarL", new Vector3(0.12f, 0.16f, 0.06f), new Vector3(0.09f, 0.09f, 0.04f), new Vector3(-15f, 0, -14f), pinkDark);
            PrimPart(headPivot, PrimitiveType.Cube, "EarR", new Vector3(-0.12f, 0.16f, 0.06f), new Vector3(0.09f, 0.09f, 0.04f), new Vector3(-15f, 0, 14f), pinkDark);

            var tailPivot = new GameObject("TailPivot").transform;
            tailPivot.SetParent(root.transform, false);
            tailPivot.localPosition = new Vector3(0, 0.52f, -0.44f);
            PrimPart(tailPivot, PrimitiveType.Cube, "Tail1", new Vector3(0, 0.02f, -0.04f), new Vector3(0.05f, 0.05f, 0.09f), new Vector3(30f, 0, 0), pinkDark);
            PrimPart(tailPivot, PrimitiveType.Cube, "Tail2", new Vector3(0, 0.07f, -0.09f), new Vector3(0.04f, 0.04f, 0.07f), new Vector3(-40f, 0, 0), pinkDark);

            Transform Leg(string name, float x, float z)
            {
                var pivot = new GameObject(name).transform;
                pivot.SetParent(root.transform, false);
                pivot.localPosition = new Vector3(x, 0.24f, z);
                PrimPart(pivot, PrimitiveType.Cube, "Bone", new Vector3(0, -0.10f, 0), new Vector3(0.10f, 0.24f, 0.10f), Vector3.zero, pinkDark);
                return pivot;
            }
            var fl = Leg("LegFL", 0.15f, 0.30f);
            var fr = Leg("LegFR", -0.15f, 0.30f);
            var rl = Leg("LegRL", 0.15f, -0.30f);
            var rr = Leg("LegRR", -0.15f, -0.30f);

            var gait = root.AddComponent<QuadrupedGait>();
            SetRef(gait, "legFL", fl); SetRef(gait, "legFR", fr);
            SetRef(gait, "legRL", rl); SetRef(gait, "legRR", rr);
            SetRef(gait, "head", headPivot); SetRef(gait, "tail", tailPivot);
            SetFloat(gait, "strideLength", 0.32f);
            SetFloat(gait, "swingDegrees", 20f);
            SetFloat(gait, "grazeDegrees", 30f);

            var wander = root.AddComponent<PenWanderer>();
            var so = new SerializedObject(wander);
            so.FindProperty("penCenter").vector3Value = PEN_CENTER;
            so.FindProperty("penHalfExtents").vector2Value = new Vector2(1.25f, 1.25f);
            so.ApplyModifiedPropertiesWithoutUndo();
            SetRef(wander, "gait", gait);
        }

        // ---------------------------------------------------------------- greenery

        static void ScatterGreenery()
        {
            var green = new GameObject("Greenery").transform;
            green.SetParent(envRoot, false);

            string[] ALIVE = { "Tree_1", "Tree_2", "Tree_3" };
            string[] BRUSH = { "Bush_1x1", "Bush_Round", "Bush_Large", "Bush_2x1", "Grass" };

            // hand-set trees framing the village
            (Vector3 pos, float sc)[] villageTrees =
            {
                (new Vector3(-12.5f, 0f,  7.5f), 1.25f),
                (new Vector3(-12.8f, 0f, 19.5f), 1.1f),
                (new Vector3( 12.6f, 0f,  4.5f), 1.2f),
                (new Vector3( 12.9f, 0f, 16.0f), 1.35f),
                (new Vector3(  7.6f, 0f, 26.5f), 1.15f),
                (new Vector3( -6.4f, 0f, 27.6f), 1.0f),
                (new Vector3( -3.4f, 0f, -10.5f), 1.2f),
                (new Vector3(  4.2f, 0f, -13.5f), 1.1f),
            };
            foreach (var (p, sc) in villageTrees)
                PlacePiece(ALIVE[rng.Next(ALIVE.Length)], new Vector3(p.x, GroundHeight(p.x, p.z), p.z),
                           Range(0, 360), green, scale: sc);

            // grass tufts and bushes inside the village, clear of streets/square/pen
            for (int i = 0; i < 46; i++)
            {
                float x = Range(-14f, 14f), z = Range(-15f, 32f);
                if (Mathf.Abs(x) < 2.6f && z < 10f) continue;                          // main street
                if (Mathf.Abs(x) < 8.4f && z > 8.3f && z < 22.1f) continue;            // square
                if (RectDist(x, z, PEN_CENTER.x, PEN_CENTER.z, 2.6f, 2.6f) < 0.5f) continue;
                PlacePiece(BRUSH[rng.Next(BRUSH.Length)], new Vector3(x, GroundHeight(x, z), z),
                           Range(0, 360), green, collider: false);
            }

            // meadow-and-treeline ring outside the village bounds — wider and denser now that
            // the world continues over the ridge, and always CLEAR of the roads
            var rock = MatAsset("VillageRock.mat", new Color(0.55f, 0.53f, 0.49f), 0f, 0.08f);
            int placed = 0, guard = 0;
            while (placed < 230 && guard++ < 9000)
            {
                float x = Range(-95f, 95f), z = Range(-60f, 125f);
                float d = RectDist(x, z, VILLAGE_CX, VILLAGE_CZ, VILLAGE_HX, VILLAGE_HZ);
                if (d < 3f) continue;
                if (DistToRoads(x, z) < 3.6f) continue;
                double roll = rng.NextDouble();
                bool isTree = roll < 0.58;
                float far = Mathf.Clamp01((d - 6f) / 30f);
                float scale = Range(0.9f, 1.35f) * (isTree ? Mathf.Lerp(1.0f, 1.8f, far) : 1f);
                PlacePiece(isTree ? ALIVE[rng.Next(ALIVE.Length)] : BRUSH[rng.Next(BRUSH.Length)],
                           new Vector3(x, GroundHeight(x, z), z), Range(0, 360), green,
                           scale: scale, collider: isTree);
                placed++;
            }

            // boulders breaking up the hillside and the meadow edges
            for (int i = 0; i < 26; i++)
            {
                float x = Range(-70f, 70f), z = Range(-45f, 115f);
                if (RectDist(x, z, VILLAGE_CX, VILLAGE_CZ, VILLAGE_HX, VILLAGE_HZ) < 2f) continue;
                if (DistToRoads(x, z) < 3f) continue;
                float s = Range(0.4f, 1.5f);
                PrimPart(green, PrimitiveType.Sphere, "Boulder",
                         new Vector3(x, GroundHeight(x, z) + s * 0.18f, z),
                         new Vector3(s, s * Range(0.45f, 0.7f), s * Range(0.7f, 1f)),
                         new Vector3(Range(-8f, 8f), Range(0, 360f), Range(-8f, 8f)), rock, collider: s > 0.9f);
            }

            // grass tufts thickening the near meadow (cheap, and exactly what "bland" was)
            int tufts = 0; guard = 0;
            while (tufts < 90 && guard++ < 4000)
            {
                float x = Range(-45f, 45f), z = Range(-35f, 95f);
                if (RectDist(x, z, VILLAGE_CX, VILLAGE_CZ, VILLAGE_HX, VILLAGE_HZ) < 1.5f) continue;
                if (DistToRoads(x, z) < 2.4f) continue;
                PlacePiece(rng.NextDouble() < 0.7 ? "Grass" : BRUSH[rng.Next(BRUSH.Length)],
                           new Vector3(x, GroundHeight(x, z), z), Range(0, 360), green, collider: false);
                tufts++;
            }
        }

        // ---------------------------------------------------------------- animators

        static RuntimeAnimatorController CreatePlayerAnimator()
        {
            string path = GEN + "/VillagePlayerAnimator.controller";
            AssetDatabase.DeleteAsset(path);
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
                ("RunB",      "Animations/UAL1.fbx", "Jog_Fwd_Loop"),
                ("Interact",  "Animations/UAL1.fbx", "Interact"),
                ("MistWalk",  "Animations/UAL1.fbx", "Push_Loop"),
            };
            foreach (var (state, fbx, clipName) in map)
            {
                var st = sm.AddState(state);
                var clip = Clip(fbx, clipName);
                st.motion = clip;
                float natural = clip.averageSpeed.magnitude;
                float Sync(float desired, float fallback) => natural < 0.5f ? fallback : Mathf.Clamp(desired / natural, 0.75f, 1.5f);
                st.speed = state switch
                {
                    "Run" => Sync(4.3f, 1.30f),
                    "RunB" => -Sync(4.3f, 1.30f),
                    "Sprint" => Sync(7.0f, 1.18f),
                    "Walk" => Sync(1.7f, 1.05f),
                    "MistWalk" => natural < 0.5f ? 0.6f : Mathf.Clamp(1.25f / natural, 0.4f, 1f),
                    _ => 1f,
                };
                if (state == "Idle") sm.defaultState = st;
            }
            return ctrl;
        }

        // stationary humanoid NPC (the fishmonger): the standard two-state Idle/Talking pair
        static RuntimeAnimatorController CreateMonkNpcAnimator()
        {
            string path = GEN + "/VillageMonkAnimator.controller";
            AssetDatabase.DeleteAsset(path);
            var ctrl = AnimatorController.CreateAnimatorControllerAtPath(path);
            var sm = ctrl.layers[0].stateMachine;
            var idle = sm.AddState("Idle"); idle.motion = Clip("Animations/UAL1.fbx", "Idle_Loop");
            var talk = sm.AddState("Talking"); talk.motion = Clip("Animations/UAL1.fbx", "Idle_Talking_Loop");
            sm.defaultState = idle;
            return ctrl;
        }

        // strolling generic-rig NPC: Idle / Walk / Talking from the rig's own embedded clips
        static RuntimeAnimatorController CreateVillagerAnimator(string assetName, string fbxRel,
                                                                string idleClip, string walkClip, string talkClip)
        {
            string path = GEN + "/" + assetName + ".controller";
            AssetDatabase.DeleteAsset(path);
            var ctrl = AnimatorController.CreateAnimatorControllerAtPath(path);
            var sm = ctrl.layers[0].stateMachine;
            var idle = sm.AddState("Idle"); idle.motion = Clip(fbxRel, idleClip);
            var walk = sm.AddState("Walk"); walk.motion = Clip(fbxRel, walkClip);
            walk.speed = 0.95f;   // ~0.85 m/s stroll on these packs' in-place Walk cycles
            var talk = sm.AddState("Talking"); talk.motion = Clip(fbxRel, talkClip);
            sm.defaultState = idle;
            return ctrl;
        }

        // strolling NPC (Fenn on the Monk rig): Idle/Walk retargeted from UAL, but Talking uses
        // the same Wizard Spell1 gesture as Odo (shared CharacterArmature skeleton) for a visible
        // talking animation; the humanoid rig also keeps the procedural head-nod
        static RuntimeAnimatorController CreateMonkStrollerAnimator()
        {
            string path = GEN + "/VillageFennAnimator.controller";
            AssetDatabase.DeleteAsset(path);
            var ctrl = AnimatorController.CreateAnimatorControllerAtPath(path);
            var sm = ctrl.layers[0].stateMachine;
            var idle = sm.AddState("Idle"); idle.motion = Clip("Animations/UAL1.fbx", "Idle_Loop");
            var walkClip = Clip("Animations/UAL1.fbx", "Walk_Loop");
            var walk = sm.AddState("Walk"); walk.motion = walkClip;
            float natural = walkClip.averageSpeed.magnitude;
            walk.speed = natural < 0.5f ? 0.72f : Mathf.Clamp(0.85f / natural, 0.4f, 1.2f);   // stroll pace
            var talk = sm.AddState("Talking"); talk.motion = Clip("Characters/Wizard.fbx", "Spell1");
            sm.defaultState = idle;
            return ctrl;
        }

        // background villager: one looping state from the rig's own clip set
        static RuntimeAnimatorController CreateExtraAnimator(string assetName, string fbxRel, string idleClip)
        {
            string path = GEN + "/" + assetName + ".controller";
            AssetDatabase.DeleteAsset(path);
            var ctrl = AnimatorController.CreateAnimatorControllerAtPath(path);
            var sm = ctrl.layers[0].stateMachine;
            var idle = sm.AddState("Idle"); idle.motion = Clip(fbxRel, idleClip);
            sm.defaultState = idle;
            return ctrl;
        }

        // ---------------------------------------------------------------- player + camera

        static GameObject BuildPlayer(RuntimeAnimatorController ctrl)
        {
            var root = new GameObject("Player");
            root.tag = "Player";
            root.layer = 2;
            root.transform.position = PLAYER_SPAWN;
            root.transform.rotation = Quaternion.identity;   // facing up the street into the village

            var cc = root.AddComponent<CharacterController>();
            cc.center = new Vector3(0, 0.95f, 0);
            cc.height = 1.8f;
            cc.radius = 0.35f;
            cc.slopeLimit = 50f;

            var model = (GameObject)PrefabUtility.InstantiatePrefab(LoadModel("Characters/Warrior.fbx"));
            model.name = "WarriorModel";
            model.transform.SetParent(root.transform, false);
            SetLayerRecursive(model, 2);
            Bounds b = RenderererSafeBounds(model);
            float scale = b.size.y > 0.01f ? 1.8f / b.size.y : 1f;
            model.transform.localScale *= scale;
            GroundModel(model, root.transform.position.y);
            ApplyCharacterTexture(model, "Warrior_Texture.png", "VillagePlayerWarrior", Color.white);

            var anim = model.GetComponent<Animator>();
            if (anim == null) anim = model.AddComponent<Animator>();
            anim.runtimeAnimatorController = ctrl;
            anim.applyRootMotion = false;
            anim.cullingMode = AnimatorCullingMode.AlwaysAnimate;

            // an armed traveler walking into a market town — sword on the mount, shield in fist
            Transform weaponMount = FindDeep(model.transform, "Weapon.R");
            GameObject sword = weaponMount != null
                ? AttachToTransform(weaponMount, LoadModel("Weapons/Sword.fbx"), "Sword")
                : AttachToBone(anim, HumanBodyBones.RightHand, LoadModel("Weapons/Sword.fbx"), "Sword");
            GameObject shield = AttachToBone(anim, HumanBodyBones.LeftHand, LoadModel("Weapons/Shield_Heater.fbx"), "Shield");
            NormalizeWorldSize(sword, 1.15f);
            NormalizeWorldSize(shield, 0.80f);
            if (shield != null)
                shield.transform.localRotation = Quaternion.Euler(270f, 0f, 0f) * shield.transform.localRotation;

            root.AddComponent<BreathingIdle>();

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

        static GameObject BuildHealFlask()
        {
            var gold = MatAsset("VillageFlaskGold.mat", new Color(0.85f, 0.65f, 0.2f), 0.6f, 0.7f);
            var flask = new GameObject("HealFlask");
            PrimPart(flask.transform, PrimitiveType.Sphere, "Body", new Vector3(0, 0.5f, 0), new Vector3(1f, 1.2f, 1f), Vector3.zero, gold);
            PrimPart(flask.transform, PrimitiveType.Cylinder, "Neck", new Vector3(0, 1.15f, 0), new Vector3(0.35f, 0.25f, 0.35f), Vector3.zero, gold);
            flask.SetActive(true);
            return flask;
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

        static GameObject AttachToBone(Animator anim, HumanBodyBones bone, GameObject prefab, string name)
        {
            Transform t = anim.GetBoneTransform(bone);
            if (t == null) { Debug.LogWarning("[TradingVillageBuilder] missing bone " + bone); return null; }
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

        static void NormalizeWorldSize(GameObject item, float targetSize)
        {
            if (item == null) return;
            Bounds b = RenderererSafeBounds(item);
            float longest = Mathf.Max(b.size.x, Mathf.Max(b.size.y, b.size.z));
            if (longest > 0.0001f) item.transform.localScale *= targetSize / longest;
        }

        static Transform FindDeep(Transform root, string name)
        {
            foreach (var t in root.GetComponentsInChildren<Transform>(true))
                if (t.name == name) return t;
            return null;
        }

        // ---------------------------------------------------------------- NPCs

        // All three villagers run the SAME (model, quant, KV-cap) tuple on purpose: LLMPool keys
        // instances on it, so one Qwen3.5-0.8B int8 on the GPU serves the whole village. That
        // means one shared maxContextLength too — and since all three ResumeFromCompact, the pool
        // cap (ctx + compact headroom) matches across them.
        const string VILLAGE_MODEL = "Qwen3.5-0.8B";
        const int VILLAGE_CONTEXT = 3000;

        static void ConfigureVillagerChat(NPCChatBase npc, string npcName, string persona,
                                          string bakedVoice, AudioClip cloneClip, float pitch)
        {
            SetString(npc, "NpcName", npcName);
            SetString(npc, "descriptionAndRules", persona);
            SetString(npc, "model", VILLAGE_MODEL);
            SetEnum(npc, "historyMode", (int)NPCChatBase.HistoryMode.ResumeFromCompact);
            SetInt(npc, "maxContextLength", VILLAGE_CONTEXT);
            SetEnum(npc, "conversationMode", (int)NPCChatBase.ConversationMode.LlmPlusTts);
            SetEnum(npc, "ttsModel", (int)NPCChatBase.TtsModel.PocketTTS);
            SetString(npc, "ttsVoice", bakedVoice);
            if (cloneClip != null) SetObject(npc, "clonedVoiceClip", cloneClip);
            SetFloat(npc, "voicePitch", pitch);
            SetFloat(npc, "voiceVolume", 4f);
            SetFloat(npc, "worldAudioWhileInteracting", 0.6f);
            SetInt(npc, "clausesPerChunk", 2);
            SetBool(npc, "usePrefetchZone", true);
        }

        static GameObject BuildFishmonger(RuntimeAnimatorController ctrl)
        {
            var root = new GameObject("NPC_Bram");
            root.layer = 2;
            root.transform.position = BRAM_POS;
            root.transform.rotation = Quaternion.Euler(0f, -90f, 0f);   // behind the counter, facing the square

            var body = root.AddComponent<CapsuleCollider>();
            body.center = new Vector3(0, 0.9f, 0);
            body.height = 1.8f;
            body.radius = 0.4f;

            var trigger = root.AddComponent<SphereCollider>();
            trigger.isTrigger = true;
            trigger.radius = 2.4f;   // slightly generous: the counter stands between him and the customer
            trigger.center = new Vector3(0, 1f, 0);

            var model = (GameObject)PrefabUtility.InstantiatePrefab(LoadModel("Characters/Monk.fbx"));
            model.name = "BramModel";
            model.transform.SetParent(root.transform, false);
            SetLayerRecursive(model, 2);
            Bounds b = RenderererSafeBounds(model);
            float scale = b.size.y > 0.01f ? 1.8f / b.size.y : 1f;
            model.transform.localScale *= scale;
            GroundModel(model, root.transform.position.y);
            // river-grey working clothes — NOT Velmire's ghost-pale robes
            ApplyCharacterTexture(model, "Monk_Texture.png", "VillageBram", new Color(0.72f, 0.78f, 0.80f));

            var anim = model.GetComponent<Animator>();
            if (anim == null) anim = model.AddComponent<Animator>();
            anim.runtimeAnimatorController = ctrl;
            anim.applyRootMotion = false;
            anim.cullingMode = AnimatorCullingMode.CullUpdateTransforms;

            var camPoint = new GameObject("DialogueCameraPoint").transform;
            camPoint.SetParent(root.transform, false);
            Vector3 worldCamPos = BRAM_POS + root.transform.forward * 2.4f + root.transform.right * 1.0f + Vector3.up * 1.65f;
            camPoint.position = worldCamPos;
            camPoint.rotation = Quaternion.LookRotation((BRAM_POS + Vector3.up * 1.45f) - worldCamPos);

            root.AddComponent<BreathingIdle>();
            var npc = root.AddComponent<VillageInteractor>();
            SetRef(npc, "dialogueCameraPoint", camPoint);
            ConfigureVillagerChat(npc, "Bram, the Fishmonger", BramPersona(), "jean",
                AssetDatabase.LoadAssetAtPath<AudioClip>(ROOT + "/Voices/Moore.mp3"), 0.93f);
            SetFloat(npc, "prefetchRadius", 12f);

            // his beat is handing a fish over at a price — that is GiveItem, and only GiveItem
            var f = typeof(NPCChatBase).GetField("descriptionAndRules",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
            f.SetValue(npc, npc.WithToolsBlock((string)f.GetValue(npc), askUserQuestion: false, giveItem: true, toolsFirst: false));
            return root;
        }

        static GameObject BuildStroller(string goName, string npcName, string persona,
                                        string modelFbx, RuntimeAnimatorController ctrl, float height,
                                        string tintMat, Color tint, string[] hideFragments,
                                        string cloneClipPath, float pitch, Vector3 startPos)
        {
            var root = new GameObject(goName);
            root.layer = 2;
            root.transform.position = startPos;

            var body = root.AddComponent<CapsuleCollider>();
            body.center = new Vector3(0, 0.9f, 0);
            body.height = 1.8f;
            body.radius = 0.38f;

            var trigger = root.AddComponent<SphereCollider>();
            trigger.isTrigger = true;
            trigger.radius = 2.2f;
            trigger.center = new Vector3(0, 1f, 0);

            var model = (GameObject)PrefabUtility.InstantiatePrefab(LoadModel(modelFbx));
            model.name = goName + "Model";
            model.transform.SetParent(root.transform, false);
            SetLayerRecursive(model, 2);
            Bounds b = RenderererSafeBounds(model);
            float scale = b.size.y > 0.01f ? height / b.size.y : 1f;
            model.transform.localScale *= scale;
            GroundModel(model, root.transform.position.y);
            if (tintMat != null)
            {
                string texFile = Path.GetFileNameWithoutExtension(modelFbx) + "_Texture.png";
                ApplyCharacterTexture(model, texFile, tintMat, tint);
            }
            HideParts(model, hideFragments);

            var anim = model.GetComponent<Animator>();
            if (anim == null) anim = model.AddComponent<Animator>();
            anim.runtimeAnimatorController = ctrl;
            anim.applyRootMotion = false;
            // they cross the whole square — pose pops at the screen edge would be visible, and
            // two rigs animating is cheap
            anim.cullingMode = AnimatorCullingMode.AlwaysAnimate;

            var camPoint = new GameObject("DialogueCameraPoint").transform;
            camPoint.SetParent(root.transform, false);
            camPoint.localPosition = new Vector3(1.0f, 1.65f, 2.4f);   // recomputed live on open

            var npc = root.AddComponent<VillageStroller>();
            SetRef(npc, "dialogueCameraPoint", camPoint);
            var clip = AssetDatabase.LoadAssetAtPath<AudioClip>(cloneClipPath);
            if (clip == null) Debug.LogWarning("[TradingVillageBuilder] clone clip missing: " + cloneClipPath +
                                               " — " + npcName + " falls back to the baked voice.");
            ConfigureVillagerChat(npc, npcName, persona, "jean", clip, pitch);
            // the pair MOVES: a generous zone means the models stream in while the player is
            // anywhere near the village core, not just brushing past the NPC
            SetFloat(npc, "prefetchRadius", 22f);
            return root;
        }

        // ---------------------------------------------------------------- stroll director

        static void BuildStrollDirector(TMP_FontAsset cinzel, GameObject odo, GameObject fenn,
                                        GameObject cow, QuadrupedGait cowGait)
        {
            var dir = new GameObject("StrollDirector");
            var group = dir.AddComponent<VillageStrollGroup>();
            var banter = dir.AddComponent<VillageBanter>();

            var so = new SerializedObject(group);
            var wp = so.FindProperty("waypoints");
            wp.arraySize = STROLL_LOOP.Length;
            for (int i = 0; i < STROLL_LOOP.Length; i++)
                wp.GetArrayElementAtIndex(i).vector3Value = STROLL_LOOP[i];
            // start on the leg that leaves the square heading south, so the player's first
            // sight of the pair is them coming from the market toward the spawn
            float startDist = 0.3f;
            for (int i = 0; i < 8; i++)
            {
                Vector3 wa = STROLL_LOOP[i], wb = STROLL_LOOP[i + 1];
                wa.y = 0; wb.y = 0;
                startDist += Vector3.Distance(wa, wb);
            }
            so.FindProperty("startDistance").floatValue = startDist;
            so.ApplyModifiedPropertiesWithoutUndo();

            var a = odo.GetComponent<VillageStroller>();
            var h = fenn.GetComponent<VillageStroller>();
            SetRef(group, "strollerA", a);
            SetRef(group, "strollerB", h);
            SetRef(group, "cow", cow.transform);
            SetRef(group, "cowGait", cowGait);
            SetRef(group, "banter", banter);
            SetRef(a, "group", group);
            SetRef(h, "group", group);

            // the leash: Odo's hand (or a shoulder socket if the generic rig hides its hand
            // bones under other names) to the cow's halter
            Transform holder = null;
            foreach (var candidate in new[] { "Fist.L", "Palm.L", "Hand.L", "hand_l", "LowerArm.L" })
            {
                holder = FindDeep(odo.transform, candidate);
                if (holder != null) break;
            }
            if (holder == null)
            {
                holder = new GameObject("LeashSocket").transform;
                holder.SetParent(odo.transform, false);
                holder.localPosition = new Vector3(-0.28f, 0.98f, -0.05f);
                Debug.LogWarning("[TradingVillageBuilder] no left-hand bone found on the Wizard rig — " +
                                 "leash holds from a fixed hip-height socket instead.");
            }
            var neckAnchor = FindDeep(cow.transform, "NeckAnchor");
            var leashGO = new GameObject("CowLeash");
            var line = leashGO.AddComponent<LineRenderer>();
            line.positionCount = 9;
            line.widthMultiplier = 0.035f;
            line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            line.receiveShadows = false;
            line.sharedMaterial = MatAsset("VillageRope.mat", new Color(0.42f, 0.38f, 0.30f), 0f, 0.05f);
            line.numCapVertices = 2;
            var leash = leashGO.AddComponent<CowLeash>();
            SetRef(leash, "holder", holder);
            SetRef(leash, "cowAnchor", neckAnchor);
            SetRef(leash, "line", line);

            // speech bubbles above their heads, revealed in step with the spoken clauses
            var bubbleMat = new Material(cinzel.material);
            bubbleMat.SetFloat(ShaderUtilities.ID_OutlineWidth, 0.22f);
            bubbleMat.SetColor(ShaderUtilities.ID_OutlineColor, new Color(0.05f, 0.04f, 0.03f, 1f));
            AssetDatabase.CreateAsset(bubbleMat, GEN + "/VillageBubbleText.mat");
            var bubA = BuildBubble(odo, a, cinzel, bubbleMat);
            var bubB = BuildBubble(fenn, h, cinzel, bubbleMat);

            SetRef(banter, "strollerA", a);
            SetRef(banter, "strollerB", h);
            SetRef(banter, "bubbleA", bubA);
            SetRef(banter, "bubbleB", bubB);

            // the gossip loop. 0 = Odo, 1 = Fenn. Short lines: each is one or two clauses of
            // real-time pocket-tts, and the pair should feel chatty, not lecture-y.
            (int who, string text)[] lines =
            {
                (0, "Fish again at Bram's stall. Third week running, I swear."),
                (1, "Better fish than nothing. The south road is mud to the knees."),
                (0, "Aye, the carters will not come till it dries. Salt costs double already."),
                (1, "And you will still buy it, you old goat. Your cabbage soup demands it."),
                (0, "My soup is famous, Fenn. Even the miller asks after it."),
                (1, "The miller asks after everything that is free."),
                (0, "Ha! True enough. At least the old girl back there never complains."),
                (1, "She is on a rope, Odo. She has no say in the matter."),
            };
            var bso = new SerializedObject(banter);
            var lp = bso.FindProperty("lines");
            lp.arraySize = lines.Length;
            for (int i = 0; i < lines.Length; i++)
            {
                var el = lp.GetArrayElementAtIndex(i);
                el.FindPropertyRelative("speaker").intValue = lines[i].who;
                el.FindPropertyRelative("text").stringValue = lines[i].text;
            }
            bso.ApplyModifiedPropertiesWithoutUndo();
        }

        static VillageSpeechBubble BuildBubble(GameObject npcRoot, VillageStroller owner,
                                               TMP_FontAsset cinzel, Material bubbleMat)
        {
            var go = new GameObject("SpeechBubble", typeof(RectTransform));
            go.transform.SetParent(npcRoot.transform, false);
            go.transform.localPosition = new Vector3(0f, 2.2f, 0f);   // just above the hairline

            // notification plate behind the text (user 2026-08-10): SEMI-TRANSPARENT light board
            // with BLACK text and a little tail pointing at whoever is speaking. Sliced sprites so
            // the runtime height-fit keeps the corners.
            var bgSprite = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/Background.psd");
            SpriteRenderer Plate(string name, Color color, float z, int order, bool sliced = true)
            {
                var pgo = new GameObject(name);
                pgo.transform.SetParent(go.transform, false);
                pgo.transform.localPosition = new Vector3(0f, 0.3f, z);
                var sr = pgo.AddComponent<SpriteRenderer>();
                sr.sprite = bgSprite;
                sr.drawMode = sliced ? SpriteDrawMode.Sliced : SpriteDrawMode.Simple;
                if (sliced) sr.size = new Vector2(1.0f, 0.3f);
                sr.color = color;
                sr.sortingOrder = order;
                return sr;
            }
            var rim = Plate("PlateRim", new Color(0.36f, 0.30f, 0.20f, 0.75f), 0.020f, 1);
            var plate = Plate("Plate", new Color(0.94f, 0.91f, 0.83f, 0.62f), 0.012f, 2);
            // the arrow: a rotated square, same tone as the plate, half of it peeking out below —
            // the bubble is a child of its speaker, so pointing down IS pointing at them
            var tailSr = Plate("Tail", new Color(0.94f, 0.91f, 0.83f, 0.62f), 0.010f, 3, sliced: false);
            tailSr.transform.localPosition = new Vector3(0f, -0.095f, 0.010f);
            tailSr.transform.localRotation = Quaternion.Euler(0f, 0f, 45f);
            tailSr.transform.localScale = Vector3.one * 0.34f;   // 0.32 m sprite -> ~0.11 m diamond

            var tmp = go.AddComponent<TextMeshPro>();
            tmp.rectTransform.sizeDelta = new Vector2(1.7f, 0.3f);
            tmp.rectTransform.pivot = new Vector2(0.5f, 0f);
            tmp.font = cinzel;
            // plain font material: black text needs no outline on a light board (the outlined
            // bubbleMat is what made the old floating text readable against the sky)
            tmp.fontSharedMaterial = cinzel.material;
            tmp.fontSize = 1.05f;   // half of the first pass — the board read as a billboard (user 2026-08-10)
            tmp.alignment = TextAlignmentOptions.Bottom;
            tmp.enableWordWrapping = true;
            tmp.color = new Color(0.09f, 0.08f, 0.06f, 0f);   // near-black ink; alpha driven at runtime
            tmp.GetComponent<MeshRenderer>().sortingOrder = 5;   // text above the plates

            var bub = go.AddComponent<VillageSpeechBubble>();
            SetRef(bub, "owner", owner);
            SetRef(bub, "text", tmp);
            SetRef(bub, "plate", plate);
            SetRef(bub, "plateRim", rim);
            SetRef(bub, "tail", tailSr);
            return bub;
        }

        // ---------------------------------------------------------------- personas

        static string BramPersona() =>
            "You are Bram, the fishmonger of Birchbrook, a small trading village. You stand at your " +
            "stall on the market square, apron stained, scale in hand.\n\n" +
            "VOICE. One or two short sentences, plain friendly market talk. Answer exactly what was " +
            "asked, then stop. Call the player 'friend'. Never describe your own actions. Stay in " +
            "character always: if asked about the real world, wave it off and return to your fish.\n\n" +
            "THE STALL. You sell what the weir gave this morning: fresh trout for 4 coins, salted " +
            "herring for 2, smoked eel for 6. You praise your fish honestly - caught at dawn, packed " +
            "in wet grass. You will knock one coin off for friendly haggling, never more; below that " +
            "you refuse plainly - 'That is the price, friend. The river does not haggle.' When the " +
            "player agrees to buy a fish at a price, call GiveItem in that same reply, with the item " +
            "named plainly (trout, herring or eel) and that price.\n\n" +
            "THE VILLAGE. The south road is mud until it dries, so salt costs double. Odo the peddler " +
            "and Fenn the herbalist stroll the square all day gossiping, Odo's cow in tow - good " +
            "people, endless tongues. The lord raised the taxes in spring and everyone grumbles.\n\n" +
            "MEMORY. When you are asked to 'Compact the conversation.', that is not the customer " +
            "speaking - it is your own memory being written down, and whatever you leave out is " +
            "forgotten for good. Answer with a plain third-person summary: who you spoke with, what " +
            "they asked, what fish and prices were discussed, whether anything was sold, and what is " +
            "still unsettled. Facts only, under 80 words.";

        static string OdoPersona() =>
            "You are Odo, an old peddler in the trading village of Birchbrook. You walk the market " +
            "loop with your friend Fenn the herbalist, as you do every day, leading your brown cow " +
            "on a rope behind you - she follows the sound of your voice and you pretend to mind.\n\n" +
            "VOICE. One or two short sentences, plain speech, dry humor. Answer exactly what was " +
            "asked, then stop. Call the player 'traveler'. Never describe your own actions. Stay in " +
            "character always: if asked about the real world, chuckle it off and return to village " +
            "matters.\n\n" +
            "WHAT YOU KNOW. The south road is mud to the knees until it dries, so the carters stay " +
            "away and salt costs double. Bram the fishmonger at the square sells trout, herring and " +
            "eel - fresh, whatever he claims about the eel. The lord raised the taxes in spring. " +
            "Your cabbage soup is famous, whatever Fenn says.\n\n" +
            "MEMORY. When you are asked to 'Compact the conversation.', that is your own memory " +
            "being written down, not the traveler speaking; whatever you leave out is forgotten for " +
            "good. Answer with a plain third-person summary of who you spoke with, what they asked, " +
            "what you told them and what is unsettled. Facts only, under 80 words.";

        static string FennPersona() =>
            "You are Fenn, the herbalist of Birchbrook, a small trading village. You walk the " +
            "market loop with your old friend Odo the peddler, needling him as you go; Odo leads " +
            "his brown cow on a rope behind the pair of you.\n\n" +
            "VOICE. One to three short sentences, dry and sharp, warm underneath. Answer exactly " +
            "what was asked, then stop. Call the player 'traveler'. Never describe your own " +
            "actions. Stay in character always: if asked about the real world, wave it off and " +
            "return to village matters.\n\n" +
            "WHAT YOU KNOW. You grow every herb worth knowing: willowbark for fevers, comfrey for " +
            "bruises, mint for a sour stomach - free advice, given freely. The well water is sweet " +
            "this year. Bram's fish is honest, his eel less so. Odo's cabbage soup is edible at " +
            "best, and you say so. The lord's spring taxes have everyone grumbling.\n\n" +
            "MEMORY. When you are asked to 'Compact the conversation.', that is your own memory " +
            "being written down, not the visitor speaking; whatever you leave out is forgotten for " +
            "good. Answer with a plain third-person summary of who you spoke with, what they asked, " +
            "what you told them and what is unsettled. Facts only, under 80 words.";

        // ---------------------------------------------------------------- UI

        static void BuildUI(TMP_FontAsset cinzel, GameObject[] npcGOs)
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

            // --- "[ E ] Talk" prompt, bottom center
            var promptGO = MakeRect("InteractPrompt", canvasGO.transform, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f),
                                    new Vector2(330, 58), new Vector2(0, 96));
            var promptBG = promptGO.AddComponent<Image>(); promptBG.color = darkBG;
            AddThinBorder(promptGO.transform, gold);
            MakeTMP("Text", promptGO.transform, "Talk   —   [ E ]", cinzel, 26, parchment,
                    TextAlignmentOptions.Center, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            promptGO.AddComponent<DeepUnity.NPCInteractPrompt>();
            promptGO.SetActive(false);

            // --- chat panel (right-docked, slides in) — the village skin of the souls window
            var panelGO = MakeRect("VillageChatWindow", canvasGO.transform, new Vector2(1f, 0f), new Vector2(1f, 1f),
                                   new Vector2(680, -56), new Vector2(-24, 0));
            ((RectTransform)panelGO.transform).pivot = new Vector2(1f, 0.5f);
            var borderImg = panelGO.AddComponent<Image>(); borderImg.color = gold;

            var bgGO = MakeRect("BG", panelGO.transform, Vector2.zero, Vector2.one, new Vector2(-4, -4), Vector2.zero);
            bgGO.AddComponent<Image>().color = darkBG;

            var titleGO = MakeTMP("Title", panelGO.transform, "—", cinzel, 30, parchment, TextAlignmentOptions.Center,
                                  new Vector2(0, 1), new Vector2(1, 1), new Vector2(-40, 64), new Vector2(0, -40));
            var divGO = MakeRect("Divider", panelGO.transform, new Vector2(0, 1), new Vector2(1, 1), new Vector2(-70, 2), new Vector2(0, -78));
            divGO.AddComponent<Image>().color = gold;

            var infoGO = MakeTMP("InfoText", panelGO.transform, "", null, 19, new Color(0.62f, 0.58f, 0.49f),
                                 TextAlignmentOptions.Center, new Vector2(0, 1), new Vector2(1, 1), new Vector2(-50, 30), new Vector2(0, -100));
            infoGO.GetComponent<TMP_Text>().fontStyle = FontStyles.Italic;

            var scrollGO = MakeRect("Messages", panelGO.transform, Vector2.zero, Vector2.one, new Vector2(-36, -210), new Vector2(0, -22));
            var scroll = scrollGO.AddComponent<ScrollRect>();
            var viewportGO = MakeRect("Viewport", scrollGO.transform, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            viewportGO.AddComponent<RectMask2D>();
            var vpImg = viewportGO.AddComponent<Image>(); vpImg.color = new Color(0, 0, 0, 0.01f);

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

            var rowGO = MakeRect("InputRow", panelGO.transform, new Vector2(0, 0), new Vector2(1, 0), new Vector2(-36, 54), new Vector2(0, 48));
            var rowHlg = rowGO.AddComponent<HorizontalLayoutGroup>();
            rowHlg.spacing = 10;
            rowHlg.childControlWidth = true; rowHlg.childControlHeight = true;
            rowHlg.childForceExpandWidth = false; rowHlg.childForceExpandHeight = true;

            var inputGO = BuildInputField(rowGO.transform, cinzel, parchment, out TMP_InputField inputField);
            inputGO.AddComponent<LayoutElement>().flexibleWidth = 1f;

            var sendBtn = BuildPanelButton(rowGO.transform, "Speak", cinzel, gold, parchment, 104);
            var leaveBtn = BuildPanelButton(rowGO.transform, "Leave", cinzel, gold, new Color(0.72f, 0.55f, 0.45f), 104);

            var win = panelGO.AddComponent<VillageChatWindow>();
            SetRef(win, "panel", (RectTransform)panelGO.transform);
            SetRef(win, "messageContainer", contentGO.transform);
            SetRef(win, "inputField", inputField);
            SetRef(win, "sendButton", sendBtn);
            SetRef(win, "leaveButton", leaveBtn);
            SetRef(win, "messageTemplate", msgGO);
            SetRef(win, "scrollRect", scroll);
            SetRef(win, "infoText", infoGO.GetComponent<TMP_Text>());
            SetRef(win, "titleText", titleGO.GetComponent<TMP_Text>());

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

            // ONE window, one prompt, three NPCs: only the NPC in interaction reacts to the
            // buttons — AskNPC/CloseInteraction are no-ops on idle NPCs by design
            var prompt = promptGO.GetComponent<DeepUnity.NPCInteractPrompt>();
            foreach (var go in npcGOs)
            {
                var it = go.GetComponent<NPCChatBase>();
                SetRef(it, "chatWindow", win);
                SetRef(it, "interactPrompt", prompt);
                UnityEventTools.AddPersistentListener(sendBtn.onClick, new UnityAction(it.AskNPC));
                UnityEventTools.AddPersistentListener(leaveBtn.onClick, new UnityAction(it.CloseInteraction));
                UnityEventTools.AddVoidPersistentListener(inputField.onSubmit, new UnityAction(it.AskNPC));
            }
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
            field.caretColor = new Color(0.77f, 0.66f, 0.42f);
            field.customCaretColor = true;
            field.caretWidth = 3;
            field.caretBlinkRate = 0.85f;
            field.selectionColor = new Color(0.45f, 0.38f, 0.22f, 0.6f);
            field.targetGraphic = bg;
            return go;
        }

        static Button BuildPanelButton(Transform parent, string label, TMP_FontAsset cinzel,
                                       Color gold, Color textColor, float width)
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
            (Vector2 min, Vector2 max, Vector2 size, Vector2 pos)[] edges =
            {
                (new Vector2(0,1), new Vector2(1,1), new Vector2(0,2), new Vector2(0,-1)),
                (new Vector2(0,0), new Vector2(1,0), new Vector2(0,2), new Vector2(0, 1)),
                (new Vector2(0,0), new Vector2(0,1), new Vector2(2,0), new Vector2(1, 0)),
                (new Vector2(1,0), new Vector2(1,1), new Vector2(2,0), new Vector2(-1,0)),
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

        // ---------------------------------------------------------------- layout screenshots
        // Characters render in bind pose (animators only run in play mode) — these are for
        // checking LAYOUT: positions, scales, sightlines, nothing buried or floating.
        public static void ScreenshotBatch()
        {
            try
            {
                EditorSceneManager.OpenScene(SCENE_PATH);
                string dir = "ProbeLogs/village_shots";
                Directory.CreateDirectory(dir);

                // ExecuteAlways scripts get no LateUpdate tick before an offline render — string
                // the leash by hand so it shows in the shots
                foreach (var leash in UnityEngine.Object.FindObjectsOfType<CowLeash>())
                    leash.SendMessage("LateUpdate");

                var camGO = new GameObject("shotcam");
                var cam = camGO.AddComponent<Camera>();
                cam.fieldOfView = 55f;
                cam.nearClipPlane = 0.1f;
                cam.farClipPlane = 500f;

                void At(string name, Vector3 pos, Vector3 lookAt, float fov = 55f)
                {
                    cam.fieldOfView = fov;
                    camGO.transform.position = pos;
                    camGO.transform.rotation = Quaternion.LookRotation(lookAt - pos);
                    Shoot(cam, dir + "/" + name + ".png", 1600, 900);
                }

                At("overview", new Vector3(-30f, 26f, -16f), new Vector3(0f, 2f, 24f));
                At("street_from_spawn", PLAYER_SPAWN + new Vector3(0f, 1.7f, -1f), new Vector3(0f, 1.2f, 14f));
                At("square", new Vector3(-1f, 3.2f, 6.5f), new Vector3(0.5f, 1.0f, 16f));
                At("fish_stall", FISH_STALL_POS + new Vector3(-4.5f, 1.8f, -1.5f), FISH_STALL_POS + Vector3.up * 1.1f, 50f);
                At("pen", PEN_CENTER + new Vector3(-4.5f, 2.5f, -3f), PEN_CENTER + Vector3.up * 0.4f, 50f);
                At("strollers_leash", new Vector3(2.4f, 1.7f, 3.2f), new Vector3(2.2f, 1.1f, 8.6f), 60f);
                At("pair_cow_side", new Vector3(6.2f, 1.6f, 8.6f), new Vector3(1.9f, 1.0f, 8.8f), 55f);
                At("well_center", new Vector3(-3.4f, 2.0f, 12.6f), new Vector3(0.6f, 1.0f, 16.8f), 60f);
                At("hill_road", new Vector3(0.6f, 2.4f, 27f), new Vector3(0f, GroundHeight(0f, 70f) + 2f, 70f));
                At("fork_lookback", new Vector3(-3.5f, GroundHeight(-3.5f, 80f) + 2.0f, 80f),
                   new Vector3(0f, 2f, 15f));

                Debug.Log("[TradingVillageBuilder] SHOTS OK -> " + dir);
                EditorApplication.Exit(0);
            }
            catch (Exception e)
            {
                Debug.LogError("[TradingVillageBuilder] SHOTS FAILED: " + e);
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

        static TMP_FontAsset CreateCinzelFont()
        {
            string path = GEN + "/Cinzel SDF.asset";
            var existing = AssetDatabase.LoadAssetAtPath<TMP_FontAsset>(path);
            if (existing != null) return existing;   // shared with the other two 3D scenes

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

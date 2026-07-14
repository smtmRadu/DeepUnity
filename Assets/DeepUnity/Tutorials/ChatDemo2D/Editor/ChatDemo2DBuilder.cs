using System;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEditor;
using UnityEditor.Events;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.EventSystems;
using UnityEngine.Rendering;
using UnityEngine.TextCore.LowLevel;
using UnityEngine.UI;

namespace DeepUnity.Tutorials.ChatDemo2D.EditorTools
{
    /// <summary>
    /// Deterministically builds the ChatDemo2D scene: a Stardew-flavored mini farm out of Kenney
    /// CC0 "tiny" tiles — a 40x30-tile meadow with a farmhouse and a general store, a fenced
    /// field of 24 workable plots (hoe → plant → water → grow → harvest, three crops), a critter
    /// pen, a day/night tint cycle and two LLM-driven villagers with streaming TTS.
    /// The ground layer is baked into one PNG at build time (no Tilemap package dependency);
    /// everything above it is Y-sorted sprites. Idempotent: texture imports are configured in
    /// place, Generated assets are rebuilt or reused, the scene file is rebuilt from scratch.
    /// Run from the menu (DeepUnity/Tutorials/Build ChatDemo2D Scene) or in batch mode via
    /// -executeMethod DeepUnity.Tutorials.ChatDemo2D.EditorTools.ChatDemo2DBuilder.BuildBatch
    /// </summary>
    public static class ChatDemo2DBuilder
    {
        const string ROOT = "Assets/DeepUnity/Tutorials/ChatDemo2D";
        const string ART = ROOT + "/Art";
        const string TILES = ART + "/Tiles";
        const string GEN = ROOT + "/Generated";
        const string SCENE_PATH = ROOT + "/ChatDemo2D.unity";

        // map in tiles; 16 px tiles at 16 PPU = 1 world unit per tile, map centered on the origin
        const int MW = 40, MH = 30;

        static readonly System.Random rng = new System.Random(20260711);

        /// <summary>Center of tile (tx, ty) in world units (tile 0,0 = bottom-left corner).</summary>
        static Vector2 T(float tx, float ty) => new Vector2(tx - MW / 2f + 0.5f, ty - MH / 2f + 0.5f);

        // ---------------------------------------------------------------- entry points

        [MenuItem("DeepUnity/Tutorials/Build ChatDemo2D Scene")]
        public static void BuildScene()
        {
            ConfigureImports();
            BuildEverything();
            Debug.Log("[ChatDemo2DBuilder] Done. Scene at " + SCENE_PATH);
        }

        public static void BuildBatch()
        {
            try
            {
                ConfigureImports();
                BuildEverything();
                Debug.Log("[ChatDemo2DBuilder] BATCH OK");
                EditorApplication.Exit(0);
            }
            catch (Exception e)
            {
                Debug.LogError("[ChatDemo2DBuilder] BATCH FAILED: " + e);
                EditorApplication.Exit(1);
            }
        }

        // ---------------------------------------------------------------- import configuration

        static void ConfigureImports()
        {
            // Flat art (drawn below/above the Y-sorted world) keeps a centered pivot; everything
            // that stands IN the world gets a bottom pivot so the camera's Y-axis transparency
            // sort compares ground positions.
            string[] centered = { "ground_", "soil_", "icon_", "seed_" };
            foreach (string guid in AssetDatabase.FindAssets("t:Texture2D", new[] { TILES }))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                string name = Path.GetFileNameWithoutExtension(path);
                bool flat = Array.Exists(centered, p => name.StartsWith(p));
                ConfigureSprite(path, 16f, flat ? new Vector2(0.5f, 0.5f) : new Vector2(0.5f, 0f));
            }
            AssetDatabase.SaveAssets();
        }

        static void ConfigureSprite(string path, float ppu, Vector2 pivot, FilterMode filter = FilterMode.Point)
        {
            var imp = AssetImporter.GetAtPath(path) as TextureImporter;
            if (imp == null) throw new Exception("Missing texture: " + path);

            var settings = new TextureImporterSettings();
            imp.ReadTextureSettings(settings);
            bool dirty = imp.textureType != TextureImporterType.Sprite
                      || imp.spriteImportMode != SpriteImportMode.Single
                      || !Mathf.Approximately(imp.spritePixelsPerUnit, ppu)
                      || imp.filterMode != filter
                      || imp.textureCompression != TextureImporterCompression.Uncompressed
                      || imp.mipmapEnabled
                      || settings.spriteAlignment != (int)SpriteAlignment.Custom
                      || settings.spritePivot != pivot;
            if (!dirty) return;

            imp.textureType = TextureImporterType.Sprite;
            imp.spriteImportMode = SpriteImportMode.Single;
            imp.spritePixelsPerUnit = ppu;
            imp.filterMode = filter;
            imp.textureCompression = TextureImporterCompression.Uncompressed;
            imp.mipmapEnabled = false;
            imp.alphaIsTransparency = true;
            imp.ReadTextureSettings(settings);
            settings.spriteAlignment = (int)SpriteAlignment.Custom;
            settings.spritePivot = pivot;
            imp.SetTextureSettings(settings);
            imp.SaveAndReimport();
        }

        // ---------------------------------------------------------------- shared helpers

        static Sprite S(string name)
        {
            var sp = AssetDatabase.LoadAssetAtPath<Sprite>(TILES + "/" + name + ".png");
            if (sp == null) throw new Exception("Missing sprite: " + TILES + "/" + name + ".png");
            return sp;
        }

        static void SetRef(Component c, string field, UnityEngine.Object value)
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

        static void SetFloat(Component c, string field, float value)
        {
            var so = new SerializedObject(c);
            so.FindProperty(field).floatValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        static void SetBool(Component c, string field, bool value)
        {
            var so = new SerializedObject(c);
            var prop = so.FindProperty(field);
            if (prop == null) throw new Exception($"No serialized field '{field}' on {c.GetType().Name}");
            prop.boolValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        static void SetEnum(Component c, string field, int value)
        {
            var so = new SerializedObject(c);
            var prop = so.FindProperty(field);
            if (prop == null) throw new Exception($"No serialized field '{field}' on {c.GetType().Name}");
            prop.enumValueIndex = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        static GameObject WorldSprite(string goName, Sprite sprite, Vector2 pos, Transform parent = null, int order = 0)
        {
            var go = new GameObject(goName);
            if (parent != null) go.transform.SetParent(parent, false);
            go.transform.position = pos;
            var sr = go.AddComponent<SpriteRenderer>();
            sr.sprite = sprite;
            sr.sortingOrder = order;
            return go;
        }

        // ---------------------------------------------------------------- build

        static void BuildEverything()
        {
            if (!AssetDatabase.IsValidFolder(GEN))
                AssetDatabase.CreateFolder(ROOT, "Generated");

            var font = CreateKenneyFont();
            var vignette = CreateVignetteSprite();
            var white = CreateWhiteSprite();
            var highlightSprite = CreateHighlightSprite();
            var ground = BakeGround();

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            GameObject cameraGO = BuildCamera();
            WorldSprite("GroundBaked", ground, Vector2.zero, null, -1000);
            BuildColliders();
            BuildBuildingsAndDecor();
            FarmPlot[] plots = BuildField();
            GameObject player = BuildPlayer();
            var npcs = BuildNpcs(font, white);
            BuildCritters();
            GameObject dayGO = BuildDayTint(white);

            var ui = BuildUI(font, vignette, npcs);

            // farming system
            var sysGO = new GameObject("FarmingSystem");
            var sfx = sysGO.AddComponent<AudioSource>();
            sfx.playOnAwake = false;
            sfx.spatialBlend = 0f;
            var sys = sysGO.AddComponent<FarmingSystem>();
            var highlightGO = WorldSprite("PlotHighlight", highlightSprite, Vector2.zero, sysGO.transform, 800);
            var highlightSr = highlightGO.GetComponent<SpriteRenderer>();
            highlightSr.color = new Color(1f, 1f, 1f, 0.55f);
            highlightSr.enabled = false;
            WireFarmingSystem(sys, player, plots, highlightSr, ui.hud, sfx);
            // give-items flow: both villagers read/empty the same basket and pay into the same purse
            foreach (var npc in npcs)
                SetRef(npc, "farm", sys);

            // day cycle drives the tint overlay + the HUD clock
            SetRef(dayGO.GetComponent<DayCycle2D>(), "hud", ui.hud);

            // final cross-wiring
            SetRef(player.GetComponent<PlayerController2D>(), "cam", cameraGO.GetComponent<CameraFollow2D>());
            SetRef(cameraGO.GetComponent<CameraFollow2D>(), "target", player.transform);

            EditorSceneManager.SaveScene(scene, SCENE_PATH);
            AssetDatabase.SaveAssets();
            Debug.Log("[ChatDemo2DBuilder] Scene saved (" + SCENE_PATH + ")");
        }

        // ---------------------------------------------------------------- camera

        static GameObject BuildCamera()
        {
            var go = new GameObject("Main Camera", typeof(Camera), typeof(AudioListener));
            go.tag = "MainCamera";
            var cam = go.GetComponent<Camera>();
            cam.orthographic = true;
            cam.orthographicSize = 5.5f;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.13f, 0.16f, 0.10f);
            // top-down Y sorting: sprites lower on the map render in front (world sprites have
            // bottom pivots, so this compares their ground lines)
            cam.transparencySortMode = TransparencySortMode.CustomAxis;
            cam.transparencySortAxis = Vector3.up;
            go.transform.position = new Vector3(0f, 0f, -10f);
            go.AddComponent<CameraFollow2D>();
            return go;
        }

        // ---------------------------------------------------------------- baked ground

        // Dirt rectangles (tile coords, inclusive) — the farmyard, the paths, the store yard.
        static readonly (int x0, int y0, int x1, int y1)[] DIRT_RECTS =
        {
            (5, 17, 10, 20),    // farmhouse yard
            (7, 15,  8, 20),    // path from the farmhouse door
            (5, 15, 31, 16),    // main east-west path
            (27, 17, 30, 20),   // store yard
            (17, 12, 18, 14),   // branch into the field gate (stops above the top plot row)
        };

        static Sprite BakeGround()
        {
            // Bytes → LoadImage decodes the PNGs regardless of importer settings, so the bake
            // never depends on import order. Deterministic RNG scatters grass variants/flowers.
            Texture2D Load(string name)
            {
                var t = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                if (!t.LoadImage(File.ReadAllBytes(TILES + "/" + name + ".png")))
                    throw new Exception("Failed to decode " + name);
                return t;
            }

            var grassA = Load("ground_grass_a");
            var grassB = Load("ground_grass_b");
            var flowers = Load("ground_flowers");
            var dirtA = Load("ground_dirt_a");
            var dirtB = Load("ground_dirt_b");

            bool[,] dirt = new bool[MW, MH];
            foreach (var (x0, y0, x1, y1) in DIRT_RECTS)
                for (int x = x0; x <= x1; x++)
                    for (int y = y0; y <= y1; y++)
                        dirt[x, y] = true;

            var tex = new Texture2D(MW * 16, MH * 16, TextureFormat.RGBA32, false);
            for (int ty = 0; ty < MH; ty++)
                for (int tx = 0; tx < MW; tx++)
                {
                    Texture2D src;
                    if (dirt[tx, ty])
                        src = rng.NextDouble() < 0.7 ? dirtA : dirtB;
                    else
                    {
                        double r = rng.NextDouble();
                        src = r < 0.05 ? flowers : (r < 0.45 ? grassB : grassA);
                    }
                    tex.SetPixels32(tx * 16, ty * 16, 16, 16, src.GetPixels32());
                }
            tex.Apply();

            string pngPath = GEN + "/GroundBaked.png";
            File.WriteAllBytes(pngPath, tex.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(tex);
            UnityEngine.Object.DestroyImmediate(grassA);
            UnityEngine.Object.DestroyImmediate(grassB);
            UnityEngine.Object.DestroyImmediate(flowers);
            UnityEngine.Object.DestroyImmediate(dirtA);
            UnityEngine.Object.DestroyImmediate(dirtB);
            AssetDatabase.ImportAsset(pngPath);
            ConfigureSprite(pngPath, 16f, new Vector2(0.5f, 0.5f));
            return AssetDatabase.LoadAssetAtPath<Sprite>(pngPath);
        }

        // ---------------------------------------------------------------- world colliders

        static void BuildColliders()
        {
            var root = new GameObject("WorldColliders");

            void Box(string name, float cx, float cy, float w, float h)
            {
                var go = new GameObject(name);
                go.transform.SetParent(root.transform, false);
                go.transform.position = new Vector3(cx, cy, 0f);
                var col = go.AddComponent<BoxCollider2D>();
                col.size = new Vector2(w, h);
            }

            // low, feet-height boxes spanning tile ranges (the fence lines)
            void TileSpan(string name, int x0, int y0, int x1, int y1)
            {
                Vector2 a = T(x0, y0);
                Vector2 b = T(x1, y1);
                bool horizontal = y0 == y1;
                Box(name, (a.x + b.x) / 2f, (a.y + b.y) / 2f,
                    horizontal ? b.x - a.x + 1f : 0.5f,
                    horizontal ? 0.5f : b.y - a.y + 1f);
            }

            // map edges
            Box("EdgeN", 0f, MH / 2f + 1f, MW + 4f, 2f);
            Box("EdgeS", 0f, -MH / 2f - 1f, MW + 4f, 2f);
            Box("EdgeE", MW / 2f + 1f, 0f, 2f, MH + 4f);
            Box("EdgeW", -MW / 2f - 1f, 0f, 2f, MH + 4f);

            // field fence (gate gap at tiles 17-18, y14)
            TileSpan("FieldFenceTopL", 12, 14, 16, 14);
            TileSpan("FieldFenceTopR", 19, 14, 23, 14);
            TileSpan("FieldFenceBottom", 12, 5, 23, 5);
            TileSpan("FieldFenceLeft", 12, 6, 12, 13);
            TileSpan("FieldFenceRight", 23, 6, 23, 13);

            // critter pen (gate gap at tile 29, y10)
            TileSpan("PenFenceTopL", 26, 10, 28, 10);
            TileSpan("PenFenceTopR", 30, 10, 32, 10);
            TileSpan("PenFenceBottom", 26, 5, 32, 5);
            TileSpan("PenFenceLeft", 26, 6, 26, 9);
            TileSpan("PenFenceRight", 32, 6, 32, 9);
        }

        // ---------------------------------------------------------------- buildings, fences, flora

        static void BuildBuildingsAndDecor()
        {
            var world = new GameObject("World");

            // --- houses: 3 wall tiles + 3 roof tiles, SortingGroup so the whole building sorts
            // as one unit at its base line
            void House(string name, int bx, int by, string roofPrefix, string wallA, string wallDoor, string wallB)
            {
                var root = new GameObject(name);
                root.transform.SetParent(world.transform, false);
                root.transform.position = T(bx + 1, by) + new Vector2(0f, -0.5f);   // base line under the door
                root.AddComponent<SortingGroup>().sortingOrder = 0;

                string[] walls = { wallA, wallDoor, wallB };
                string[] roofs = { roofPrefix + "_l", roofPrefix + "_m", roofPrefix + "_r" };
                for (int i = 0; i < 3; i++)
                {
                    Vector2 wallPos = T(bx + i, by) + new Vector2(0f, -0.5f);
                    WorldSprite("Wall" + i, S(walls[i]), wallPos, root.transform);
                    WorldSprite("Roof" + i, S(roofs[i]), wallPos + Vector2.up, root.transform);
                }

                var col = root.AddComponent<BoxCollider2D>();
                col.offset = new Vector2(0f, 1f);
                col.size = new Vector2(3f, 2f);
            }

            House("Farmhouse", 6, 21, "roof_slate", "wall_wood_a", "wall_wood_door", "wall_wood_b");
            House("GeneralStore", 28, 21, "roof_red", "wall_stone_a", "wall_stone_door", "wall_stone_b");

            // store sign
            var sign = WorldSprite("StoreSign", S("sign"), T(31, 21) + new Vector2(0f, -0.5f), world.transform);
            var signCol = sign.AddComponent<BoxCollider2D>();
            signCol.size = new Vector2(0.6f, 0.35f);
            signCol.offset = new Vector2(0f, 0.18f);

            // well on the main path junction
            var well = WorldSprite("Well", S("well"), T(19, 19) + new Vector2(0f, -0.5f), world.transform);
            var wellCol = well.AddComponent<BoxCollider2D>();
            wellCol.size = new Vector2(0.9f, 0.6f);
            wellCol.offset = new Vector2(0f, 0.3f);

            // --- fences (visuals; the physics is the segment colliders). Bottom pivot at the
            // cell's lower edge so characters sort correctly walking in front/behind.
            //
            // Sprite geometry (verified pixel-level): fence_h's rail band spans the FULL tile
            // width and fence_v's post spans the FULL tile height, both X-symmetric — straight
            // runs never need mirroring. fence_corner is drawn as the TOP-LEFT corner only: its
            // rail exits the RIGHT edge and its post exits the BOTTOM edge (connects right+down).
            // The old code stamped that same unmirrored piece on all four corners, so the two
            // RIGHT corners had their rail sticking OUT to the right with a visible gap toward
            // the fence_h run on their left, and the two BOTTOM corners dangled their post
            // downward instead of joining the fence_v run above. Each corner now gets the
            // mirror that makes its connections point INTO the rectangle:
            //   top-left     (x0,y1): native          — connects right + down
            //   top-right    (x1,y1): flipX           — connects left  + down
            //   bottom-left  (x0,y0): flipY           — connects right + up
            //   bottom-right (x1,y0): flipX + flipY   — connects left  + up
            // SpriteRenderer.flipX/flipY mirror about the sprite pivot (bottom-center here):
            // flipX is in-place, but flipY would reflect the art into the cell BELOW — so
            // flipY tiles anchor at the cell's TOP edge (+0.5) instead of the bottom (-0.5),
            // which puts the mirrored art back into the same cell. Sorting is unaffected: the
            // camera Y-sorts by sprite CENTER (default SpriteSortPoint), which stays at the
            // cell center either way.
            var fences = new GameObject("Fences");
            fences.transform.SetParent(world.transform, false);
            void FenceTile(string sprite, int tx, int ty, bool flipX = false, bool flipY = false)
            {
                var go = WorldSprite("Fence", S(sprite),
                                     T(tx, ty) + new Vector2(0f, flipY ? 0.5f : -0.5f), fences.transform);
                var sr = go.GetComponent<SpriteRenderer>();
                sr.flipX = flipX;
                sr.flipY = flipY;
            }

            void FenceRect(int x0, int y0, int x1, int y1, (int x, int y)[] gaps)
            {
                bool IsGap(int x, int y)
                {
                    foreach (var g in gaps) if (g.x == x && g.y == y) return true;
                    return false;
                }
                for (int x = x0; x <= x1; x++)
                {
                    // top edge (y1): corners turn DOWN into the side runs
                    if (!IsGap(x, y1))
                    {
                        if (x == x0)      FenceTile("fence_corner", x, y1);                // right+down
                        else if (x == x1) FenceTile("fence_corner", x, y1, flipX: true);   // left+down
                        else              FenceTile("fence_h", x, y1);
                    }
                    // bottom edge (y0): corners turn UP into the side runs
                    if (!IsGap(x, y0))
                    {
                        if (x == x0)      FenceTile("fence_corner", x, y0, flipY: true);               // right+up
                        else if (x == x1) FenceTile("fence_corner", x, y0, flipX: true, flipY: true);  // left+up
                        else              FenceTile("fence_h", x, y0);
                    }
                }
                for (int y = y0 + 1; y < y1; y++)
                {
                    if (!IsGap(x0, y)) FenceTile("fence_v", x0, y);
                    if (!IsGap(x1, y)) FenceTile("fence_v", x1, y);
                }
            }

            FenceRect(12, 5, 23, 14, new[] { (17, 14), (18, 14) });   // field
            FenceRect(26, 5, 32, 10, new[] { (29, 10) });             // critter pen

            // --- trees (with small base colliders) and soft decor
            var flora = new GameObject("Flora");
            flora.transform.SetParent(world.transform, false);
            void Tree(string sprite, int tx, int ty)
            {
                var t = WorldSprite("Tree", S(sprite), T(tx, ty) + new Vector2(0f, -0.5f), flora.transform);
                var c = t.AddComponent<BoxCollider2D>();
                c.size = new Vector2(0.5f, 0.3f);
                c.offset = new Vector2(0f, 0.15f);
            }
            void Decor(string sprite, int tx, int ty)
                => WorldSprite(sprite, S(sprite), T(tx, ty) + new Vector2(0f, -0.5f), flora.transform);

            string[] treePool = { "tree_pine", "tree_round", "tree_pine_big" };
            int[] topX = { 2, 5, 9, 13, 17, 21, 25, 29, 33, 37 };
            foreach (int x in topX) Tree(treePool[rng.Next(treePool.Length)], x, 26 + rng.Next(3));
            int[] leftY = { 5, 9, 13, 18, 22 };
            foreach (int y in leftY) Tree(treePool[rng.Next(treePool.Length)], 1 + rng.Next(2), y);
            int[] rightY = { 4, 8, 12, 17, 21, 25 };
            foreach (int y in rightY) Tree(treePool[rng.Next(treePool.Length)], 36 + rng.Next(3), y);
            int[] botX = { 3, 8, 13, 19, 25, 31, 36 };
            foreach (int x in botX) Tree(treePool[rng.Next(treePool.Length)], x, 1 + rng.Next(2));
            Tree("tree_round", 11, 22);
            Tree("tree_pine_big", 24, 18);
            Tree("tree_pine_big", 34, 19);
            Tree("tree_round", 4, 12);

            Decor("apple_bush", 4, 20);
            Decor("sunflower", 10, 21);
            Decor("sunflower", 26, 16);
            Decor("sunflower", 3, 6);
            Decor("bush", 12, 17);
            Decor("bush", 25, 12);
            Decor("stones", 33, 12);
            Decor("stones", 15, 3);
            Decor("hay", 27, 9);
            Decor("mushrooms", 2, 14);
            Decor("mushrooms", 36, 7);
            Decor("grass_tuft", 6, 13);
            Decor("grass_tuft", 20, 17);
            Decor("grass_tuft", 31, 13);
            Decor("grass_tuft", 14, 19);
            Decor("grass_tuft", 9, 3);
        }

        // ---------------------------------------------------------------- the field

        static FarmPlot[] BuildField()
        {
            var fieldGO = new GameObject("Field");
            var plots = new List<FarmPlot>();
            Sprite dry = S("soil_dry"), wet = S("soil_wet");

            // 3 rows x 8 columns with walking gaps between the rows
            int[] rows = { 7, 9, 11 };
            for (int r = 0; r < rows.Length; r++)
                for (int x = 14; x <= 21; x++)
                {
                    var go = new GameObject($"Plot_{r}_{x - 14}");
                    go.transform.SetParent(fieldGO.transform, false);
                    go.transform.position = T(x, rows[r]);

                    var soil = go.AddComponent<SpriteRenderer>();
                    soil.sprite = dry;
                    soil.sortingOrder = -500;   // flat on the ground, under everything that Y-sorts
                    soil.enabled = false;       // invisible until hoed

                    var cropGO = new GameObject("Crop");
                    cropGO.transform.SetParent(go.transform, false);
                    cropGO.transform.localPosition = new Vector3(0f, -0.5f, 0f);   // bottom pivot on the cell's ground line
                    var crop = cropGO.AddComponent<SpriteRenderer>();
                    crop.sortingOrder = 0;
                    crop.enabled = false;

                    var plot = go.AddComponent<FarmPlot>();
                    SetRef(plot, "soil", soil);
                    SetRef(plot, "crop", crop);
                    SetRef(plot, "soilDry", dry);
                    SetRef(plot, "soilWet", wet);
                    plots.Add(plot);
                }

            return plots.ToArray();
        }

        static void WireFarmingSystem(FarmingSystem sys, GameObject player, FarmPlot[] plots,
                                      SpriteRenderer highlight, FarmHud hud, AudioSource sfx)
        {
            SetRef(sys, "player", player.GetComponent<PlayerController2D>());
            SetRef(sys, "highlight", highlight);
            SetRef(sys, "hud", hud);
            SetRef(sys, "sfx", sfx);
            SetRef(sys, "actionClip", AssetDatabase.LoadAssetAtPath<AudioClip>(ART + "/Audio/UI/type_3.ogg"));

            var so = new SerializedObject(sys);
            var plotsProp = so.FindProperty("plots");
            plotsProp.arraySize = plots.Length;
            for (int i = 0; i < plots.Length; i++)
                plotsProp.GetArrayElementAtIndex(i).objectReferenceValue = plots[i];

            // coin value scales with ripening time — slow crops pay best
            (string name, string prefix, float secs, int coins)[] defs =
            {
                ("Carrot", "crop_carrot", 22f, 2),
                ("Turnip", "crop_turnip", 32f, 3),
                ("Tomato", "crop_tomato", 45f, 5),
            };
            var cropsProp = so.FindProperty("crops");
            cropsProp.arraySize = defs.Length;
            for (int i = 0; i < defs.Length; i++)
            {
                var d = cropsProp.GetArrayElementAtIndex(i);
                d.FindPropertyRelative("name").stringValue = defs[i].name;
                d.FindPropertyRelative("secondsPerStage").floatValue = defs[i].secs;
                d.FindPropertyRelative("coinValue").intValue = defs[i].coins;
                var stages = d.FindPropertyRelative("stageSprites");
                stages.arraySize = 3;
                for (int s = 0; s < 3; s++)
                    stages.GetArrayElementAtIndex(s).objectReferenceValue = S($"{defs[i].prefix}_{s}");
            }
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        // ---------------------------------------------------------------- player, NPCs, critters

        static GameObject BuildPlayer()
        {
            var go = new GameObject("Player");
            go.tag = "Player";
            go.transform.position = T(8, 18);

            var rb = go.AddComponent<Rigidbody2D>();
            rb.gravityScale = 0f;
            rb.freezeRotation = true;
            rb.collisionDetectionMode = CollisionDetectionMode2D.Continuous;
            rb.interpolation = RigidbodyInterpolation2D.Interpolate;

            // feet-sized box so the farmer's head can overlap building fronts (top-down depth feel)
            var col = go.AddComponent<BoxCollider2D>();
            col.size = new Vector2(0.55f, 0.30f);
            col.offset = new Vector2(0f, 0.15f);

            var visual = new GameObject("Visual");
            visual.transform.SetParent(go.transform, false);
            var sr = visual.AddComponent<SpriteRenderer>();
            sr.sprite = S("char_player");
            sr.sortingOrder = 0;

            var anim = go.AddComponent<CharacterAnimator2D>();
            SetRef(anim, "target", sr);

            var pc = go.AddComponent<PlayerController2D>();
            SetRef(pc, "anim", anim);
            return go;
        }

        static NPCInteractor2D[] BuildNpcs(TMP_FontAsset font, Sprite white)
        {
            var hobb = BuildNpc(font, white, "OldHobb", "char_farmer", T(16, 15), "Old Hobb",
                "You are Old Hobb, a weathered old farmer leaning on the fence of Butterbrook Farm, where the player " +
                "is tending the field right beside you. You are gruff but kind, plain-spoken, dry-humored, and full of " +
                "practical wisdom. You know exactly how this farm works and coach the player when asked: first hoe a plot, " +
                "then sow seeds, then water — a crop only grows while its soil is dark and wet, and every growth spurt " +
                "drinks the water dry, so it must be watered again after each stage. Carrots ripen quickest, turnips take " +
                "a while longer, tomatoes are the slowest but the proudest harvest. Granny Marla runs the general store " +
                "with the red roof up the path and gives seeds away on the honor system. The pen by the store holds two " +
                "hens, a cow and a sheep you're fond of. You like remarking on the time of day and the weather. " +
                "Sometimes the player hands you vegetables from their harvest — a bracketed note will tell you " +
                "exactly what they gave and how many coins you pay for it. Accept the produce, thank them in your " +
                "own gruff way, judge the vegetables like the old farmer you are, and mention the coins you're handing over. " +
                "Stay in character at all times. Keep your replies to one to three short sentences.",
                "The old farmer leans on the fence and squints at you...",
                // text-only 2D demo (user pick): no TTS on either villager — the talk bob
                // follows the token stream; the Kokoro voice fields stay baked for easy re-enable
                NPCInteractor2D.ConversationMode.LlmOnly, "am_onyx", 0.95f,
                // history-mode A/B spread: Hobb forgets you the moment the chat closes
                NPCInteractor2D.HistoryMode.ResetEveryTime,
                // Qwen3.5-0.8B: the coalesced-kernel model (fast decode + disk-KV restore); Hobb
                // is the perf arm of the 2D A/B (Marla stays on MiniCPM5-1B)
                "Qwen3.5-0.8B", think: false);

            var marla = BuildNpc(font, white, "GrannyMarla", "char_granny", T(30, 19), "Granny Marla",
                "You are Granny Marla, the warm, talkative grandmother who runs the little red-roofed general store on " +
                "Butterbrook Farm's lane. You call everyone 'dear', you always have tea brewing, and you hand out carrot, " +
                "turnip and tomato seeds for free on the honor system — folk pay you back in vegetables and gossip. You " +
                "know the farm well: the fenced field down the path where crops need hoeing, sowing and watering after " +
                "every growth spurt; the old well; the pen with the hens, the cow and the sheep. Old Hobb, the farmer by " +
                "the field gate, is an old friend you tease fondly — you suspect he names the crows. " +
                "Sometimes the player brings you vegetables from their harvest — a bracketed note will tell you " +
                "exactly what they brought and how many coins you pay for it. Fuss over the produce warmly, thank " +
                "them like the dear they are, and mention the coins you're pressing into their hands. " +
                "Stay in character at all times. Keep your replies to one to three short sentences.",
                "The shopkeeper looks up from her knitting with a smile...",
                NPCInteractor2D.ConversationMode.LlmOnly, "granny", 0.92f,
                // history-mode A/B spread: Granny REMEMBERS across dialogues (live KV while
                // resident, transcript re-prefill after release — MiniCPM has no disk-KV restore
                // yet). Residency is the zone's job. She is also the THINKING arm of the A/B.
                NPCInteractor2D.HistoryMode.ContinueWhereLeftOff,
                "MiniCPM5-1B", think: true);

            return new[] { hobb, marla };
        }

        static NPCInteractor2D BuildNpc(TMP_FontAsset font, Sprite white, string goName, string sprite,
                                        Vector2 pos, string displayName, string systemPrompt,
                                        string approach, NPCInteractor2D.ConversationMode mode,
                                        string voice, float pitch,
                                        NPCInteractor2D.HistoryMode history,
                                        string model, bool think)
        {
            var go = new GameObject(goName);
            go.transform.position = pos;

            // solid body (can't walk through them) + generous talk-range trigger
            var body = go.AddComponent<BoxCollider2D>();
            body.size = new Vector2(0.6f, 0.3f);
            body.offset = new Vector2(0f, 0.15f);
            var trigger = go.AddComponent<CircleCollider2D>();
            trigger.isTrigger = true;
            trigger.radius = 2.0f;
            trigger.offset = new Vector2(0f, 0.4f);

            var visual = new GameObject("Visual");
            visual.transform.SetParent(go.transform, false);
            var sr = visual.AddComponent<SpriteRenderer>();
            sr.sprite = S(sprite);
            sr.sortingOrder = 0;

            var anim = go.AddComponent<CharacterAnimator2D>();
            SetRef(anim, "target", sr);

            var npc = go.AddComponent<NPCInteractor2D>();
            SetRef(npc, "charAnim", anim);
            SetString(npc, "npc_name", displayName);
            SetString(npc, "system_prompt", systemPrompt);
            SetString(npc, "approach_text", approach);
            // Per-villager LLM (user pick): MiniCPM5-1B on both, thinking as a live A/B — Hobb
            // answers directly, Marla REASONS in <think> first (never shown/voiced; the window
            // pulses 'Thinking…' until her actual answer starts). LLMRegistry id string.
            SetString(npc, "model", model);
            SetBool(npc, "allowThinking", think);
            SetEnum(npc, "historyMode", (int)history);
            // modern voice wiring: Kokoro-only, mode as enum, voicepack by manifest name
            // (field unified with the 3D demo as "ttsVoice" when the NPCs moved onto NPCChatBase)
            SetEnum(npc, "conversationMode", (int)mode);
            SetString(npc, "ttsVoice", voice);
            SetFloat(npc, "voicePitch", pitch);
            // latent loading on by default: walking into the green circle slow-prefetches the
            // LLM + Kokoro weights; wandering off while Idle unloads both (7 tiles clears the
            // 2-tile talk trigger with a comfortable walk-up)
            SetBool(npc, "usePrefetchZone", true);
            SetFloat(npc, "prefetchRadius", 7f);

            // --- floating nameplate, always drawn over the world
            var label = new GameObject("NameLabel");
            label.transform.SetParent(go.transform, false);
            label.transform.localPosition = new Vector3(0f, 1.45f, 0f);

            var bg = new GameObject("BG");
            bg.transform.SetParent(label.transform, false);
            var bgSr = bg.AddComponent<SpriteRenderer>();
            bgSr.sprite = white;
            bgSr.color = new Color(0.07f, 0.05f, 0.03f, 0.62f);
            bgSr.transform.localScale = new Vector3(4.6f, 0.85f, 1f);   // white.png is 8px = 0.5 units
            bgSr.sortingOrder = 1490;

            var textGO = new GameObject("Text");
            textGO.transform.SetParent(label.transform, false);
            var tmp = textGO.AddComponent<TextMeshPro>();
            tmp.text = displayName;
            tmp.font = font;
            tmp.fontSize = 2.4f;
            tmp.color = new Color(0.95f, 0.90f, 0.75f);
            tmp.alignment = TextAlignmentOptions.Center;
            tmp.enableWordWrapping = false;
            tmp.rectTransform.sizeDelta = new Vector2(6f, 0.8f);
            textGO.GetComponent<MeshRenderer>().sortingOrder = 1500;

            return npc;
        }

        static void BuildCritters()
        {
            var root = new GameObject("Critters");
            // pen interior: tiles x27..31, y6..9
            Vector2 a = T(27, 6), b = T(31, 9);
            var area = new Rect(a.x - 0.4f, a.y - 0.4f, b.x - a.x + 0.8f, b.y - a.y + 0.8f);

            (string sprite, float speed, Vector2 at)[] critters =
            {
                ("critter_chicken", 1.4f, T(28, 7)),
                ("critter_chicken", 1.2f, T(30, 8)),
                ("critter_cow",     0.7f, T(29, 6)),
                ("critter_sheep",   0.8f, T(31, 9)),
            };
            foreach (var (sprite, speed, at) in critters)
            {
                var go = new GameObject(sprite.Replace("critter_", "Critter_"));
                go.transform.SetParent(root.transform, false);
                go.transform.position = at;

                var visual = new GameObject("Visual");
                visual.transform.SetParent(go.transform, false);
                var sr = visual.AddComponent<SpriteRenderer>();
                sr.sprite = S(sprite);
                sr.sortingOrder = 0;

                var anim = go.AddComponent<CharacterAnimator2D>();
                SetRef(anim, "target", sr);

                var wander = go.AddComponent<CritterWander2D>();
                SetRef(wander, "anim", anim);
                var so = new SerializedObject(wander);
                so.FindProperty("area").rectValue = area;
                so.FindProperty("speed").floatValue = speed;
                so.ApplyModifiedPropertiesWithoutUndo();
            }
        }

        static GameObject BuildDayTint(Sprite white)
        {
            var go = new GameObject("DayCycle");
            var overlayGO = WorldSprite("TintOverlay", white, Vector2.zero, go.transform, 3000);
            // white.png is 8px = 0.5 world units — scale it past the map so the edges never show
            overlayGO.transform.localScale = new Vector3((MW + 4) * 2f, (MH + 4) * 2f, 1f);
            var sr = overlayGO.GetComponent<SpriteRenderer>();
            sr.color = new Color(0f, 0f, 0f, 0f);

            var day = go.AddComponent<DayCycle2D>();
            SetRef(day, "overlay", sr);
            return go;
        }

        // ---------------------------------------------------------------- UI

        struct UiRefs { public ChatWindow2D win; public GameObject prompt; public FarmHud hud; }

        static UiRefs BuildUI(TMP_FontAsset font, Sprite vignette, NPCInteractor2D[] npcs)
        {
            var canvasGO = new GameObject("UI", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            var canvas = canvasGO.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGO.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
            scaler.matchWidthOrHeight = 0.5f;   // between 16:9 and ultrawide, scale stays sane

            new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));

            // cozy palette
            Color woodDark = new Color(0.13f, 0.10f, 0.07f, 0.96f);
            Color trim = new Color(0.55f, 0.40f, 0.22f, 0.95f);
            Color cream = new Color(0.95f, 0.90f, 0.75f);
            Color nameGold = new Color(0.90f, 0.72f, 0.35f);

            // --- soft vignette (always on, never blocks clicks)
            var vinGO = MakeRect("Vignette", canvasGO.transform, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var vinImg = vinGO.AddComponent<Image>();
            vinImg.sprite = vignette;
            vinImg.color = new Color(0f, 0f, 0f, 0.25f);
            vinImg.raycastTarget = false;

            // --- "[ E ] Talk" prompt, bottom center (above the toolbar)
            var promptGO = MakeRect("InteractPrompt", canvasGO.transform, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f),
                                    new Vector2(300, 58), new Vector2(0, 100));
            var promptBG = promptGO.AddComponent<Image>();
            promptBG.color = woodDark;
            AddThinBorder(promptGO.transform, trim);
            MakeTMP("Text", promptGO.transform, "[ E ]  Talk", font, 26, cream,
                    TextAlignmentOptions.Center, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            promptGO.SetActive(false);

            // --- chat overlay (no box): dialogue lines float as drop-shadowed text above a thin
            // 20%-alpha input strip, bottom-center. Center-anchored fixed width: identical
            // framing at 16:9 and ultrawide — extra horizontal space just shows more farm.
            var panelGO = MakeRect("ChatWindow2D", canvasGO.transform, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f),
                                   new Vector2(940, 640), new Vector2(0, 14));
            ((RectTransform)panelGO.transform).pivot = new Vector2(0.5f, 0f);
            var panelGroup = panelGO.AddComponent<CanvasGroup>();   // ChatWindow2D fades this open/closed

            // float region: lines spawn pinned at its bottom edge, drift up and fade out near its top
            var linesGO = MakeRect("Lines", panelGO.transform, new Vector2(0, 0), new Vector2(1, 0),
                                   new Vector2(-80, 460), new Vector2(0, 136));
            ((RectTransform)linesGO.transform).pivot = new Vector2(0.5f, 0f);

            // line template: ONE rich-text TMP per line (name tint + body color ride in via
            // <color> tags; body keeps the default TMP font — the pixel font is charming for
            // chrome but harder to read in sentences) on a shared underlay material so the
            // floating text stays readable over the bright farm without any panel behind it
            var lineGO = MakeRect("LineTemplate", linesGO.transform, new Vector2(0, 0), new Vector2(1, 0),
                                  new Vector2(0, 30), Vector2.zero);
            ((RectTransform)lineGO.transform).pivot = new Vector2(0.5f, 0f);
            lineGO.AddComponent<CanvasGroup>();
            var lineTmp = lineGO.AddComponent<TextMeshProUGUI>();
            lineTmp.fontSize = 27;   // bumped from 22 — dialogue must read effortlessly over the farm
            lineTmp.color = Color.white;   // per-speaker tints come from the rich text
            lineTmp.alignment = TextAlignmentOptions.BottomLeft;
            lineTmp.raycastTarget = false;
            lineTmp.fontSharedMaterial = CreateChatLineMaterial(lineTmp.font);

            // small gold NPC-name label above the strip (the minimal stand-in for the old header)
            var titleGO = MakeTMP("Title", panelGO.transform, "-", font, 20,
                                  new Color(nameGold.r, nameGold.g, nameGold.b, 0.95f), TextAlignmentOptions.Left,
                                  new Vector2(0, 0), new Vector2(0, 0), new Vector2(320, 26), new Vector2(230, 68));

            // approach flavor line, italic grey, at the base of the float region
            var infoGO = MakeTMP("InfoText", panelGO.transform, "", null, 22, new Color(0.88f, 0.83f, 0.70f, 0.95f),
                                 TextAlignmentOptions.Center, new Vector2(0, 0), new Vector2(1, 0), new Vector2(-160, 28), new Vector2(0, 102));
            infoGO.GetComponent<TMP_Text>().fontStyle = FontStyles.Italic;

            // input row: the only visible chrome — a very light strip with an underline input
            // and text-only Say/Leave buttons
            var rowGO = MakeRect("InputRow", panelGO.transform, new Vector2(0, 0), new Vector2(1, 0), new Vector2(-140, 46), new Vector2(0, 34));
            var stripImg = rowGO.AddComponent<Image>();
            // solid dark-wood input bar (was 20%-alpha black — invisible over the bright farm);
            // gives the cream input text + Say/Leave labels a consistent dark backing to read against
            stripImg.color = new Color(0.11f, 0.08f, 0.05f, 0.80f);
            AddThinBorder(rowGO.transform, trim);   // wooden trim so the bar has a defined edge
            var rowHlg = rowGO.AddComponent<HorizontalLayoutGroup>();
            rowHlg.padding = new RectOffset(14, 10, 6, 6);
            rowHlg.spacing = 8;
            rowHlg.childControlWidth = true; rowHlg.childControlHeight = true;
            rowHlg.childForceExpandWidth = false; rowHlg.childForceExpandHeight = true;

            var inputGO = BuildLineInput(rowGO.transform, cream, nameGold, out TMP_InputField inputField);
            inputGO.AddComponent<LayoutElement>().flexibleWidth = 1f;

            var sendBtn = BuildTextButton(rowGO.transform, "Say", font, cream, 64);
            // Give-items: appears only while the basket has harvest in it (NPCInteractor2D toggles it)
            var giveBtn = BuildTextButton(rowGO.transform, "Give", font, nameGold, 74);
            giveBtn.gameObject.SetActive(false);
            var leaveBtn = BuildTextButton(rowGO.transform, "Leave", font, new Color(0.85f, 0.62f, 0.48f), 84);

            // --- component wiring
            var win = panelGO.AddComponent<ChatWindow2D>();
            SetRef(win, "canvasGroup", panelGroup);
            SetRef(win, "linesContainer", (RectTransform)linesGO.transform);
            SetRef(win, "lineTemplate", lineGO);
            SetRef(win, "inputField", inputField);
            SetRef(win, "sendButton", sendBtn);
            SetRef(win, "giveButton", giveBtn);
            SetRef(win, "leaveButton", leaveBtn);
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

            // Both villagers share the one chat window, prompt box and buttons. Every listener
            // fires on both interactors, but only the one actually in interaction reacts:
            // AskNPC guards on WaitingInInteraction and CloseInteraction is a no-op for an NPC
            // that is already Idle (same pattern as ChatDemo3D's two NPCs).
            foreach (var npc in npcs)
            {
                SetRef(npc, "chatWindow", win);
                SetRef(npc, "interactPrompt", promptGO);
                UnityEventTools.AddPersistentListener(sendBtn.onClick, new UnityAction(npc.AskNPC));
                UnityEventTools.AddPersistentListener(giveBtn.onClick, new UnityAction(npc.GiveItems));
                UnityEventTools.AddPersistentListener(leaveBtn.onClick, new UnityAction(npc.CloseInteraction));
                UnityEventTools.AddVoidPersistentListener(inputField.onSubmit, new UnityAction(npc.AskNPC));
            }
            UnityEventTools.AddPersistentListener(sendBtn.onClick, new UnityAction(win.PlayButtonClick));
            UnityEventTools.AddPersistentListener(giveBtn.onClick, new UnityAction(win.PlayButtonClick));
            UnityEventTools.AddPersistentListener(leaveBtn.onClick, new UnityAction(win.PlayButtonClick));

            var hud = BuildHud(canvasGO.transform, font, woodDark, trim, cream);

            return new UiRefs { win = win, prompt = promptGO, hud = hud };
        }

        // --- farm HUD: toolbar (bottom-left), harvest counters above it, clock (top-right)
        static FarmHud BuildHud(Transform canvas, TMP_FontAsset font, Color woodDark, Color trim, Color cream)
        {
            var hudGO = MakeRect("FarmHud", canvas, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var hud = hudGO.AddComponent<FarmHud>();

            (string icon, string key)[] slots =
            {
                ("icon_hoe", "1"), ("icon_water", "2"),
                ("seed_carrot", "3"), ("seed_turnip", "4"), ("seed_tomato", "5"),
                ("icon_harvest", "6"),
            };
            var frames = new Image[slots.Length];
            for (int i = 0; i < slots.Length; i++)
            {
                var slot = MakeRect("Slot_" + slots[i].icon, hudGO.transform, Vector2.zero, Vector2.zero,
                                    new Vector2(56, 56), new Vector2(56 + 62 * i, 56));
                var frame = slot.AddComponent<Image>();
                frame.color = new Color(0.35f, 0.27f, 0.18f, 0.9f);
                frame.raycastTarget = false;
                frames[i] = frame;
                var inner = MakeRect("BG", slot.transform, Vector2.zero, Vector2.one, new Vector2(-6, -6), Vector2.zero);
                var innerImg = inner.AddComponent<Image>();
                innerImg.color = woodDark;
                innerImg.raycastTarget = false;
                var iconGO = MakeRect("Icon", slot.transform, Vector2.zero, Vector2.one, new Vector2(-16, -16), Vector2.zero);
                var iconImg = iconGO.AddComponent<Image>();
                iconImg.sprite = S(slots[i].icon);
                iconImg.preserveAspect = true;
                iconImg.raycastTarget = false;
                MakeTMP("Key", slot.transform, slots[i].key, font, 15, cream, TextAlignmentOptions.TopLeft,
                        Vector2.zero, Vector2.one, new Vector2(-8, -4), new Vector2(6, 0));
            }

            // harvest counters, stacked above the toolbar
            string[] cropIcons = { "icon_carrot", "icon_turnip", "icon_tomato" };
            var counters = new TMP_Text[cropIcons.Length];
            for (int i = 0; i < cropIcons.Length; i++)
            {
                var row = MakeRect("Count_" + cropIcons[i], hudGO.transform, Vector2.zero, Vector2.zero,
                                   new Vector2(110, 30), new Vector2(83, 111 + 34 * i));
                var bg = row.AddComponent<Image>();
                bg.color = new Color(0f, 0f, 0f, 0.35f);
                bg.raycastTarget = false;
                var iconGO = MakeRect("Icon", row.transform, new Vector2(0, 0.5f), new Vector2(0, 0.5f),
                                      new Vector2(24, 24), new Vector2(18, 0));
                var img = iconGO.AddComponent<Image>();
                img.sprite = S(cropIcons[i]);
                img.preserveAspect = true;
                img.raycastTarget = false;
                var txt = MakeTMP("Count", row.transform, "x 0", font, 19, cream, TextAlignmentOptions.Left,
                                  new Vector2(0, 0), new Vector2(1, 1), new Vector2(-44, 0), new Vector2(20, 0));
                counters[i] = txt.GetComponent<TMP_Text>();
            }

            // coin purse, stacked right above the harvest counters (gold text — it's money)
            var coinRow = MakeRect("Coins", hudGO.transform, Vector2.zero, Vector2.zero,
                                   new Vector2(110, 30), new Vector2(83, 111 + 34 * cropIcons.Length));
            var coinBg = coinRow.AddComponent<Image>();
            coinBg.color = new Color(0f, 0f, 0f, 0.35f);
            coinBg.raycastTarget = false;
            var coinTxt = MakeTMP("Count", coinRow.transform, "0 g", font, 19,
                                  new Color(0.95f, 0.80f, 0.40f), TextAlignmentOptions.Center,
                                  new Vector2(0, 0), new Vector2(1, 1), Vector2.zero, Vector2.zero);

            // clock, top-right
            var clockGO = MakeRect("Clock", hudGO.transform, Vector2.one, Vector2.one, new Vector2(210, 44), new Vector2(-129, -46));
            var clockBg = clockGO.AddComponent<Image>();
            clockBg.color = new Color(0f, 0f, 0f, 0.4f);
            clockBg.raycastTarget = false;
            AddThinBorder(clockGO.transform, trim);
            var clockTxt = MakeTMP("Text", clockGO.transform, "Day 1   08:00", font, 21, cream,
                                   TextAlignmentOptions.Center, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);

            // controls hint, bottom-right
            MakeTMP("Hint", hudGO.transform, "WASD move   1-6 tools   Space use   E talk", font, 16,
                    new Color(0.95f, 0.90f, 0.75f, 0.55f), TextAlignmentOptions.BottomRight,
                    new Vector2(1, 0), new Vector2(1, 0), new Vector2(560, 30), new Vector2(-296, 26));

            var hudSO = new SerializedObject(hud);
            var framesProp = hudSO.FindProperty("slotFrames");
            framesProp.arraySize = frames.Length;
            for (int i = 0; i < frames.Length; i++)
                framesProp.GetArrayElementAtIndex(i).objectReferenceValue = frames[i];
            var countersProp = hudSO.FindProperty("counters");
            countersProp.arraySize = counters.Length;
            for (int i = 0; i < counters.Length; i++)
                countersProp.GetArrayElementAtIndex(i).objectReferenceValue = counters[i];
            hudSO.FindProperty("clockText").objectReferenceValue = clockTxt.GetComponent<TMP_Text>();
            hudSO.FindProperty("coinText").objectReferenceValue = coinTxt.GetComponent<TMP_Text>();
            hudSO.ApplyModifiedPropertiesWithoutUndo();

            return hud;
        }

        static GameObject BuildLineInput(Transform parent, Color cream, Color gold, out TMP_InputField field)
        {
            var go = new GameObject("InputField", typeof(RectTransform));
            go.transform.SetParent(parent, false);
            // invisible click target — the 20%-alpha strip behind the row is the only visible chrome
            var bg = go.AddComponent<Image>();
            bg.color = new Color(0f, 0f, 0f, 0f);

            field = go.AddComponent<TMP_InputField>();

            var areaGO = MakeRect("Text Area", go.transform, Vector2.zero, Vector2.one, new Vector2(-12, -8), Vector2.zero);
            areaGO.AddComponent<RectMask2D>();

            // placeholder + typed text bumped to a legible weight over the dark input bar
            // (placeholder was 55%-alpha and hard to read); typed text is full-opacity cream
            var phGO = MakeTMP("Placeholder", areaGO.transform, "Say something...", null, 20,
                               new Color(0.86f, 0.81f, 0.66f, 0.85f), TextAlignmentOptions.Left, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            phGO.GetComponent<TMP_Text>().fontStyle = FontStyles.Italic;
            var txtGO = MakeTMP("Text", areaGO.transform, "", null, 20, cream,
                                TextAlignmentOptions.Left, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);

            // thin underline instead of a box — gold, near-opaque so the field edge is clearly visible
            var underGO = MakeRect("Underline", go.transform, new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 2), new Vector2(0, 2));
            var underImg = underGO.AddComponent<Image>();
            underImg.color = new Color(gold.r, gold.g, gold.b, 0.85f);
            underImg.raycastTarget = false;

            field.textViewport = (RectTransform)areaGO.transform;
            field.textComponent = txtGO.GetComponent<TMP_Text>();
            field.placeholder = phGO.GetComponent<TMP_Text>();
            field.lineType = TMP_InputField.LineType.SingleLine;
            // clearly visible blinking caret: gold accent, thick (same treatment as the 3D souls input)
            field.caretColor = gold;
            field.customCaretColor = true;
            field.caretWidth = 3;
            field.caretBlinkRate = 0.85f;
            field.selectionColor = new Color(0.50f, 0.38f, 0.18f, 0.6f);
            field.targetGraphic = bg;
            return go;
        }

        static Button BuildTextButton(Transform parent, string label, TMP_FontAsset font,
                                      Color textColor, float width)
        {
            // minimal, boxless button: the label itself is the target graphic (hover/press tints)
            var go = new GameObject(label + "Button", typeof(RectTransform));
            go.transform.SetParent(parent, false);
            go.AddComponent<LayoutElement>().preferredWidth = width;

            var labelGO = MakeTMP("Label", go.transform, label, font, 21, textColor, TextAlignmentOptions.Center,
                                  Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var labelTmp = labelGO.GetComponent<TMP_Text>();
            labelTmp.raycastTarget = true;   // the text IS the button

            var btn = go.AddComponent<Button>();
            btn.targetGraphic = labelTmp;
            var colors = btn.colors;
            colors.highlightedColor = new Color(1.3f, 1.25f, 1.1f, 1f);
            colors.pressedColor = new Color(0.65f, 0.65f, 0.65f, 1f);
            colors.disabledColor = new Color(0.55f, 0.55f, 0.55f, 0.6f);
            btn.colors = colors;
            return btn;
        }

        static void AddThinBorder(Transform parent, Color color)
        {
            // four 2px edge images — cheap wooden trim without a 9-slice sprite
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
                var img = e.AddComponent<Image>();
                img.color = color;
                img.raycastTarget = false;
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

        // Duplicate of the chat-line font material with a soft dark underlay (drop shadow):
        // the floating dialogue lines have no panel behind them, so they need baked-in
        // contrast to stay readable over the bright pixel-art farm. One shared asset — the
        // runtime line clones all reference it, no per-line material instances.
        static Material CreateChatLineMaterial(TMP_FontAsset baseFont)
        {
            if (baseFont == null) baseFont = TMP_Settings.defaultFontAsset;
            if (baseFont == null) throw new Exception("No TMP default font asset — import TMP Essentials first");

            string path = GEN + "/ChatLine Underlay.mat";
            var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (mat == null)
            {
                mat = new Material(baseFont.material) { name = "ChatLine Underlay" };
                AssetDatabase.CreateAsset(mat, path);
            }
            // ALWAYS (re)apply the values — rebuilding the scene must be able to strengthen an
            // existing asset (the old early-return froze the first-ever bake forever). A near-black
            // dilated shadow behind every glyph keeps the floating text readable over bright grass.
            mat.EnableKeyword(ShaderUtilities.Keyword_Underlay);
            mat.SetColor(ShaderUtilities.ID_UnderlayColor, new Color(0f, 0f, 0f, 0.95f));
            mat.SetFloat(ShaderUtilities.ID_UnderlayOffsetX, 0.85f);
            mat.SetFloat(ShaderUtilities.ID_UnderlayOffsetY, -0.85f);
            mat.SetFloat(ShaderUtilities.ID_UnderlayDilate, 0.4f);
            mat.SetFloat(ShaderUtilities.ID_UnderlaySoftness, 0.15f);
            EditorUtility.SetDirty(mat);
            return mat;
        }

        static TMP_FontAsset CreateKenneyFont()
        {
            string path = GEN + "/Kenney Mini SDF.asset";
            var existing = AssetDatabase.LoadAssetAtPath<TMP_FontAsset>(path);
            if (existing != null) return existing;

            var font = AssetDatabase.LoadAssetAtPath<Font>(ART + "/Fonts/Kenney Mini.ttf");
            if (font == null) throw new Exception("Missing " + ART + "/Fonts/Kenney Mini.ttf");

            var fa = TMP_FontAsset.CreateFontAsset(font, 64, 6, GlyphRenderMode.SDFAA, 1024, 1024,
                                                   AtlasPopulationMode.Dynamic);
            fa.name = "Kenney Mini SDF";
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
                const int SZ = 512;
                var tex = new Texture2D(SZ, SZ, TextureFormat.RGBA32, false);
                var px = new Color32[SZ * SZ];
                for (int y = 0; y < SZ; y++)
                    for (int x = 0; x < SZ; x++)
                    {
                        float dx = (x - SZ * 0.5f) / (SZ * 0.5f);
                        float dy = (y - SZ * 0.5f) / (SZ * 0.5f);
                        float r = Mathf.Sqrt(dx * dx + dy * dy);
                        float a = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01((r - 0.62f) / 0.60f));
                        px[y * SZ + x] = new Color32(0, 0, 0, (byte)(a * 255));
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

        static Sprite CreateWhiteSprite()
        {
            string pngPath = GEN + "/White.png";
            if (!File.Exists(pngPath))
            {
                var tex = new Texture2D(8, 8, TextureFormat.RGBA32, false);
                var px = new Color32[64];
                for (int i = 0; i < 64; i++) px[i] = new Color32(255, 255, 255, 255);
                tex.SetPixels32(px);
                tex.Apply();
                File.WriteAllBytes(pngPath, tex.EncodeToPNG());
                UnityEngine.Object.DestroyImmediate(tex);
                AssetDatabase.ImportAsset(pngPath);
            }
            ConfigureSprite(pngPath, 16f, new Vector2(0.5f, 0.5f));
            return AssetDatabase.LoadAssetAtPath<Sprite>(pngPath);
        }

        static Sprite CreateHighlightSprite()
        {
            string pngPath = GEN + "/PlotHighlight.png";
            if (!File.Exists(pngPath))
            {
                const int SZ = 16, B = 2;
                var tex = new Texture2D(SZ, SZ, TextureFormat.RGBA32, false);
                var px = new Color32[SZ * SZ];
                for (int y = 0; y < SZ; y++)
                    for (int x = 0; x < SZ; x++)
                    {
                        bool border = x < B || x >= SZ - B || y < B || y >= SZ - B;
                        px[y * SZ + x] = border ? new Color32(255, 255, 255, 255) : new Color32(255, 255, 255, 28);
                    }
                tex.SetPixels32(px);
                tex.Apply();
                File.WriteAllBytes(pngPath, tex.EncodeToPNG());
                UnityEngine.Object.DestroyImmediate(tex);
                AssetDatabase.ImportAsset(pngPath);
            }
            ConfigureSprite(pngPath, 16f, new Vector2(0.5f, 0.5f));
            return AssetDatabase.LoadAssetAtPath<Sprite>(pngPath);
        }
    }
}

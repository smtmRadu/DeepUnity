using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity.Tutorials.ChatDemo3D.EditorTools
{
    /// <summary>
    /// Rendering audit + A/B render harness for the ChatDemo3D scene. Works on whatever scene is
    /// already OPEN (it never opens or saves a scene) and restores everything it toggles, so it is
    /// safe to invoke through the ClaudeBridge while the editor is live.
    ///
    /// Inventory()          — dumps every perf-relevant fact to ProbeLogs/scene_opt/inventory.txt
    /// FloorTileAB()        — renders each pose with the courtyard floor tiles ON vs OFF
    /// ShadowDistanceAB()   — renders each pose at QualitySettings.shadowDistance 150 / 90 / 70 / 55
    /// </summary>
    public static class ChatDemo3DPerfProbe
    {
        const string OUT = "ProbeLogs/scene_opt";

        // the exact framings used for the before/after parity shots
        static readonly (string name, Vector3 pos, Vector3 euler)[] POSES =
        {
            ("overview",    new Vector3(0f, 24f, -34f),      new Vector3(34f, 0f, 0f)),
            ("playerview",  new Vector3(0f, 2.0f, -9.6f),    new Vector3(6f, 0f, 0f)),
            ("npc_closeup", new Vector3(4.0f, 1.7f, 2.6f),   new Vector3(6f, 25f, 0f)),
            ("gate_south",  new Vector3(0f, 1.9f, -5f),      new Vector3(4f, 180f, 0f)),
            ("bossroom",    new Vector3(1f, 2.2f, 14.2f),    new Vector3(4f, 0f, 0f)),
            ("shadowcheck", new Vector3(-12f, 8f, -18f),     new Vector3(22f, 35f, 0f)),
        };

        // ------------------------------------------------------------------ inventory

        public static void Inventory()
        {
            var sb = new StringBuilder();
            var scene = EditorSceneManager.GetActiveScene();
            sb.AppendLine("scene: " + scene.path);

            var roots = scene.GetRootGameObjects();
            var renderers = new List<Renderer>();
            foreach (var r in roots) renderers.AddRange(r.GetComponentsInChildren<Renderer>(true));

            // ---- quality settings (built-in RP shadow/light budget lives here, not in the scene)
            sb.AppendLine("\n=== QualitySettings (level " + QualitySettings.GetQualityLevel() + " '" +
                          QualitySettings.names[QualitySettings.GetQualityLevel()] + "') ===");
            sb.AppendLine($"pixelLightCount   {QualitySettings.pixelLightCount}");
            sb.AppendLine($"shadows           {QualitySettings.shadows}");
            sb.AppendLine($"shadowResolution  {QualitySettings.shadowResolution}");
            sb.AppendLine($"shadowProjection  {QualitySettings.shadowProjection}");
            sb.AppendLine($"shadowCascades    {QualitySettings.shadowCascades}");
            sb.AppendLine($"shadowDistance    {QualitySettings.shadowDistance}");
            sb.AppendLine($"cascade4Split     {QualitySettings.shadowCascade4Split}");
            sb.AppendLine($"antiAliasing      {QualitySettings.antiAliasing}");
            sb.AppendLine($"vSyncCount        {QualitySettings.vSyncCount}");
            sb.AppendLine($"softParticles     {QualitySettings.softParticles}");
            sb.AppendLine($"realtimeRefProbes {QualitySettings.realtimeReflectionProbes}");
            sb.AppendLine($"lodBias           {QualitySettings.lodBias}");
            sb.AppendLine($"skinWeights       {QualitySettings.skinWeights}");

            // ---- render settings
            sb.AppendLine("\n=== RenderSettings ===");
            sb.AppendLine($"fog {RenderSettings.fog} mode {RenderSettings.fogMode} density {RenderSettings.fogDensity} color {RenderSettings.fogColor}");
            sb.AppendLine($"ambientMode {RenderSettings.ambientMode}");
            sb.AppendLine($"skybox {(RenderSettings.skybox != null ? RenderSettings.skybox.shader.name : "none")}");
            sb.AppendLine($"reflectionIntensity {RenderSettings.reflectionIntensity} bounces {RenderSettings.reflectionBounces}");
            sb.AppendLine($"defaultReflectionMode {RenderSettings.defaultReflectionMode} res {RenderSettings.defaultReflectionResolution}");

            // ---- cameras
            sb.AppendLine("\n=== Cameras ===");
            foreach (var c in roots.SelectMany(r => r.GetComponentsInChildren<Camera>(true)))
                sb.AppendLine($"{c.name}: HDR {c.allowHDR} MSAA {c.allowMSAA} dynRes {c.allowDynamicResolution} " +
                              $"near {c.nearClipPlane} far {c.farClipPlane} fov {c.fieldOfView} " +
                              $"clear {c.clearFlags} occlusionCulling {c.useOcclusionCulling} depthTexture {c.depthTextureMode}");

            // ---- lights
            var lights = roots.SelectMany(r => r.GetComponentsInChildren<Light>(true)).ToList();
            sb.AppendLine($"\n=== Lights ({lights.Count}) ===");
            foreach (var g in lights.GroupBy(l => $"{l.type} shadows={l.shadows} renderMode={l.renderMode}"))
                sb.AppendLine($"{g.Count(),4} x {g.Key}  ranges [{g.Min(l => l.range):0.0}..{g.Max(l => l.range):0.0}] " +
                              $"intensity [{g.Min(l => l.intensity):0.00}..{g.Max(l => l.intensity):0.00}]");
            foreach (var l in lights.Where(l => l.shadows != LightShadows.None))
                sb.AppendLine($"SHADOW CASTER LIGHT: {Path(l.transform)} {l.type} strength {l.shadowStrength} bias {l.shadowBias}");

            // ---- renderers, materials, shadow casters
            int slots = renderers.Sum(r => r.sharedMaterials.Length);
            int casters = renderers.Count(r => r.shadowCastingMode != UnityEngine.Rendering.ShadowCastingMode.Off);
            int casterSlots = renderers.Where(r => r.shadowCastingMode != UnityEngine.Rendering.ShadowCastingMode.Off)
                                       .Sum(r => r.sharedMaterials.Length);
            int nonStatic = renderers.Count(r => !r.gameObject.isStatic);
            sb.AppendLine($"\n=== Renderers ===");
            sb.AppendLine($"renderers {renderers.Count} | material slots {slots} | shadow casters {casters} (slots {casterSlots})");
            sb.AppendLine($"non-static renderers {nonStatic}: " +
                          string.Join(", ", renderers.Where(r => !r.gameObject.isStatic).Take(30).Select(r => Path(r.transform))));
            sb.AppendLine($"receiveShadows off: {renderers.Count(r => !r.receiveShadows)}");

            long tris = 0;
            foreach (var r in renderers)
            {
                Mesh m = r is SkinnedMeshRenderer smr ? smr.sharedMesh : r.GetComponent<MeshFilter>()?.sharedMesh;
                if (m != null) tris += m.triangles.Length / 3;
            }
            sb.AppendLine($"triangles in scene (sum of unique renderer meshes) {tris:N0}");

            sb.AppendLine("\n--- renderers per top-level branch ---");
            foreach (var g in renderers.GroupBy(r => Branch(r.transform)).OrderByDescending(g => g.Count()))
                sb.AppendLine($"{g.Count(),5}  {g.Key,-28} slots {g.Sum(r => r.sharedMaterials.Length),5}  casters {g.Count(r => r.shadowCastingMode != UnityEngine.Rendering.ShadowCastingMode.Off),5}");

            sb.AppendLine("\n--- top materials by renderer count (static batching potential) ---");
            foreach (var g in renderers.SelectMany(r => r.sharedMaterials.Select(m => new { r, m }))
                                       .GroupBy(x => x.m == null ? "<null>" : x.m.name + " | " + (x.m.shader != null ? x.m.shader.name : "?"))
                                       .OrderByDescending(g => g.Count()).Take(25))
                sb.AppendLine($"{g.Count(),5}  {g.Key}");

            // ---- per-pixel point-light overlap = extra ForwardAdd passes per renderer
            var points = lights.Where(l => l.type == LightType.Point && l.enabled).ToArray();
            var hist = new Dictionary<int, int>();
            long addPasses = 0;
            int cap = QualitySettings.pixelLightCount;
            foreach (var r in renderers)
            {
                Bounds b = r.bounds;
                int n = points.Count(l => b.SqrDistance(l.transform.position) <= l.range * l.range);
                hist.TryGetValue(n, out int c); hist[n] = c + 1;
                addPasses += Math.Min(n, cap) * r.sharedMaterials.Length;
            }
            sb.AppendLine("\n--- point lights whose range sphere touches each renderer (ForwardAdd passes) ---");
            foreach (var kv in hist.OrderBy(k => k.Key))
                sb.AppendLine($"{kv.Value,5} renderers touched by {kv.Key} point light(s)");
            sb.AppendLine($"estimated extra ForwardAdd draw calls per camera pass (capped at pixelLightCount={cap}): {addPasses:N0}");

            // ---- the coplanar-floor question
            sb.AppendLine("\n=== Floor tiles vs ground plane ===");
            var tiles = FloorTileRenderers();
            if (tiles.Count > 0)
            {
                sb.AppendLine($"floor tile renderers {tiles.Count} | top-Y min {tiles.Min(r => r.bounds.max.y):0.0000} " +
                              $"max {tiles.Max(r => r.bounds.max.y):0.0000} | bottom-Y min {tiles.Min(r => r.bounds.min.y):0.0000}");
                sb.AppendLine($"floor tile material slots {tiles.Sum(r => r.sharedMaterials.Length)} | " +
                              $"casters {tiles.Count(r => r.shadowCastingMode != UnityEngine.Rendering.ShadowCastingMode.Off)}");
            }
            var ground = roots.FirstOrDefault(g => g.name == "Ground");
            if (ground != null)
            {
                var gr = ground.GetComponent<Renderer>();
                var gm = ground.GetComponent<MeshFilter>()?.sharedMesh;
                sb.AppendLine($"Ground renderer bounds {gr.bounds.min} .. {gr.bounds.max} | verts {gm?.vertexCount} tris {(gm != null ? gm.triangles.Length / 3 : 0)} " +
                              $"| casts {gr.shadowCastingMode} | static {ground.isStatic}");
            }

            // ---- animators
            sb.AppendLine("\n=== Animators ===");
            foreach (var a in roots.SelectMany(r => r.GetComponentsInChildren<Animator>(true)))
                sb.AppendLine($"{Path(a.transform),-46} culling {a.cullingMode} rootMotion {a.applyRootMotion} " +
                              $"ctrl {(a.runtimeAnimatorController != null ? a.runtimeAnimatorController.name : "none")}");

            // ---- UI
            sb.AppendLine("\n=== UI ===");
            foreach (var cv in roots.SelectMany(r => r.GetComponentsInChildren<Canvas>(true)))
            {
                var graphics = cv.GetComponentsInChildren<Graphic>(true);
                sb.AppendLine($"Canvas {Path(cv.transform)} mode {cv.renderMode} overrideSorting {cv.overrideSorting} " +
                              $"nested-canvases {cv.GetComponentsInChildren<Canvas>(true).Length - 1} " +
                              $"graphics {graphics.Length} raycastTargets {graphics.Count(g => g.raycastTarget)} " +
                              $"layoutGroups {cv.GetComponentsInChildren<LayoutGroup>(true).Length} " +
                              $"sizeFitters {cv.GetComponentsInChildren<ContentSizeFitter>(true).Length}");
            }

            Directory.CreateDirectory(OUT);
            string path = System.IO.Path.Combine(OUT, "inventory.txt");
            File.WriteAllText(path, sb.ToString());
            Debug.Log($"[PerfProbe] inventory -> {path}\n" +
                      $"renderers {renderers.Count} slots {slots} casters {casters}/{casterSlots} tris {tris:N0} " +
                      $"lights {lights.Count} addPasses~{addPasses:N0} floorTiles {tiles.Count} " +
                      $"shadowDistance {QualitySettings.shadowDistance} cascades {QualitySettings.shadowCascades} " +
                      $"HDR {roots.SelectMany(r => r.GetComponentsInChildren<Camera>(true)).FirstOrDefault()?.allowHDR}");
        }

        // ------------------------------------------------------------------ A/B renders

        /// The courtyard + boss-room floor tiles are placed with their TOP face at y=0, which is
        /// exactly where the flat part of the ground mesh sits. Render both ways to find out
        /// whether a single pixel of them is ever visible.
        public static void FloorTileAB()
        {
            var tiles = FloorTileRenderers();
            RenderPoses("ab_tilesOn");
            var were = tiles.Select(r => r.enabled).ToArray();
            try
            {
                foreach (var r in tiles) r.enabled = false;
                RenderPoses("ab_tilesOff");
            }
            finally
            {
                for (int i = 0; i < tiles.Count; i++) tiles[i].enabled = were[i];
            }
            Debug.Log($"[PerfProbe] floor-tile A/B done over {tiles.Count} renderers -> {OUT}");
        }

        /// Shadow distance is a QualitySettings value in the built-in pipeline, so sweep it here
        /// (and put it back) to see at what point trimming it becomes visible through the fog.
        public static void ShadowDistanceAB()
        {
            float original = QualitySettings.shadowDistance;
            try
            {
                foreach (float d in new[] { 150f, 90f, 70f, 55f })
                {
                    QualitySettings.shadowDistance = d;
                    RenderPoses("ab_sd" + (int)d);
                }
            }
            finally
            {
                QualitySettings.shadowDistance = original;
            }
            Debug.Log($"[PerfProbe] shadow-distance A/B done (restored to {original}) -> {OUT}");
        }

        // ------------------------------------------------------------------ helpers

        static void RenderPoses(string prefix)
        {
            Directory.CreateDirectory(OUT);
            var camGO = new GameObject("PerfProbeCam");
            try
            {
                var cam = camGO.AddComponent<Camera>();
                cam.fieldOfView = 55f;
                cam.nearClipPlane = 0.05f;
                cam.farClipPlane = 500f;
                var rt = new RenderTexture(1600, 900, 24);
                foreach (var (name, pos, euler) in POSES)
                {
                    camGO.transform.SetPositionAndRotation(pos, Quaternion.Euler(euler));
                    cam.targetTexture = rt;
                    cam.Render();
                    RenderTexture.active = rt;
                    var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
                    tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                    tex.Apply();
                    RenderTexture.active = null;
                    cam.targetTexture = null;
                    File.WriteAllBytes(System.IO.Path.Combine(OUT, prefix + "_" + name + ".png"), tex.EncodeToPNG());
                    UnityEngine.Object.DestroyImmediate(tex);
                }
                rt.Release();
                UnityEngine.Object.DestroyImmediate(rt);
            }
            finally { UnityEngine.Object.DestroyImmediate(camGO); }
        }

        static List<Renderer> FloorTileRenderers()
        {
            var list = new List<Renderer>();
            foreach (var root in EditorSceneManager.GetActiveScene().GetRootGameObjects())
                foreach (var t in root.GetComponentsInChildren<Transform>(true))
                    if (t.name.StartsWith("Floor_"))
                        list.AddRange(t.GetComponentsInChildren<Renderer>(true));
            return list;
        }

        static string Branch(Transform t)
        {
            var chain = new List<string>();
            for (Transform c = t; c != null; c = c.parent) chain.Add(c.name);
            chain.Reverse();
            return chain.Count <= 2 ? string.Join("/", chain) : chain[0] + "/" + chain[1];
        }

        static string Path(Transform t)
        {
            string s = t.name;
            for (Transform c = t.parent; c != null; c = c.parent) s = c.name + "/" + s;
            return s;
        }
    }
}

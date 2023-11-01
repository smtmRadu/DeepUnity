using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// GridSensor returns the observations of all objects reached by their tags. <br></br>
    /// No hit -> value 0    <br />
    /// Tag 1 hit -> value 1 <br />
    /// Tag 2 hit -> value 2 <br />
    /// ...
    /// </summary>
    [AddComponentMenu("DeepUnity/GridSensor")]
    public class GridSensor : MonoBehaviour, ISensor
    {
        private GridCellInfo[,,] Observations;
        [SerializeField, Tooltip("@scene type")] World world = World.World3d;
        [SerializeField, Tooltip("@LayerMask used when casting the rays")] LayerMask layerMask = ~0;
        [SerializeField] string[] detectableTags;
        [SerializeField, Range(0.01f, 100f)] float scale = 1f;
        [SerializeField, Range(0.01f, 1f), Tooltip("@cast overlap raio")] float castScale = 0.95f;
        [SerializeField, Range(1, 20f)] int width = 8;
        [SerializeField, Range(1, 20f)] int height = 8;
        [SerializeField, Range(1, 20f)] int depth = 8;

        [Space(10)]
        [SerializeField, Range(-4.5f, 4.5f), Tooltip("@grid X axis offset")] float xOffset = 0;
        [SerializeField, Range(-4.5f, 4.5f), Tooltip("@grid Y axis offset")] float yOffset = 0;
        [SerializeField, Range(-4.5f, 4.5f), Tooltip("@grid Z axis offset\n@not used in 2D world")] float zOffset = 0;

        [Space(10)]
        [SerializeField] Color missColor = Color.gray;
        Color defaultHitColor = Color.black;

        private void Awake()
        {
            Observations = new GridCellInfo[depth, height, width];
        }
        private void OnDrawGizmos()
        {
            Vector3 origin000 = transform.position + (Vector3.one - new Vector3(width, height, depth)) * scale / 2f + new Vector3(xOffset, yOffset, zOffset) * scale;

            // Compute positions
            for (int d = 0; d < depth; d++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        Vector3 position = origin000 + new Vector3(w, h, d) * scale;

                        if (world == World.World3d)
                        {
                            Collider[] hits = Physics.OverlapBox(position, Vector3.one * scale * castScale / 2f, new Quaternion(0, 0, 0, 1), layerMask);
 
                            if (hits.Length > 0)
                            {
                                Renderer rend;
                                hits[0].gameObject.TryGetComponent(out rend);
                                Gizmos.color = rend != null ? rend.sharedMaterial.color : defaultHitColor;
                            }
                            else
                                Gizmos.color = missColor;

                            Gizmos.color = new Color(Gizmos.color.r, Gizmos.color.g, Gizmos.color.b, 0.6f);
                            Gizmos.DrawWireCube(position, Vector3.one * scale * castScale);
                            Gizmos.color = new Color(Gizmos.color.r, Gizmos.color.g, Gizmos.color.b, 0.2f);
                            Gizmos.DrawCube(position, Vector3.one * scale * castScale);
                            
                           
                        }
                        else if(world == World.World2d)
                        {
                            if (d > 0)
                                return;

                            Collider2D hit = Physics2D.OverlapBox(position, Vector2.one * scale * castScale, 0);
                            bool gotHit = hit != null;

                            if (gotHit)
                            {
                                SpriteRenderer sr;
                                hit.gameObject.TryGetComponent(out sr);
                                Gizmos.color = sr != null ? sr.color : defaultHitColor;
                            }
                            else
                                Gizmos.color = missColor;
                            Gizmos.color = new Color(Gizmos.color.r, Gizmos.color.g, Gizmos.color.b, 0.6f);
                            Gizmos.DrawWireCube(new Vector3(position.x, position.y, transform.position.z), Vector3.one * scale * castScale);
                            Gizmos.color = new Color(Gizmos.color.r, Gizmos.color.g, Gizmos.color.b, 0.4f);
                            Gizmos.DrawCube(new Vector3(position.x, position.y, transform.position.z), Vector3.one * scale * castScale);
                        }
                       
                    }
                }
            }
           

        }

        /// <summary>
        /// - HasOverlappedObject is 1 if true, otherwise 0.<br></br>
        /// - OverlappedObjectTagIndex is One Hot Encoded,  where the first spot represents a non-detectable tag. <br></br>
        /// Example: <br></br> 
        /// <b>[cell 1: <em>HasOverlappedObject</em>, NonDetectableTag, DetectableTag[0], DetectableTag[1], ... DetectableTag[n-1], <br></br>
        /// cell 2: <em>HasOverlappedObject</em>, NonDetectableTag, DetectableTag[0], DetectableTag[1], ... DetectableTag[n-1], ..... ]</b>
        /// </summary>
        /// <returns>Returns a float[] of <b>length = width * height * depth * (2 + num_detectable_tags)</b></returns>
        public float[] GetObservationsVector()
        {
            CastGrid();
            int cellDataSize = 2 + detectableTags.Length;
            float[] vector = new float[cellDataSize * Observations.GetLength(0) * Observations.GetLength(1) * Observations.GetLength(2)];
            int index = 0; 
            for (int k = 0; k < depth; k++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        GridCellInfo cell = Observations[k, h, w];
                        vector[index++] = cell.HasOverlappedObject ? 1f : 0f;

                        // OneHotEncode
                        if (cell.OverlappedObjectTagIndex == -1)
                            vector[index++] = 1f;
                        else
                            vector[index++] = 0f;


                        for (int i = 0; i < detectableTags.Length; i++)
                            if (cell.OverlappedObjectTagIndex == i)
                                vector[index++] = 1f;
                            else
                                vector[index++] = 0f;
                    }
                }
            }
            return vector;
        }

        /// <summary>
        ///  - HasOverlappedObject is 1 if true, otherwise 0.<br></br>
        ///  - If DetectableTags.Length > 0, OverlappedObejctTagIndex is normalized in range [0, 1]. If the OverlappedObjectTag is not contained into DetectableTags, the value is -1. <br></br>
        /// Example: <br></br>
        /// if DetectableTags.Length > 0: <b>[cell 1: HasOverlappedObject, NormalizedOverlappedObjectTagIndex, cell 2: HasOverlappedObject, NormalizedOverlappedObjectTagIndex, .....]</b> <br></br>
        /// else: <b>[cell 1: HasOverlappedObject, cell 2: HasOverlappedObject, .....]</b>
        /// 
        /// </summary>
        /// <returns>Returns a float[] of <b>length = width * height * depth * 2</b> if DetectableTags.Length > 0 else <b>width * height * depth</b>.</returns>
        public float[] GetCompressedObservationsVector()
        {
            CastGrid();

            if(detectableTags.Length > 0)
            {
                float[] vector = new float[2 * Observations.GetLength(0) * Observations.GetLength(1) * Observations.GetLength(2)];
                int index = 0;
                for (int k = 0; k < depth; k++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            GridCellInfo cell = Observations[k, h, w];
                            vector[index++] = cell.HasOverlappedObject ? 1f : 0f;

                            // OneHotEncode
                            if (cell.OverlappedObjectTagIndex == -1)
                                vector[index++] = -1f;
                            else
                                vector[index++] = cell.OverlappedObjectTagIndex / (float)cell.OverlappedObjectTagIndex;
                        }
                    }
                }
                return vector;
            }
            else
            {
                float[] vector = new float[Observations.GetLength(0) * Observations.GetLength(1) * Observations.GetLength(2)];
                int index = 0;
                for (int k = 0; k < depth; k++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            GridCellInfo cell = Observations[k, h, w];
                            vector[index++] = cell.HasOverlappedObject ? 1f : 0f;
                        }
                    }
                }
                return vector;
            }
        }
        /// <summary>
        /// Returns information of all grid cells in (depth, height, width) dimensions. In 2D, first dimension (depth) is 1.
        /// </summary>
        /// <returns></returns>
        public GridCellInfo[,,] GetObservationsGridCells()
        {
            CastGrid();
            return Observations.Clone() as GridCellInfo[,,];
        }

        private void CastGrid()
        {
            Vector3 origin000 = transform.position + (Vector3.one - new Vector3(width, height, depth)) * scale / 2f + new Vector3(xOffset, yOffset, zOffset) * scale;

            // Compute positions
            for (int d = 0; d < depth; d++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        Vector3 position = origin000 + new Vector3(w, h, d) * scale;
                        // string[] tags = UnityEditorInternal.InternalEditorUtility.tags;

                        if (world == World.World3d)
                        {
                            Collider[] hits = Physics.OverlapBox(position, Vector3.one * scale * castScale / 2f, new Quaternion(0, 0, 0, 1), layerMask);

                            GridCellInfo cellInfo = new GridCellInfo();
                            cellInfo.HasOverlappedObject = hits.Length > 0;
                            cellInfo.OverlappedObjectTagIndex = hits.Length > 0 && detectableTags != null ? Array.IndexOf(detectableTags, hits[0].tag) : -1;
                            Observations[d,h,w] = cellInfo;
                        }
                        else if (world == World.World2d)
                        {
                            if (d > 0)
                                return;

                            Collider2D hit = Physics2D.OverlapBox(position, Vector2.one * scale * castScale, 0);
                            GridCellInfo cellInfo = new GridCellInfo();
                            cellInfo.HasOverlappedObject = hit;
                            cellInfo.OverlappedObjectTagIndex = hit && detectableTags != null ? Array.IndexOf(detectableTags, hit.tag) : -1;
                            Observations[d, h, w] = cellInfo;

                        }

                    }
                }
            }

        }
    }
    [CustomEditor(typeof(GridSensor)), CanEditMultipleObjects]
    class ScriptlessGridSensor : Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            List<string> _dontDrawMe = new List<string>() { "m_Script" };


            SerializedProperty sr = serializedObject.FindProperty("world");

            if (sr.enumValueIndex == (int)World.World2d)
            {
                _dontDrawMe.Add("deep");
                _dontDrawMe.Add("zOffset");

            }

            SerializedProperty detTags = serializedObject.FindProperty("detectableTags");
            SerializedProperty width = serializedObject.FindProperty("width");
            SerializedProperty height = serializedObject.FindProperty("height");
            SerializedProperty depth = serializedObject.FindProperty("depth");

            CheckTags(detTags);
           
            DrawPropertiesExcluding(serializedObject, _dontDrawMe.ToArray());

            int vecDim = sr.enumValueIndex == (int)World.World2d ?
              (2 + detTags.arraySize) * width.intValue * height.intValue :
              (2 + detTags.arraySize) * width.intValue * height.intValue * depth.intValue;

            int compVecDim = sr.enumValueIndex == (int)World.World2d ?
                width.intValue * height.intValue :
                width.intValue * height.intValue * depth.intValue;
            if (detTags.arraySize > 0)
                compVecDim *= 2;
            
            EditorGUILayout.HelpBox($"Observations Vector contains {vecDim} float values. Compressed Observations Vector contains {compVecDim} float values.", MessageType.Info);


            serializedObject.ApplyModifiedProperties();
        }
        private void CheckTags(SerializedProperty detectableTags)
        {
            List<string> tagsToList = new List<string>();
            for (int i = 0; i < detectableTags.arraySize; i++)
                tagsToList.Add(detectableTags.GetArrayElementAtIndex(i).stringValue);

            List<(int, string)> tags_that_are_not_existing = new List<(int, string)>();
            for (int i = 0; i < tagsToList.Count; i++)
            {
                if (!UnityEditorInternal.InternalEditorUtility.tags.Contains(tagsToList[i]))
                {
                    tags_that_are_not_existing.Add((i, tagsToList[i]));
                }
            }

            if (tags_that_are_not_existing.Count == 1)
                EditorGUILayout.HelpBox(
                    $"Detectable tag '{tags_that_are_not_existing[0].Item2}' at index {tags_that_are_not_existing[0].Item1} is not defined!", MessageType.Warning);

            else if (tags_that_are_not_existing.Count > 1)
            {
                string wrongTagsList = "";
                foreach (var item in tags_that_are_not_existing)
                {
                    wrongTagsList += $"'{item.Item2}' ({item.Item1}), ";
                }
                wrongTagsList = wrongTagsList.Substring(0, wrongTagsList.Length - 2);
                EditorGUILayout.HelpBox(
                    $"Detectable tags {wrongTagsList} are not defined!", MessageType.Warning);

            }


            // Check for doubles
            var set = tagsToList.ToHashSet();
            if (tagsToList.Count != set.Count)
                EditorGUILayout.HelpBox(
                   $"Detectable tags contains doubles!", MessageType.Warning);
        }
    }
}


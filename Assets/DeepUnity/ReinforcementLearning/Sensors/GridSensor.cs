using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// No hit value: 0       <br />
    /// Tag 0 hit value: 1    <br />
    /// Tag 1 hit value: 2    <br />
    /// ...
    /// </summary>
    [AddComponentMenu("DeepUnity/Grid Sensor")]
    public class GridSensor : MonoBehaviour, ISensor
    {
 	    private List<int> Observations = new List<int>();
        [SerializeField, Tooltip("@scene type")] World world = World.World3d;
        [SerializeField, Tooltip("@LayerMask used when casting the rays")] LayerMask layerMask = ~0;
        [SerializeField, Range(0.01f, 100f)] float scale = 1f;
        [SerializeField, Range(0.01f, 0.99f), Tooltip("@cast overlap raio")] float castScale = 0.95f;
        [SerializeField, Range(1, 10f)] int width = 8;
        [SerializeField, Range(1, 10f)] int height = 8;
        [SerializeField, Range(1, 10f)] int deep = 8;

        [Space(10)]
        [SerializeField, Range(-4.5f, 4.5f), Tooltip("@grid X axis offset")] float xOffset = 0;
        [SerializeField, Range(-4.5f, 4.5f), Tooltip("@grid Y axis offset")] float yOffset = 0;
        [SerializeField, Range(-4.5f, 4.5f), Tooltip("@grid Z axis offset\n@not used in 2D world")] float zOffset = 0;

        [Space(10)]
        [SerializeField] Color missColor = new Color(Color.red.r, Color.red.g, Color.red.b, 0.5f);
        [SerializeField, Tooltip("Color drawn for hit objects with the specific tag.")] Color[] hitColor = 
            new Color[] { Color.green, Color.blue, Color.magenta, Color.yellow, Color.cyan};


        private void Start()
        {
            CastGrid();
        }
        private void Update()
        {
            CastGrid();
        }
        private void OnDrawGizmos()
        {
            Vector3 origin000 = transform.position + (Vector3.one - new Vector3(width, height, deep)) * scale / 2f + new Vector3(xOffset, yOffset, zOffset) * scale;

            // Compute positions
            for (int d = 0; d < deep; d++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        Vector3 position = origin000 + new Vector3(w, h, d) * scale;
                        string[] tags = UnityEditorInternal.InternalEditorUtility.tags;

                        if (world == World.World3d)
                        {
                            Collider[] hits = Physics.OverlapBox(position, Vector3.one * scale * castScale / 2f, new Quaternion(0, 0, 0, 1), layerMask);

                            if (hits.Length > 0)
                            {
                                
                                int index = tags.ToList().IndexOf(hits[0].tag);
                                try
                                {
                                    Gizmos.color = hitColor[index];
                                }
                                catch
                                {
                                    Gizmos.color = Color.green;
                                }
                            }
                            else
                            {

                                Gizmos.color = missColor;
                            }
                            Gizmos.DrawWireCube(position, Vector3.one * scale * castScale);
                        }
                        else if(world == World.World2d)
                        {
                            if (d == 1)
                                return;

                            Collider2D hit = Physics2D.OverlapBox(position, Vector2.one * scale * castScale, 0);

                            if(hit != null)
                            {
                                int index = tags.ToList().IndexOf(hit.tag);
                                try
                                {
                                    Gizmos.color = hitColor[index];
                                }
                                catch
                                {
                                    Gizmos.color = Color.green;
                                }
                            }
                            else
                            {

                                Gizmos.color = missColor;
                            }

                            Gizmos.DrawWireCube(new Vector3(position.x, position.y, transform.position.z), Vector3.one * scale * castScale);
                        }
                       
                    }
                }
            }
           

        }

        /// <summary>
        /// <b>Length</b> = <b>Width</b> * <b>Height</b> * (if World == 3D <b>Deep</b> else <b>1</b>)
        /// </summary>
        /// <returns>IEnumerable of int values</returns>
        public IEnumerable GetObservations()
        {
            return Observations;
        }

        private void CastGrid()
        {
            Observations.Clear();
            Vector3 origin000 = transform.position + (Vector3.one - new Vector3(width, height, deep)) * scale / 2f + new Vector3(xOffset, yOffset, zOffset) * scale;

            // Compute positions
            for (int d = 0; d < deep; d++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        Vector3 position = origin000 + new Vector3(w, h, d) * scale;
                        string[] tags = UnityEditorInternal.InternalEditorUtility.tags;

                        if (world == World.World3d)
                        {
                            Collider[] hits = Physics.OverlapBox(position, Vector3.one * scale * castScale / 2f, new Quaternion(0, 0, 0, 1), layerMask);

                            if (hits.Length > 0)
                            {
                                int index = tags.ToList().IndexOf(hits[0].tag);
                                Observations.Add(index + 1);
                            }
                            else
                            {
                                Observations.Add(0);
                            }
                        }
                        else if (world == World.World2d)
                        {
                            if (d == 1)
                                return;

                            Collider2D hit = Physics2D.OverlapBox(position, Vector2.one * scale * castScale, 0);

                            if (hit != null)
                            {
                                int index = tags.ToList().IndexOf(hit.tag);
                                Observations.Add(index + 1);
                            }
                            else
                            {
                                Observations.Add(0);
                            }
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

            var script = target as GridSensor;

            List<string> _dontDrawMe = new List<string>() { "m_Script" };


            SerializedProperty sr = serializedObject.FindProperty("world");

            if (sr.enumValueIndex == (int)World.World2d)
            {
                _dontDrawMe.Add("deep");
                _dontDrawMe.Add("zOffset");

            }
            DrawPropertiesExcluding(serializedObject, _dontDrawMe.ToArray());

            serializedObject.ApplyModifiedProperties();
        }
    }
}


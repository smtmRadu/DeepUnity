using System;
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Collections;

namespace DeepUnity
{
    [AddComponentMenu("DeepUnity/Ray Sensor")]
    public class RaySensor : MonoBehaviour, ISensor
    {
        private List<float> Observations = new List<float>();

        [SerializeField, Tooltip("@scene type")] World world = World.World3d;
        [SerializeField, Tooltip("@LayerMask used when casting the rays")] LayerMask layerMask = ~0;
        [SerializeField, Tooltip("@observation value returned by the rays")] RayInfo info = RayInfo.Distance;
        [SerializeField, Range(1, 50), Tooltip("@size of the buffer equals the number of rays")] int rays = 5;
        [SerializeField, Range(1, 360)] int fieldOfView = 45;
        [SerializeField, Range(0, 359)] int rotationOffset = 0;
        [SerializeField, Range(1, 1000), Tooltip("@maximum length of the rays\n@when no collision, value of the observation is 0")] int distance = 100;
        [SerializeField, Range(0.01f, 10)] float sphereCastRadius = 0.5f;

        [Space(10)]
        [SerializeField, Range(-45, 45), Tooltip("@ray vertical tilt\n@not used in 2D world")] float tilt = 0;
        [SerializeField, Range(-5, 5), Tooltip("@ray X axis offset")] float xOffset = 0;
        [SerializeField, Range(-5, 5), Tooltip("@ray Y axis offset")] float yOffset = 0;
        [SerializeField, Range(-5, 5), Tooltip("@ray Z axis offset\n@not used in 2D world")] float zOffset = 0;

        [Space(10)]
        [SerializeField] Color rayColor = Color.green;
        [SerializeField] Color missRayColor = Color.red;




        private void Start()
        {
            CastRays();
        }
        private void Update()
        {
            CastRays();
        }
        private void OnDrawGizmos()
        {
            float oneAngle = rays == 1 ? 0 : -fieldOfView / (rays - 1f);

            float begin = (float)-oneAngle * (rays - 1f) / 2f + rotationOffset;
            Vector3 startAngle;

            if (world == World.World3d)
                startAngle = Quaternion.AngleAxis(tilt, transform.right) * Quaternion.AngleAxis(tilt, transform.forward) * Quaternion.AngleAxis(begin, transform.up) * transform.forward;
            else //world2d
                startAngle = Quaternion.AngleAxis(begin, transform.forward) * transform.up;

            Vector3 castOrigin = transform.position + (transform.right * xOffset + transform.up * yOffset + transform.forward * zOffset) * transform.lossyScale.magnitude;

            float currentAngle = 0;

            for (int r = 0; r < rays; r++)
            {
                Vector3 rayDirection;
                if (world == World.World3d) //3d
                {
                    rayDirection = Quaternion.AngleAxis(currentAngle, transform.up) * startAngle;

                    RaycastHit hit;
                    bool isHit = Physics.SphereCast(castOrigin, sphereCastRadius, rayDirection, out hit, distance, layerMask);
                    
                    if (isHit == true)
                    {
                        Gizmos.color = rayColor;
                        Gizmos.DrawRay(castOrigin, rayDirection * hit.distance);
                        Gizmos.DrawWireSphere(castOrigin + rayDirection * hit.distance, sphereCastRadius);
                    }
                    else
                    {
                        Gizmos.color = missRayColor;
                        Gizmos.DrawRay(castOrigin, rayDirection * distance);
                    }
                }
                else //2d
                {
                    rayDirection = Quaternion.AngleAxis(currentAngle, transform.forward) * startAngle;
                    
                    RaycastHit2D hit2D = Physics2D.CircleCast(castOrigin, sphereCastRadius, rayDirection, distance, layerMask);
                    if (hit2D == true)
                    {
                        Gizmos.color = rayColor;
                        Gizmos.DrawRay(castOrigin, rayDirection * hit2D.distance);
                        Gizmos.DrawWireSphere(castOrigin + rayDirection * hit2D.distance, sphereCastRadius);
                    }
                    else
                    {
                        Gizmos.color = missRayColor;
                        Gizmos.DrawRay(castOrigin, rayDirection * distance);
                    }
                }

                currentAngle += oneAngle;
            }


        }
    
        public IEnumerable GetObservations()
        {
            return Observations;
        }

       

        /// <summary>
        /// This methods casts the necessary rays.
        /// </summary>
        private void CastRays()
        {
            Observations.Clear();
            float oneAngle = rays == 1 ? 0 : -fieldOfView / (rays - 1f);

            float begin = (float)-oneAngle * (rays - 1f) / 2f + rotationOffset;
            Vector3 startAngle;
            if (world == World.World3d)
                startAngle = Quaternion.AngleAxis(tilt, transform.right) * Quaternion.AngleAxis(tilt, transform.forward) * Quaternion.AngleAxis(begin, transform.up) * transform.forward;
            else //world2d
                startAngle = Quaternion.AngleAxis(begin, transform.forward) * transform.up;


            Vector3 castOrigin = transform.position + (transform.right * xOffset + transform.up * yOffset + transform.forward * zOffset) * transform.lossyScale.magnitude;

            float currentAngle = 0;

            for (int r = 0; r < rays; r++)
            {
                
                if (world == World.World3d)
                {
                    Vector3 rayDirection = Quaternion.AngleAxis(currentAngle, transform.up) * startAngle;
                    CastRay3D(castOrigin, sphereCastRadius, rayDirection, distance, layerMask);
                }
                else
                {
                    Vector3 rayDirection = Quaternion.AngleAxis(currentAngle, transform.forward) * startAngle;
                    CastRay2D(castOrigin, sphereCastRadius, rayDirection, distance, layerMask);
                }
              
                currentAngle += oneAngle;
            }
        }
        /// <summary>
        /// This method casts only rays for 3D worlds. It is called from CastRays().
        /// </summary>
        private void CastRay3D(Vector3 castOrigin, float sphereCastRadius, Vector3 rayDirection, float distance, LayerMask layerMask)
        {
            RaycastHit hit;
            if (Physics.SphereCast(castOrigin, sphereCastRadius, rayDirection, out hit, distance, layerMask))
            {
                switch(info)
                {
                    case RayInfo.Distance:
                        Observations.Add(hit.distance);
                        break;
                    case RayInfo.Layer:
                        Observations.Add(hit.collider.gameObject.layer);
                        break;
                    case RayInfo.Angle:
                        Observations.Add(Vector3.Angle(hit.normal, rayDirection));
                        break;
                    case RayInfo.All:
                        Observations.Add(hit.distance);
                        Observations.Add(hit.collider.gameObject.layer);
                        Observations.Add(Vector3.Angle(hit.normal, rayDirection));
                        break;
                }
            }
            else
            {
                switch (info)
                {
                    case RayInfo.Distance:
                        Observations.Add(0);
                        break;
                    case RayInfo.Layer:
                        Observations.Add(0);
                        break;
                    case RayInfo.Angle:
                        Observations.Add(0);
                        break;
                    case RayInfo.All:
                        Observations.Add(0);
                        Observations.Add(0);
                        Observations.Add(0);
                        break;
                }
            }
        }
        /// <summary>
        /// This method casts only rays for 2D worlds. It is called from CastRays().
        /// </summary>
        private void CastRay2D(Vector3 castOrigin, float sphereCastRadius, Vector3 rayDirection, float distance, LayerMask layerMask)
        {
            RaycastHit2D hit = Physics2D.CircleCast(castOrigin, sphereCastRadius, rayDirection, distance, layerMask);
            if (hit == true)
            {
                switch (info)
                {
                    case RayInfo.Distance:
                        Observations.Add(hit.distance);
                        break;
                    case RayInfo.Layer:
                        Observations.Add(hit.collider.gameObject.layer);
                        break;
                    case RayInfo.Angle:
                        Observations.Add(Vector3.Angle(hit.normal, rayDirection));
                        break;
                    case RayInfo.All:
                        Observations.Add(hit.distance);
                        Observations.Add(hit.collider.gameObject.layer);
                        Observations.Add(Vector3.Angle(hit.normal, rayDirection));
                        break;
                }
            }
            else
            {
                switch (info)
                {
                    case RayInfo.Distance:
                        Observations.Add(0);
                        break;
                    case RayInfo.Layer:
                        Observations.Add(0);
                        break;
                    case RayInfo.Angle:
                        Observations.Add(0);
                        break;
                    case RayInfo.All:
                        Observations.Add(0);
                        Observations.Add(0);
                        Observations.Add(0);
                        break;
                }
            }
        }        
    }
  
    public enum RayInfo
    {
        [Tooltip("1 float value per ray")]
        Distance,
        [Tooltip("1 float value per ray")]
        Layer,
        [Tooltip("1 float value per ray")]
        Angle,
        [Tooltip("3 float values per ray")]
        All,
    }

    [CustomEditor(typeof(RaySensor)), CanEditMultipleObjects]
    class ScriptlessRaySensor : Editor
    {
        private static readonly string[] _dontIncludeMe = new string[] { "m_Script" };
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            
            DrawPropertiesExcluding(serializedObject, _dontIncludeMe);

            serializedObject.ApplyModifiedProperties();    
        }
    }
    public class ReadOnlyAttribute : PropertyAttribute
    {

    }

    [CustomPropertyDrawer(typeof(ReadOnlyAttribute))]
    public class ReadOnlyDrawer : PropertyDrawer
    {
        public override float GetPropertyHeight(SerializedProperty property,
                                                GUIContent label)
        {
            return EditorGUI.GetPropertyHeight(property, label, true);
        }

        public override void OnGUI(Rect position,
                                   SerializedProperty property,
                                   GUIContent label)
        {
            GUI.enabled = false;
            EditorGUI.PropertyField(position, property, label, true);
            GUI.enabled = true;
        }
    }
    public static class EditorGUILayoutUtility
    {
        public static readonly Color DEFAULT_COLOR = new Color(0f, 0f, 0f, 0.3f);
        public static readonly Vector2 DEFAULT_LINE_MARGIN = new Vector2(2f, 2f);

        public const float DEFAULT_LINE_HEIGHT = 1f;

        public static void HorizontalLine(Color color, float height, Vector2 margin)
        {
            GUILayout.Space(margin.x);

            EditorGUI.DrawRect(EditorGUILayout.GetControlRect(false, height), color);

            GUILayout.Space(margin.y);
        }
        public static void HorizontalLine(Color color, float height) => EditorGUILayoutUtility.HorizontalLine(color, height, DEFAULT_LINE_MARGIN);
        public static void HorizontalLine(Color color, Vector2 margin) => EditorGUILayoutUtility.HorizontalLine(color, DEFAULT_LINE_HEIGHT, margin);
        public static void HorizontalLine(float height, Vector2 margin) => EditorGUILayoutUtility.HorizontalLine(DEFAULT_COLOR, height, margin);

        public static void HorizontalLine(Color color) => EditorGUILayoutUtility.HorizontalLine(color, DEFAULT_LINE_HEIGHT, DEFAULT_LINE_MARGIN);
        public static void HorizontalLine(float height) => EditorGUILayoutUtility.HorizontalLine(DEFAULT_COLOR, height, DEFAULT_LINE_MARGIN);
        public static void HorizontalLine(Vector2 margin) => EditorGUILayoutUtility.HorizontalLine(DEFAULT_COLOR, DEFAULT_LINE_HEIGHT, margin);

        public static void HorizontalLine() => EditorGUILayoutUtility.HorizontalLine(DEFAULT_COLOR, DEFAULT_LINE_HEIGHT, DEFAULT_LINE_MARGIN);
    }

}
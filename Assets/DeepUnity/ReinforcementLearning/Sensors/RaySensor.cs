using System;
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;

namespace DeepUnity
{
    /// <summary>
    /// ObservationVector.Length = <see cref="rays"/> * (2 +  <see cref="detectableTags"/>.Length)
    /// </summary>
    [AddComponentMenu("DeepUnity/RaySensor")]
    public class RaySensor : MonoBehaviour, ISensor
    {
        private readonly LinkedList<RayInfo> Observations = new LinkedList<RayInfo>();

        [SerializeField, Tooltip("@scene type")] World world = World.World3d;
        [SerializeField, Tooltip("@LayerMask used when casting the rays")] LayerMask layerMask = ~0;
        [SerializeField, Tooltip("@tags that can provide information")] string[] detectableTags;
        [SerializeField, Range(1, 50), Tooltip("@size of the buffer equals the number of rays")] int rays = 5;
        [SerializeField, Range(1, 360)] int fieldOfView = 45;
        [SerializeField, Range(0, 359)] int rotationOffset = 0;
        [SerializeField, Range(1f, 1000f), Tooltip("@maximum length of the rays")] float distance = 100;
        [SerializeField, Range(0.01f, 10)] float sphereCastRadius = 0.5f;

        [Space(10)]
        [SerializeField, Range(-5, 5), Tooltip("@ray X axis offset")] float xOffset = 0;
        [SerializeField, Range(-5, 5), Tooltip("@ray Y axis offset")] float yOffset = 0;
        [SerializeField, Range(-5, 5), Tooltip("@ray Z axis offset\n@not used in 2D world")] float zOffset = 0;
        [SerializeField, Range(-90, 90), Tooltip("@ray vertical tilt\n@not used in 2D world")] float tilt = 0;

        [Space(10)]
        [SerializeField] Color rayColor = Color.green;
        [SerializeField] Color missRayColor = Color.red;

        private void OnDrawGizmos()
        {
            float oneAngle = rays == 1 ? 0 : -fieldOfView / (rays - 1f);

            float begin = -oneAngle * (rays - 1f) / 2f + rotationOffset;
            Vector3 startAngle;

            if (world == World.World3d)
            {
                Quaternion rotationToTheLeft = Quaternion.AngleAxis(begin, transform.up);
                Vector3 rotatedForward = rotationToTheLeft * transform.forward;
                Vector3 rotationAxis = Vector3.Cross(rotatedForward, transform.up).normalized;
                Quaternion secondaryRotation = Quaternion.AngleAxis(tilt, rotationAxis);
                startAngle = secondaryRotation * rotatedForward;
            }
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
                    
                    if (isHit)
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
                    if (hit2D)
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

        /// <summary>
        /// - The Distance is normalized in range [0, 1] (hit_distance / ray_max_distance).  If no hit happend, the value is -1. <br></br>
        /// - HitTagIndex is One Hot Encoded, where the first spot represents a non-detectable tag. <br></br>
        /// Example: <br></br>
        /// <b>[ray 1: <em>NormalizedDistance</em>, NonDetectableTag, DetectableTag[0], DetectableTag[1], ... DetectableTag[n-1], <br></br>
        ///  ray 2: <em>NormalizedDistance</em>, NonDetectableTag, DetectableTag[0], DetectableTag[1], ... DetectableTag[n-1], .....]</b>
        /// </summary>
        /// <returns> a float[] of <b>length = num_rays * (2 + num_detectable_tags)</b>.</returns>
        public float[] GetObservationsVector()
        {
            CastRays();
            int rayInfoDim = 2 + detectableTags.Length;
            float[] vector = new float[rays * rayInfoDim];
            int index = 0;
            foreach (var rayInfo in Observations)
            {
                vector[index++] = rayInfo.NormalizedDistance;

                // OneHotEncode
                if (rayInfo.HitTagIndex == -1)
                    vector[index++] = 1f;
                else
                    vector[index++] = 0f;


                for (int i = 0; i < detectableTags.Length; i++)
                    if (rayInfo.HitTagIndex == i)
                        vector[index++] = 1f;
                    else
                        vector[index++] = 0f;
            }
            return vector;
        }
        /// <summary>
        /// - The Distance is normalized in range [0, 1] (hit_distance / ray_max_distance).  If no hit happend, the value is -1. <br></br>
        /// - If DetectableTags.Length > 0, HitTagIndex is normalized in range [0, 1]. If no detectable tag hit, the value is -1. <br></br>
        /// Example: <br></br>
        /// if DetectableTags.Length > 0: <b>[ray 1: NormalizedDistance, NormalizedHitTagIndex, ray 2: NormalizedDistance, NormalizedHitTagIndex, ..... ]</b> <br></br>
        /// else: <b>[ray 1: NormalizedDistance, ray 2: NormalizedDistance, ..... ]</b>
        /// </summary>
        /// <returns>a float[] of <b>length = num_rays * 2</b> if DetectableTags.Length > 0 else <b>num_rays</b></returns>
        public float[] GetCompressedObservationsVector()
        {
            CastRays();

            if(detectableTags != null && detectableTags.Length > 0)
            {
                float[] vector = new float[rays * 2];
                int index = 0;
                foreach (var rayInfo in Observations)
                {
                    vector[index++] = rayInfo.NormalizedDistance;

                    if (rayInfo.HitTagIndex == -1)
                        vector[index++] = -1f;
                    else
                        vector[index++] = rayInfo.HitTagIndex / (float)detectableTags.Length;
                }
                return vector;
            }
            else
            {
                float[] vector = new float[rays];
                int index = 0;
                foreach (var rayInfo in Observations)
                {
                    vector[index++] = rayInfo.NormalizedDistance;
                }
                return vector;
            }
            
        }
        /// <summary>
        /// Returns information of all rays.
        /// </summary>
        /// <returns></returns>
        public RayInfo[] GetObservationRays()
        {
            CastRays();
            return Observations.ToArray();
        }
        
       

        /// <summary>
        /// This methods casts the necessary rays.
        /// </summary>
        private void CastRays()
        {
            Observations.Clear();
            float oneAngle = rays == 1 ? 0 : -fieldOfView / (rays - 1f);

            float begin = -oneAngle * (rays - 1f) / 2f + rotationOffset;
            Vector3 startAngle;

            if (world == World.World3d)
            {
                Quaternion rotationToTheLeft = Quaternion.AngleAxis(begin, transform.up);
                Vector3 rotatedForward = rotationToTheLeft * transform.forward;
                Vector3 rotationAxis = Vector3.Cross(rotatedForward, transform.up).normalized;
                Quaternion secondaryRotation = Quaternion.AngleAxis(tilt, rotationAxis);
                startAngle = secondaryRotation * rotatedForward;
            }
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

            // var str = "";
            // foreach (var item in Observations)
            // {
            //     str += item.ToString() + " ";
            // }
            // print(str);
        }
        /// <summary>
        /// This method casts only rays for 3D worlds. It is called from CastRays().
        /// </summary>
        private void CastRay3D(Vector3 castOrigin, float sphereCastRadius, Vector3 rayDirection, float distance, LayerMask layerMask)
        {
            RaycastHit hit;
            bool success = Physics.SphereCast(castOrigin, sphereCastRadius, rayDirection, out hit, distance, layerMask);
            
            RayInfo rayInfo = new RayInfo();
            rayInfo.NormalizedDistance = success ? hit.distance / distance : -1f;
            rayInfo.HitTagIndex = success && detectableTags != null ? Array.IndexOf(detectableTags, hit.collider.tag) : -1;
            Observations.AddLast(rayInfo);
        }
        /// <summary>
        /// This method casts only rays for 2D worlds. It is called from CastRays().
        /// </summary>
        private void CastRay2D(Vector3 castOrigin, float sphereCastRadius, Vector3 rayDirection, float distance, LayerMask layerMask)
        {
            RaycastHit2D hit = Physics2D.CircleCast(castOrigin, sphereCastRadius, rayDirection, distance, layerMask);

            RayInfo rayInfo = new RayInfo();
            rayInfo.NormalizedDistance = hit ? hit.distance / distance : -1f;
            rayInfo.HitTagIndex = hit && detectableTags != null ? Array.IndexOf(detectableTags, hit.collider.tag) : -1;
            Observations.AddLast(rayInfo);
        }   
    }



#if UNITY_EDITOR

    [CustomEditor(typeof(RaySensor)), CanEditMultipleObjects]
    class ScriptlessRaySensor : Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            List<string> _dontDrawMe = new List<string>() { "m_Script" };


            SerializedProperty sr = serializedObject.FindProperty("world");
            if (sr.enumValueIndex == (int)World.World2d)
            {
                _dontDrawMe.Add("tilt");
                _dontDrawMe.Add("zOffset");

            }


            SerializedProperty detectableTags = serializedObject.FindProperty("detectableTags");
            CheckTags(detectableTags);

            DrawPropertiesExcluding(serializedObject, _dontDrawMe.ToArray());

            SerializedProperty rays_num = serializedObject.FindProperty("rays");
            int vecSize = (2 + detectableTags.arraySize) * rays_num.intValue;
            int compVecSize = detectableTags.arraySize == 0 ? rays_num.intValue : 2 * rays_num.intValue;
            EditorGUILayout.HelpBox($"Observations Vector contains {vecSize} float values. Compressed Observations Vector contains {compVecSize} float values.", MessageType.Info);


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
            if(tagsToList.Count != set.Count)
                EditorGUILayout.HelpBox(
                   $"Detectable tags contains doubles!", MessageType.Warning);
        }
    }
#endif
}
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    [AddComponentMenu("DeepUnity/Grid Sensor")]
    public class GridSensor : MonoBehaviour, ISensor
    {
 	    private List<float> Observations = new List<float>();
        [SerializeField, Tooltip("@scene type")] World world = World.World2d;
        [SerializeField, Tooltip("@LayerMask used when casting the rays")] LayerMask layerMask = ~0;
        [SerializeField, Range(0.01f, 10f)] public float scale = 1f;
        [SerializeField, Range(1, 10f)] public int width = 10;
        [SerializeField, Range(1, 10f)] public int height = 10;
        [SerializeField, Range(1, 10f)] public int deep = 10;

        [Space(10)]
        [SerializeField, Range(-5, 5), Tooltip("@grid X axis offset")] float xOffset = 0;
        [SerializeField, Range(-5, 5), Tooltip("@grid Y axis offset")] float yOffset = 0;
        [SerializeField, Range(-5, 5), Tooltip("@grid Z axis offset\n@not used in 2D world")] float zOffset = 0;


        private Vector3[,,] gridOrigins;

        private void Start()
        {
            
        }
        private void Update()
        {
            gridOrigins = new Vector3[deep, height, width];
        }
        private void OnDrawGizmos()
        {
            gridOrigins = new Vector3[deep, height, width];

        }

        public IEnumerable GetObservations()
        {
            return Observations;
        }

        private void CastGrid()
        {

        }
        private void CastGrid3D(Vector3 origin, Vector3 halfExtents, Quaternion quaternion, LayerMask layerMask)
        {
            
        }
        private void CastGrid2D(Vector3 origin, Vector3 halfExtents, Quaternion quaternion, LayerMask layerMask)
        {
            
        }
    }


    public enum GridInfo
    {

    }
}


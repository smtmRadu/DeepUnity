using UnityEngine;

// To be implemented... (though not necessary cause the grid sensor is already perfect)
// Notes:
// It basically works, all we need to implement next is cast method and the GetObs methods that return the info..
namespace DeepUnity
{
    public class HexSensor : MonoBehaviour, ISensor
    {
        [SerializeField, Range(0.01f, 100f)] public float scale = 1f;   // Radius of each sphere.
        [SerializeField, Range(3, 40)] private int size = 5;        // Number of rows in the honeycomb.

        [SerializeField] Color missColor = Color.gray;
        [SerializeField] Color missingMaterialColor = new Color(1f, 0f, 0.95f);//pink

        private void OnDrawGizmos()
        {
            float yOffset = Mathf.Sqrt(3) * scale;
            float xOffset = 2 * scale;

            int centralRow = size / 2;

            for (int row = 0; row < size; row++)
            {
                int numSpheres = size - Mathf.Abs(centralRow - row);

                float startX = -(numSpheres - 1) * xOffset * 0.5f;

                for (int col = 0; col < numSpheres; col++)
                {
                    float x = startX + col * xOffset;
                    float y = (row - centralRow) * yOffset;

                    Vector3 position = new Vector3(x, 0, y);
                    Vector3 worldCenter = transform.position + transform.rotation * position;

                    Collider[] hits = Physics.OverlapSphere(worldCenter, scale);
                    if (hits.Length > 0)
                    {
                        Renderer rend;
                        hits[0].gameObject.TryGetComponent(out rend);
                        Gizmos.color = rend != null ? rend.sharedMaterial.color : missingMaterialColor;
                    }
                    else
                        Gizmos.color = missColor;

                    // Gizmos.color = new Color(Gizmos.color.r, Gizmos.color.g, Gizmos.color.b, 0.6f);
                    // Gizmos.DrawWireSphere(worldCenter, scale);
                    Gizmos.color = new Color(Gizmos.color.r, Gizmos.color.g, Gizmos.color.b, 0.2f);
                    Gizmos.DrawSphere(worldCenter, scale);
                }
            }
        }

        private void Cast()
        {
            // TODO
        }




        public float[] GetObservationsVector()
        {
            return null;
        }

        public float[] GetCompressedObservationsVector()
        {
            return null;
        }
    }

}



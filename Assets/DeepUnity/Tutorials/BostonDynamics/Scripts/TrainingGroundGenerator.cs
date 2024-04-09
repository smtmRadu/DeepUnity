using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity.Tutorials
{
    public class TrainingGroundGenerator : MonoBehaviour
    {
        
        public List<GameObject> tiles;
        [Space(20)]
        [Button("Generate")]
        [Range(1, 100)] public int height = 10;
        [Range(1, 100)] public int width = 10;
        [MinMax(0.1f, 10f)] public float scale = 1f;

        public void Generate()
        {
            if (tiles == null || tiles.Count == 0)
                return;

            var parent = new GameObject($"Training Ground [{height}x{width}]");
            parent.transform.position = Vector3.zero;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    var random_tile = Utils.Random.Sample(tiles);
                    var tile = Instantiate(random_tile, new Vector3(i * 2, 0, j * 2) * scale - new Vector3(height, 0, width) * scale, Quaternion.identity, parent.transform);
                    tile.name = $"Tile [{i+1}, {j+1}]";
                    tile.transform.localScale = Vector3.one * scale;
                    float rotation = Utils.Random.Range(0, 4) * 90f;
                    tile.transform.rotation *= Quaternion.Euler(0, rotation, 0);
                }
            }
        }

    }

}



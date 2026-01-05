using System.Collections;
using System.Collections.Generic;
using UnityEngine;



namespace DeepUnity.Tutorials.ChatDemo
{
    public class WalkabilityMap : MonoBehaviour
    {
        public SpriteRenderer mapRenderer;

        public bool IsWalkable(Vector2 worldPos)
        {
            Sprite sprite = mapRenderer.sprite;
            Texture2D tex = sprite.texture;

            // Convert world position to local sprite space
            Vector2 localPos = mapRenderer.transform.InverseTransformPoint(worldPos);

            Rect rect = sprite.rect;
            Vector2 pivot = sprite.pivot;

            // Convert local to pixel coordinates
            float pixelsPerUnit = sprite.pixelsPerUnit;
            int x = Mathf.RoundToInt(pivot.x + localPos.x * pixelsPerUnit);
            int y = Mathf.RoundToInt(pivot.y + localPos.y * pixelsPerUnit);

            // Bounds check
            if (x < 0 || y < 0 || x >= rect.width || y >= rect.height)
                return false;

            Color pixel = tex.GetPixel(
                (int)(rect.x + x),
                (int)(rect.y + y)
            );

            return pixel.a > 0.1f; // transparent = not walkable
        }
    }
}
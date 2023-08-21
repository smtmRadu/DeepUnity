using System.Collections;
using UnityEngine;

namespace DeepUnity
{
    public interface ISensor
    {
        /// <summary>
        /// Returns all sensor's observation in array format.
        /// </summary>
        /// <returns></returns>
        public float[] GetObservationsVector();
        /// <summary>
        /// Returns all sensor's important observations in array format.
        /// </summary>
        /// <returns></returns>
        public float[] GetCompressedObservationsVector();
    }
    public enum World
    {
        World3d,
        World2d,
    }
    public enum CaptureType
    {
        RGB,
        Grayscale,
    }
    public enum CompressionType
    {
        PNG,
        JPG,
        EXG,
        TGA
    }
    /// <summary>
    /// The embedded float[] size of a RayInfo is 2 + num_detectable_tags.
    /// </summary>
    public struct RayInfo
    {
        /// <summary>
        /// Normalized distance to the hit object (in range [0, 1] by hit_distance / ray_max_distance). If no hit happend, the value is -1.
        /// </summary>
        public float NormalizedDistance { get; set; }    
        /// <summary>
        /// The index of the hit object's tag in the DetectableTags list, or -1 if there was no hit, or the hit object has a different tag.
        /// </summary>
        public int HitTagIndex { get; set; }

        public string ToString()
        {
            return $"[HitFraction={NormalizedDistance}, HitTagIndex={HitTagIndex}]";
        }

    }
    /// <summary>
    /// The emebedded value of a GridCellInfo is 2 * num_detectable_tags.
    /// </summary>
    public struct GridCellInfo
    {
        /// <summary>
        /// Whether or not the grid cell overlapped an object.
        /// </summary>
        public bool HasOverlappedObject { get; set; }
        /// <summary>
        /// The index of the overlapped object's tag in the DetectableTags list, or -1 if there is no overlap, or the overlapped object has a different tag.
        /// </summary>
        public int OverlappedObjectTagIndex { get; set; }

        public string ToString()
        {
            return $"[HasOverlappedObject={HasOverlappedObject}, OverlapTagIndex={OverlappedObjectTagIndex}]";
        }
    }
}


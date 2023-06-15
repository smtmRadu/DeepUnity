using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    public static class Settings
    {
        public static Device Device = Device.CPU;

        /// <summary>
        /// Reference to MatMulCS Compute Shader.
        /// </summary>
        internal readonly static ComputeShader MatMulCS;

        /// <summary>
        /// Reference to Conv2DCS Compute Shader.
        /// </summary>
        internal readonly static ComputeShader Conv2DCS;

        /// <summary>
        /// Threads displacement for Compute Shaders usage.
        /// </summary>
        internal readonly static int[] numthreads = new int[] { 10, 10, 8 };


        static Settings()
        {
            try
            {
                string csguid = AssetDatabase.FindAssets("MatMulCS")[0];
                string cspath = AssetDatabase.GUIDToAssetPath(csguid);
                MatMulCS = AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;

                csguid = AssetDatabase.FindAssets("Conv2DCS")[0];
                cspath = AssetDatabase.GUIDToAssetPath(csguid);
                Conv2DCS = AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;
            }
            catch { }
        }
 	   
    }
    public enum InitType
    {
        Default,
        HE,
        Xavier,
        Normal,
        Uniform,
    }
    public enum Device
    {
        CPU,
        GPU
    }
    public enum NormType
    {
        ManhattanL1,
        EuclideanL2,
        Frobenius
    }
    public enum PaddingType
    {
        Zeros,
        Mirror,
        // Replicate
        // Circular
    }
}


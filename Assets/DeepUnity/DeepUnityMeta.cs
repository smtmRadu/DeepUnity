using System;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    public static class DeepUnityMeta
    {
        public static Device Device = Device.CPU;

        /// <summary>
        /// Reference to MatMulCS Compute Shader.
        /// </summary>
        internal readonly static ComputeShader MatMulCS;

        /// <summary>
        /// Reference to Conv2DCS Compute Shader.
        /// </summary>
        internal readonly static ComputeShader Correlation2DCS;

        /// <summary>
        /// Threads displacement for Compute Shaders usage.
        /// </summary>
        internal readonly static int[] numthreads = new int[] { 8, 8, 8 };

        /// <summary>
        /// MaxDegreeOfParallelism of 8 for multi-threaded tasks.
        /// </summary>
        internal readonly static ParallelOptions threadLimit8 = new ParallelOptions { MaxDegreeOfParallelism = 8 };

        static DeepUnityMeta()
        {
            try
            {
                string csguid = AssetDatabase.FindAssets("MatMulCS")[0];
                string cspath = AssetDatabase.GUIDToAssetPath(csguid);
                MatMulCS = AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;

                csguid = AssetDatabase.FindAssets("Correlation2DCS")[0];
                cspath = AssetDatabase.GUIDToAssetPath(csguid);
                Correlation2DCS = AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;
            }
            catch { }
        }
 	   
    }
    public static class Timer
    {
        static DateTime start;
        static TimeSpan time;
        public static void Start()
        {
            start = DateTime.Now;
        }
        public static TimeSpan Stop()
        {
            time = DateTime.Now - start;
            Debug.Log("[Timer] : " +  time);
            return time;
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
    public enum TDim
    {
        batch,
        channel,
        height,
        width
    }
    public enum CorrelationMode
    {
        Valid,
        Full,
        Same
    }
}


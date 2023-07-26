using System;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    public static class DeepUnityMeta
    {
        internal readonly static ComputeShader TensorCS;
        internal readonly static ComputeShader DenseCS;
        internal readonly static ComputeShader Conv2DCS;
        internal readonly static int THREADS_NUM = 256;

        static DeepUnityMeta()
        {
            try
            {
                var csguid = AssetDatabase.FindAssets("TensorCS")[0];
                var cspath = AssetDatabase.GUIDToAssetPath(csguid);
                TensorCS = AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;

                csguid = AssetDatabase.FindAssets("DenseCS")[0];
                cspath = AssetDatabase.GUIDToAssetPath(csguid);
                DenseCS = AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;

                csguid = AssetDatabase.FindAssets("Conv2DCS")[0];
                cspath = AssetDatabase.GUIDToAssetPath(csguid);
                Conv2DCS = AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;
            }
            catch { }
        }
 	   
    }
    public static class TimerX
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
    public enum InitType
    {
        Default,
        HE,
        Xavier,
        Normal,
        Uniform,
        [Tooltip("[DO NOT USE] A hardcoded value set for the parameters. Used on development.")]
        Debug,
    }
    public enum Device
    {
        CPU,
        GPU
    }
    public enum NormType
    {
        NonZeroL0,
        ManhattanL1,
        EuclideanL2,
        MaxLInf
    }
    public enum PaddingType
    {
        Zeros,
        Mirror,
        // Replicate
        // Circular
    }
    public enum Dim
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
    /// <summary>
    /// Nonlinearity used for RNNCells.
    /// </summary>
    public enum NonLinearity
    {
        Tanh,
        ReLU
    }

    public enum DatasetSettings
    {
        LoadAll,
        LoadTrainOnly,
        LoadTestOnly
    }
}


using System;
using System.Diagnostics;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Notes and bugs:
    /// 1. Do not rename any Sequencial or RNN asset (even repair doesn t work for now).
    /// 2. Do not use TensorGPU (it was experimental).
    /// 3. Do not use RL because is in development.
    /// 4. CamSensor requires a different architecture containing Conv2D modules in order for max efficiency.
    /// 
    /// </summary>
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
    public static class ClockTimer
    {
        static Stopwatch clock;
        public static void Start()
        {
            clock = Stopwatch.StartNew();
        }
        public static TimeSpan Stop()
        {
            clock.Stop();
            UnityEngine.Debug.Log("[Timer] : " +  clock.Elapsed);
            return clock.Elapsed;
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
        [Tooltip("N(0, s) where s = sqrt(2 / fan_in). Works well with ReLU activation function.")]
        HE_Normal,
        [Tooltip("U(-k, k) where k = sqrt(6 / fan_in). Works well with ReLU activation function.")]
        HE_Uniform, 
        
        [Tooltip("N(0, s) where s = sqrt(2 / (fan_in + fan_out)). Works well with Sigmoid activation function.")]
        Glorot_Normal,
        [Tooltip("U(-k, k) where k = sqrt(6 / (fan_in + fan_out)). Works well with Sigmoid activation function.")]
        Glorot_Uniform,

        [Tooltip("N(0, s) where s = sqrt(1 / (fan_in + fan_out)). Works well with Tanh/LReLU activation function.")]
        LeCun_Normal,
        [Tooltip("U(-k, k) where k = sqrt(3 / (fan_in + fan_out)). Works well with Tanh/LReLU activation function.")]
        LeCun_Uniform,

        [Tooltip("N(0, 1).")]
        Random_Normal,
        [Tooltip("U(-1, 1).")]
        Random_Uniform,

        [Tooltip("0")]
        Zeros,
        [Tooltip("1")]
        Ones


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

    public enum DecisionRequestType
    {
        [Tooltip("The agent performs an action on each FixedUpdate() call.")]
        OnceEachFrame,
        [Tooltip("The agent performs an action once every X seconds denoted by the field below.")]
        OnPeriodInterval,
        [Tooltip("The agent performs an action whenever RequestAction() method is called.")]
        WhenRequested
    }
    public enum BehaviourType
    {
        [Tooltip("Latent behaviour. Learning: NO. Scene resets: NO.")]
        Inactive,
        [Tooltip("Active behaviour. Learning: NO. Scene resets: YES.")]
        Active,
        [Tooltip("Exploring behaviour. Learning: YES. Scene resets: YES.")]
        Learn,
        [Tooltip("Manual control. Learning: NO. Scene resets: YES.")]
        Manual,
    }

    public enum OnEpisodeEndType
    {
        [Tooltip("When the episode ends, only the agent is repositioned to the initial state.")]
        ResetAgent,
        [Tooltip("When the episode ends, the agent and it's parent environment are repositioned to the initial state.")]
        ResetEnvironment
    }
    public enum ActionType
    {
        Continuous,
        Discrete
    }

    // public enum NormalizationType
    // {
    //     None,
    //     Scale,
    //     ZScore,
    //     LogScale
    // }

}


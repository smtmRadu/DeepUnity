using System;
using System.Diagnostics;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Notes:
    /// 1. Do not use TensorGPU (it was experimental).
    /// 2. CamSensor requires a different architecture containing Conv2D modules in order for max efficiency.
    /// 
    /// </summary>
    public static class DeepUnityMeta
    {
        internal readonly static ComputeShader TensorCS;
        internal readonly static ComputeShader DenseCS;
        internal readonly static ComputeShader Conv2DCS;
        internal readonly static ComputeShader LinearCS;
        internal readonly static int THREADS_NUM = 256;
        internal readonly static ParallelOptions MULTITHREADS_8 = new ParallelOptions() { MaxDegreeOfParallelism = 8 };
        internal readonly static ParallelOptions MULTITHREADS_4 = new ParallelOptions() { MaxDegreeOfParallelism = 4 };

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
            catch 
            {
                ConsoleMessage.Warning("Compute Shader files where not found! Make sure DeepUnity framework files were not modified or deleted");          
            }
        }
 	   
    }
    public static class TimeKeeper
    {
        static Stopwatch clock;
        public static void Start()
        {
            clock = Stopwatch.StartNew();
        }
        public static TimeSpan Stop()
        {
            clock.Stop();
            ConsoleMessage.Info("[Timer] : " + clock.Elapsed);
            return clock.Elapsed;
        }
    }
   
    public enum InitType
    {
        [Tooltip("[Kaiming He] N(0, s) where s = sqrt(2 / fan_in). Works well with ReLU / LeakyReLU activation function.")]
        HE_Normal,
        [Tooltip("[Kaiming He] U(-k, k) where k = sqrt(6 / fan_in). Works well with ReLU / LeakyReLU  activation function.")]
        HE_Uniform, 
        
        [Tooltip("[Xavier Glorot] N(0, s) where s = sqrt(2 / (fan_in + fan_out)). Works well with Tanh / Sigmoid activation function.")]
        Glorot_Normal,
        [Tooltip("[Xavier Gloro] U(-k, k) where k = sqrt(6 / (fan_in + fan_out)). Works well with Tanh / Sigmoid activation function.")]
        Glorot_Uniform,

        [Tooltip("[Yann LeCun] N(0, s) where s = sqrt(1 / (fan_in + fan_out)). Works well for activation differentiable in z = 0. (Tanh / Sigmoid)")]
        LeCun_Normal,
        [Tooltip("[Yann LeCun] U(-k, k) where k = sqrt(3 / (fan_in + fan_out)).  Works well for activation differentiable in z = 0. (Tanh / Sigmoid)")]
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

    public enum DatasetSettings
    {
        LoadAll,
        LoadTrainOnly,
        LoadTestOnly
    }
    public enum BehaviourType
    {
        [Tooltip("Latent behaviour. Learning: NO. Scene resets: NO.")]
        Off,
        [Tooltip("Active behaviour. Learning: NO. Scene resets: YES.")]
        Inference,
        [Tooltip("Exploring behaviour. Learning: YES. Scene resets: YES.")]
        Learn,
        [Tooltip("Manual control. Learning: NO. Scene resets: YES.")]
        Manual,
    }

    public enum OnEpisodeEndType
    {
        [Tooltip("When the episode ends, OnEpisodeBegin() method is called.")]
        Nothing,
        [Tooltip("When the episode ends, agent's transforms and rigidbodies are reinitialized. OnEpisodeBegin() is called afterwards.")]
        ResetAgent,
        [Tooltip("When the episode ends, environment's transforms and rigidbodies (including the agent) are reinitialized. OnEpisodeBegin() is called afterwards.")]
        ResetEnvironment
    }

    public enum StandardDeviationType
    {
        Fixed,
        Trainable
    }

    public enum AverageType
    {
        Weighted,
        Micro,
    }

    public enum ModelType
    {
        NN,
        CNN,
        RNN
    }

    public enum KLType
    {
        [Tooltip("No calculation of Kullback-Leibler Divergence")]
        Off,
        [Tooltip("Use of early stopping.")]
        KLE_Stop,
        // [Tooltip("If KLD > KL_target, the policy is rollbacked to the old state.")]
        // KLE_Rollback
    }

    public enum UseSensorsType
    {
        [Tooltip("Does not collect automatically the observation values from attached sensors. All attached sensors observation vectors can be added manually inside CollectObservations() method.")]
        Off,
        [Tooltip("Automatically collects ObservationsVector from attached sensors.")]
        ObservationsVector,
        [Tooltip("Automatically collects CompressedObservationsVector from attached sensors.")]
        CompressedObservationsVector
    }

    public enum TrainerType
    {
        [Tooltip("Proximal Policy Optimization")]
        PPO,
        [Tooltip("Soft Actor-Critic")]
        SAC,
        [Tooltip("Generative Adversial Imitation Learning")]
        GAIL
    }

    public enum TimescaleAdjustmentType
    {
        [Tooltip("Dynamic adjustment of timescale during training to get the maximum efficiency.")]
        Dynamic,
        [Tooltip("Manual adjustment of timescale during training.")]
        Static
    }
}


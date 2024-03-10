using System;
using System.Diagnostics;
using System.Threading.Tasks;
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
        internal readonly static ComputeShader OptimizerCS;
        internal readonly static ComputeShader RNNCellCS;

        internal readonly static int THREADS_NUM = 256;
        internal readonly static ParallelOptions MULTITHREADS_8 = new ParallelOptions() { MaxDegreeOfParallelism = 8 };
        internal readonly static ParallelOptions MULTITHREADS_4 = new ParallelOptions() { MaxDegreeOfParallelism = 4 };

        static DeepUnityMeta()
        {
            try
            {
                // I have no clue how can i make them loadable using Resouces.Load without moving them into a folder called Resources

                var csguid = UnityEditor.AssetDatabase.FindAssets("TensorCS")[0];
                var cspath = UnityEditor.AssetDatabase.GUIDToAssetPath(csguid);
                TensorCS = UnityEditor.AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;

                csguid = UnityEditor.AssetDatabase.FindAssets("DenseCS")[0];
                cspath = UnityEditor.AssetDatabase.GUIDToAssetPath(csguid);
                DenseCS = UnityEditor.AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;
  
                csguid = UnityEditor.AssetDatabase.FindAssets("Conv2DCS")[0]; 
                cspath = UnityEditor.AssetDatabase.GUIDToAssetPath(csguid);
                Conv2DCS = UnityEditor.AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;

                csguid = UnityEditor.AssetDatabase.FindAssets("OptimizerCS")[0];
                cspath = UnityEditor.AssetDatabase.GUIDToAssetPath(csguid);
                OptimizerCS = UnityEditor.AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;

                csguid = UnityEditor.AssetDatabase.FindAssets("RNNCellCS")[0];
                cspath = UnityEditor.AssetDatabase.GUIDToAssetPath(csguid);
                RNNCellCS = UnityEditor.AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;
            }
            catch 
            {
                ConsoleMessage.Warning("Compute Shader files where not found! Make sure DeepUnity framework files were not modified or deleted");          
            }
        }
 	   
    }
    public static class BenchmarkClock
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
        [Tooltip("[Kaiming He] N(0, s) where s = sqrt(2 / fan_in). Works well with ReLU / LeakyReLU activation functions.")]
        HE_Normal,
        [Tooltip("[Kaiming He] U(-k, k) where k = sqrt(6 / fan_in). Works well with ReLU / LeakyReLU  activation functions.")]
        HE_Uniform, 
        
        [Tooltip("[Xavier Glorot] N(0, s) where s = sqrt(2 / (fan_in + fan_out)). Works well with Tanh / Sigmoid activation functions.")]
        Glorot_Normal,
        [Tooltip("[Xavier Gloro] U(-k, k) where k = sqrt(6 / (fan_in + fan_out)). Works well with Tanh / Sigmoid activation functions.")]
        Glorot_Uniform,

        [Tooltip("[Yann LeCun] N(0, s) where s = sqrt(1 / fan_in). Works well for activation differentiable in z = 0. (Tanh / Sigmoid)")]
        LeCun_Normal,
        [Tooltip("[Yann LeCun] U(-k, k) where k = sqrt(3 / fan_in).  Works well for activation differentiable in z = 0. (Tanh / Sigmoid)")]
        LeCun_Uniform,

        [Tooltip("N(0, 1).")]
        Normal,
        [Tooltip("N(0, 0.1).")]
        Normal0_1,
        [Tooltip("N(0, 0.01).")]
        Normal0_01,
        [Tooltip("N(0, 0.001).")]
        Normal0_001,

        [Tooltip("U(-1, 1).")]
        Uniform,
        [Tooltip("U(-0.1, 0.1).")]
        Uniform0_1,
        [Tooltip("U(-0.01, 0.01).")]
        Uniform0_01,
        [Tooltip("U(-0.001, 0.001).")]
        Uniform0_001,


        [Tooltip("0")]
        Zeros,
        [Tooltip("1")]
        Ones,

        [Tooltip("Orthogonal Initialization using Gram-Schmidt Decomposition (used only for 2 dimensional matrices - Dense). Shows very good results in finding a good local minimum.")]
        Orthogonal,
      
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
    }

    public enum TimescaleAdjustmentType
    {
        [Tooltip("Dynamic adjustment of timescale during training to get the maximum efficiency.")]
        Dynamic,
        [Tooltip("Manual adjustment of timescale during training.")]
        Static
    }
    public enum NonLinearity
    {
        Tanh,
        ReLU
    }
    public enum HiddenStates
    {
        [Tooltip("Returns all hidden states in the sequence h(1), h(2), ... h(L)")]
        ReturnAll,
        [Tooltip("Returns only the last hidden state in the sequence h(L)")]
        ReturnLast
    }

    public enum ArchitectureType
    {
        MLP,
        CNN,
        RNN
    }
}


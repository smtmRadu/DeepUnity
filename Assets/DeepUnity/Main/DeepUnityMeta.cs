using System;
using System.Diagnostics;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    public static class DeepUnityMeta
    {
        internal readonly static ComputeShader TensorCS;
        internal readonly static ComputeShader DenseCS;
        internal readonly static ComputeShader Conv2DCS;
        internal readonly static ComputeShader RNNCellCS;
        internal readonly static ComputeShader ConvTranpose2DCS;
        // internal readonly static ComputeShader MatMulFP16CS;

        internal readonly static int THREADS_NUM = 256;
        internal readonly static Lazy<ParallelOptions> MULTITHREADS_8 = new Lazy<ParallelOptions>(() => new ParallelOptions { MaxDegreeOfParallelism = 8 });
        internal readonly static Lazy<ParallelOptions> MULTITHREADS_4 = new Lazy<ParallelOptions>(() => new ParallelOptions { MaxDegreeOfParallelism = 4 });
        static DeepUnityMeta()
        {
            try
            {
                // I have no clue how can i make them loadable using Resouces.Load without moving them into a folder called Resources

                TensorCS = Resources.Load<ComputeShader>("ComputeShaders/TensorCS");
                DenseCS = Resources.Load<ComputeShader>("ComputeShaders/DenseCS");
                Conv2DCS = Resources.Load<ComputeShader>("ComputeShaders/Conv2DCS");
                RNNCellCS = Resources.Load<ComputeShader>("ComputeShaders/RNNCellCS");
                ConvTranpose2DCS = Resources.Load<ComputeShader>("ComputeShaders/ConvTranpose2DCS");
                // MatMulFP16CS = Resources.Load<ComputeShader>("ComputeShaders/MatMulFP16CS");

                if (TensorCS == null)
                    throw new Exception("The Compute Shader scripts were moved from Resources/ComputeShaders folder. Please move them back or modify this script that finds them by adjusting the path.");


                // var csguid = UnityEditor.AssetDatabase.FindAssets("TensorCS")[0];
                // var cspath = UnityEditor.AssetDatabase.GUIDToAssetPath(csguid);
                // TensorCS = UnityEditor.AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;
                // 
                // csguid = UnityEditor.AssetDatabase.FindAssets("DenseCS")[0];
                // cspath = UnityEditor.AssetDatabase.GUIDToAssetPath(csguid);
                // DenseCS = UnityEditor.AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;
                // 
                // csguid = UnityEditor.AssetDatabase.FindAssets("Conv2DCS")[0]; 
                // cspath = UnityEditor.AssetDatabase.GUIDToAssetPath(csguid);
                // Conv2DCS = UnityEditor.AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;
                // 
                // csguid = UnityEditor.AssetDatabase.FindAssets("RNNCellCS")[0];
                // cspath = UnityEditor.AssetDatabase.GUIDToAssetPath(csguid);
                // RNNCellCS = UnityEditor.AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;
                // 
                // csguid = UnityEditor.AssetDatabase.FindAssets("ConvTranspose2DCS")[0];
                // cspath = UnityEditor.AssetDatabase.GUIDToAssetPath(csguid);
                // ConvTranpose2DCS = UnityEditor.AssetDatabase.LoadAssetAtPath(cspath, typeof(ComputeShader)) as ComputeShader;
            }
            catch 
            {
                ConsoleMessage.Error("Compute Shader files where not found! Make sure DeepUnity framework files were not modified or deleted.");          
            }
        }
 	   
    }
    public static class Benckmark
    {
        static Stopwatch _clock;
        public static void Start()
        {
            _clock = Stopwatch.StartNew();
        }
        public static TimeSpan Stop()
        {
            _clock.Stop();
            ConsoleMessage.Info("[TIMER] : " + _clock.Elapsed);
            return _clock.Elapsed;
        }
    }
   
    public enum InitType
    {
        [Tooltip("[Kaiming He] N(0, s) where s = sqrt(2 / fan_in). Works well with ReLU / LeakyReLU activation functions.")]
        Kaiming_Uniform,
        [Tooltip("[Kaiming He] U(-k, k) where k = sqrt(6 / fan_in). Works well with ReLU / LeakyReLU  activation functions.")]
        Kaiming_Normal, 
        
        [Tooltip("[Xavier Glorot] N(0, s) where s = sqrt(2 / (fan_in + fan_out)). Works well with Tanh / Sigmoid activation functions.")]
        Xavier_Normal,
        [Tooltip("[Xavier Glorot] U(-k, k) where k = sqrt(6 / (fan_in + fan_out)). Works well with Tanh / Sigmoid activation functions.")]
        Xavier_Uniform,

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

    public enum Stochasticity
    {
        [Tooltip("Used for stochastic policies (PPO)")]
        FixedStandardDeviation,
        [Tooltip("Used for stochastic policies (PPO, SAC)")]
        TrainebleStandardDeviation,
        [Tooltip("Zero-mean Gaussian noise is added over the actions. Used for deterministic policies (TD3, DDPG)")]
        ActiveNoise,
        [Tooltip("Actions are drawn from an uniform distribution. Used at the beginning in off-policy algorithms (SAC, DDPG, TD3)")]
        Random
    }

    public enum AverageType
    {
        Weighted,
        Micro,
    }

    public enum KLType
    {
        [Tooltip("No calculation of Kullback-Leibler Divergence")]
        Off,
        [Tooltip("Use of early stopping.")]
        KLE_Stop,
        // [Tooltip("If KLD > KL_target, the policy is rollbacked to the old state.")] // disabled because it doesn t worth to cache the old state of the network.. it consumes resources.
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
        [Tooltip("Twin Delayed Deep Deterministic Policy Gradient")]
        TD3,
        [Tooltip("Deep Deterministic Policy Gradient")]
        DDPG
    }

    public enum TimescaleAdjustmentType
    {
        [Tooltip("The timescale remains constant, but can be modified during training.")]
        Constant,
        [Tooltip("Dynamic adjustment of timescale during training (it can be used for fast sketch ups when you are too lazy to see what constant value matches your pc performance).")]
        Dynamic      
    }
    public enum NonLinearity
    {
        Tanh,
        Relu
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
        [Tooltip("Multilayer Perceptron")]
        MLP,
        [Tooltip("Multilayer Perceptron with Normalization Layers")]
        LnMLP,
        [Tooltip("Convolutional Neural Network")]
        CNN,
        [Tooltip("Recurrent Neural Network")]
        RNN,
        [Tooltip("Self-Attention Neural Network")]
        ATT
    }

    public enum FloatingPointPrecision
    {
        FP32,
        FP16
    }
}


using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    public static class DeepUnityMeta
    {
        // Compute shaders are loaded LAZILY, each on first access. The old eager static ctor loaded
        // all 13 shaders the first time anything touched DeepUnityMeta, so e.g. constructing an LLM
        // paid ~100 ms of Resources.Load for shaders it never uses — a visible frame hitch.
        internal static ComputeShader TensorCS          => Get(ref _tensorCS, "TensorCS");
        internal static ComputeShader DenseCS           => Get(ref _denseCS, "DenseCS");
        internal static ComputeShader Conv2DCS          => Get(ref _conv2DCS, "Conv2DCS");
        internal static ComputeShader RNNCellCS         => Get(ref _rnnCellCS, "RNNCellCS");
        internal static ComputeShader ConvTranpose2DCS  => Get(ref _convTranspose2DCS, "ConvTranspose2DCS");

        internal static ComputeShader GQAInferenceCS    => Get(ref _gqaInferenceCS, "GQAInferenceCS");
        internal static ComputeShader GLUInferenceCS    => Get(ref _gluInferenceCS, "GLUInferenceCS");
        internal static ComputeShader FFNInferenceCS    => Get(ref _ffnInferenceCS, "FFNInferenceCS");
        internal static ComputeShader LmHeadInferenceCS => Get(ref _lmHeadInferenceCS, "LmHeadInferenceCS");
        internal static ComputeShader Gemma3FP32CS      => Get(ref _gemma3FP32CS, "Gemma3FP32CS");
        internal static ComputeShader Gemma3CS          => Get(ref _gemma3CS, "Gemma3CS");
        internal static ComputeShader Gemma3OriginalCS  => Get(ref _gemma3OriginalCS, "Gemma3OriginalCS");
        internal static ComputeShader Qwen3_5CS         => Get(ref _qwen3_5CS, "Qwen3_5CS");

        static ComputeShader _tensorCS, _denseCS, _conv2DCS, _rnnCellCS, _convTranspose2DCS;
        static ComputeShader _gqaInferenceCS, _gluInferenceCS, _ffnInferenceCS, _lmHeadInferenceCS;
        static ComputeShader _gemma3FP32CS, _gemma3CS, _gemma3OriginalCS, _qwen3_5CS;

        static ComputeShader Get(ref ComputeShader field, string name)
        {
            if (field == null)
            {
                field = Resources.Load<ComputeShader>("ComputeShaders/" + name);
                if (field == null)
                    ConsoleMessage.Error($"Compute Shader '{name}' was not found in Resources/ComputeShaders. " +
                                         "Make sure DeepUnity framework files were not modified or deleted.");
            }
            return field;
        }

        internal readonly static int THREADS_NUM = 256;
        internal readonly static Lazy<ParallelOptions> MULTITHREADS_8 = new Lazy<ParallelOptions>(() => new ParallelOptions { MaxDegreeOfParallelism = 8 });
        internal readonly static Lazy<ParallelOptions> MULTITHREADS_4 = new Lazy<ParallelOptions>(() => new ParallelOptions { MaxDegreeOfParallelism = 4 });
    }


    public static class Benckmark
    {
        static Stopwatch _clock;
        public static void Start()
        {
            _clock = Stopwatch.StartNew();
        }
        public static TimeSpan Stop(string tag = "TIMER")
        {
            _clock.Stop();
            ConsoleMessage.Info($"[{tag}] : " + _clock.Elapsed);
            return _clock.Elapsed;
        }
    }
   
    public enum InitType
    {
        [Tooltip("N(0, s) where s = sqrt(2 / fan_in). Works well with ReLU / LeakyReLU activation functions.")]
        Kaiming_Uniform,
        [Tooltip("U(-k, k) where k = sqrt(6 / fan_in). Works well with ReLU / LeakyReLU  activation functions.")]
        Kaiming_Normal, 
        
        [Tooltip("N(0, s) where s = sqrt(2 / (fan_in + fan_out)). Works well with Tanh / Sigmoid activation functions.")]
        Xavier_Normal,
        [Tooltip("U(-k, k) where k = sqrt(6 / (fan_in + fan_out)). Works well with Tanh / Sigmoid activation functions.")]
        Xavier_Uniform,

        [Tooltip("N(0, s) where s = sqrt(1 / fan_in). Works well for activation differentiable in z = 0. (Tanh / Sigmoid)")]
        LeCun_Normal,
        [Tooltip("U(-k, k) where k = sqrt(3 / fan_in).  Works well for activation differentiable in z = 0. (Tanh / Sigmoid)")]
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
        NothingHappens,
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
        TrainableStandardDeviation,
        [Tooltip("Zero-mean Gaussian noise is added over the actions. Used for deterministic policies (TD3, DDPG)")]
        ActiveNoise,
        [Tooltip("Ornstein-Uhlenbeck noise is added over the actions. Used in DPPG.")]
        OUNoise,
        [Tooltip("Actions are drawn from an uniform distribution. Used at the beginning in off-policy algorithms (SAC, DDPG, TD3)")]
        Random
    }

    public enum AverageType
    {
        Weighted,
        Micro,
    }

    public enum EarlyStopType
    {
        [Tooltip("No Early Stopping")]
        Off,
        [Tooltip("Early Stopping")]
        Stop,
        [Tooltip("Early Stopping + Rollback")]
        Rollback
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
        // Explicit values: enums serialize as ints, so existing Config.asset files must keep
        // pointing at the SAME implementation across the 2026-06 rename (the full-GPU trainers
        // took the standard PPO/SAC names; the old CPU implementations became *Depr and the
        // legacy trainers are locked in the inspector).
        [Tooltip("DEPRECATED CPU Proximal Policy Optimization — locked; use PPO (full-GPU).")]
        PPODepr = 0,
        [Tooltip("DEPRECATED CPU Soft Actor-Critic — locked; use SAC (full-GPU).")]
        SACDepr = 1,
        [Tooltip("Twin Delayed Deep Deterministic Policy Gradient — legacy, locked.")]
        TD3 = 2,
        [Tooltip("Deep Deterministic Policy Gradient — legacy, locked.")]
        DDPG = 3,
        [Tooltip("Vanilla Policy Gradient — legacy, locked.")]
        VPG = 4,
        [Tooltip("Proximal Policy Optimization (standard, full-GPU). Forward, backward and AdamW step run entirely on GPU. MLP / LnMLP networks only.")]
        PPO = 5,
        [Tooltip("Soft Actor-Critic (standard, full-GPU). Forward, backward and AdamW step run entirely on GPU. MLP / LnMLP networks only.")]
        SAC = 6,
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
        Relu, // Fast end efficient
        Tanh, // Slower but stable
        Silu, // Most expressive
        Gelu  // Transformer-style smooth gating
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

    /// <summary>
    /// Replay strategy <b>S</b> used in HER (Hindsight Experience Replay).
    /// </summary>
    public enum ReplayStrategy
    {
        Final,
        Future,
        Episode,
        Random
    }
}


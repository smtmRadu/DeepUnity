using DeepUnity.Activations;
using DeepUnity.Models;
using DeepUnity.Modules;
using DeepUnity.Optimizers;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEditor;
using UnityEngine;


namespace DeepUnity.ReinforcementLearning
{
    [Serializable]
    public sealed class AgentBehaviour : ScriptableObject
    {
        [Header("Behaviour Properties")]
        [SerializeField, ViewOnly, Tooltip("A counter of the number of saves, to keep track which version is the most trained.")] private int version = 1;
        [SerializeField, ViewOnly] public string behaviourName;
        [SerializeField, HideInInspector] private bool assetCreated = false;
        [SerializeField, ViewOnly] public int observationSize;
        [SerializeField, ViewOnly, Min(1)] public int stackedInputs;
        [Tooltip("Number of continuous actions this model uses.")]
        [SerializeField, ViewOnly] public int continuousDim;
        [Tooltip("Number of discrete actions this model uses.")]
        [SerializeField, ViewOnly] public int discreteDim;

        [Header("Hyperparameters")]
        [Tooltip("The scriptable object file containing the training hyperparameters.")]
        [SerializeField] public Hyperparameters config;

        [Header("Critic")]
        [SerializeField] public Sequential vNetwork;
        [SerializeField] public Sequential q1Network;
        [SerializeField] public Sequential q2Network;

        [Space]

        [Header("Policy")]
        [SerializeField] public Sequential muNetwork;
        [SerializeField] public Sequential sigmaNetwork;
        [SerializeField] public Sequential discreteNetwork;
        [Space]


        [Header("Behaviour Configurations")]
        
        [SerializeField, Tooltip("Network forward progapation is runned on this device when the agents interfere with the environment.")]
        public Device inferenceDevice = Device.CPU;

        [SerializeField, Tooltip("Network computation is runned on this device when training on batches. It is highly recommended to be set on GPU if it is available.")]
        public Device trainingDevice = Device.GPU;

        [SerializeField, Tooltip("The frames per second runned by the physics engine. [Time.fixedDeltaTime = 1 / targetFPS]")]
        [Range(30, 100)]
        public int targetFPS = 50;

        [Range(1f, 10f), SerializeField, Tooltip("Observations are clipped [after normarlization] in range [-clip, clip]. \n Rewards (per timestep) are clipped in range [-clip, clip]. \nNote that using a low clipping (c > 5) may induce instability on large number of inputs.")]
        public float clipping = 5f;

        [SerializeField, Tooltip("Auto-normalize input observations and rewards for a stable training.")]
        public bool normalize = false;

        [ViewOnly, SerializeField, Tooltip("Observations normalizer.")]
        public RunningNormalizer observationsNormalizer;    

        [HideInInspector, SerializeField, ToolboxItem("Rewards normalizer")]
        public RewardsNormalizer rewardsNormalizer;

        [Header("Exploration in Continuous Action space")]
        [SerializeField, Tooltip("The Continuous Actions form of exploration.")]
        public Stochasticity stochasticity = Stochasticity.FixedStandardDeviation;
        [Tooltip("Modify this value to change the exploration/exploitation ratio.")]
        [SerializeField, Min(0.001f)]
        public float standardDeviationValue = 1f;

        [Tooltip("Modify this value to change the exploration/exploitation ratio. The standard deviation obtained by softplus(std_dev) * standardDeviationScale. 1.5scale  ~ 1fixed, 3scale  ~ 1.5fixed, 4.5scale ~ 2fixed")]
        [SerializeField, Min(0.1f)]
        public float standardDeviationScale = 1.5f;

        [SerializeField, Min(0f)]
        public float noiseValue = 0.1f;



        public bool IsUsingContinuousActions { get => continuousDim > 0; }
        public bool IsUsingDiscreteActions { get => discreteDim > 0; }

        const string VALUE_NET_NAMING_CONVENTION = "V";
        const string Q1_NET_NAMING_CONVENTION = "Q1";
        const string Q2_NET_NAMING_CONVENTION = "Q2";
        const string MU_NET_NAMING_CONVENTION = "Mu";
        const string SIGMA_NET_NAMING_CONVENTION = "Sigma";
        const string DISCRETE_NET_NAMING_CONVENTION = "Discrete";
        private AgentBehaviour(in int STATE_SIZE, in int STACKED_INPUTS, in int VISUAL_INPUT_WIDTH, in int VISUAL_INPUT_HEIGHT, in int VISUAL_INPUT_CHANNELS,
            in int CONTINUOUS_ACTIONS_NUM, in int DISCRETE_ACTIONS_NUM, in int NUM_LAYERS, in int HIDDEN_UNITS, in ArchitectureType ARCHITECTURE, in NonLinearity NONLINEARITY)
        {

           
           
            static IActivation HiddenActivation(NonLinearity activ) => activ == NonLinearity.Relu ? new ReLU(in_place:true) : new Tanh(in_place:true);

            static IModule[] CreateMLP(int inputs, int stack, int outputs, int layers, int hidUnits, NonLinearity activ)
            {
                InitType INIT_W = activ == NonLinearity.Relu ? InitType.Kaiming_Uniform : InitType.Xavier_Uniform;
                InitType INIT_B = InitType.Zeros;
                if (layers == 1)
                {
                    return new IModule[] {
                        new Dense(inputs * stack, hidUnits, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new Dense(hidUnits, outputs, weight_init: INIT_W, bias_init : INIT_B)};
                }
                if (layers == 2)
                {
                    return new IModule[] {
                        new Dense(inputs * stack, hidUnits, weight_init: INIT_W, bias_init: INIT_B),
                        HiddenActivation(activ),
                        new Dense(hidUnits, hidUnits, weight_init: INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new Dense(hidUnits, outputs, weight_init: INIT_W, bias_init : INIT_B)};
                }
                if (layers == 3)
                {
                    return new IModule[] {
                        new Dense(inputs * stack, hidUnits, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new Dense(hidUnits, hidUnits, weight_init: INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new Dense(hidUnits, hidUnits, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new Dense(hidUnits, outputs, weight_init: INIT_W, bias_init : INIT_B)};
                }
                throw new ArgumentException("Unhandled numLayers outside range 1 - 3");
            }
            static IModule[] CreateLnMLP(int inputs, int stack, int outputs, int layers, int hidUnits, NonLinearity activ)
            {
                InitType INIT_W = activ == NonLinearity.Relu ? InitType.Kaiming_Uniform : InitType.Xavier_Uniform;
                InitType INIT_B = InitType.Zeros;
                if (layers == 1)
                {
                    return new IModule[] {
                        new Dense(inputs * stack, hidUnits, weight_init : INIT_W, bias_init : INIT_B),
                        new LayerNorm(hidUnits),
                        HiddenActivation(activ),
                        new Dense(hidUnits, outputs, weight_init: INIT_W, bias_init : INIT_B)};
                }
                if (layers == 2)
                {
                    return new IModule[] {
                        new Dense(inputs * stack, hidUnits, weight_init: INIT_W, bias_init: INIT_B),
                        new LayerNorm(hidUnits),
                        HiddenActivation(activ),
                        new Dense(hidUnits, hidUnits, weight_init: INIT_W, bias_init : INIT_B),
                        new LayerNorm(hidUnits),
                        HiddenActivation(activ),
                        new Dense(hidUnits, outputs, weight_init: INIT_W, bias_init : INIT_B)};
                }
                if (layers == 3)
                {
                    return new IModule[] {
                        new Dense(inputs * stack, hidUnits, weight_init : INIT_W, bias_init : INIT_B),
                        new LayerNorm(hidUnits),
                        HiddenActivation(activ),
                        new Dense(hidUnits, hidUnits, weight_init: INIT_W, bias_init : INIT_B),
                        new LayerNorm(hidUnits),
                        HiddenActivation(activ),
                        new Dense(hidUnits, hidUnits, weight_init : INIT_W, bias_init : INIT_B),
                        new LayerNorm(hidUnits),
                        HiddenActivation(activ),
                        new Dense(hidUnits, outputs, weight_init: INIT_W, bias_init : INIT_B)};
                }
                throw new ArgumentException("Unhandled numLayers outside range 1 - 3");
            }
            static IModule[] CreateRNN(int inputs, int stack, int outputs, int layers, int hidUnits, NonLinearity activ)
            {
                if (layers == 1)
                {
                    return new IModule[] {
                        new Reshape(new int[]{ inputs * stack}, new int[]{stack, inputs}),
                        new RNNCell(inputs, hidUnits, HiddenStates.ReturnLast),
                        new Dense(hidUnits, outputs)};
                }
                if (layers == 2)
                {
                    return new IModule[] {
                        new Reshape(new int[]{ inputs * stack}, new int[]{stack, inputs}),
                        new RNNCell(inputs, hidUnits, HiddenStates.ReturnAll),
                        new RNNCell(hidUnits, hidUnits, HiddenStates.ReturnLast),
                        new Dense(hidUnits, outputs)};
                }
                if (layers == 3)
                {
                    return new IModule[] {
                        new Reshape(new int[]{ inputs * stack}, new int[]{stack, inputs}),
                        new RNNCell(inputs, hidUnits, HiddenStates.ReturnAll),
                        new RNNCell(hidUnits, hidUnits, HiddenStates.ReturnAll),
                        new RNNCell(hidUnits, hidUnits, HiddenStates.ReturnLast),
                        new Dense(hidUnits, outputs)};
                }
                throw new ArgumentException("Unhandled numLayers outside range 1 - 3");
            }
            static IModule[] CreateCNN(int width, int height, int channels, int outputs, int layers, int hidunits, NonLinearity activ)
            {
                InitType INIT_W = activ == NonLinearity.Relu ? InitType.Kaiming_Uniform : InitType.Xavier_Uniform;
                InitType INIT_B = InitType.Zeros;

                if (layers == 1)
                {
                    return new IModule[]
                    {
                        new LazyConv2D(channels * 3, kernel_size: 3, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new AvgPool2D(2),

                        new Flatten(),
                        new LazyDense(hidunits, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new LazyDense(outputs, weight_init : INIT_W, bias_init : INIT_B)
                    };
                }
                if (layers == 2)
                {
                    return new IModule[]
                   {
                        new LazyConv2D(channels * 2, kernel_size: 3, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new AvgPool2D(2),

                        new LazyConv2D(channels * 4, kernel_size: 3, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new AvgPool2D(2),

                        new Flatten(),
                        new LazyDense(hidunits, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new LazyDense(hidunits, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new LazyDense(outputs, weight_init : INIT_W, bias_init : INIT_B)
                   };
                }
                if (layers == 3)
                {
                    return new IModule[]
                    {
                        new LazyConv2D(channels * 2, kernel_size: 3, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new AvgPool2D(2),

                        new LazyConv2D(channels * 4, kernel_size: 3, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new AvgPool2D(2),

                        new LazyConv2D(channels * 8, kernel_size: 3, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new AvgPool2D(2),

                        new Flatten(),
                        new LazyDense(hidunits, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new LazyDense(hidunits, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new LazyDense(hidunits, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new LazyDense(outputs, weight_init : INIT_W, bias_init : INIT_B)
                    };
                }
                throw new ArgumentException("Unhandled numLayers outside range 1 - 3");
            }
            static IModule[] CreateATT(int inputs, int stack, int outputs, int layers, int hidUnits, NonLinearity activ)
            {
                InitType INIT_W = activ == NonLinearity.Relu ? InitType.Kaiming_Uniform : InitType.Xavier_Uniform;
                InitType INIT_B = InitType.Zeros;
                if (layers == 1)
                {
                    return new IModule[] {
                        new Reshape(new int[]{ inputs * stack}, new int[]{stack, inputs}),
                        new Attention(inputs, hidUnits),
                        new LastSequence1DElementModule(),
                        new Dense(hidUnits, outputs, weight_init : INIT_W, bias_init : INIT_B)};
                }
                if (layers == 2)
                {
                    return new IModule[] {
                        new Reshape(new int[]{ inputs * stack}, new int[]{stack, inputs}),
                        new Attention(inputs, hidUnits),
                        new LastSequence1DElementModule(),
                        new Dense (hidUnits, hidUnits, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new Dense(hidUnits, outputs, weight_init : INIT_W, bias_init : INIT_B)};
                }
                if (layers == 3)
                {
                    return new IModule[] {
                        new Reshape(new int[]{ inputs * stack}, new int[]{stack, inputs}),
                        new Attention(inputs, hidUnits),
                        new LastSequence1DElementModule(),
                        new Dense(hidUnits, hidUnits, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new Dense(hidUnits, hidUnits, weight_init : INIT_W, bias_init : INIT_B),
                        HiddenActivation(activ),
                        new Dense(hidUnits, outputs, weight_init : INIT_W, bias_init : INIT_B)};
                }
                throw new ArgumentException("Unhandled numLayers outside range 1 - 3");
            }
            //------------------ NETWORK INITIALIZATION ----------------//


            if (ARCHITECTURE == ArchitectureType.MLP)
            {
                vNetwork = new Sequential(CreateMLP(STATE_SIZE, STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY));
                if (CONTINUOUS_ACTIONS_NUM > 0)
                {
                    muNetwork = new Sequential(CreateMLP(STATE_SIZE, STACKED_INPUTS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Tanh() }).ToArray());
                    sigmaNetwork = new Sequential(CreateMLP(STATE_SIZE, STACKED_INPUTS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Softplus() }).ToArray());
                    q1Network = new Sequential(CreateMLP((STATE_SIZE + CONTINUOUS_ACTIONS_NUM), STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY));
                    q2Network = new Sequential(CreateMLP((STATE_SIZE + CONTINUOUS_ACTIONS_NUM), STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY));
                }

                if (DISCRETE_ACTIONS_NUM > 0)
                    discreteNetwork = new Sequential(CreateMLP(STATE_SIZE, STACKED_INPUTS, DISCRETE_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Softmax() }).ToArray());
            }
            else if (ARCHITECTURE == ArchitectureType.LnMLP)
            {
                vNetwork = new Sequential(CreateLnMLP(STATE_SIZE, STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY));
                if (CONTINUOUS_ACTIONS_NUM > 0)
                {
                    muNetwork = new Sequential(CreateLnMLP(STATE_SIZE, STACKED_INPUTS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Tanh() }).ToArray());
                    sigmaNetwork = new Sequential(CreateLnMLP(STATE_SIZE, STACKED_INPUTS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Softplus() }).ToArray());
                    q1Network = new Sequential(CreateLnMLP((STATE_SIZE + CONTINUOUS_ACTIONS_NUM), STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY));
                    q2Network = new Sequential(CreateLnMLP((STATE_SIZE + CONTINUOUS_ACTIONS_NUM), STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY));
                }

                if (DISCRETE_ACTIONS_NUM > 0)
                    discreteNetwork = new Sequential(CreateLnMLP(STATE_SIZE, STACKED_INPUTS, DISCRETE_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Softmax() }).ToArray());

            }
            else if (ARCHITECTURE == ArchitectureType.RNN)
            {
                vNetwork = new Sequential(CreateRNN(STATE_SIZE, STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY));

                if (CONTINUOUS_ACTIONS_NUM > 0)
                {
                    muNetwork = new Sequential(CreateRNN(STATE_SIZE, STACKED_INPUTS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Tanh() }).ToArray());
                    sigmaNetwork = new Sequential(CreateRNN(STATE_SIZE, STACKED_INPUTS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Softplus() }).ToArray());
                    q1Network = new Sequential(CreateRNN((STATE_SIZE + CONTINUOUS_ACTIONS_NUM), STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY));
                    q2Network = new Sequential(CreateRNN((STATE_SIZE + CONTINUOUS_ACTIONS_NUM), STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY));
                }

                if (DISCRETE_ACTIONS_NUM > 0)
                    discreteNetwork = new Sequential(CreateRNN(STATE_SIZE, STACKED_INPUTS, DISCRETE_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Softmax() }).ToArray());
            }
            else if (ARCHITECTURE == ArchitectureType.CNN)
            {
                vNetwork = new Sequential(CreateCNN(VISUAL_INPUT_WIDTH, VISUAL_INPUT_HEIGHT, VISUAL_INPUT_CHANNELS, 1, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY));

                if (CONTINUOUS_ACTIONS_NUM > 0)
                {
                    muNetwork = new Sequential(CreateCNN(VISUAL_INPUT_WIDTH, VISUAL_INPUT_HEIGHT, VISUAL_INPUT_CHANNELS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Tanh() }).ToArray());
                    sigmaNetwork = new Sequential(CreateCNN(VISUAL_INPUT_WIDTH, VISUAL_INPUT_HEIGHT, VISUAL_INPUT_CHANNELS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Softplus() }).ToArray());
                }

                if (DISCRETE_ACTIONS_NUM > 0)
                    discreteNetwork = new Sequential(CreateCNN(VISUAL_INPUT_WIDTH, VISUAL_INPUT_HEIGHT, VISUAL_INPUT_CHANNELS, DISCRETE_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Softmax() }).ToArray());

                Tensor input = Tensor.Ones(VISUAL_INPUT_CHANNELS, VISUAL_INPUT_HEIGHT, VISUAL_INPUT_WIDTH);
                vNetwork.Predict(input);
                muNetwork?.Predict(input);
                sigmaNetwork?.Predict(input);
                discreteNetwork?.Predict(input);
            
            }
            else if (ARCHITECTURE == ArchitectureType.ATT)
            {
                vNetwork = new Sequential(CreateATT(STATE_SIZE, STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY));

                if (CONTINUOUS_ACTIONS_NUM > 0)
                {
                    muNetwork = new Sequential(CreateATT(STATE_SIZE, STACKED_INPUTS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Tanh() }).ToArray());
                    sigmaNetwork = new Sequential(CreateATT(STATE_SIZE, STACKED_INPUTS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Softplus() }).ToArray());
                    q1Network = new Sequential(CreateATT((STATE_SIZE + CONTINUOUS_ACTIONS_NUM), STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY));
                    q2Network = new Sequential(CreateATT((STATE_SIZE + CONTINUOUS_ACTIONS_NUM), STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY));
                }

                if (DISCRETE_ACTIONS_NUM > 0)
                    discreteNetwork = new Sequential(CreateATT(STATE_SIZE, STACKED_INPUTS, DISCRETE_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS, NONLINEARITY).Concat(new IModule[] { new Softmax() }).ToArray());

            }
            else
                throw new NotImplementedException("Unhandled ArchType");

        }

        /// <summary>
        /// Input: <paramref name="state"/> - <em>sₜ</em> | <see cref="Tensor"/> (<em>Observations</em>) or <see cref="Tensor"/>  (<em>Batch</em>, <em>Observations</em>)<br></br>
        /// Output: <paramref name="action"/> - <em>aₜ</em> |  <see cref="Tensor"/>  (<em>Continuous Actions</em>)  or <see cref="Tensor"/>  (<em>Batch</em>, <em>Continuous Actions</em>)<br></br>
        /// Extra Output: <paramref name="probs"/> - <em>πθ(aₜ|sₜ)</em> | <see cref="Tensor"/>  (<em>Continuous Actions</em>) or <see cref="Tensor"/> (<em>Batch</em>, <em>Continuous Actions</em>)
        /// </summary>
        public void ContinuousEval(Tensor state, out Tensor action, out Tensor probs)
        {
            if (!IsUsingContinuousActions)
            {
                action = null;
                probs = null;
                return;
            }

            Tensor mu;
            Tensor sigma;
            switch (stochasticity)
            {
                case Stochasticity.FixedStandardDeviation:
                    mu = muNetwork.Predict(state);
                    sigma = Tensor.Fill(standardDeviationValue, mu.Shape);
                    action = mu.Select(x => Utils.Random.Normal(x, standardDeviationValue, threadsafe: false));
                    probs = Tensor.Probability(action, mu, sigma);
                    break;
                case Stochasticity.TrainebleStandardDeviation:
                    mu = muNetwork.Predict(state);
                    sigma = sigmaNetwork.Predict(state) * standardDeviationScale;
                    action = mu.Zip(sigma, (x, y) => Utils.Random.Normal(x, y, threadsafe: false));
                    probs = Tensor.Probability(action, mu, sigma);
                    break;
                case Stochasticity.ActiveNoise:
                    mu = muNetwork.Predict(state);                  
                    action = (mu + Tensor.RandomNormal((0, noiseValue), mu.Shape)).Clip(-1f, 1f);// Check openai spinningup documentation. They fkin forgot to say that the e~N std is configurable (i don't think they forgot to clip also, maybe no clipping involved)
                    probs = null;
                    break;
                case Stochasticity.Random:
                    bool isBatched = state.Rank == 2; 
                    action = Tensor.RandomRange((-1f, 1f), isBatched ? new int[] { state.Size(0), continuousDim } : new int[] { continuousDim });
                    probs = null;
                    break;
                default:
                    throw new NotImplementedException("Unhandled Stochasticity type");
            }
        }
        /// <summary>
        /// Input: <paramref name="stateBatch"/> - <em>s</em> | <see cref="Tensor"/> (<em>Observations</em>) or <see cref="Tensor"/>  (<em>Batch</em>, <em>Observations</em>)<br></br>
        /// Output: <paramref name="muBatch"/> - <em>μ</em> | <see cref="Tensor"/>  (<em>Continuous Actions</em>)  or <see cref="Tensor"/>  (<em>Batch</em>, <em>Continuous Actions</em>)<br></br>
        /// Output: <paramref name="sigmaBatch"/> - <em>σ</em> | <see cref="Tensor"/>  (<em>Continuous Actions</em>) or <see cref="Tensor"/> (<em>Batch</em>, <em>Continuous Actions</em>)
        /// </summary>
        public void ContinuousForward(Tensor stateBatch, out Tensor muBatch, out Tensor sigmaBatch)
        {
            if (!IsUsingContinuousActions)
            {
                muBatch = null;
                sigmaBatch = null;
                return;
            }

            switch(stochasticity)
            {
                case Stochasticity.FixedStandardDeviation:
                    muBatch = muNetwork.Forward(stateBatch);
                    sigmaBatch = Tensor.Fill(standardDeviationValue, muBatch.Shape);
                    break;
                case Stochasticity.TrainebleStandardDeviation:
                    muBatch = muNetwork.Forward(stateBatch);
                    sigmaBatch = sigmaNetwork.Forward(stateBatch) * standardDeviationScale;
                    break;
                case Stochasticity.ActiveNoise:
                    muBatch = muNetwork.Forward(stateBatch);
                    sigmaBatch = null;
                    break;
                case Stochasticity.Random:
                    throw new Exception("How the fuck did you manage to get here?");
                default:
                    throw new NotImplementedException("Unhandled Stochasticity type");
            }                       
        }
        /// <summary>
        /// Input: <paramref name="state"/> - <em>sₜ</em> | Tensor (<em>Observations</em>) <br></br>
        /// Output: <paramref name="action"/> - <em>aₜ</em> |  Tensor (<em>Discrete Actions</em>) (one hot embedding)<br></br>
        /// Extra Output: <paramref name="phi"/> - <em>φₜ</em> | Tensor (<em>Discrete Actions</em>)
        /// </summary>
        public void DiscreteEval(Tensor state, out Tensor action, out Tensor phi)
        {

            if (!IsUsingDiscreteActions)
            {
                action = null;
                phi = null;
                return;
            }
            phi = discreteNetwork.Predict(state);
            int batchSize = phi.Rank == 2 ? phi.Size(0) : 1;
            action = Tensor.Zeros(phi.Shape);

            // φₜ - Normalzed Probabilities (through softmax) - parametrizes Multinomial probability distribution
            // δₜ - Multinomial Probability Distribution
            int[] discreteActionsIndexes = new int[discreteDim];//   Tensor.Arange(0, discreteDim, 1f).ToArray().Select(x => (int)x).ToArray();
            for (int i = 0; i < discreteDim; i++)
            {
                discreteActionsIndexes[i] = i;
            }

            if(batchSize == 1)
            {
                int sample = -1;
                try
                {
                    sample = Utils.Random.Sample(collection: discreteActionsIndexes, probs: phi.ToArray());
                }
                catch // probably this part is in case phi has some NaNs i supppose.
                {
                    sample = Utils.Random.Range(0, discreteDim);
                }

                action[sample] = 1f;
            }
            else
            {
                Tensor[] phis = phi.Split(0, 1);
                for (int b = 0; b < batchSize; b++)
                {
                    int sample = -1;
                    try
                    {
                        sample = Utils.Random.Sample(collection: discreteActionsIndexes, probs: phis[b].ToArray());
                    }
                    catch // probably this part is in case phi has some NaNs i supppose.
                    {
                        sample = Utils.Random.Range(0, discreteDim);
                    }

                    action[b, sample] = 1f;
                }
            }
          

            
        }
        /// <summary>
        /// Input: <paramref name="stateBatch"/> - <em>s</em> | Tensor (<em>Batch Size, Observations</em>) <br></br>
        /// Output: <paramref name="phi"/> - <em>φ </em> | Tensor (<em>Batch Size, Discrete Actions</em>) <br></br>
        /// </summary>
        public void DiscreteForward(Tensor stateBatch, out Tensor phi)
        {
            if (!IsUsingDiscreteActions)
            {
                phi = null;
                return;
            }

            phi = discreteNetwork.Forward(stateBatch);
        }


        /// <summary>
        /// Creates a new Agent behaviour folder containing all auxiliar neural networks, or loads it if already exists one for this behaviour.
        /// </summary>
        /// <returns></returns>
        public static AgentBehaviour CreateOrLoadAsset(string name, int stateSize, int stackedInputs, int widthSize, int heightSize, int channelSize, int continuousActions, int discreteActions, int numLayers, int hidUnits, ArchitectureType aType, NonLinearity nonlinearity)
        {
            if (stateSize == 0 && (aType == ArchitectureType.CNN || aType  == ArchitectureType.ATT))
                stateSize = 1;
#if UNITY_EDITOR
            var instance = UnityEditor.AssetDatabase.LoadAssetAtPath<AgentBehaviour>($"Assets/{name}/_{name}.asset");

            if (instance != null)
            {
                ConsoleMessage.Info($"Behaviour {name} asset loaded");
                return instance;
            }
#endif

            AgentBehaviour newAgBeh = new AgentBehaviour(stateSize, stackedInputs, widthSize, heightSize, channelSize, continuousActions, discreteActions, numLayers, hidUnits, aType, nonlinearity);
            newAgBeh.behaviourName = name;
            newAgBeh.observationSize = stateSize;
            newAgBeh.stackedInputs = stackedInputs;
            newAgBeh.continuousDim = continuousActions;
            newAgBeh.discreteDim = discreteActions;
            newAgBeh.observationsNormalizer = new RunningNormalizer(stateSize * stackedInputs);
            newAgBeh.rewardsNormalizer = new RewardsNormalizer();
            newAgBeh.assetCreated = true;

            // Create the asset
            if (!Directory.Exists($"Assets/{name}"))
                Directory.CreateDirectory($"Assets/{name}");

#if UNITY_EDITOR
            UnityEditor.AssetDatabase.CreateAsset(newAgBeh, $"Assets/{name}/_{name}.asset");
#endif

            // Create aux assets
            newAgBeh.config = Hyperparameters.CreateOrLoadAsset(name);
            newAgBeh.vNetwork?.CreateAsset($"{name}/{VALUE_NET_NAMING_CONVENTION}");
            newAgBeh.muNetwork?.CreateAsset($"{name}/{MU_NET_NAMING_CONVENTION}");
            newAgBeh.sigmaNetwork?.CreateAsset($"{name}/{SIGMA_NET_NAMING_CONVENTION}");
            newAgBeh.q1Network?.CreateAsset($"{name}/{Q1_NET_NAMING_CONVENTION}");
            newAgBeh.q2Network?.CreateAsset($"{name}/{Q2_NET_NAMING_CONVENTION}");
            newAgBeh.discreteNetwork?.CreateAsset($"{name}/{DISCRETE_NET_NAMING_CONVENTION}");


            return newAgBeh;
        }
        /// <summary>
        /// Updates the state of the Behaviour parameters.
        /// </summary>
        public void Save()
        {
            if (!assetCreated)
            {
                ConsoleMessage.Warning("Cannot save the Behaviour because it requires compilation first");
                return;
            }

            version++;
            ConsoleMessage.Info($"<b>[AUTOSAVE]</b> Agent behaviour <b><i>{behaviourName}</i></b> saved");

#if UNITY_EDITOR
            EditorUtility.SetDirty(this);
            AssetDatabase.SaveAssetIfDirty(this);
#endif

            vNetwork?.Save();
            q1Network?.Save();
            q2Network?.Save();
            muNetwork?.Save();
            sigmaNetwork?.Save();
            discreteNetwork?.Save();

#if !UNITY_EDITOR
            string folderPath = Path.Combine(Utils.GetDesktopPath(), $"{this.behaviourName}_Trained");
            if(!Directory.Exists(folderPath))
                Directory.CreateDirectory(folderPath);

            {
                string beh = JsonUtility.ToJson(this);
                File.WriteAllText(Path.Combine(folderPath, $"_{this.behaviourName}.json"), beh);
            }

            if(vNetwork!=null)
            {
                string vnet = JsonUtility.ToJson(vNetwork);
                File.WriteAllText(Path.Combine(folderPath, $"{VALUE_NET_NAMING_CONVENTION}.json"), vnet);
            }
            if (q1Network != null)
            {
                string q1net = JsonUtility.ToJson(vNetwork);
                File.WriteAllText(Path.Combine(folderPath, $"{Q1_NET_NAMING_CONVENTION}.json"), q1net);
            }
            if (q2Network != null)
            {
                string q2net = JsonUtility.ToJson(vNetwork);
                File.WriteAllText(Path.Combine(folderPath, $"{Q2_NET_NAMING_CONVENTION}.json"), q2net);
            }
            if (muNetwork != null)
            {
                string munet = JsonUtility.ToJson(muNetwork);
                File.WriteAllText(Path.Combine(folderPath, $"{MU_NET_NAMING_CONVENTION}.json"), munet);
            }
            if (sigmaNetwork != null)
            {
                string sigmanet = JsonUtility.ToJson(muNetwork);
                File.WriteAllText(Path.Combine(folderPath, $"{SIGMA_NET_NAMING_CONVENTION}.json"), sigmanet);
            }
            if (discreteNetwork != null)
            {
                string discretenet = JsonUtility.ToJson(discreteNetwork);
                File.WriteAllText(Path.Combine(folderPath, $"{DISCRETE_NET_NAMING_CONVENTION}.json"), discretenet);
            }
#endif
        }
        /// <summary>
        /// Before using, checks if config file or neural networks are not attached to this scriptable object.
        /// </summary>
        /// <returns></returns>
        public List<string> CheckForMissingAssets()
        {
            TryReassignReference_EditorOnly();

            if (config == null)
            {
                return new List<string>() { "Config" };
            }

            var trainer = config.trainer;
            var whatIsMissing = new List<string>();

            if (trainer == TrainerType.PPO)
            {
                if (!vNetwork)
                    whatIsMissing.Add("Value Network");

                if (IsUsingContinuousActions)
                {
                    if (!muNetwork)
                        whatIsMissing.Add("Mu Network");

                    if (!sigmaNetwork)
                        whatIsMissing.Add("Sigma Network");
                }

                if (IsUsingDiscreteActions)
                {
                    if (!discreteNetwork)
                        whatIsMissing.Add("Discrete Network");
                }

            }
            else if (trainer == TrainerType.SAC)
            {
                if (!q1Network)
                    whatIsMissing.Add("Q Network 1");

                if (!q2Network)
                    whatIsMissing.Add("Q Network 2");

                if (!muNetwork)
                    whatIsMissing.Add("Mu Network");

                if (!sigmaNetwork)
                    whatIsMissing.Add("Sigma Network");

            }
            else if (trainer == TrainerType.TD3)
            {
                if (!q1Network)
                    whatIsMissing.Add("Q Network 1");

                if (!q2Network)
                    whatIsMissing.Add("Q Network 2");

                if (!muNetwork)
                    whatIsMissing.Add("Mu Network");
            }
            else if (trainer == TrainerType.DDPG)
            {
                if (!q1Network)
                    whatIsMissing.Add("Q Network 1");

                if (!muNetwork)
                    whatIsMissing.Add("Mu Network");
            }
            else
                throw new NotImplementedException();
            return whatIsMissing;
        }
        /// <summary>
        /// This was implemented due the fact the Unity Editor is sometimes losing references of the networks, so to not reassign them manually this exists.
        /// </summary>
        private void TryReassignReference_EditorOnly()
        {
#if UNITY_EDITOR
            if(config == null)
            {
                string path = AssetDatabase.GetAssetPath(GetInstanceID());
                string dirpath = Path.GetDirectoryName(path);

                string[] guids = AssetDatabase.FindAssets("t:Object", new[] { dirpath });
                List<UnityEngine.Object> allBehaviorAssets = new();
                foreach (string guid in guids)
                {
                    string assetPath = AssetDatabase.GUIDToAssetPath(guid);
                    allBehaviorAssets.Add(AssetDatabase.LoadAssetAtPath(assetPath, typeof(UnityEngine.Object)));
                }

                this.config = allBehaviorAssets.OfType<Hyperparameters>().FirstOrDefault();

                if (vNetwork == null)
                {
                    var networks = allBehaviorAssets.OfType<Sequential>();
                    vNetwork = networks.FirstOrDefault(x => x.name == VALUE_NET_NAMING_CONVENTION);
                    muNetwork = networks.FirstOrDefault(x => x.name == MU_NET_NAMING_CONVENTION);
                    sigmaNetwork = networks.FirstOrDefault(x => x.name == SIGMA_NET_NAMING_CONVENTION);
                    discreteNetwork = networks.FirstOrDefault(x => x.name == DISCRETE_NET_NAMING_CONVENTION);
                    q1Network = networks.FirstOrDefault(x => x.name == Q1_NET_NAMING_CONVENTION);
                    q2Network = networks.FirstOrDefault(x => x.name == Q2_NET_NAMING_CONVENTION);
                }
            }
#endif
        }
        /// <summary>
        /// When the training is within a build, the trained weights are saved on the desktop. <br></br>
        /// A button to update the weights in editor will appear on the behaviour asset.
        /// </summary>
        public void TryUpdateWeightsFromDesktop()
        {
            if (!Directory.Exists(Path.Combine(Utils.GetDesktopPath(), $"{behaviourName}_Trained")))
                return;

            // it seems like the overwrite doesn t work.
            string path = Path.Combine(Utils.GetDesktopPath(), $"{this.behaviourName}_Trained");

            if(vNetwork != null)
            {
                string vnetPath = Path.Combine(path, $"{VALUE_NET_NAMING_CONVENTION}.json");
                string jsonData = File.ReadAllText(vnetPath);
                JsonUtility.FromJsonOverwrite(jsonData, vNetwork);
            }
            
            if (q1Network != null)
            {
                string q1netPath = Path.Combine(path, $"{Q1_NET_NAMING_CONVENTION}.json");
                string jsonData = File.ReadAllText(q1netPath);
                JsonUtility.FromJsonOverwrite(jsonData, q1Network);
            }
            
            if (q2Network != null)
            {
                string q2netPath = Path.Combine(path, $"{Q2_NET_NAMING_CONVENTION}.json");
                string jsonData = File.ReadAllText(q2netPath);
                JsonUtility.FromJsonOverwrite(jsonData, q2Network);
            }
            
            if (muNetwork != null)
            {
                string munetPath = Path.Combine(path, $"{MU_NET_NAMING_CONVENTION}.json");
                string jsonData = File.ReadAllText(munetPath);
                JsonUtility.FromJsonOverwrite(jsonData, muNetwork);
            }
            
            if (sigmaNetwork != null)
            {
                string sigmanetPath = Path.Combine(path, $"{SIGMA_NET_NAMING_CONVENTION}.json");
                string jsonData = File.ReadAllText(sigmanetPath);
                JsonUtility.FromJsonOverwrite(jsonData, sigmaNetwork);
            }
            
            if (discreteNetwork != null)
            {
                string discretenetPath = Path.Combine(path, $"{DISCRETE_NET_NAMING_CONVENTION}.json");
                string jsonData = File.ReadAllText(discretenetPath);
                JsonUtility.FromJsonOverwrite(jsonData, discreteNetwork);
            }

            // The behaviour is last because is changes the reference to the correct networks and the reupdate will not work.
            string behPath = Path.Combine(path, $"_{this.behaviourName}.json");
            string jsonDataBeh = File.ReadAllText(behPath);
            JsonUtility.FromJsonOverwrite(jsonDataBeh, this);

            TryReassignReference_EditorOnly();
            Save();
            ConsoleMessage.Info($"<b>[OVERWRITE]</b> Agent behaviour <b><i>{behaviourName}</i></b> was overriden with new weights from desktop");
            // well this complete reassignation.. a lot of stuff going on i'm to lazy to implement it for now.
        }
    }
#if UNITY_EDITOR
    [UnityEditor.CustomEditor(typeof(AgentBehaviour), true), UnityEditor.CanEditMultipleObjects]
    sealed class CustomAgentBehaviourEditor : UnityEditor.Editor
    {
        const string updateButtonMessage = "A version of this behavior is available on the desktop. \nPress this button to update this policy with the weights of that version.";
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            List<string> dontDrawMe = new() { "m_Script" };

            AgentBehaviour script = (AgentBehaviour)target;

            
            if(Directory.Exists(Path.Combine(Utils.GetDesktopPath(), $"{script.behaviourName}_Trained")) && GUILayout.Button(updateButtonMessage))
            {
                // GUILayout.Box .HelpBox("A new version of this behavior is available on your desktop. Press the button above to take the new weights.", MessageType.Info);
                script.TryUpdateWeightsFromDesktop();
              
            }

            if(SystemInfo.graphicsDeviceType == UnityEngine.Rendering.GraphicsDeviceType.Null)
            {
                if(script.inferenceDevice == Device.CPU || script.inferenceDevice == Device.GPU)
                {
                    EditorGUILayout.HelpBox("Your device doesn't have a graphics card. The inference and training devices will be set on CPU by default.", MessageType.Warning);
                }
            }
            // See or not standard deviation
            if (!script.IsUsingContinuousActions)
            {
                dontDrawMe.Add("stochasticity");
                dontDrawMe.Add("standardDeviationValue");
                dontDrawMe.Add("standardDeviationScale");
                dontDrawMe.Add("noiseValue");
            }
            else
            {
                if (script.stochasticity == Stochasticity.TrainebleStandardDeviation)
                {
                    dontDrawMe.Add("standardDeviationValue");
                    dontDrawMe.Add("noiseValue");
                }
                else if (script.stochasticity == Stochasticity.FixedStandardDeviation)
                {
                    dontDrawMe.Add("standardDeviationScale");
                    dontDrawMe.Add("noiseValue");
                }
                else if (script.stochasticity == Stochasticity.ActiveNoise)
                {
                    dontDrawMe.Add("standardDeviationValue");
                    dontDrawMe.Add("standardDeviationScale");
                }
                else if(script.stochasticity == Stochasticity.Random)
                {
                    dontDrawMe.Add("standardDeviationValue");
                    dontDrawMe.Add("standardDeviationScale");
                    dontDrawMe.Add("noiseValue");
                }
                else
                    throw new NotImplementedException("Unhandled stochasticity type");
            }



            if (script.standardDeviationValue <= 0)
            {
                script.standardDeviationValue = 1f;
            }

            if (script.standardDeviationScale <= 0)
            {
                script.standardDeviationScale = 1f;
            }
            if (!script.normalize)
            {
                dontDrawMe.Add("observationsNormalizer");
            }

            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
#endif
}

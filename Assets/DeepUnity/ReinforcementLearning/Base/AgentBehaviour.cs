using DeepUnity.Activations;
using DeepUnity.Models;
using DeepUnity.Modules;
using DeepUnity.Optimizers;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;


namespace DeepUnity.ReinforcementLearning
{
    [Serializable]
    public sealed class AgentBehaviour : ScriptableObject
    {
        [Header("Behaviour Properties")]
        [SerializeField, ViewOnly] public string behaviourName;
        [SerializeField, HideInInspector] private bool assetCreated = false;
        [SerializeField, ViewOnly] public int observationSize;
        [SerializeField, ViewOnly, Min(1)] public int stackedInputs;
        [SerializeField, ViewOnly] public int continuousDim;
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

        // [Header("Discriminator")]
        // [Tooltip("Neural Network used for Behavioral Cloning")]
        // [SerializeField] public NeuralNetwork discContNetwork;
        // [Tooltip("Neural Network used for Behavioral Cloning")]
        // [SerializeField] public NeuralNetwork discDiscNetwork;




        [Header("Behaviour Configurations")]
        [SerializeField, Tooltip("The frames per second runned by the physics engine. [Time.fixedDeltaTime = 1 / targetFPS]")]
        [Range(30, 100)]
        public int targetFPS = 50;

        [SerializeField, Tooltip("Network forward progapation is runned on this device when the agents interfere with the environment. It is recommended to be kept on CPU." +
           " The best way to find the optimal device is to check the number of fps when running out multiple environments.")]
        public Device inferenceDevice = Device.CPU;

        [SerializeField, Tooltip("Network computation is runned on this device when training on batches. It is highly recommended to be set on GPU if it is available.")]
        public Device trainingDevice = Device.GPU;

        [SerializeField, Tooltip("Auto-normalize input observations and rewards for a stable training.")]
        public bool normalize = false;

        [ViewOnly, SerializeField, Tooltip("Observations normalizer.")]
        public RunningNormalizer observationsNormalizer;

        [Range(1f, 10f), SerializeField, Tooltip("The observations are clipped [after normarlization] in range [-clip, clip]")]
        public float observationsClip = 5f; // old 1.5f - 3.5f

        [HideInInspector, SerializeField, ToolboxItem("Rewards normalizer")]
        public RewardsNormalizer rewardsNormalizer;

        [Header("Standard Deviation for Continuous Actions")]
        [SerializeField, Tooltip("The standard deviation for Continuous Actions")]
        public StandardDeviationType standardDeviation = StandardDeviationType.Trainable;
        [Tooltip("Modify this value to change the exploration/exploitation ratio.")]
        [SerializeField, Min(0.001f)]
        public float standardDeviationValue = 1f;

        [Tooltip("Modify this value to change the exploration/exploitation ratio. The standard deviation obtained by softplus(std_dev) * standardDeviationScale. 1.5scale  ~ 1fixed, 3scale  ~ 1.5fixed, 4.5scale ~ 2fixed")]
        [SerializeField, Min(0.1f)]
        public float standardDeviationScale = 1.5f;

        public Optimizer vOptimizer { get; private set; }
        public Optimizer q1Optimizer { get; private set; }
        public Optimizer q2Optimizer { get; private set; }
        public Optimizer muOptimizer { get; private set; }
        public Optimizer sigmaOptimizer { get; private set; }
        public Optimizer discreteOptimizer { get; private set; }
        public Optimizer dContOptimizer { get; private set; }
        public Optimizer dDiscOptimizer { get; private set; }

        public LRScheduler vScheduler { get; private set; }
        public LRScheduler q1Scheduler { get; private set; }
        public LRScheduler q2Scheduler { get; private set; }
        public LRScheduler muScheduler { get; private set; }
        public LRScheduler sigmaScheduler { get; private set; }
        public LRScheduler discreteScheduler { get; private set; }
        public LRScheduler dContScheduler { get; private set; }
        public LRScheduler dDiscScheduler { get; private set; }

        public bool IsUsingContinuousActions { get => continuousDim > 0; }
        public bool IsUsingDiscreteActions { get => discreteDim > 0; }


        private AgentBehaviour(in int STATE_SIZE, in int STACKED_INPUTS, in int VISUAL_INPUT_WIDTH, in int VISUAL_INPUT_HEIGHT, in int VISUAL_INPUT_CHANNELS,
            in int CONTINUOUS_ACTIONS_NUM, in int DISCRETE_ACTIONS_NUM, in int NUM_LAYERS, in int HIDDEN_UNITS, in ArchitectureType ARCHITECTURE)
        {

            const InitType INIT_W = InitType.HE_Normal;
            const InitType INIT_B = InitType.Zeros;
            static IActivation HiddenActivation() => new Tanh();

            static IModule[] CreateMLP(int inputs, int stack, int outputs, int layers, int hidUnits)
            {
                if (layers == 1)
                {
                    return new IModule[] {
                        new Dense(inputs * stack, hidUnits, INIT_W, INIT_B),
                        HiddenActivation(),
                        new Dense(hidUnits, outputs, INIT_W, INIT_B)};
                }
                if (layers == 2)
                {
                    return new IModule[] {
                        new Dense(inputs * stack, hidUnits, INIT_W, INIT_B),
                        HiddenActivation(),
                        new Dense(hidUnits, hidUnits, INIT_W, INIT_B),
                        HiddenActivation(),
                        new Dense(hidUnits, outputs, INIT_W, INIT_B)};
                }
                if (layers == 3)
                {
                    return new IModule[] {
                        new Dense(inputs * stack, hidUnits, INIT_W, INIT_B),
                        HiddenActivation(),
                        new Dense(hidUnits, hidUnits, INIT_W, INIT_B),
                        HiddenActivation(),
                        new Dense(hidUnits, hidUnits, INIT_W, INIT_B),
                        HiddenActivation(),
                        new Dense(hidUnits, outputs, INIT_W, INIT_B)};
                }
                throw new ArgumentException("Unhandled numLayers outside range 1 - 3");
            }
            static IModule[] CreateRNN(int inputs, int stack, int outputs, int layers, int hidUnits)
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
            static IModule[] CreateCNN(int width, int height, int channels, int outputs, int layers, int hidUnits)
            {

                int conv_hout = height - 3 + 1;
                int conv_wout = width - 3 + 1;
                int pool_hout = (int)MathF.Floor((conv_hout - 2f) / 2f + 1);
                int pool_wout = (int)MathF.Floor((conv_wout - 2f) / 2f + 1);
                if (layers == 1)
                {
                    return new IModule[]
                    {
                        new Conv2D(channels, 1, 3),
                        new MaxPool2D(2),
                        new Flatten(),
                        HiddenActivation(),
                        new Dense(pool_hout * pool_wout, outputs)
                    };
                }
                if (layers == 2)
                {
                    return new IModule[]
                   {
                        new Conv2D(channels, 1, 3),
                        new MaxPool2D(2),
                        new Flatten(),
                        HiddenActivation(),
                        new Dense(pool_hout * pool_wout, hidUnits),
                        HiddenActivation(),
                        new Dense(hidUnits, outputs),
                   };
                }
                if (layers == 3)
                {
                    return new IModule[]
                    {
                        new Conv2D(channels, 1, 3),
                        new MaxPool2D(2),
                        new Flatten(),
                        HiddenActivation(),
                        new Dense(pool_hout * pool_wout, hidUnits),
                        HiddenActivation(),
                        new Dense(hidUnits, hidUnits),
                        HiddenActivation(),
                        new Dense(hidUnits, outputs),
                    };
                }
                throw new ArgumentException("Unhandled numLayers outside range 1 - 3");
            }
            //------------------ NETWORK INITIALIZATION ----------------//


            if (ARCHITECTURE == ArchitectureType.MLP)
            {
                vNetwork = new Sequential(CreateMLP(STATE_SIZE, STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS));
                if (CONTINUOUS_ACTIONS_NUM > 0)
                {
                    muNetwork = new Sequential(CreateMLP(STATE_SIZE, STACKED_INPUTS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS));
                    sigmaNetwork = new Sequential(CreateMLP(STATE_SIZE, STACKED_INPUTS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS).Concat(new IModule[] { new Softplus() }).ToArray());
                    q1Network = new Sequential(CreateMLP((STATE_SIZE + CONTINUOUS_ACTIONS_NUM), STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS));
                    q2Network = new Sequential(CreateMLP((STATE_SIZE + CONTINUOUS_ACTIONS_NUM), STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS));
                }

                if (DISCRETE_ACTIONS_NUM > 0)
                    discreteNetwork = new Sequential(CreateMLP(STATE_SIZE, STACKED_INPUTS, DISCRETE_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS).Concat(new IModule[] { new Softmax() }).ToArray());
            }
            else if (ARCHITECTURE == ArchitectureType.RNN)
            {
                vNetwork = new Sequential(CreateRNN(STATE_SIZE, STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS));

                if (CONTINUOUS_ACTIONS_NUM > 0)
                {
                    muNetwork = new Sequential(CreateRNN(STATE_SIZE, STACKED_INPUTS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS));
                    sigmaNetwork = new Sequential(CreateRNN(STATE_SIZE, STACKED_INPUTS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS).Concat(new IModule[] { new Softplus() }).ToArray());
                    q1Network = new Sequential(CreateRNN((STATE_SIZE + CONTINUOUS_ACTIONS_NUM), STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS));
                    q2Network = new Sequential(CreateRNN((STATE_SIZE + CONTINUOUS_ACTIONS_NUM), STACKED_INPUTS, 1, NUM_LAYERS, HIDDEN_UNITS));
                }

                if (DISCRETE_ACTIONS_NUM > 0)
                    discreteNetwork = new Sequential(CreateRNN(STATE_SIZE, STACKED_INPUTS, DISCRETE_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS).Concat(new IModule[] { new Softmax() }).ToArray());
            }
            else if (ARCHITECTURE == ArchitectureType.CNN)
            {
                vNetwork = new Sequential(CreateCNN(VISUAL_INPUT_WIDTH, VISUAL_INPUT_HEIGHT, VISUAL_INPUT_CHANNELS, 1, NUM_LAYERS, HIDDEN_UNITS));

                if (CONTINUOUS_ACTIONS_NUM > 0)
                {
                    muNetwork = new Sequential(CreateCNN(VISUAL_INPUT_WIDTH, VISUAL_INPUT_HEIGHT, VISUAL_INPUT_CHANNELS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS));
                    sigmaNetwork = new Sequential(CreateCNN(VISUAL_INPUT_WIDTH, VISUAL_INPUT_HEIGHT, VISUAL_INPUT_CHANNELS, CONTINUOUS_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS).Concat(new IModule[] { new Softplus() }).ToArray());
                }

                if (DISCRETE_ACTIONS_NUM > 0)
                    discreteNetwork = new Sequential(CreateCNN(VISUAL_INPUT_WIDTH, VISUAL_INPUT_HEIGHT, VISUAL_INPUT_CHANNELS, DISCRETE_ACTIONS_NUM, NUM_LAYERS, HIDDEN_UNITS).Concat(new IModule[] { new Softmax() }).ToArray());
            }



        }

        public void InitOptimisers(Hyperparameters hp, TrainerType trainer)
        {
            const float lambda = 0f;
            const float epsilon = 1e-5F; // PPO openAI eps they use :D, but in Andrychowicz, et al. (2021) they use tf default 1e-7
            if (trainer == TrainerType.SAC)
            {
                vOptimizer = new Adam(vNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
                q1Optimizer = new Adam(q1Network.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
                q2Optimizer = new Adam(q2Network.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
                muOptimizer = new Adam(muNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
                sigmaOptimizer = new Adam(sigmaNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
            }
            else if (trainer == TrainerType.PPO)
            {
                vOptimizer = new Adam(vNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);

                if (IsUsingContinuousActions)
                {
                    muOptimizer = new Adam(muNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
                    sigmaOptimizer = new Adam(sigmaNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
                }

                if (IsUsingDiscreteActions)
                {
                    discreteOptimizer = new Adam(discreteNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
                }
            }
        }
        public void InitSchedulers(Hyperparameters hp, TrainerType trainer)
        {
            int total_epochs = (int)hp.maxSteps / hp.bufferSize * hp.numEpoch; // THIS IS FOR PPO, but for now i will let it for SAC as well

            if (trainer == TrainerType.SAC)
            {
                vScheduler = new LinearLR(vOptimizer, start_factor: 1f, end_factor: 0f, epochs: total_epochs);
                q1Scheduler = new LinearLR(q1Optimizer, start_factor: 1f, end_factor: 0f, epochs: total_epochs);
                q2Scheduler = new LinearLR(q1Optimizer, start_factor: 1f, end_factor: 0f, epochs: total_epochs);
                muScheduler = new LinearLR(muOptimizer, start_factor: 1f, end_factor: 0f, epochs: total_epochs);
                sigmaScheduler = new LinearLR(sigmaOptimizer, start_factor: 1f, end_factor: 0f, epochs: total_epochs);
            }
            else if (trainer == TrainerType.PPO)
            {
                vScheduler = new LinearLR(vOptimizer, start_factor: 1f, end_factor: 0f, epochs: total_epochs);

                if (IsUsingContinuousActions)
                {
                    muScheduler = new LinearLR(muOptimizer, start_factor: 1f, end_factor: 0f, epochs: total_epochs);
                    sigmaScheduler = new LinearLR(sigmaOptimizer, start_factor: 1f, end_factor: 0f, epochs: total_epochs);
                }

                if (IsUsingDiscreteActions)
                {
                    discreteScheduler = new LinearLR(discreteOptimizer, start_factor: 1f, end_factor: 0f, epochs: total_epochs);
                }
            }
        }


        /// <summary>
        /// Input: <paramref name="state"/> - <em>sₜ</em> | Tensor (<em>Observations</em>) <br></br>
        /// Output: <paramref name="action"/> - <em>aₜ</em> |  Tensor (<em>Continuous Actions</em>) <br></br>
        /// Extra Output: <paramref name="probs"/> - <em>πθ(aₜ|sₜ)</em> | Tensor (<em>Continuous Actions</em>)
        /// </summary>
        public void ContinuousPredict(Tensor state, out Tensor action, out Tensor probs)
        {
            if (!IsUsingContinuousActions)
            {
                action = null;
                probs = null;
                return;
            }

            Tensor mu = muNetwork.Predict(state);
            Tensor sigma = standardDeviation == StandardDeviationType.Trainable ?
                            sigmaNetwork.Predict(state) * standardDeviationScale :
                            Tensor.Fill(standardDeviationValue, mu.Shape);

            action = mu.Zip(sigma, (x, y) => Utils.Random.Normal(x, y));
            probs = Tensor.Probability(action, mu, sigma);


        }
        /// <summary>
        /// Input: <paramref name="stateBatch"/> - <em>s</em> | Tensor (<em>Batch Size, Observations</em>) <br></br>
        /// Output: <paramref name="muBatch"/> - <em>μ</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// Output: <paramref name="sigmaBatch"/> - <em>σ</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// </summary>
        public void ContinuousForward(Tensor stateBatch, out Tensor muBatch, out Tensor sigmaBatch)
        {
            if (!IsUsingContinuousActions)
            {
                muBatch = null;
                sigmaBatch = null;
                return;
            }

            muBatch = muNetwork.Forward(stateBatch);
            sigmaBatch = standardDeviation == StandardDeviationType.Trainable ?
                           sigmaNetwork.Forward(stateBatch) * standardDeviationScale :
                           Tensor.Fill(standardDeviationValue, muBatch.Shape);
        }
        /// <summary>
        /// Input: <paramref name="state"/> - <em>sₜ</em> | Tensor (<em>Observations</em>) <br></br>
        /// Output: <paramref name="action"/> - <em>aₜ</em> |  Tensor (<em>Discrete Actions</em>) (one hot embedding)<br></br>
        /// Extra Output: <paramref name="phi"/> - <em>φₜ</em> | Tensor (<em>Discrete Actions</em>)
        /// </summary>
        public void DiscretePredict(Tensor state, out Tensor action, out Tensor phi)
        {

            if (!IsUsingDiscreteActions)
            {
                action = null;
                phi = null;
                return;
            }

            phi = discreteNetwork.Predict(state);

            // φₜ - Normalzed Probabilities (through softmax) - parametrizes Multinomial probability distribution
            // δₜ - Multinomial Probability Distribution
            int[] discreteActionsIndexes = Tensor.Arange(0, discreteDim, 1f).ToArray().Select(x => (int)x).ToArray();
            int sample = -1;
            try
            {
                sample = Utils.Random.Sample(collection: discreteActionsIndexes, probs: phi.ToArray());
            }
            catch
            {
                sample = Utils.Random.Range(0, discreteDim);
            }
            action = Tensor.Zeros(phi.Shape);
            action[sample] = 1f;
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
        public static AgentBehaviour CreateOrLoadAsset(string name, int stateSize, int stackedInputs, int widthSize, int heightSize, int channelSize, int continuousActions, int discreteActions, int numLayers, int hidUnits, ArchitectureType aType)
        {
            var instance = UnityEditor.AssetDatabase.LoadAssetAtPath<AgentBehaviour>($"Assets/{name}/{name}.asset");

            if (instance != null)
            {
                ConsoleMessage.Info($"Behaviour {name} asset loaded");
                return instance;
            }


            AgentBehaviour newAgBeh = new AgentBehaviour(stateSize, stackedInputs, widthSize, heightSize, channelSize, continuousActions, discreteActions, numLayers, hidUnits, aType);
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
            UnityEditor.AssetDatabase.CreateAsset(newAgBeh, $"Assets/{name}/_{name}.asset");

            // Create aux assets
            newAgBeh.config = Hyperparameters.CreateOrLoadAsset(name);
            newAgBeh.vNetwork?.CreateAsset($"{name}/V");
            newAgBeh.muNetwork?.CreateAsset($"{name}/Mu");
            newAgBeh.sigmaNetwork?.CreateAsset($"{name}/Sigma");
            newAgBeh.q1Network?.CreateAsset($"{name}/Q1");
            newAgBeh.q2Network?.CreateAsset($"{name}/Q2");
            newAgBeh.discreteNetwork?.CreateAsset($"{name}/Discrete");


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
            }

            ConsoleMessage.Info($"Agent behaviour <b><i>{behaviourName}</i></b> autosaved");

            vNetwork?.Save();
            q1Network?.Save();
            q2Network?.Save();
            muNetwork?.Save();
            sigmaNetwork?.Save();
            discreteNetwork?.Save();
            // discContNetwork?.Save();
            // discDiscNetwork?.Save();
        }
        /// <summary>
        /// Before using, checks if config file or neural networks are not attached to this scriptable object.
        /// </summary>
        /// <returns></returns>
        public List<string> CheckForMissingAssets()
        {
            TryReassignReference();

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
                if (!vNetwork)
                    whatIsMissing.Add("Value Network 1");

                if (!q1Network)
                    whatIsMissing.Add("Q Network 1");

                if (!q2Network)
                    whatIsMissing.Add("Q Network 2");

                if (IsUsingContinuousActions)
                {
                    if (!muNetwork)
                        whatIsMissing.Add("Mu Network");

                    if (!sigmaNetwork)
                        whatIsMissing.Add("Sigma Network");
                }
            }
            return whatIsMissing;
        }
        private void TryReassignReference()
        {
            return;// it seems like it cannot find the object anyways idk why


            // if(config == null)
            // {
            //     string path = AssetDatabase.GetAssetPath(GetInstanceID());
            //     path = Path.GetDirectoryName(path);
            //     config = AssetDatabase.LoadAllAssetsAtPath(path).OfType<Hyperparameters>().FirstOrDefault();
            // 
            //     if (vNetwork == null)
            //     {
            //         var networks = AssetDatabase.LoadAllAssetsAtPath(path).OfType<NeuralNetwork>();
            // 
            //         vNetwork = networks.FirstOrDefault(x => x.name == "V");
            //         muNetwork = networks.FirstOrDefault(x => x.name == "Mu");
            //         sigmaNetwork = networks.FirstOrDefault(x => x.name == "Sigma");
            //         discreteNetwork = networks.FirstOrDefault(x => x.name == "Discrete");
            //         q1Network = networks.FirstOrDefault(x => x.name == "Q1");
            //         q2Network = networks.FirstOrDefault(x => x.name == "Q2");
            //     }
            // }
        }
    }
#if UNITY_EDITOR
    [UnityEditor.CustomEditor(typeof(AgentBehaviour), true), UnityEditor.CanEditMultipleObjects]
    sealed class CustomAgentBehaviourEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            List<string> dontDrawMe = new() { "m_Script" };

            AgentBehaviour script = (AgentBehaviour)target;

            // See or not standard deviation
            if (!script.IsUsingContinuousActions)
            {
                dontDrawMe.Add("standardDeviation");
                dontDrawMe.Add("standardDeviationValue");
                dontDrawMe.Add("standardDeviationScale");
            }
            else
            {
                if (script.standardDeviation == StandardDeviationType.Trainable)
                {
                    dontDrawMe.Add("standardDeviationValue");
                }
                else
                {
                    dontDrawMe.Add("standardDeviationScale");
                }
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

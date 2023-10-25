using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;


namespace DeepUnity
{
    /// <summary>
    /// AgentBehaviour asset, along with the entire folder of networks is recommended to be kept inside main Assets folder,
    /// until the agent is completely finished.
    /// </summary>
    [Serializable]
    public sealed class AgentBehaviour : ScriptableObject
    {
        [Header("Behaviour Properties")]
        [SerializeField, ReadOnly] public string behaviourName;
        [SerializeField, HideInInspector] private bool assetCreated = false;
        [SerializeField, ReadOnly] public int observationSize;
        [SerializeField, ReadOnly, Min(1)] public int stackedInputsNum;
        [SerializeField, ReadOnly] public int continuousDim;
        [SerializeField, ReadOnly] public int discreteDim;

        [Header("Neural Networks & Hyperparameters")]    
        [SerializeField] public NeuralNetwork critic;
        [SerializeField] public NeuralNetwork actorContinuousMu;
        [SerializeField] public NeuralNetwork actorContinuousSigma;
        [Tooltip("Neural Network used for Behavioral Cloning")]
        [SerializeField] public NeuralNetwork discriminatorContinuous;

        [Space]
        [SerializeField] public NeuralNetwork actorDiscrete;
        [Tooltip("Neural Network used for Behavioral Cloning")]
        [SerializeField] public NeuralNetwork discriminatorDiscrete;
        [Space, Tooltip("The scriptable object file containing the training hyperparameters.")]
        [SerializeField] public Hyperparameters config;



        [ Header("Behaviour Configurations")]
        [SerializeField, Tooltip("The frames per second runned by the physics engine. [Time.fixedDeltaTime = 1 / targetFPS]")]
        [Range(30, 100)]
        public int targetFPS = 50;

        [SerializeField, Tooltip("Network forward progapation is runned on this device when the agents interfere with the environment. It is recommended to be kept on CPU." +
           " The best way to find the optimal device is to check the number of fps when running out multiple environments.")]
        public Device inferenceDevice = Device.CPU;

        [SerializeField, Tooltip("Network computation is runned on this device when training on batches. It is highly recommended to be set on GPU if it is available.")]
        public Device trainingDevice = Device.GPU;

        [SerializeField, Tooltip("Auto-normalize input observations.")]
        public bool normalizeObservations = false;

        [ReadOnly, SerializeField, Tooltip("Observations normalizer data.")] 
        public ZScoreNormalizer normalizer;

        [Header("Standard Deviation for Continuous Actions")]
        [SerializeField, Tooltip("The standard deviation for Continuous Actions")] 
        public StandardDeviationType standardDeviation = StandardDeviationType.Trainable;
        [Tooltip("Modify this value to change the exploration/exploitation ratio.")]
        [SerializeField, Range(0.001f, 3f)] 
        public float standardDeviationValue = 1f;
        [Tooltip("Sigma network's output is multiplied by this number. Modify this value to change the exploration/exploitation ratio.")]
        [SerializeField, Range(0.1f, 3f)]
        public float standardDeviationScale = 1f;
        

        public Optimizer criticOptimizer { get; private set; }
        public Optimizer actorMuOptimizer { get; private set; }
        public Optimizer actorSigmaOptimizer { get; private set; }
        public Optimizer actorDiscreteOptimizer { get; private set; }
        public Optimizer discriminatorContinuousOptimizer { get; private set; }
        public Optimizer discriminatorDiscreteOptimizer { get; private set; }

        public LRScheduler criticScheduler { get; private set; }
        public LRScheduler actorMuScheduler { get; private set; }
        public LRScheduler actorSigmaScheduler { get; private set; }
        public LRScheduler actorDiscreteScheduler { get; private set; }
        public LRScheduler discriminatorContinuousScheduler{ get; private set; }
        public LRScheduler discriminatorDiscreteScheduler { get; private set; }

        public bool IsUsingContinuousActions { get => continuousDim > 0; }
        public bool IsUsingDiscreteActions { get => discreteDim > 0; }

        private AgentBehaviour(in int STATE_SIZE, in int STACKED_INPUTS, in int CONTINUOUS_ACTIONS_NUM, in int DISCRETE_ACTIONS_NUM, in ModelType arch, in int NUM_LAYERS, in int HIDDEN_UNITS)
        {
          
            const InitType INIT_W = InitType.HE_Uniform;
            const InitType INIT_B = InitType.Zeros;
            //------------------ NETWORK INITIALIZATION ----------------//

            switch(arch)
            {
                case ModelType.NN:
                    critic = new NeuralNetwork(
                   new IModule[] {
                       new Dense(STATE_SIZE * STACKED_INPUTS, HIDDEN_UNITS, INIT_W, INIT_B), CreateActivation() }.
                       Concat(CreateHiddenLayers(NUM_LAYERS, HIDDEN_UNITS, INIT_W, INIT_B)).
                       Concat(new IModule[] { new Dense(HIDDEN_UNITS, 1, INIT_W, INIT_B) }).ToArray()
                    );
                    if (CONTINUOUS_ACTIONS_NUM > 0)
                    {
                        actorContinuousMu = new NeuralNetwork(
                                new IModule[] {
                            new Dense(STATE_SIZE * STACKED_INPUTS, HIDDEN_UNITS, INIT_W, INIT_B), CreateActivation() }.
                                    Concat(CreateHiddenLayers(NUM_LAYERS, HIDDEN_UNITS, INIT_W, INIT_B)).
                                    Concat(new IModule[] { new Dense(HIDDEN_UNITS, CONTINUOUS_ACTIONS_NUM, INIT_W, INIT_B), new Tanh() }).ToArray()
                            );

                        actorContinuousSigma = new NeuralNetwork(
                                new IModule[] {
                            new Dense(STATE_SIZE * STACKED_INPUTS, HIDDEN_UNITS, INIT_W, INIT_B), CreateActivation() }.
                                    Concat(CreateHiddenLayers(NUM_LAYERS - 1, HIDDEN_UNITS, INIT_W, INIT_B)). // Sigma network is a bit smaller for efficiency. This network is typically not important to be large
                                    Concat(new IModule[] { new Dense(HIDDEN_UNITS, CONTINUOUS_ACTIONS_NUM, INIT_W, INIT_B), new Exp() }).ToArray()  // Also Softplus can be used
                            );

                        discriminatorContinuous = new NeuralNetwork(
                                new IModule[] {
                            new Dense(CONTINUOUS_ACTIONS_NUM, HIDDEN_UNITS, INIT_W, INIT_B),CreateActivation() }.
                                    Concat(CreateHiddenLayers(NUM_LAYERS, HIDDEN_UNITS, INIT_W, INIT_B)).
                                    Concat(new IModule[] { new Dense(HIDDEN_UNITS, 1, INIT_W, INIT_B), new Sigmoid() }).ToArray()
                            );
                    }

                    if (DISCRETE_ACTIONS_NUM > 0)
                    {
                        actorDiscrete = new NeuralNetwork(
                                new IModule[] {
                            new Dense(STATE_SIZE * STACKED_INPUTS, HIDDEN_UNITS, INIT_W, INIT_B),new LeakyReLU() }.
                                    Concat(CreateHiddenLayers(NUM_LAYERS, HIDDEN_UNITS, INIT_W, INIT_B)).
                                    Concat(new IModule[] { new Dense(HIDDEN_UNITS, DISCRETE_ACTIONS_NUM, INIT_W, INIT_B), new Softmax() }).ToArray()
                            );
                        discriminatorDiscrete = new NeuralNetwork(
                                new IModule[] {
                            new Dense(DISCRETE_ACTIONS_NUM, HIDDEN_UNITS, INIT_W, INIT_B), CreateActivation() }.
                                    Concat(CreateHiddenLayers(NUM_LAYERS, HIDDEN_UNITS, INIT_W, INIT_B)).
                                    Concat(new IModule[] { new Dense(HIDDEN_UNITS, 1, INIT_W, INIT_B), new Sigmoid() }).ToArray()
                            );
                    }

                    break;
                case ModelType.CNN:
                    throw new ArgumentException("CNN was not introduced yet to RL");
                case ModelType.RNN:
                    throw new ArgumentException("RNN was not introduced yet to RL");
            }

            static Activation CreateActivation() => new LeakyReLU();

            static IModule[] CreateHiddenLayers(int numLayers, int hidUnits, InitType INIT_W, InitType INIT_B)
            {
                if (numLayers == 1)
                    return new IModule[] { };
                else if (numLayers == 2)
                    return new IModule[] { new Dense(hidUnits, hidUnits, INIT_W, INIT_B), CreateActivation() };
                else if (numLayers == 3)
                    return new IModule[] { new Dense(hidUnits, hidUnits, INIT_W, INIT_B), CreateActivation(), 
                                           new Dense(hidUnits, hidUnits, INIT_W, INIT_B), CreateActivation() };
                else
                    throw new ArgumentException("Unhandled numLayers outside range 1 - 3");

            }
        }

        public void SetActorDevice(Device device)
        {
            if (IsUsingContinuousActions)
            { 
                var learnables = actorContinuousMu.Parameters();
                foreach (var item in learnables)
                {
                    item.device = device;
                }

                learnables = actorContinuousSigma.Parameters();
                foreach (var item in learnables)
                {
                    item.device = device;
                }

                learnables = discriminatorContinuous.Parameters();
                foreach (var item in learnables)
                {
                    item.device = device;
                }
            }

            if (IsUsingDiscreteActions)
            {
                var learnables = actorDiscrete.Parameters();
                foreach (var item in learnables)
                {
                    item.device = device;
                }

                learnables = discriminatorDiscrete.Parameters();
                foreach (var item in learnables)
                {
                    item.device = device;
                }
            }
        }
        public void SetCriticDevice(Device device)
        {
            var learnables = critic.Parameters();
            for (int i = 0; i < learnables.Length - 1; i++)
            {
                learnables[i].device = device;
            }
            learnables[learnables.Length - 1].device = Device.CPU;

            // Due to benchmark performances, critic last dense (which has the output features = 1), is working faster on CPU.
        }
        public void InitOptimisers(Hyperparameters hp)
        {
            if (criticOptimizer != null)
                return;

            if (critic == null || config == null)
            {
                ConsoleMessage.Warning($"Critic Neural Network & Config assets are not attached to {behaviourName} behaviour asset!");
                EditorApplication.isPlaying = false;
                return;
            }

            criticOptimizer = new Adam(critic.Parameters(), hp.learningRate);  
            
            if(IsUsingContinuousActions)
            {
                if (actorContinuousMu == null || actorContinuousSigma == null || discriminatorContinuous == null)
                {
                    ConsoleMessage.Warning($"Neural Network assets for continuous actions are not attached to {behaviourName} behaviour asset!");
                    EditorApplication.isPlaying = false;
                    return;
                }
                actorMuOptimizer = new Adam(actorContinuousMu.Parameters(), hp.learningRate);
                actorSigmaOptimizer = new Adam(actorContinuousSigma.Parameters(), hp.learningRate);
                discriminatorContinuousOptimizer = new Adam(discriminatorContinuous.Parameters(), hp.learningRate);
            }
           
            if(IsUsingDiscreteActions)
            {
                if (actorDiscrete == null || discriminatorDiscrete == null)
                {
                    ConsoleMessage.Warning($"Neural Network assets for discrete actions are not attached to {behaviourName} behaviour asset!");
                    EditorApplication.isPlaying = false;
                    return;
                }
                actorDiscreteOptimizer = new Adam(actorDiscrete.Parameters(), hp.learningRate);
                discriminatorDiscreteOptimizer = new Adam(discriminatorDiscrete.Parameters(), hp.learningRate);
            
            }
        }
        public void InitSchedulers(Hyperparameters hp)
        {
            if (criticScheduler != null) // One way init
                return;

            if (critic == null)
            {
                ConsoleMessage.Warning($"Neural Network & Config assets are not attached to {behaviourName} behaviour asset!");
                EditorApplication.isPlaying = false;
                return;
            }

            // LR 0 = initialLR * (gamma^n) => gamma = nth-root(initialLr);
            int total_epochs = (int)hp.maxSteps / hp.bufferSize * hp.numEpoch;
            int step_size = 1;
            float gamma = Mathf.Pow(hp.learningRate, 1f / total_epochs);


            criticScheduler = new LRScheduler(criticOptimizer, step_size, gamma);

            if (IsUsingContinuousActions)
            {
                actorMuScheduler = new LRScheduler(actorMuOptimizer, step_size, gamma);
                actorSigmaScheduler = new LRScheduler(actorSigmaOptimizer, step_size, gamma);
                discriminatorContinuousScheduler = new LRScheduler(discriminatorContinuousOptimizer, step_size, gamma);
            }
          
            if(IsUsingDiscreteActions)
            {
                actorDiscreteScheduler = new LRScheduler(actorDiscreteOptimizer, step_size, gamma);
                discriminatorDiscreteScheduler = new LRScheduler(discriminatorDiscreteOptimizer, step_size, gamma);
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
            if (actorContinuousMu == null)
            {
                ConsoleMessage.Warning($"<i>NeuralNetwork</i> assets are not attached to <b>{behaviourName}</b> behaviour asset");
                EditorApplication.isPlaying = false;
                action = null;
                probs = null;
                return;
            }

            Tensor mu = actorContinuousMu.Predict(state);
            Tensor sigma = standardDeviation == StandardDeviationType.Trainable ?
                            actorContinuousSigma.Predict(state) * standardDeviationScale :
                            Tensor.Fill(standardDeviationValue, mu.Shape);
            action = mu.Zip(sigma, (x, y) => Utils.Random.Gaussian(x, y));
            probs = Tensor.Probability(action, mu, sigma);
        }
        /// <summary>
        /// Input: <paramref name="stateBatch"/> - <em>s</em> | Tensor (<em>Batch Size, Observations</em>) <br></br>
        /// Output: <paramref name="muBatch"/> - <em>μ</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// OutputL <paramref name="sigmaBatch"/> - <em>σ</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// </summary>
        public void ContinuousForward(Tensor stateBatch, out Tensor muBatch, out Tensor sigmaBatch)
        {
            if (!IsUsingContinuousActions)
            {
                muBatch = null;
                sigmaBatch = null;
                return;
            }

            muBatch = actorContinuousMu.Forward(stateBatch);
            sigmaBatch = standardDeviation == StandardDeviationType.Trainable ?
                           actorContinuousSigma.Forward(stateBatch) * standardDeviationScale :
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

            if (actorDiscrete == null)
            {
                ConsoleMessage.Warning($"<i>NeuralNetwork</i> assets are not attached to <b>{behaviourName}</b> behaviour asset");
                EditorApplication.isPlaying = false;
                action = null;
                phi = null;
                return;
            }
            
            phi = actorDiscrete.Predict(state);

            // φₜ - Normalzed Probabilities (through softmax) - parametrizes Multinomial probability distribution
            // δₜ - Multinomial Probability Distribution
            int[] discreteActionsIndexes = Tensor.Arange(0, discreteDim, 1f).ToArray().Select(x => (int)x).ToArray();
            int sample = Utils.Random.Sample(collection: discreteActionsIndexes, probs: phi.ToArray());
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

            phi = actorDiscrete.Forward(stateBatch);        
        }




        /// <summary>
        /// Creates a new Agent behaviour folder containing all auxiliar neural networks, or loads it if already exists one for this behaviour.
        /// </summary>
        /// <returns></returns>
        public static AgentBehaviour CreateOrLoadAsset(string name, int stateSize, int stackedInputs, int continuousActions, int discreteActions, ModelType architecture, int numLayers, int hidUnits)
        {          
            var instance = AssetDatabase.LoadAssetAtPath<AgentBehaviour>($"Assets/{name}/{name}.asset");

            if (instance != null)
            {
                ConsoleMessage.Info($"Behaviour {name} asset loaded");
                return instance;
            }


            AgentBehaviour newAgBeh = new AgentBehaviour(stateSize, stackedInputs, continuousActions, discreteActions, architecture, numLayers, hidUnits);
            newAgBeh.behaviourName = name;
            newAgBeh.observationSize = stateSize;
            newAgBeh.stackedInputsNum = stackedInputs;
            newAgBeh.continuousDim = continuousActions;
            newAgBeh.discreteDim = discreteActions;
            newAgBeh.normalizer = new ZScoreNormalizer(stateSize);
            newAgBeh.assetCreated = true;



            // Create the asset
            if (!Directory.Exists($"Assets/{name}"))
                Directory.CreateDirectory($"Assets/{name}");
            AssetDatabase.CreateAsset(newAgBeh, $"Assets/{name}/{name}.asset");

            // Create aux assets
            newAgBeh.config = Hyperparameters.CreateOrLoadAsset(name);
            newAgBeh.critic.CreateAsset($"{name}/critic");

            if(newAgBeh.IsUsingContinuousActions)
            {
                newAgBeh.actorContinuousMu.CreateAsset($"{name}/actorContinuousMu");
                newAgBeh.actorContinuousSigma.CreateAsset($"{name}/actorContinuousSigma");
                newAgBeh.discriminatorContinuous.CreateAsset($"{name}/discriminatorContinuous");
            }
            
            if(newAgBeh.IsUsingDiscreteActions)
            {
                newAgBeh.actorDiscrete.CreateAsset($"{name}/actorDiscrete");
                newAgBeh.discriminatorDiscrete.CreateAsset($"{name}/discriminatorDiscrete");
            }

            return newAgBeh;
        }
        /// <summary>
        /// Updates the state of the Behaviour parameters.
        /// </summary>
        public void Save()
        {
            if(!assetCreated)
            {
                ConsoleMessage.Warning("Cannot save the Behaviour because it requires compilation first");
            }

            ConsoleMessage.Info($"Agent behaviour <b><i>{behaviourName}</i></b> autosaved");

            critic.Save(); 
            actorContinuousMu?.Save();
            actorContinuousSigma?.Save();
            discriminatorContinuous?.Save();
            actorDiscrete?.Save();
            discriminatorDiscrete?.Save();
        }
    }

    [CustomEditor(typeof(AgentBehaviour), true), CanEditMultipleObjects]
    sealed class CustomAgentBehaviourEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            List<string> dontDrawMe = new() { "m_Script" };

            AgentBehaviour script = (AgentBehaviour)target;

            if(!script.IsUsingContinuousActions)
            {
                dontDrawMe.Add("actorContinuousMu");
                dontDrawMe.Add("actorContinuousSigma");
                dontDrawMe.Add("discriminatorContinuous");
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

            if(!script.IsUsingDiscreteActions)
            {
                dontDrawMe.Add("actorDiscrete");
                dontDrawMe.Add("discriminatorDiscrete");
            }

            if(script.standardDeviationValue <= 0)
            {
                script.standardDeviationValue = 1f;
            }

            if(!script.normalizeObservations)
            {
                dontDrawMe.Add("normalizer");
            }

            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
}


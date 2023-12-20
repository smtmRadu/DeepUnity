using System;
using System.Collections.Generic;
using System.ComponentModel;
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
        [SerializeField, ReadOnly, Min(1)] public int stackedInputs;
        [SerializeField, ReadOnly] public int continuousDim;
        [SerializeField, ReadOnly] public int discreteDim;

        [Header("Hyperparameters")]
        [Tooltip("The scriptable object file containing the training hyperparameters.")]    
        [SerializeField] public Hyperparameters config;
        
        [Header("Critic")]
        [SerializeField] public NeuralNetwork vNetwork;
        [SerializeField] public NeuralNetwork q1Network;
        [SerializeField] public NeuralNetwork q2Network;
        
        [Space]

        [Header("Policy")]
        [SerializeField] public NeuralNetwork muNetwork;
        [SerializeField] public NeuralNetwork sigmaNetwork;
        [SerializeField] public NeuralNetwork discreteNetwork;
        [Space]

        [Header("Discriminator")]
        [Tooltip("Neural Network used for Behavioral Cloning")]
        [SerializeField] public NeuralNetwork discContNetwork;
        [Tooltip("Neural Network used for Behavioral Cloning")]
        [SerializeField] public NeuralNetwork discDiscNetwork;
        



        [ Header("Behaviour Configurations")]
        [SerializeField, Tooltip("The frames per second runned by the physics engine. [Time.fixedDeltaTime = 1 / targetFPS]")]
        [Range(30, 100)]
        public int targetFPS = 50;

        [SerializeField, Tooltip("Network forward progapation is runned on this device when the agents interfere with the environment. It is recommended to be kept on CPU." +
           " The best way to find the optimal device is to check the number of fps when running out multiple environments.")]
        public Device inferenceDevice = Device.CPU;

        [SerializeField, Tooltip("Network computation is runned on this device when training on batches. It is highly recommended to be set on GPU if it is available.")]
        public Device trainingDevice = Device.GPU;

        [SerializeField, Tooltip("Auto-normalize input observations and rewards for a stable training.")]
        public bool normalize = true;

        [ReadOnly, SerializeField, Tooltip("Observations normalizer.")] 
        public RunningNormalizer observationsNormalizer;

        [Range(1.5f, 3.5f), SerializeField, Tooltip("The observations are clipped [after normarlization] in range [-clip, clip]")]
        public float observationsClip = 3f;

        [Header("Standard Deviation for Continuous Actions")]
        [SerializeField, Tooltip("The standard deviation for Continuous Actions")] 
        public StandardDeviationType standardDeviation = StandardDeviationType.Trainable;
        [Tooltip("Modify this value to change the exploration/exploitation ratio.")]
        [SerializeField, Range(0.001f, 3f)] 
        public float standardDeviationValue = 1f;

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
        public LRScheduler dContScheduler{ get; private set; }
        public LRScheduler dDiscScheduler { get; private set; }

        public bool IsUsingContinuousActions { get => continuousDim > 0; }
        public bool IsUsingDiscreteActions { get => discreteDim > 0; }

        private AgentBehaviour(in int STATE_SIZE, in int STACKED_INPUTS, in int CONTINUOUS_ACTIONS_NUM, in int DISCRETE_ACTIONS_NUM, in int NUM_LAYERS, in int HIDDEN_UNITS)
        {
            static Activation HiddenActivation() => new Tanh();
            const InitType INIT_W = InitType.HE_Normal;
            const InitType INIT_B = InitType.Zeros;

            //------------------ NETWORK INITIALIZATION ----------------//

            // Initialize value network V(st)
            vNetwork = new NeuralNetwork(
               new IModule[] 
               { new Dense(STATE_SIZE * STACKED_INPUTS, HIDDEN_UNITS, INIT_W, INIT_B), HiddenActivation() }.
               Concat(CreateHiddenLayers(NUM_LAYERS, HIDDEN_UNITS, INIT_W, INIT_B)).
               Concat(new IModule[] { new Dense(HIDDEN_UNITS, 1, INIT_W, INIT_B) }).ToArray()
            );

            // Initialize pi
            if (CONTINUOUS_ACTIONS_NUM > 0)
            {
                muNetwork = new NeuralNetwork(
                        new IModule[]
                        { new Dense(STATE_SIZE * STACKED_INPUTS, HIDDEN_UNITS, INIT_W, INIT_B), HiddenActivation() }.
                            Concat(CreateHiddenLayers(NUM_LAYERS, HIDDEN_UNITS, INIT_W, INIT_B)).
                            Concat(new IModule[] { new Dense(HIDDEN_UNITS, CONTINUOUS_ACTIONS_NUM, INIT_W, INIT_B), new Tanh() }).ToArray()
                    );

                sigmaNetwork = new NeuralNetwork(
                           new IModule[]
                        { new Dense(STATE_SIZE * STACKED_INPUTS, HIDDEN_UNITS, INIT_W, INIT_B), HiddenActivation() }.
                            Concat(CreateHiddenLayers(NUM_LAYERS - 1, HIDDEN_UNITS, INIT_W, INIT_B)).
                            Concat(new IModule[] { new Dense(HIDDEN_UNITS, CONTINUOUS_ACTIONS_NUM, INIT_W, INIT_B), new Softplus(1.5f, 6f) }).ToArray() // softplus (1.2, 3.5)
                    );

                // Initialize q networks Q(st,at)
                q1Network = new NeuralNetwork(
                   new IModule[]
                   { new Dense((STATE_SIZE + CONTINUOUS_ACTIONS_NUM) * STACKED_INPUTS, HIDDEN_UNITS, INIT_W, INIT_B), HiddenActivation() }.
                   Concat(CreateHiddenLayers(NUM_LAYERS, HIDDEN_UNITS, INIT_W, INIT_B)).
                   Concat(new IModule[] { new Dense(HIDDEN_UNITS, 1, INIT_W, INIT_B) }).ToArray()
                );

                q2Network = new NeuralNetwork(
                   new IModule[]
                   { new Dense((STATE_SIZE + CONTINUOUS_ACTIONS_NUM) * STACKED_INPUTS, HIDDEN_UNITS, INIT_W, INIT_B), HiddenActivation() }.
                   Concat(CreateHiddenLayers(NUM_LAYERS, HIDDEN_UNITS, INIT_W, INIT_B)).
                   Concat(new IModule[] { new Dense(HIDDEN_UNITS, 1, INIT_W, INIT_B) }).ToArray()
                );

                discContNetwork = new NeuralNetwork(
                        new IModule[] 
                        { new Dense(CONTINUOUS_ACTIONS_NUM, HIDDEN_UNITS, INIT_W, INIT_B),HiddenActivation() }.
                            Concat(CreateHiddenLayers(NUM_LAYERS, HIDDEN_UNITS, INIT_W, INIT_B)).
                            Concat(new IModule[] { new Dense(HIDDEN_UNITS, 1, INIT_W, INIT_B), new Sigmoid() }).ToArray()
                    );
            }

            if (DISCRETE_ACTIONS_NUM > 0)
            {
                discreteNetwork = new NeuralNetwork(
                        new IModule[] 
                        { new Dense(STATE_SIZE * STACKED_INPUTS, HIDDEN_UNITS, INIT_W, INIT_B), HiddenActivation() }.
                            Concat(CreateHiddenLayers(NUM_LAYERS, HIDDEN_UNITS, INIT_W, INIT_B)).
                            Concat(new IModule[] { new Dense(HIDDEN_UNITS, DISCRETE_ACTIONS_NUM, INIT_W, INIT_B), new Softmax() }).ToArray()
                    );
                discDiscNetwork = new NeuralNetwork(
                        new IModule[] 
                        { new Dense(DISCRETE_ACTIONS_NUM, HIDDEN_UNITS, INIT_W, INIT_B), HiddenActivation() }.
                            Concat(CreateHiddenLayers(NUM_LAYERS, HIDDEN_UNITS, INIT_W, INIT_B)).
                            Concat(new IModule[] { new Dense(HIDDEN_UNITS, 1, INIT_W, INIT_B), new Sigmoid() }).ToArray()
                    );
            }         

            static IModule[] CreateHiddenLayers(int numLayers, int hidUnits, InitType INIT_W, InitType INIT_B)
            {
                if (numLayers == 1)
                    return new IModule[] { };
                else if (numLayers == 2)
                    return new IModule[] { new Dense(hidUnits, hidUnits, INIT_W, INIT_B), HiddenActivation() };
                else if (numLayers == 3)
                    return new IModule[] { new Dense(hidUnits, hidUnits, INIT_W, INIT_B), HiddenActivation(), 
                                           new Dense(hidUnits, hidUnits, INIT_W, INIT_B), HiddenActivation() };
                else
                    throw new ArgumentException("Unhandled numLayers outside range 1 - 3");

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
            else if(trainer == TrainerType.PPO)
            {
                vOptimizer = new Adam(vNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);

                if (IsUsingContinuousActions)
                {
                    muOptimizer = new Adam(muNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
                    sigmaOptimizer = new Adam(sigmaNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
                }
                
                if(IsUsingDiscreteActions)
                {
                    discreteOptimizer = new Adam(discreteNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
                }
            }
            else if(trainer == TrainerType.GAIL)
            {
                if (IsUsingContinuousActions)
                {
                    muOptimizer = new Adam(muNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda) ;
                    sigmaOptimizer = new Adam(sigmaNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
                    dContOptimizer = new Adam(discContNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
                }

                if (IsUsingDiscreteActions)
                {
                    discreteOptimizer = new Adam(discreteNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
                    dDiscOptimizer = new Adam(discDiscNetwork.Parameters(), hp.learningRate, eps: epsilon, weightDecay: lambda);
                }
            }                    
        }
        public void InitSchedulers(Hyperparameters hp, TrainerType trainer)
        {
            // LR 0 = initialLR * (gamma^n) => gamma = nth-root(initial_lr);

            int total_epochs = (int)hp.maxSteps / hp.bufferSize * hp.numEpoch;
            int step_size = 1;
            float gamma = Mathf.Pow(hp.learningRate, 1f / total_epochs);


            if (trainer == TrainerType.SAC)
            {
                vScheduler = new LRScheduler(vOptimizer, step_size, gamma);
                q1Scheduler = new LRScheduler(q1Optimizer, step_size, gamma);
                q2Scheduler = new LRScheduler(q1Optimizer, step_size, gamma);
                muScheduler = new LRScheduler(muOptimizer, step_size, gamma);
                sigmaScheduler = new LRScheduler(sigmaOptimizer, step_size, gamma);
            }
            else if (trainer == TrainerType.PPO)
            {
                vScheduler = new LRScheduler(vOptimizer, step_size, gamma);

                if (IsUsingContinuousActions)
                {
                    muScheduler = new LRScheduler(muOptimizer, step_size, gamma);
                    sigmaScheduler = new LRScheduler(sigmaOptimizer, step_size, gamma);
                }

                if (IsUsingDiscreteActions)
                {
                    discreteScheduler = new LRScheduler(discreteOptimizer, step_size, gamma);
                }
            }
            else if (trainer == TrainerType.GAIL)
            {
                if (IsUsingContinuousActions)
                { 
                    muScheduler = new LRScheduler(muOptimizer, step_size, gamma);
                    sigmaScheduler = new LRScheduler(sigmaOptimizer, step_size, gamma);
                    dContScheduler = new LRScheduler(dContOptimizer, step_size, gamma);
                }

                if (IsUsingDiscreteActions)
                {
                    discreteScheduler = new LRScheduler(discreteOptimizer, step_size, gamma);
                    discreteScheduler = new LRScheduler(dDiscOptimizer, step_size, gamma);   
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
                            sigmaNetwork.Predict(state) :
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
                           sigmaNetwork.Forward(stateBatch):
                           Tensor.Fill(standardDeviationValue, muBatch.Shape);
        }
        /// <summary>
        /// Input: <paramref name="stateBatch"/> - <em>s</em> | Tensor (<em>Batch Size, Observations</em>) <br></br>
        /// Output: <paramref name="muBatch"/> - <em>μ</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// Output: <paramref name="sigmaBatch"/> - <em>σ</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// </summary>
        public void ContinuousReparametrizedForward(Tensor statesBatch,  out Tensor muBatch, out Tensor sigmaBatch, out Tensor ksiBatch)
        {
            if (!IsUsingContinuousActions)
            {
                muBatch = null;
                sigmaBatch = null;
                ksiBatch = null;
                return;
            }

            muBatch = muNetwork.Forward(statesBatch);
            sigmaBatch = standardDeviation == StandardDeviationType.Trainable ?
                           sigmaNetwork.Forward(statesBatch) :
                           Tensor.Fill(standardDeviationValue, muBatch.Shape);

            ksiBatch = Tensor.RandomNormal(sigmaBatch.Shape);
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

        public Tensor Q1Forward(Tensor stateBatch, Tensor actionBatch)
        {
            int batch_size = stateBatch.Size(0);
            int state_size = stateBatch.Size(1);
            int action_size = actionBatch.Size(1);
            Tensor input = Tensor.Zeros(batch_size, state_size + action_size);
            for (int i = 0; i < batch_size; i++)
            {
                for (int f = 0; f < state_size; f++)
                {
                    input[i, f] = stateBatch[i, f];
                }
                for (int f = 0; f < action_size; f++)
                {
                    input[i, state_size + f] = actionBatch[i, f];
                }
            }
            return q1Network.Forward(input);           
        }
        public Tensor Q2Forward(Tensor stateBatch, Tensor actionBatch)
        {
            int batch_size = stateBatch.Size(0);
            int state_size = stateBatch.Size(1);
            int action_size = actionBatch.Size(1);
            Tensor input = Tensor.Zeros(batch_size, state_size + action_size);
            for (int i = 0; i < batch_size; i++)
            {
                for (int f = 0; f < state_size; f++)
                {
                    input[i, f] = stateBatch[i, f];
                }
                for (int f = 0; f < action_size; f++)
                {
                    input[i, state_size + f] = actionBatch[i, f];
                }
            }

            return q2Network.Forward(input);
        }



        /// <summary>
        /// Creates a new Agent behaviour folder containing all auxiliar neural networks, or loads it if already exists one for this behaviour.
        /// </summary>
        /// <returns></returns>
        public static AgentBehaviour CreateOrLoadAsset(string name, int stateSize, int stackedInputs, int continuousActions, int discreteActions, int numLayers, int hidUnits)
        {          
            var instance = AssetDatabase.LoadAssetAtPath<AgentBehaviour>($"Assets/{name}/{name}.asset");

            if (instance != null)
            {
                ConsoleMessage.Info($"Behaviour {name} asset loaded");
                return instance;
            }


            AgentBehaviour newAgBeh = new AgentBehaviour(stateSize, stackedInputs, continuousActions, discreteActions, numLayers, hidUnits);
            newAgBeh.behaviourName = name;
            newAgBeh.observationSize = stateSize;
            newAgBeh.stackedInputs = stackedInputs;
            newAgBeh.continuousDim = continuousActions;
            newAgBeh.discreteDim = discreteActions;
            newAgBeh.observationsNormalizer = new RunningNormalizer(stateSize * stackedInputs);
            newAgBeh.assetCreated = true;



            // Create the asset
            if (!Directory.Exists($"Assets/{name}"))
                Directory.CreateDirectory($"Assets/{name}");
            AssetDatabase.CreateAsset(newAgBeh, $"Assets/{name}/_{name}.asset");

            // Create aux assets
            newAgBeh.config = Hyperparameters.CreateOrLoadAsset(name);
            newAgBeh.vNetwork?.CreateAsset($"{name}/V");
            newAgBeh.muNetwork?.CreateAsset($"{name}/Mu");
            newAgBeh.sigmaNetwork?.CreateAsset($"{name}/Sigma");
            newAgBeh.discContNetwork?.CreateAsset($"{name}/Disc_Cont");
            newAgBeh.q1Network?.CreateAsset($"{name}/Q1");
            newAgBeh.q2Network?.CreateAsset($"{name}/Q2");
            newAgBeh.discreteNetwork?.CreateAsset($"{name}/Discrete");
            newAgBeh.discDiscNetwork?.CreateAsset($"{name}/Disc_Disc");
            

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

            vNetwork?.Save();
            q1Network?.Save();
            q2Network?.Save();  
            muNetwork?.Save();
            sigmaNetwork?.Save();
            discreteNetwork?.Save();
            discContNetwork?.Save();
            discDiscNetwork?.Save();
        }
        /// <summary>
        /// Before using, checks if config file or neural networks are not attached to this scriptable object.
        /// </summary>
        /// <returns></returns>
        public List<string> CheckForMissingAssets()
        {
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
            else if (trainer == TrainerType.GAIL)
            {
                if (IsUsingContinuousActions)
                {
                    if (!muNetwork)
                        whatIsMissing.Add("Mu Network");

                    if (!sigmaNetwork)
                        whatIsMissing.Add("Sigma Network");

                    if (!discContNetwork)
                        whatIsMissing.Add("Discriminator Continuous Network");
                }

                if (IsUsingDiscreteActions)
                {
                    if (!discreteNetwork)
                        whatIsMissing.Add("Discrete Network");

                    if (!discDiscNetwork)
                        whatIsMissing.Add("Discriminator Discrete Network");
                }
            }

            return whatIsMissing;
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

            // See or not standard deviation
            if (!script.IsUsingContinuousActions)
            {
                dontDrawMe.Add("standardDeviation");
                dontDrawMe.Add("standardDeviationValue");
            }
            else
            {
                if (script.standardDeviation == StandardDeviationType.Trainable)
                {
                    dontDrawMe.Add("standardDeviationValue");
                }
            }



            if (script.standardDeviationValue <= 0)
            {
                script.standardDeviationValue = 1f;
            }

            if(!script.normalize)
            {
                dontDrawMe.Add("observationsNormalizer");
            }

            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
}


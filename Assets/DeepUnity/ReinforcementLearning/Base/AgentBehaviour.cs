using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;


namespace DeepUnity
{
    /// <summary>
    /// AgentBehaviour asset, along with the entire folder of networks is recommended to be kept inside main Assets folder,
    /// until the agent is completely finished.
    /// </summary>
    [Serializable]
    public class AgentBehaviour : ScriptableObject
    {

        [Header("Behaviour Properties")]
        [SerializeField, ReadOnly] public string behaviourName;
        [SerializeField, HideInInspector] private bool assetCreated = false;
        [SerializeField, ReadOnly] public int observationSize;
        [SerializeField, ReadOnly] public int continuousDim;
        [SerializeField, ReadOnly] public int[] discreteBranches;

        [Header("Neural Networks")]
        [SerializeField] public NeuralNetwork critic;
        [SerializeField] public NeuralNetwork actorMu;
        [SerializeField] public NeuralNetwork actorSigma;
        [SerializeField] public NeuralNetwork[] actorDiscretes;




        [Space(25), Header("Behaviour Configurations")]
        [SerializeField, Tooltip("The frames per second runned by the physics engine. Time.fixedDeltaTime = 1 / targetFPS")]
        [Range(30, 100)]
        public int targetFPS = 50;

        [Space]
        [SerializeField, Tooltip("Auto-normalize input observations.")]
        public bool normalizeObservations = false;
        [ReadOnly, SerializeField, Tooltip("Observations normalizer data.")] 
        public ZScoreNormalizer normalizer;

        [Space]
        [SerializeField, Tooltip("The standard deviation for Continuous Actions")] 
        public StandardDeviationType standardDeviation = StandardDeviationType.Fixed;
        [Tooltip("Modify this value to change the exploration/exploitation ratio.")]
        [SerializeField, Range(0.001f, 3f)] 
        public float standardDeviationValue = 1.5f;
        [Tooltip("Sigma network's output is multiplied by this number. Modify this value to change the exploration/exploitation ratio.")]
        [SerializeField, Range(0.1f, 3f)]
        public float standardDeviationScale = 1f;

        [Space]
        [SerializeField, Tooltip("Network forward progapation is runned on this device when the agents interfere with the environment. It is recommended to be kept on CPU." +
           " The best way to find the optimal device is to check the number of fps when running out multiple environments.")]
        public Device inferenceDevice = Device.CPU;
        [SerializeField, Tooltip("Network computation is runned on this device when training on batches. It is highly recommended to be set on GPU if it is available.")]
        public Device trainingDevice = Device.GPU;

        

        public Optimizer criticOptimizer { get; private set; }
        public Optimizer actorMuOptimizer { get; private set; }
        public Optimizer actorSigmaOptimizer { get; private set; }
        public Optimizer[] actorDiscretesOptimizers { get; private set; }

        public LRScheduler criticScheduler { get; private set; }
        public LRScheduler actorMuScheduler { get; private set; }
        public LRScheduler actorSigmaScheduler { get; private set; }
        public LRScheduler[] actorDiscretesSchedulers { get; private set; }

        public bool IsUsingContinuousActions { get => continuousDim > 0; }
        public bool IsUsingDiscreteActions { get => discreteBranches != null && discreteBranches.Length > 0; }


        private AgentBehaviour(string behaviourName, int stateSize, int continuousActions, int[] discreteBranches)
        {
            this.behaviourName = behaviourName;
            this.observationSize = stateSize;
            this.continuousDim = continuousActions;
            this.discreteBranches = discreteBranches;
            normalizer = new ZScoreNormalizer(stateSize);
            assetCreated = true;

            //------------------ NETWORK INITIALIZATION ----------------//
            // const int H_128 = 128;
            const int H_64 = 64;
            // const int H_32 = 32;
            const InitType INIT_W = InitType.LeCun_Uniform;
            const InitType INIT_B = InitType.LeCun_Uniform;
            const Device dev = Device.CPU;

            critic = new NeuralNetwork(
                new Dense(stateSize, H_64, INIT_W, INIT_B, device: dev),
                new ReLU(),
                new Dense(H_64, H_64, INIT_W, INIT_B, device: dev),
                new ReLU(),
                new Dense(H_64, 1, INIT_W, INIT_B, device: dev));

            if(IsUsingContinuousActions)
            {
                actorMu = new NeuralNetwork(
                new Dense(stateSize, H_64, INIT_W, INIT_B, device: dev),
                new ReLU(),
                new Dense(H_64, H_64, INIT_W, INIT_B, device: dev),
                new ReLU(),
                new Dense(H_64, continuousActions, INIT_W, INIT_B, device: dev),
                new Tanh());

                actorSigma = new NeuralNetwork(
                    new Dense(stateSize, H_64, INIT_W, INIT_B, device: dev),
                    new ReLU(),
                    new Dense(H_64, continuousActions, INIT_W, INIT_B, device: dev),
                    new Exp()); // Softplus()
            }
            if(IsUsingDiscreteActions)
            {
                actorDiscretes = new NeuralNetwork[discreteBranches.Length];
                for (int i = 0; i < actorDiscretes.Length; i++)
                {
                    if (discreteBranches[i] < 2)
                    {
                        ConsoleMessage.Warning("Any discrete branch's value must be greater than 1.");
                        return;
                    }

                    actorDiscretes[i] = new NeuralNetwork(
                        new Dense(stateSize, H_64, INIT_W, INIT_B, device: dev),
                        new ReLU(),
                        new Dense(H_64, H_64, INIT_W, INIT_B, device: dev),
                        new ReLU(),
                        new Dense(H_64, discreteBranches[i], INIT_W, INIT_B, device: dev),
                        new Softmax());
                }
            }

            
            //----------------------------------------------------------//
        }

        public void SetActorDevice(Device device)
        {
            if (IsUsingContinuousActions)
            { 
                var learnables = actorMu.Parameters();
                foreach (var item in learnables)
                {
                    item.device = device;
                }

                learnables = actorSigma.Parameters();
                foreach (var item in learnables)
                {
                    item.device = device;
                }
            }

            if (IsUsingDiscreteActions)
            {
                for (int i = 0; i < actorDiscretes.Length; i++)
                {
                    var learnables = actorDiscretes[i].Parameters();
                    foreach (var item in learnables)
                    {
                        item.device = device;
                    }
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

            if (critic == null)
            {
                ConsoleMessage.Warning($"Some network assets are not attached to {behaviourName} behaviour asset!");
                EditorApplication.isPlaying = false;
                return;
            }

            criticOptimizer = new Adam(critic.Parameters(), hp.learningRate);  
            
            if(IsUsingContinuousActions)
            {
                actorMuOptimizer = new Adam(actorMu.Parameters(), hp.learningRate);
                actorSigmaOptimizer = new Adam(actorSigma.Parameters(), hp.learningRate);
            }
           
            if(IsUsingDiscreteActions)
            {
                actorDiscretesOptimizers = new Optimizer[discreteBranches == null ? 0 : discreteBranches.Length];
                for (int i = 0; i < actorDiscretes.Length; i++)
                {
                    actorDiscretesOptimizers[i] = new Adam(actorDiscretes[i].Parameters(), hp.learningRate);
                }
            }
        }
        public void InitSchedulers(Hyperparameters hp)
        {
            if (criticScheduler != null)
                return;

            if (critic == null)
            {
                ConsoleMessage.Warning($"Some network assets are not attached to {behaviourName} behaviour asset!");
                EditorApplication.isPlaying = false;
                return;
            }

            int step_size = hp.learningRateSchedule ? hp.schedulerStepSize : 1_000_000;
            float gamma = hp.learningRateSchedule ? hp.schedulerDecay : 1f;

            criticScheduler = new LRScheduler(criticOptimizer, step_size, gamma);

            if (IsUsingContinuousActions)
            {
                actorMuScheduler = new LRScheduler(actorMuOptimizer, step_size, gamma);
                actorSigmaScheduler = new LRScheduler(actorSigmaOptimizer, step_size, gamma);
            }
          
            if(IsUsingDiscreteActions)
            {
                actorDiscretesSchedulers = new LRScheduler[discreteBranches == null ? 0 : discreteBranches.Length];
                for (int i = 0; i < actorDiscretes.Length; i++)
                {
                    actorDiscretesSchedulers[i] = new LRScheduler(actorDiscretesOptimizers[i], step_size, gamma);
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
            if (actorMu == null)
            {
                ConsoleMessage.Warning($"Some network assets are not attached to {behaviourName} behaviour asset!");
                EditorApplication.isPlaying = false;
                action = null;
                probs = null;
                return;
            }

            Tensor mu = actorMu.Predict(state);
            Tensor sigma = standardDeviation == StandardDeviationType.Trainable ?
                            actorSigma.Predict(state) * standardDeviationScale :
                            Tensor.Fill(standardDeviationValue, mu.Shape);
            action = mu.Zip(sigma, (x, y) => Utils.Random.Gaussian(x, y));
            probs = Tensor.Probability(action, mu, sigma);
        }
        /// <summary>
        /// Input: <paramref name="states"/> - <em>s</em> | Tensor (<em>Batch Size, Observations</em>) <br></br>
        /// Output: <paramref name="mus"/> - <em>μ</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// OutputL <paramref name="sigmas"/> - <em>σ</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// </summary>
        public void ContinuousForward(Tensor states, out Tensor mus, out Tensor sigmas)
        {
            if (!IsUsingContinuousActions)
            {
                mus = null;
                sigmas = null;
                return;
            }

            mus = actorMu.Forward(states);
            sigmas = standardDeviation == StandardDeviationType.Trainable ?
                           actorSigma.Forward(states) * standardDeviationScale :
                           Tensor.Fill(standardDeviationValue, mus.Shape);
        }
        public void DiscretePredict(Tensor state, out Tensor action, out Tensor probs)
        {
            if (!IsUsingDiscreteActions)
            {
                action = null;
                probs = null;
                return;
            }

            if (actorDiscretes[0] == null)
            {
                ConsoleMessage.Warning($"Some network assets are not attached to {behaviourName} behaviour asset!");
                EditorApplication.isPlaying = false;
                action = null;
                probs = null;
                return;
            }

            action = null;
            probs = null;
        }
        public void DiscreteForward(Tensor statesBatch, out Tensor probs)
        {
            if (!IsUsingDiscreteActions)
            {
                probs = null;
                return;
            }

            probs = null;
        }




        /// <summary>
        /// Creates a new Agent behaviour folder containing all auxiliar neural networks, or loads it if already exists one for this behaviour.
        /// </summary>
        /// <returns></returns>
        public static AgentBehaviour CreateOrLoadAsset(string name, int stateSize, int continuousActions, int[] discreteActions)
        {          
            var instance = AssetDatabase.LoadAssetAtPath<AgentBehaviour>($"Assets/{name}/{name}.asset");

            if (instance != null)
            {
                ConsoleMessage.Info($"Behaviour {name} asset loaded.");
                return instance;
            }
                

            AgentBehaviour newAgBeh = new AgentBehaviour(name, stateSize, continuousActions, discreteActions);

            // Create the asset
            if (!Directory.Exists($"Assets/{name}"))
                Directory.CreateDirectory($"Assets/{name}");
            AssetDatabase.CreateAsset(newAgBeh, $"Assets/{name}/{name}.asset");
            AssetDatabase.SaveAssets();

            // Create aux assets
            newAgBeh.critic.CreateAsset($"{name}/critic");

            if(newAgBeh.IsUsingContinuousActions)
            {
                newAgBeh.actorMu.CreateAsset($"{name}/actorMu");
                newAgBeh.actorSigma.CreateAsset($"{name}/actorSigma");
            }
            
            if(newAgBeh.IsUsingDiscreteActions)
            {
                for (int i = 0; i < newAgBeh.actorDiscretes.Length; i++)
                {
                    newAgBeh.actorDiscretes[i].CreateAsset($"{name}/actorDiscrete{i}");
                }
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
                ConsoleMessage.Warning("Cannot save the Behaviour because it requires compilation first.");
            }

            ConsoleMessage.Info($"Agent behaviour <b><i>{behaviourName}</i></b> autosaved.");

            critic.Save();

            if(IsUsingContinuousActions)
            {
                actorMu.Save();
                actorSigma.Save();
            }
           
            if(IsUsingDiscreteActions)
                for (int i = 0; i < actorDiscretes.Length; i++)
                {
                    actorDiscretes[i].Save();
                }

        }
    }

    [CustomEditor(typeof(AgentBehaviour), true), CanEditMultipleObjects]
    sealed class CustomAgentBehaviourEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            List<string> dontDrawMe = new() { "m_Script" };

            SerializedProperty sd = serializedObject.FindProperty("standardDeviation");

            if(sd.enumValueIndex == (int)StandardDeviationType.Trainable)
            {
                dontDrawMe.Add("standardDeviationValue");
            }
            else
            {
                dontDrawMe.Add("standardDeviationScale");
            }

            SerializedProperty fixedSd = serializedObject.FindProperty("standardDeviationValue");
            if (fixedSd.floatValue <= 0f)
            {
                ConsoleMessage.Warning("Standard deviation is 0. Please make it positive to avoid erros!");
                EditorGUILayout.HelpBox("Standard deviation is 0. Please make it positive to avoid errors!", MessageType.Warning);
            }

            SerializedProperty norm = serializedObject.FindProperty("normalizeObservations");
            if(!norm.boolValue)
            {
                dontDrawMe.Add("normalizer");
            }

            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
}


using System;
using System.Collections.Generic;
using System.IO;
using System.Security.AccessControl;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class AgentBehaviour : ScriptableObject
    {
        [SerializeField] public string behaviourName;
        [SerializeField] public int observationSize;
        [SerializeField] public int continuousDim;
        [SerializeField] public int[] discreteBranches;

        [Header("Standard Deviation for Continuous Actions")]
        [SerializeField] public StandardDeviationType standardDeviation = StandardDeviationType.Fixed;
        [SerializeField, Min(0.001f)] public float fixedStandardDeviationValue = 1f;

        [Header("Normalization")]
        [SerializeField] public bool normalizeObservations = false;
        [ReadOnly, SerializeField] public ZScoreNormalizer normalizer;

        [Header("Neural Networks")]
        [SerializeField] public Sequential critic;
        [SerializeField] public Sequential actorMu;
        [SerializeField] public Sequential actorSigma;
        [SerializeField] public Sequential[] actorDiscretes;

        public Optimizer criticOptimizer { get; private set; }
        public Optimizer actorMuOptimizer { get; private set; }
        public Optimizer actorSigmaOptimizer { get; private set; }
        public Optimizer[] actorDiscretesOptimizers { get; private set; }

        public LRScheduler criticScheduler { get; private set; }
        public LRScheduler actorMuScheduler { get; private set; }
        public LRScheduler actorSigmaScheduler { get; private set; }
        public LRScheduler[] actorDiscretesScheduler { get; private set; }

        private void Awake()
        {
            if (critic == null)
            {
                Debug.Log($"<color=#fcba03>Warning! Some network assets are not attached to {behaviourName} behaviour asset! </color>");
                EditorApplication.isPlaying = false;
                return;
            }

        }
        private AgentBehaviour(string behaviourName, int stateSize, int continuousActions, int[] discreteBranches)
        {
            this.behaviourName = behaviourName;
            this.observationSize = stateSize;
            this.continuousDim = continuousActions;
            normalizer = new ZScoreNormalizer(stateSize);


            //------------------ NETWORK INITIALIZATION ----------------//
            const int H = 64;
            const InitType INIT_W = InitType.LeCun_Uniform;
            const InitType INIT_B = InitType.LeCun_Uniform;

            critic = new Sequential(
                new Dense(stateSize, H, INIT_W, INIT_B),
                new ReLU(),
                new Dense(H, H, INIT_W, INIT_B, device: Device.GPU),
                new ReLU(),
                new Dense(H, 1, INIT_W, INIT_B));

            actorMu = new Sequential(
                new Dense(stateSize, H, INIT_W, INIT_B),
                new ReLU(),
                new Dense(H, H, INIT_W, INIT_B, device: Device.GPU),
                new ReLU(),
                new Dense(H, continuousActions, INIT_W, INIT_B),
                new Tanh());

            actorSigma = new Sequential(
                new Dense(stateSize, H, INIT_W, INIT_B),
                new ReLU(),
                new Dense(H, H, INIT_W, INIT_B, device: Device.GPU),
                new ReLU(),
                new Dense(H, continuousActions, INIT_W, INIT_B),
                new Softplus());

            actorDiscretes = new Sequential[discreteBranches.Length];
            for (int i = 0; i < actorDiscretes.Length; i++)
            {
                actorDiscretes[i] = new Sequential(
                    new Dense(stateSize, H, INIT_W, INIT_B),
                    new ReLU(),
                    new Dense(H, H, INIT_W, INIT_B, device: Device.GPU),
                    new ReLU(),
                    new Dense(H, discreteBranches[i], INIT_W, INIT_B),
                    new Softmax());
            }
            //----------------------------------------------------------//
        }
        public void InitOptimisers(Hyperparameters hp)
        {
            if (criticOptimizer != null)
                return;

            if (critic == null)
            {
                Debug.Log($"<color=#fcba03>Warning! Some network assets are not attached to {behaviourName} behaviour asset! </color>");
                EditorApplication.isPlaying = false;
                return;
            }

            criticOptimizer = new Adam(critic.Parameters(), hp.learningRate);          
            actorMuOptimizer = new Adam(actorMu.Parameters(), hp.learningRate);            
            actorSigmaOptimizer = new Adam(actorSigma.Parameters(), hp.learningRate);

            actorDiscretesOptimizers = new Optimizer[discreteBranches == null? 0 : discreteBranches.Length];
            for (int i = 0; i < actorDiscretes.Length; i++)
            {
                actorDiscretesOptimizers[i] = new Adam(actorDiscretes[i].Parameters(), hp.learningRate);
            }

        }
        public void InitSchedulers(Hyperparameters hp)
        {
            if (criticScheduler != null)
                return;

            if (critic == null)
            {
                Debug.Log($"<color=#fcba03>Warning! Some network assets are not attached to {behaviourName} behaviour asset! </color>");
                EditorApplication.isPlaying = false;
                return;
            }

            int step_size = hp.learningRateSchedule ? hp.schedulerStepSize : 1_000_000;
            float gamma = hp.learningRateSchedule ? hp.schedulerDecay : 1f;

            criticScheduler = new LRScheduler(criticOptimizer, step_size, gamma);
            actorMuScheduler = new LRScheduler(actorMuOptimizer, step_size, gamma);
            actorSigmaScheduler = new LRScheduler(actorSigmaOptimizer, step_size, gamma);

            actorDiscretesScheduler = new LRScheduler[discreteBranches == null ? 0 : discreteBranches.Length];
            for (int i = 0; i < actorDiscretes.Length; i++)
            {
                actorDiscretesScheduler[i] = new LRScheduler(actorDiscretesOptimizers[i], step_size, gamma);
            }

        }



        /// <summary>
        /// Input: <paramref name="state"/> - <em>sₜ</em> | Tensor (<em>Observations</em>) <br></br>
        /// Output: <paramref name="value"/> - <em>Vtarget</em> | Tensor (<em>1</em>)
        /// </summary>
        public void ValuePredict(Tensor state, out Tensor value) 
        {
            value = critic.Predict(state);
        }
        /// <summary>
        /// Input: <paramref name="state"/> - <em>sₜ</em> | Tensor (<em>Observations</em>) <br></br>
        /// Output: <paramref name="action"/> - <em>aₜ</em> |  Tensor (<em>Continuous Actions</em>) <br></br>
        /// Extra Output: <paramref name="log_probs"/> - <em>log πθ(aₜ|sₜ)</em> | Tensor (<em>Continuous Actions</em>)
        /// </summary>
        public void ContinuousPredict(Tensor state, out Tensor action, out Tensor log_probs)
        {
            Tensor mu = actorMu.Predict(state);
            Tensor sigma = standardDeviation == StandardDeviationType.Trainable ?
                            actorSigma.Predict(state) :
                            Tensor.Fill(fixedStandardDeviationValue, mu.Shape);
            action = mu.Zip(sigma, (x, y) => Utils.Random.Gaussian(x, y));
            log_probs = Tensor.LogProbability(action, mu, sigma);
        }
        /// <summary>
        /// Input: <paramref name="states"/> - <em>s</em> | Tensor (<em>Batch Size, Observations</em>) <br></br>
        /// Output: <paramref name="mu"/> - <em>μ</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// OutputL <paramref name="sigma"/> - <em>σ</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// </summary>
        public void ContinuousForward(Tensor states, out Tensor mu, out Tensor sigma)
        {
            mu = actorMu.Forward(states);
            sigma = standardDeviation == StandardDeviationType.Trainable ?
                           actorSigma.Predict(states) :
                           Tensor.Fill(fixedStandardDeviationValue, mu.Shape);
        }
        public void DiscretePredict(Tensor state, out Tensor action, out Tensor probabilities)
        {
            probabilities = null;
            action = null;
        }
        public void DiscreteForward(Tensor statesBatch, out Tensor logProbs)
        {
            logProbs = null;
        }




        /// <summary>
        /// Creates a new Agent behaviour folder containing all auxiliar neural networks, or loads it if already exists one for this behaviour.
        /// </summary>
        /// <returns></returns>
        public static AgentBehaviour CreateOrLoadAsset(string name, int stateSize, int continuousActions, int[] discreteActions)
        {          
            var instance = AssetDatabase.LoadAssetAtPath<AgentBehaviour>($"Assets/{name}/{name}.asset");

            if (instance != null)
                return instance;

            AgentBehaviour newAgBeh = new AgentBehaviour(name, stateSize, continuousActions, discreteActions);

            // Create the asset
            if (!Directory.Exists($"Assets/{name}"))
                Directory.CreateDirectory($"Assets/{name}");
            AssetDatabase.CreateAsset(newAgBeh, $"Assets/{name}/{name}.asset");
            AssetDatabase.SaveAssets();

            // Create aux assets
            newAgBeh.critic.CreateAsset($"{name}/critic");
            newAgBeh.actorMu.CreateAsset($"{name}/actorMu");
            newAgBeh.actorSigma.CreateAsset($"{name}/actorSigma");

            for (int i = 0; i < newAgBeh.actorDiscretes.Length; i++)
            {
                newAgBeh.actorDiscretes[i].CreateAsset($"{name}/actorDiscrete{i}");
            }

            return newAgBeh;
        }
        /// <summary>
        /// Updates the state of the Behaviour parameters.
        /// </summary>
        public void Save()
        {
            var instance = AssetDatabase.LoadAssetAtPath<AgentBehaviour>($"Assets/{behaviourName}/{behaviourName}.asset");

            if (instance == null)
                throw new InvalidOperationException("Cannot save the Behaviour because it requires compilation first.");

            Debug.Log($"<color=#03a9fc>Agent behaviour <b><i>{behaviourName}</i></b> autosaved.</color>");

            critic.Save();
            actorMu.Save();
            actorSigma.Save();

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
            List<string> dontDrawMe = new() { "m_Script" };

            SerializedProperty sd = serializedObject.FindProperty("standardDeviation");

            if(sd.enumValueIndex == (int)StandardDeviationType.Trainable)
            {
                dontDrawMe.Add("fixedStandardDeviationValue");
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


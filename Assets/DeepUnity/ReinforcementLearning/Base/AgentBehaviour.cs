using System;
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

        [Header("Normalizers")]
        [SerializeField] public ZScoreNormalizer stateNormalizer;

        [Header("Neural Networks")]
        [SerializeField] public Sequential critic;
        [SerializeField] public Sequential muHead;
        [SerializeField] public Sequential sigmaHead;
        [SerializeField] public Sequential[] discreteHeads;

        public Optimizer criticOptimizer { get; private set; }
        public Optimizer muHeadOptimizer { get; private set; }
        public Optimizer sigmaHeadOptimizer { get; private set; }
        public Optimizer[] discreteHeadsOptimizers { get; private set; }

        public LRScheduler criticScheduler { get; private set; }
        public LRScheduler muHeadScheduler { get; private set; }
        public LRScheduler sigmaHeadScheduler { get; private set; }
        public LRScheduler[] discreteHeadsSchedulers { get; private set; }




        private AgentBehaviour(string behaviourName, int stateSize, int continuousActions, int[] discreteBranches)
        {
            this.behaviourName = behaviourName;
            this.observationSize = stateSize;
            this.continuousDim = continuousActions;
            stateNormalizer = new ZScoreNormalizer(stateSize);


            //------------------ NETWORK INITIALIZATION ----------------//
            int H = 64;

            critic = new Sequential(
                new Dense(stateSize, H, InitType.HE_Normal, InitType.HE_Normal),
                new ReLU(),
                new Dense(H, H, InitType.HE_Normal, InitType.HE_Normal, device: Device.GPU),
                new ReLU(),
                new Dense(H, 1, InitType.HE_Normal, InitType.HE_Normal));

            muHead = new Sequential(
                new Dense(stateSize, H, InitType.HE_Normal, InitType.HE_Normal),
                new ReLU(),
                new Dense(H, H, InitType.HE_Normal, InitType.HE_Normal, device: Device.GPU),
                new ReLU(),
                new Dense(H, continuousActions, InitType.HE_Normal, InitType.HE_Normal),
                new Tanh());

            sigmaHead = new Sequential(
                new Dense(stateSize, H, InitType.HE_Normal, InitType.HE_Normal),
                new ReLU(),
                new Dense(H, H, InitType.HE_Normal, InitType.HE_Normal, device: Device.GPU),
                new ReLU(),
                new Dense(H, continuousActions, InitType.HE_Normal, InitType.HE_Normal),
                new Softplus());

            discreteHeads = new Sequential[discreteBranches.Length];
            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeads[i] = new Sequential(
                    new Dense(stateSize, H, InitType.HE_Normal, InitType.HE_Normal),
                    new ReLU(),
                    new Dense(H, H, InitType.HE_Normal, InitType.HE_Normal, device: Device.GPU),
                    new ReLU(),
                    new Dense(H, discreteBranches[i], InitType.HE_Normal, InitType.HE_Normal),
                    new Softmax());
            }
            //----------------------------------------------------------//
        }
        public void InitOptimisers(Hyperparameters hp)
        {
            if (criticOptimizer != null)
                return;

            if (critic == null)
                throw new Exception($"Networks were not assigned to the {behaviourName} behaviour asset.");

            criticOptimizer = new Adam(critic.Parameters(), hp.learningRate);          
            muHeadOptimizer = new Adam(muHead.Parameters(), hp.learningRate);            
            sigmaHeadOptimizer = new Adam(sigmaHead.Parameters(), hp.learningRate);

            discreteHeadsOptimizers = new Optimizer[discreteBranches == null? 0 : discreteBranches.Length];
            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeadsOptimizers[i] = new Adam(discreteHeads[i].Parameters(), hp.learningRate);
            }

        }
        public void InitSchedulers(Hyperparameters hp)
        {
            if (criticScheduler != null)
                return;


            if (critic == null)
                throw new Exception($"Networks were not assigned to the {behaviourName} behaviour asset.");

            int step_size = hp.learningRateSchedule ? 10 : 1000;
            float gamma = hp.learningRateSchedule ? 0.99f : 1f;

            criticScheduler = new LRScheduler(criticOptimizer, step_size, gamma);
            muHeadScheduler = new LRScheduler(muHeadOptimizer, step_size, gamma);
            sigmaHeadScheduler = new LRScheduler(sigmaHeadOptimizer, step_size, gamma);

            discreteHeadsSchedulers = new LRScheduler[discreteBranches == null ? 0 : discreteBranches.Length];
            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeadsSchedulers[i] = new LRScheduler(discreteHeadsOptimizers[i], step_size, gamma);
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
        /// Extra Output: <paramref name="probabilities"/> - <em>πθ(aₜ|sₜ)</em> | Tensor (<em>Continuous Actions</em>)
        /// </summary>
        public void ContinuousPredict(Tensor state, out Tensor action, out Tensor probabilities)
        {
            Tensor mu = muHead.Predict(state);
            Tensor sigma = Tensor.Fill(0.1f, mu.Shape);
            action = mu.Zip(sigma, (x, y) => Utils.Random.Gaussian(x, y));
            probabilities = Tensor.PDF(action, mu, sigma);
        }
        /// <summary>
        /// Input: <paramref name="statesBatch"/> - <em>s</em> | Tensor (<em>Batch Size, Observations</em>) <br></br>
        /// Output: <paramref name="mu"/> - <em>μ</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// OutputL <paramref name="sigma"/> - <em>σ</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// </summary>
        public void ContinuousForward(Tensor statesBatch, out Tensor mu, out Tensor sigma)
        {
            mu = muHead.Forward(statesBatch);
            sigma = Tensor.Fill(0.1f, mu.Shape);
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
            newAgBeh.muHead.CreateAsset($"{name}/mu");
            newAgBeh.sigmaHead.CreateAsset($"{name}/sigma");

            for (int i = 0; i < newAgBeh.discreteHeads.Length; i++)
            {
                newAgBeh.discreteHeads[i].CreateAsset($"{name}/discrete{i}");
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
            muHead.Save();
            sigmaHead.Save();

            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeads[i].Save();
            }

        }
    }
}


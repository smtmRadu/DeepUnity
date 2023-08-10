using System;
using System.IO;
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

        [SerializeField] public ZScoreNormalizer stateStandardizer;
        [SerializeField] public ZScoreNormalizer rewardNormalizer;  
 
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


        public static readonly int default_step_size_StepLR = 10;
        public static readonly float default_gamma_StepLR = 0.99f;
        public static readonly (float, float) sigma_clip = (0.001f, 5f);

        public AgentBehaviour(int stateSize, int continuousActions, int[] discreteBranches)
        {
            this.behaviourName = name;
            this.observationSize = stateSize;
            this.continuousDim = continuousActions;

            stateStandardizer = new ZScoreNormalizer(stateSize, true);// Normalizer.Create(stateSize, normalization);
            rewardNormalizer = null; // new ZScoreNormalizer(1, true);// Normalizer.Create(1, normalization);

            int hiddenUnits = 64;

            critic = new Sequential(
                new Dense(stateSize, hiddenUnits),
                new ReLU(),
                new Dense(hiddenUnits, hiddenUnits, device: Device.GPU),
                new ReLU(),
                new Dense(hiddenUnits, 1));

            muHead = new Sequential(
                new Dense(stateSize, hiddenUnits),
                new ReLU(),
                new Dense(hiddenUnits, hiddenUnits, device: Device.GPU),
                new ReLU(),
                new Dense(hiddenUnits, continuousActions),
                new Tanh());

            sigmaHead = new Sequential(
                new Dense(stateSize, hiddenUnits),
                new ReLU(),
                new Dense(hiddenUnits, hiddenUnits, device: Device.GPU),
                new ReLU(),
                new Dense(hiddenUnits, continuousActions),
                new Softplus());

            discreteHeads = new Sequential[discreteBranches.Length];
            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeads[i] = new Sequential(
                    new Dense(stateSize, hiddenUnits),
                    new ReLU(),
                    new Dense(hiddenUnits, hiddenUnits, device: Device.GPU),
                    new ReLU(),
                    new Dense(hiddenUnits, discreteBranches[i]),
                    new Softmax());
            }
        }
        public void InitOptimisers(HyperParameters hp)
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
        public void InitSchedulers(HyperParameters hp)
        {
            if (criticScheduler != null)
                return;


            if (critic == null)
                throw new Exception($"Networks were not assigned to the {behaviourName} behaviour asset.");

            int step_size = hp.learningRateSchedule ? default_step_size_StepLR : 1000000;
            float gamma = hp.learningRateSchedule ? default_gamma_StepLR : 1f;

            criticScheduler = new LRScheduler(criticOptimizer, step_size, gamma);
            muHeadScheduler = new LRScheduler(muHeadOptimizer, step_size, gamma);
            sigmaHeadScheduler = new LRScheduler(sigmaHeadOptimizer, step_size, gamma);

            discreteHeadsSchedulers = new LRScheduler[discreteBranches == null ? 0 : discreteBranches.Length];
            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeadsSchedulers[i] = new LRScheduler(discreteHeadsOptimizers[i], step_size, gamma);
            }

        }

        public Tensor Value(Tensor state) => critic.Predict(state);
        public Tensor ContinuousPredict(Tensor state, out Tensor logProbs)
        {
            // Sample mu and sigma
            Tensor mu = muHead.Predict(state);
            // Tensor sigma = sigmaHead.Predict(state).Clip(sigma_clip.Item1, sigma_clip.Item2);
            Tensor sigma = Tensor.Fill(0.1f, mu.Shape); // (static sigma 0.1)

            // Sample actions
            Tensor actions = mu.Zip(sigma, (x, y) => Utils.Random.Gaussian(x, y));

            // Get log probs
            logProbs = Tensor.LogDensity(actions, mu, sigma);
       
            return actions;
        }
        public Tensor ContinuousForward(Tensor stateBatch, out Tensor mu, out Tensor sigma)
        {
            mu = muHead.Forward(stateBatch);
            // sigma = sigmaHead.Forward(stateBatch).Clip(sigma_clip.Item1, sigma_clip.Item2);
            sigma = Tensor.Fill(0.1f, mu.Shape); // (static sigma 0.1)

            return mu.Zip(sigma, (x, y) => Utils.Random.Gaussian(x, y));
        }


        public Tensor DiscretePredict(Tensor state, out Tensor logProbs)
        {
            logProbs = null;
            return null;
        }
        public Tensor DiscreteForward(Tensor statesBatch, out Tensor logProbs)
        {
            logProbs = null;
            return null;
        }


        public AgentBehaviour CreateAsset(string name)
        {
            this.behaviourName = name;
            var instance = AssetDatabase.LoadAssetAtPath<AgentBehaviour>($"Assets/{behaviourName}/{behaviourName}.asset");

            if (instance != null)
                throw new InvalidOperationException($"A Behaviour with the same name '{name}' already exists!");

            // Create the asset
            if (!Directory.Exists($"Assets/{behaviourName}"))
                Directory.CreateDirectory($"Assets/{behaviourName}");
            AssetDatabase.CreateAsset(this, $"Assets/{behaviourName}/{behaviourName}.asset");
            AssetDatabase.SaveAssets();

            // Create aux assets
            critic.CreateAsset($"{behaviourName}/critic");
            muHead.CreateAsset($"{behaviourName}/mu");
            sigmaHead.CreateAsset($"{behaviourName}/sigma");

            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeads[i].CreateAsset($"{behaviourName}/discrete{i}");
            }

            return this;
        }
        /// <summary>
        /// Updates the state of the Behaviour parameters.
        /// </summary>
        public void Save()
        {
            var instance = AssetDatabase.LoadAssetAtPath<AgentBehaviour>($"Assets/{behaviourName}/{behaviourName}.asset");

            if (instance == null)
                throw new InvalidOperationException("Cannot save the Behaviour because it requires compilation first.");

            // Debug.Log($"<color=#03a9fc>Agent behaviour <b>{behaviourName}<b/> saved.</color>");

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


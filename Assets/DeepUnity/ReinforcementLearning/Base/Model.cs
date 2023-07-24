using System;
using System.IO;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Model : ScriptableObject
    {
        [SerializeField] public string behaviourName;
        [SerializeField] public int observationSize;
        [SerializeField] public int continuousDim;
        [SerializeField] public int[] discreteBranches;

        [SerializeField] public RunningStandardizer stateStandardizer;
        [SerializeField] public RunningStandardizer rewardStadardizer;  
 
        [SerializeField] public Sequential critic;
        [SerializeField] public Sequential muHead;
        [SerializeField] public Sequential sigmaHead;
        [SerializeField] public Sequential[] discreteHeads;

        public Optimizer criticOptimizer { get; private set; }
        public Optimizer muHeadOptimizer { get; private set; }
        public Optimizer sigmaHeadOptimizer { get; private set; }
        public Optimizer[] discreteHeadsOptimizers { get; private set; }

        public StepLR criticScheduler { get; private set; }
        public StepLR muHeadScheduler { get; private set; }
        public StepLR sigmaHeadScheduler { get; private set; }
        public StepLR[] discreteHeadsSchedulers { get; private set; }


        public static readonly int default_step_size_StepLR = 10;
        public static readonly float default_gamma_StepLR = 0.99f;
        public static readonly (float, float) sigma_clip = (0.001f, 5f);

        public Model(int stateSize, int continuousActions, int[] discreteBranches, HyperParameters hp, string name)
        {
            this.behaviourName = name;
            this.observationSize = stateSize;
            this.continuousDim = continuousActions;

            stateStandardizer = new RunningStandardizer(stateSize);
            rewardStadardizer = new RunningStandardizer(1);


            critic = new Sequential(
                new Dense(stateSize, hp.hiddenUnits),
                new ReLU(),
                new Dense(hp.hiddenUnits, hp.hiddenUnits, device: hp.device),
                new ReLU(),
                new Dense(hp.hiddenUnits, 1),
                new Linear());

            muHead = new Sequential(
                new Dense(stateSize, hp.hiddenUnits),
                new ReLU(),
                new Dense(hp.hiddenUnits, hp.hiddenUnits, device: hp.device),
                new ReLU(),
                new Dense(hp.hiddenUnits, continuousActions),
                new Tanh());

            sigmaHead = new Sequential(
                new Dense(stateSize, hp.hiddenUnits),
                new ReLU(),
                new Dense(hp.hiddenUnits, hp.hiddenUnits, device: hp.device),
                new ReLU(),
                new Dense(hp.hiddenUnits, continuousActions),
                new Softplus());

            discreteHeads = new Sequential[discreteBranches.Length];
            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeads[i] = new Sequential(
                    new Dense(stateSize, hp.hiddenUnits),
                    new ReLU(),
                    new Dense(hp.hiddenUnits, hp.hiddenUnits),
                    new ReLU(),
                    new Dense(hp.hiddenUnits, discreteBranches[i]),
                    new Softmax());
            }
        }
        public void InitOptimisers(HyperParameters hp)
        {
            if (criticOptimizer != null)
                return;

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

            /*
             int max_step = 500_000;
             int step_size = 10;
             int no_epochs = 8;
             
             */

            int step_size = hp.learningRateSchedule ? default_step_size_StepLR : 1000;
            float gamma = hp.learningRateSchedule ? default_gamma_StepLR : 1f;

            criticScheduler = new StepLR(criticOptimizer, step_size, gamma);
            muHeadScheduler = new StepLR(muHeadOptimizer, step_size, gamma);
            sigmaHeadScheduler = new StepLR(sigmaHeadOptimizer, step_size, gamma);

            discreteHeadsSchedulers = new StepLR[discreteBranches == null ? 0 : discreteBranches.Length];
            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeadsSchedulers[i] = new StepLR(discreteHeadsOptimizers[i], step_size, gamma);
            }

        }

        public Tensor Value(Tensor state)
        {
            return critic.Predict(state);
        }
        public Tensor ContinuousPredict(Tensor state, out Tensor logProbs)
        {
            // Sample mu and sigma
            Tensor mu = muHead.Predict(state);
            //Tensor sigma = sigmaHead.Predict(state).Clip(sigma_clip.Item1, sigma_clip.Item2);
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
            //sigma = sigmaHead.Forward(stateBatch).Clip(sigma_clip.Item1, sigma_clip.Item2);
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

        /// <summary>
        /// Saves the behaviour in the Assets folder, along with the respective neural networks.
        /// </summary>
        public void Save()
        {
            if (!Directory.Exists($"Assets/{behaviourName}"))
                Directory.CreateDirectory($"Assets/{behaviourName}");

            // Save aux networks
            critic.Save($"{behaviourName}/critic");
            muHead.Save($"{behaviourName}/mu");
            sigmaHead.Save($"{behaviourName}/sigma");

            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeads[i].Save($"{behaviourName}/discrete{i}");
            }


            // Save this wrapper
            var instance = AssetDatabase.LoadAssetAtPath<Model>($"Assets/{behaviourName}/{behaviourName}.asset");
            if (instance == null)
            {
                //create instance
                AssetDatabase.CreateAsset(this, $"Assets/{behaviourName}/{behaviourName}.asset");
                AssetDatabase.SaveAssets();
            }

            EditorUtility.SetDirty(this);
            AssetDatabase.SaveAssetIfDirty(this);
            AssetDatabase.Refresh();

        }
    }
    public enum ActionType
    {
        Continuous,
        Discrete
    }
}


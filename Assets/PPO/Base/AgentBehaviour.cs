using System;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class AgentBehaviour : ScriptableObject
    {
        [SerializeField] public new string name;
        [SerializeField] public int observationSize;
        [SerializeField] public int continuousDim;
        [SerializeField] public int[] discreteBranches;

        [SerializeField] public RunningStandardizer stateStandardizer;
        [SerializeField] public RunningStandardizer rewardStadardizer;  
 
        [SerializeField] public Sequential critic;
        [SerializeField] public Sequential muHead;
        [SerializeField] public Sequential sigmaHead;
        [SerializeField] public Sequential[] discreteHeads;

        [NonSerialized] public Optimizer criticOptimizer;
        [NonSerialized] public Optimizer muHeadOptimizer;
        [NonSerialized] public Optimizer sigmaHeadOptimizer;
        [NonSerialized] public Optimizer[] discreteHeadsOptimizers;

        [NonSerialized] public StepLR criticScheduler;
        [NonSerialized] public StepLR muHeadScheduler;
        [NonSerialized] public StepLR sigmaHeadScheduler;
        [NonSerialized] public StepLR[] discreteHeadsSchedulers;

        public static readonly int default_step_size_StepLR = 10;
        public static readonly float default_gamma_StepLR = 0.99f;
        public static readonly (float, float) sigma_clip = (0.01f, 5f);

        public AgentBehaviour(int stateSize, int continuousActions, int[] discreteBranches, HyperParameters hp, string name)
        {
            this.name = name;
            this.observationSize = stateSize;
            this.continuousDim = continuousActions;

            stateStandardizer = new RunningStandardizer(stateSize);
            rewardStadardizer = new RunningStandardizer(1);


            critic = new Sequential(
                new Dense(stateSize, hp.hiddenUnits),
                new ReLU(),
                new Dense(hp.hiddenUnits, hp.hiddenUnits),
                new ReLU(),
                new Dense(hp.hiddenUnits, 1),
                new Linear());

            muHead = new Sequential(
                new Dense(stateSize, hp.hiddenUnits),
                new ReLU(),
                new Dense(hp.hiddenUnits, hp.hiddenUnits),
                new ReLU(),
                new Dense(hp.hiddenUnits, continuousActions),
                new TanH());

            sigmaHead = new Sequential(
                new Dense(stateSize, hp.hiddenUnits),
                new ReLU(),
                new Dense(hp.hiddenUnits, hp.hiddenUnits),
                new ReLU(),
                new Dense(hp.hiddenUnits, continuousActions),
                new SoftPlus());

            discreteHeads = new Sequential[discreteBranches.Length];
            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeads[i] = new Sequential(
                    new Dense(stateSize, hp.hiddenUnits),
                    new ReLU(),
                    new Dense(hp.hiddenUnits, hp.hiddenUnits),
                    new ReLU(),
                    new Dense(hp.hiddenUnits, discreteBranches[i]),
                    new SoftMax());
            }


        }
        public void InitOptimisers(HyperParameters hp)
        {
            if (criticOptimizer != null)
                return;

            criticOptimizer = new Adam(critic.Parameters(), hp.learningRate * 3f);          
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

            int ss = hp.learningRateSchedule == true ? default_step_size_StepLR : 1000;
            float gamma = hp.learningRateSchedule == true ? default_gamma_StepLR : 1f;

            criticScheduler = new StepLR(criticOptimizer, ss, gamma);
            muHeadScheduler = new StepLR(muHeadOptimizer, ss, gamma);
            sigmaHeadScheduler = new StepLR(sigmaHeadOptimizer, ss, gamma);

            discreteHeadsSchedulers = new StepLR[discreteBranches == null ? 0 : discreteBranches.Length];
            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeadsSchedulers[i] = new StepLR(discreteHeadsOptimizers[i], ss, gamma);
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
            Tensor sigma = Tensor.Fill(0.1f, mu.Shape);
            // sigma = Tensor.Clip(sigma, sigma_clip.Item1, sigma_clip.Item2);

            // Sample actions
            Tensor actions = mu.Zip(sigma, (x, y) => Utils.Random.Gaussian(x, y));

            // Get log probs
            logProbs = Tensor.LogDensity(actions, mu, sigma);
       
            return actions;
        }
        public Tensor ContinuousForward(Tensor stateBatch, out Tensor mu, out Tensor sigma)
        {
            mu = muHead.Forward(stateBatch);
            sigma = Tensor.Fill(0.1f, mu.Shape);

            Tensor actions = mu.Zip(sigma, (x, y) => Utils.Random.Gaussian(x, y));

            return actions;

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

        public void Save()
        {
            // Save aux networks
            critic.Save(name + "_critic");
            muHead.Save(name + "_muHead");
            sigmaHead.Save(name + "_sigmaHead");

            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeads[i].Save(name + $"_discreteHead{i}");
            }


            // Save this wrapper
            var instance = AssetDatabase.LoadAssetAtPath<AgentBehaviour>("Assets/" + name + ".asset");
            if (instance == null)
            {
                //create instance
                AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
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


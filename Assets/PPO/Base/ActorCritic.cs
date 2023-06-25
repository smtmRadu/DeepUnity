using System;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class ActorCritic : ScriptableObject, ISerializationCallbackReceiver
    {
 	    public new string name;
        public int observationSize;
        public int continuousDim;
        public int[] discreteBranches;

        public RunningStandardizer stateStandardizer;
        public RunningStandardizer rewardStadardizer;  
       
        public Sequential critic;
        public Sequential muHead;
        public Sequential sigmaHead;
        public Sequential[] discreteHeads;

        public Optimizer criticOptimizer;
        public Optimizer muHeadOptimizer;
        public Optimizer sigmaHeadOptimizer;
        public Optimizer[] discreteHeadsOptimizers;


        public ActorCritic(int stateSize, int continuousActions, int[] discreteBranches, HyperParameters hp, string name)
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

            criticOptimizer = new Adam(critic.Parameters(), hp.learningRate);

            muHead = new Sequential(
                new Dense(stateSize, hp.hiddenUnits),
                new ReLU(),
                new Dense(hp.hiddenUnits, hp.hiddenUnits),
                new ReLU(),
                new Dense(hp.hiddenUnits, continuousActions),
                new TanH());

            muHeadOptimizer = new Adam(muHead.Parameters(), hp.learningRate);

            sigmaHead = new Sequential(
                new Dense(stateSize, hp.hiddenUnits),
                new ReLU(),
                new Dense(hp.hiddenUnits, hp.hiddenUnits),
                new ReLU(),
                new Dense(hp.hiddenUnits, continuousActions),
                new SoftPlus());

            sigmaHeadOptimizer = new Adam(sigmaHead.Parameters(), hp.learningRate);

            discreteHeads = new Sequential[discreteBranches.Length];
            discreteHeadsOptimizers = new Optimizer[discreteBranches.Length];

            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeads[i] = new Sequential(
                    new Dense(stateSize, hp.hiddenUnits),
                    new ReLU(),
                    new Dense(hp.hiddenUnits, hp.hiddenUnits),
                    new ReLU(),
                    new Dense(hp.hiddenUnits, discreteBranches[i]),
                    new SoftMax());

                discreteHeadsOptimizers[i] = new Adam(discreteHeads[i].Parameters(), hp.learningRate);
            }
        }

        public Tensor Value(Tensor state)
        {
            return critic.Predict(state);
        }
        public Tensor ContinuousAction(Tensor state, out Tensor logProbs, out Tensor mu, out Tensor sigma)
        {
            // Sample mu and sigma
            mu = muHead.Predict(state);
            sigma = Tensor.Ones(continuousDim);

            // Sample actions
            Tensor actions = Tensor.Gaussian(mu, sigma, out _);

            // Get log probs
            logProbs = Tensor.LogDensity(actions, mu, sigma);
       
            return actions;
        }
        public Tensor DiscreteAction(Tensor state, out Tensor logProbs)
        {
            logProbs = null;
            return null;
        }

        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            criticOptimizer = new Adam(critic.Parameters(), 1e-4f);
            muHeadOptimizer = new Adam(muHead.Parameters(), 1e-4f);
            sigmaHeadOptimizer = new Adam(sigmaHead.Parameters(), 1e-4f);
            for (int i = 0; i < discreteHeads.Length; i++)
            {
                discreteHeadsOptimizers[i] = new Adam(discreteHeads[i].Parameters(), 1e-4f);
            }
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
            var instance = AssetDatabase.LoadAssetAtPath<ActorCritic>("Assets/" + name + ".asset");
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


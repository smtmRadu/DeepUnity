using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class ActorCritic : ScriptableObject
    {
 	    public new string name;
        public int observationSize;
        public ActionType actionSpace;
        public int continuousDim;
        public int[] discreteBranches;
        

        public HyperParameters hp;
        public RunningStandardizer stateStandardizer;
        public RunningStandardizer rewardStadardizer;

       
        public Sequential critic;
        public Sequential muHead;
        public Sequential sigmaHead;
        public List<Sequential> discreteHeads;

        public ActorCritic(int stateSize, int continuousActionSize, HyperParameters hp, string name)
        {
            this.name = name;
            this.observationSize = stateSize;
            this.continuousDim = continuousActionSize;
            this.actionSpace = ActionType.Continuous;
            this.hp = hp;

            stateStandardizer = new RunningStandardizer(stateSize);
            rewardStadardizer = new RunningStandardizer(1);

            muHead = new Sequential(
                new Dense(stateSize, 64),
                new ReLU(),
                new Dense(64, 64),
                new ReLU(),
                new Dense(64, 1),
                new TanH());

            critic = new Sequential(
                new Dense(stateSize, 64),
                new ReLU(),
                new Dense(64, 64),
                new ReLU(),
                new Dense(64, 1),
                new Linear());
        }
        public ActorCritic(int stateSize, int[] discreteActionSize, HyperParameters hp, string name)
        {
            throw new NotImplementedException();
        }

        public Tensor Value(Tensor state)
        {
            return critic.Predict(state);
        }
        public Tensor ContinuousAction(Tensor state, out Tensor logProbs)
        {
            // Sample mu and sigma
            Tensor mu = muHead.Predict(state);
            Tensor sigma = Tensor.Ones(continuousDim);

            // Sample actions
            Tensor actions = Tensor.Gaussian(mu, sigma, out _);

            // Get log probs
            logProbs = Tensor.LogDensity(actions, mu, sigma);

            return actions;

        }
        public Tensor DiscreteAction(Tensor state, out Tensor logProbs)
        {
            throw new NotImplementedException();
        }
       

        public void Save()
        {
            critic.Save(name + "_critic");
            muHead.Save(name + "_mu");

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


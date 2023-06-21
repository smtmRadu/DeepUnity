using UnityEngine;

namespace DeepUnity
{
    public class ActorCritic : ScriptableObject
    {
 	    public new string name;
        public int observationSize;
        public int actionSize;

        public HyperParameters hp;
        public Sequential critic;
        public Sequential muHead;

        private RunningStandardizer stateStandardizer;
        private RunningStandardizer rewardStadardizer;

        //public List<Sequential> discreteHeads;

        public ActorCritic(int stateSize, int actionSize, HyperParameters hp, string name)
        {
            this.name = name;
            this.observationSize = stateSize;
            this.actionSize = actionSize;
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


        public Tensor Action(Tensor state, out Tensor logProbs, out Tensor mus, out Tensor sigmas)
        {
            if(hp.normalize)
                state = stateStandardizer.Standardise(state);

            mus = muHead.Predict(state);
            sigmas = Tensor.Ones(actionSize);

            Tensor actions = mus.Zip(sigmas, (x, y) => Utils.Random.Gaussian(x, y));
            actions = Tensor.Clip(actions, -1f, 1f);

            logProbs = Tensor.Zeros(actionSize);
            for (int i = 0; i < actionSize; i++)
            {
                logProbs[i] = Utils.Numerics.LogDensity(actions[i], mus[i], sigmas[i]);
            }

            return actions;

        }
        public Tensor Value(Tensor state)
        {
            if (hp.normalize)
                state = stateStandardizer.Standardise(state);

            return critic.Predict(state);
        }

        public void Save()
        {
            critic.Save(name+ "_critic");
            muHead.Save(name + "_mu");


            
        }

    }
    public enum ActionType
    {
        Continuous,
        Discrete
    }
}


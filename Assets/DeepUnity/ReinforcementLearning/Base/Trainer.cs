namespace DeepUnity
{
    public static class Trainer
    {
        public static void Subscribe(Agent agent)
        {
            if(agent.behaviourType == BehaviourType.Learn)
            {
                PPOTrainer.Subscribe(agent);
            }
            else if(agent.behaviourType == BehaviourType.Heuristic)
            {
                HeuristicTrainer.Subscribe(agent);
            }
        }
    }

}



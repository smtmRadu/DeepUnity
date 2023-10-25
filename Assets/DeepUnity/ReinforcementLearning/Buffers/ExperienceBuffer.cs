using System;
using System.Linq;
using System.Collections.Generic;

namespace DeepUnity
{
    /// <summary>
    /// The common buffer of the agents that holds all experiences across all parallel environments
    /// </summary>
    public class ExperienceBuffer
    {
        public int Count { get => frames.Count; }
        public Tensor[] States { get => frames.Select(x => x.state).ToArray(); }
        public Tensor[] ContinuousActions { get => frames.Select(x => x.action_continuous).ToArray(); }
        public Tensor[] ContinuousProbabilities { get => frames.Select(x => x.prob_continuous).ToArray(); }
        public Tensor[] DiscreteActions { get => frames.Select(x => x.action_discrete).ToArray(); }
        public Tensor[] DiscreteProbabilities { get => frames.Select(x => x.prob_discrete).ToArray(); }
        public Tensor[] ValueTargets { get => frames.Select(x => x.value_target).ToArray(); }
        public Tensor[] Advantages { get => frames.Select(x => x.advantage).ToArray(); }

        private readonly List<TimestepBuffer> frames;

        public ExperienceBuffer()
        {
            frames = new List<TimestepBuffer>();
        }

        public void Shuffle()
        {
            for (int i = Count - 1; i > 0; i--)
            {
                int r = Utils.Random.Range(0, Count);
                Tensor temp;

                temp = frames[i].state;
                frames[i].state = frames[r].state;
                frames[r].state = temp;

                if (frames[i].action_continuous != null)
                {
                    temp = frames[i].action_continuous;
                    frames[i].action_continuous = frames[r].action_continuous;
                    frames[r].action_continuous = temp;

                    temp = frames[i].prob_continuous;
                    frames[i].prob_continuous = frames[r].prob_continuous;
                    frames[r].prob_continuous = temp;
                }

                if (frames[i].action_discrete != null)
                {
                    temp = frames[i].action_discrete;
                    frames[i].action_discrete = frames[r].action_discrete;
                    frames[r].action_discrete = temp;

                    temp = frames[i].prob_discrete;
                    frames[i].prob_discrete = frames[r].prob_discrete;
                    frames[r].prob_discrete = temp;
                }

                temp = frames[i].reward;
                frames[i].reward = frames[r].reward;
                frames[r].reward = temp;

                temp = frames[i].advantage;
                frames[i].advantage = frames[r].advantage;
                frames[r].advantage = temp;

                temp = frames[i].value_target;
                frames[i].value_target = frames[r].value_target;
                frames[r].value_target = temp;

                // These are not really neccesarry to be shuffled
                temp = frames[i].done;
                frames[i].done = frames[r].done;
                frames[r].done = temp;
            }
        }
        public bool IsFull(int buffer_size)
        {
            return Count == buffer_size;
        }
        public void Add(MemoryBuffer agentMemory, int buffer_size)
        {
            foreach (var frm in agentMemory.frames)
            {
                if (Count == buffer_size)
                    return;

                frames.Add(frm.Clone() as TimestepBuffer);
            }
        }
        public void NormalizeAdvantages()
        {
            float mean = frames.Average(x => x.advantage[0]);
            float variance = frames.Sum(x => (x.advantage[0] - mean) * (x.advantage[0] - mean)) / (Count - 1); // with bessel correction
            float std = MathF.Sqrt(variance);

            if (std < 0)
                return;

            for (int i = 0; i < Count; i++)
                frames[i].advantage = (frames[i].advantage - mean) / (std + Utils.EPSILON);
        }

        public void Clear()
        {
            frames.Clear();
        }
    }

}



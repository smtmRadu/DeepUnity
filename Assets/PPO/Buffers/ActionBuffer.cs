using System.Linq;
using Unity.VisualScripting;

namespace DeepUnity
{
    public class ActionBuffer
    {
        public float[] DiscreteActions { get; set; }
        public float[] ContinuousActions { get; set; }

        public ActionBuffer(int capacity)
        {
            DiscreteActions = new float[capacity];
            ContinuousActions = new float[capacity];
        }
        public void Clear()
        {
            DiscreteActions = DiscreteActions.Select(x => float.NaN).ToArray();
            ContinuousActions = ContinuousActions.Select(x => float.NaN).ToArray();
        }
        public override string ToString()
        {
            return $"(ContinuousActions {ContinuousActions.ToCommaSeparatedString()} | DiscreteActions {DiscreteActions.ToCommaSeparatedString()})";
        }
    }
}


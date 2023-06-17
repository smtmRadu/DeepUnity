namespace DeepUnity
{
    public class ActionBuffer
    {
        public Tensor DiscreteActions { get; set; }
        public Tensor ContinuousActions { get; set; }

        public ActionBuffer(int capacity)
        {
            DiscreteActions = Tensor.Zeros(capacity);
            ContinuousActions = Tensor.Zeros(capacity);
        }
        public void Clear()
        {
            DiscreteActions.ForEach(x => 0f);
            ContinuousActions.ForEach(x => 0f);
        }
        public override string ToString()
        {
            return $"(ContinuousActions {ContinuousActions} | DiscreteActions {DiscreteActions})";
        }
    }
}


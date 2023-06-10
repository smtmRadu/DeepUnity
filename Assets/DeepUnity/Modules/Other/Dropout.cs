namespace DeepUnity
{
    public class Dropout : IModule
    {
        public float dropout;
        public Tensor InputCache { get; set; }

        public Dropout(float dropout = 0.5f) => this.dropout = dropout;
      
        public Tensor Forward(Tensor input)
        {
            input.ForEach(x => Utils.Random.Value < dropout? 0f : x);
            InputCache = Tensor.Identity(input);
            return input;

        }
        public Tensor Backward(Tensor loss)
        {
            return loss.Zip(InputCache, (l, i) => i != 0f ? l : 0f);
        }
    }

}

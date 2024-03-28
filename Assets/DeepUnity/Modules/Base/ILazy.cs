namespace DeepUnity.Modules
{
    public interface ILazy
    {
        /// <summary>
        /// If the lazy layer is already initialized, it returns true.
        /// Otherwise, it initializes the layer and returns false.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public bool LazyInit(Tensor input);
    }
}





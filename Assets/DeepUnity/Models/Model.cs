using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// <b>Rename</b>: Rename the asset file normally. Right click on the network asset, and from inspector, replace the field <b>Name</b> accordingly. <br></br>
    /// </summary>
    /// <typeparam name="T">Type of the class that inherits.</typeparam>
    public abstract class Model<T> : ScriptableObject
    {
        [SerializeField] public new string name;
        [SerializeField] public string version;



        /// <summary>
        /// Backpropagates the <paramref name="lossDerivative"/> and computes the gradients.
        /// </summary>
        /// <param name="lossDerivative">Derivative of the loss function w.r.t output (dLdY).</param>
        public abstract void Backward(Tensor lossDerivative);
        /// <summary>
        /// Creates a Unity asset instance of this network in <em>Assets/</em> folder with the specified name. <br></br>
        /// Renaming an asset requires to also change the <b>Name</b> field inside the asset from inspector. <br></br>
        /// </summary>
        /// <param name="name"></param>
        /// <returns>Returns this network model.</returns>
        public abstract T Compile(string name);
        /// <summary>
        /// Updates the Unity asset with the new state of the network parameters. <br></br>
        /// Notes: <br></br>
        /// <![CDATA[-]]> Cannot save an uncompiled network. <br></br>
        /// <![CDATA[-]]> Consider saving is a costly operation, so save the network sparsely, like after the entire training session, not after every single epoch. <br></br>
        /// </summary>
        /// <param name="version">This parameter is used to keep track of the network training versions. Can be also modified manually from inspector.</param>
        public abstract void Save(string version = "1.0");
        /// <summary>
        /// Get all <typeparamref name="Learnable"/> modules.
        /// </summary>
        /// <returns></returns>
        public abstract Learnable[] Parameters();
        /// <summary>
        /// Displays information about this neural network.
        /// </summary>
        /// <returns></returns>
        public abstract string Summary();
    }

}



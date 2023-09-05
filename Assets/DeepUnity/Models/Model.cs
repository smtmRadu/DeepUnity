using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <typeparam name="T">Type of the network class that inherits.</typeparam>
    [Serializable]
    public abstract class Model<T> : ScriptableObject where T : Model<T>
    {
        [SerializeField] private int version = 1;
        [SerializeField, HideInInspector] private bool assetCreated = false;
        private bool canBackPropagate = false; // a flag that cares if a forward pass had happen before doing backpropagation.

        /// <summary>
        /// Forwards the input without caching it on the layers.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        /// <exception cref="NotSupportedException"></exception>
        public virtual Tensor Predict(Tensor input) { throw new NotSupportedException($"Tensor Predict(Tensor input) method is not available for {GetType().Name} models."); }
        /// <summary>
        /// Forwards the input and caches it on all layers.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public virtual Tensor Forward(Tensor input) { throw new NotSupportedException($"Tensor Forward(Tensor input) method is not available for {GetType().Name} models."); }
        /// <summary>
        /// Backpropagates the <paramref name="lossDerivative"/> and computes the gradients.
        /// </summary>
        /// <param name="lossDerivative">Derivative of the loss function w.r.t output (dLdY).</param>
        /// <returns>The backpropagated loss derivative. (<em>Not necessary to use in casual situations</em>)</returns>
        public abstract Tensor Backward(Tensor lossDerivative);  
        /// <summary>
        /// Get all <typeparamref name="Learnable"/> modules.
        /// </summary>
        /// <returns></returns>
        public abstract Learnable[] Parameters();
        /// <summary>
        /// Creates a Unity asset instance of this network in <em>Assets</em> folder with the specified <paramref name="name"/>. <br></br>
        /// </summary>
        /// <param name="name"></param>
        /// <returns>Returns this network model.</returns>        
        public T CreateAsset(string name)
        {
            var instance = AssetDatabase.LoadAssetAtPath<T>("Assets/" + name + ".asset");
            if (instance != null)
            {
                if (name.EndsWith($"v{version}"))
                {
                    // Replace the last version with the new version
                    name = name.Substring(0, name.Length - ("_v" + version).Length);                 
                }

                version++;
                name += $"_v{version}";
                return CreateAsset(name);
            }

            this.name = name;
            this.assetCreated = true;
            AssetDatabase.CreateAsset(this, $"Assets/{name}.asset");
            EditorUtility.SetDirty(this);
            AssetDatabase.SaveAssetIfDirty(this);
            Debug.Log($"<color=#03a9fc>[<b>{name}</b> <i>{GetType().Name}</i> asset created]</color>");
            return (T)this;
        }
        /// <summary>
        /// Saves the Unity asset with the new state of the network parameters. <br></br>
        /// Notes: <br></br>
        /// <![CDATA[-]]> Cannot save a network that was not created as an asset. <br></br>
        /// <![CDATA[-]]> Consider saving is a costly operation in terms of performance, so save the network manually sparsely throughout the training process.<br></br>
        /// </summary>
        public void Save()
        {
            if (!assetCreated)
            {
                Debug.LogError($"<color=#ff3636>[<b>{name}</b> <i>{GetType().Name}</i> model cannot be saved because the asset was not created]</color>");
                return;
            }

            // Debug.Log($"<color=#03a9fc>[{name} model saved]</color>");
            EditorUtility.SetDirty(this);
            AssetDatabase.SaveAssetIfDirty(this);
        }
        /// <summary>
        /// Displays information about this neural network.
        /// </summary>
        /// <returns></returns>
        public abstract string Summary();


        protected void BaseForward()
        {
            canBackPropagate = true;
        }
        protected void BaseBackward()
        {
            if(!canBackPropagate)
                throw new ArgumentException("Cannot backpropagate if no forward had happen before!");

            canBackPropagate = false;
        }
    }

}



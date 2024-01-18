using System;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <typeparam name="NeuralNetworkType">Type of the network class that inherits. It is used as a box to serialize models as Scriptable Objects.</typeparam>
   
    [Serializable]
    public abstract class Model<NeuralNetworkType, IOType> : ScriptableObject, ICloneable where NeuralNetworkType : Model<NeuralNetworkType, IOType>
    {
        [SerializeField, ReadOnly] private int version = 1;
        [SerializeField, HideInInspector] private bool assetCreated = false;



        /// <summary>
        /// Forwards the input without caching it on the layers.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        /// <exception cref="NotSupportedException"></exception>
        public abstract IOType Predict(IOType input);
        /// <summary>
        /// Forwards the input and caches it on all layers.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public abstract IOType Forward(IOType input);
        /// <summary>
        /// Backpropagates the <paramref name="lossGradient"/> and computes the gradients.
        /// </summary>
        /// <param name="lossGradient">Derivative of the loss function w.r.t output (dLdY).</param>
        /// <returns>The backpropagated loss derivative (dLdX).</returns>
        public abstract IOType Backward(IOType lossGradient);




        /// <summary>
        /// Get all <typeparamref name="Parameter"/>s.
        /// </summary>
        /// <returns></returns>
        public abstract Parameter[] Parameters();
        /// <summary>
        /// Displays information about this neural network.
        /// </summary>
        /// <returns></returns>
        public abstract string Summary();


     


        /// <summary>
        /// Creates a Unity asset instance of this network in <em>Assets</em> folder with the specified <paramref name="name"/>. <br></br>
        /// <br></br>
        /// <em>If already exists, this method creates a new instance called '[behaviour_name]_v[x]'</em>
        /// </summary>
        /// <param name="name"></param>
        /// <returns>Returns this network model.</returns>        
        public NeuralNetworkType CreateAsset(string name)
        {
            var instance = AssetDatabase.LoadAssetAtPath<NeuralNetworkType>("Assets/" + name + ".asset");
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
            ConsoleMessage.Info($"<b>{name}</b> <i>{GetType().Name}</i> asset created");
            return (NeuralNetworkType)this;
        }
        /// <summary>
        /// Saves the Unity asset with the new state of the network parameters. <br></br>
        /// Notes: <br></br>
        /// <![CDATA[-]]> Cannot save a network that was not created as an asset. <br></br>
        /// <![CDATA[-]]> Consider saving is a costly operation in terms of performance, so save the network manually sparsely throughout the training process.<br></br>
        /// <![CDATA[-]]> Version increments on each save.
        /// </summary>
        public void Save()
        {
            if (!assetCreated)
            {
                ConsoleMessage.Error($"<b>{name}</b> <i>{GetType().Name}</i> model cannot be saved because the asset was not created]");
                return;
            }
            version++;
            EditorUtility.SetDirty(this);
            AssetDatabase.SaveAssetIfDirty(this);
        }
        /// <summary>
        /// Clones all of this network's characteristics. All modules with all the data are cloned as new.
        /// </summary>
        /// <returns></returns>
        public abstract object Clone();

    }

}



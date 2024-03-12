using System;
using UnityEngine;
using DeepUnity.Modules;

namespace DeepUnity.Models
{
    /// <typeparam name="ModelType">Type of the model class that inherits. It is used as a box to serialize models as Scriptable Objects.</typeparam>
   
    [Serializable]
    public abstract class Model<ModelType, IOType> : ScriptableObject, ICloneable where ModelType : Model<ModelType, IOType>
    {
        [SerializeField, ViewOnly] private int version = 1;
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
        /// Clones all of this network's characteristics. All modules with all the data are cloned as new.
        /// </summary>
        /// <returns></returns>
        public abstract object Clone();
        /// <summary>
        /// Set the device of all model <see cref="IModule"/>s that are <see cref="ILearnable"/>.
        /// </summary>
        /// <param name="device"></param>
        public abstract void SetDevice(Device device);



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
#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(this);
            UnityEditor.AssetDatabase.SaveAssetIfDirty(this);
#endif
        }
        /// <summary>
        /// Creates a Unity asset instance of this network in <em>Assets</em> folder with the specified <paramref name="name"/>. <br></br>
        /// <br></br>
        /// <em>If already exists, this method creates a new instance called '[behaviour_name]_v[x]'</em>
        /// </summary>
        /// <param name="name"></param>
        /// <returns>Returns this network model.</returns>        
        public ModelType CreateAsset(string name)
        {
#if UNITY_EDITOR
            var instance = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelType>("Assets/" + name + ".asset");
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
            UnityEditor.AssetDatabase.CreateAsset(this, $"Assets/{name}.asset");
            UnityEditor.EditorUtility.SetDirty(this);
            UnityEditor.AssetDatabase.SaveAssetIfDirty(this);
#endif
            ConsoleMessage.Info($"<b>{name}</b> <i>{GetType().Name}</i> asset created");
            return (ModelType)this;

        }


       /* /// <summary>
        /// Begins a learning session. The data is shuffled automatically every epoch.
        /// </summary>
        public void Fit(Optimizer optim, Tensor[] data, Tensor[] labels, int epochs, int batch_size, bool verbose = true)
        {
            if(LearnSession.Instance != null)
            {
                throw new ArgumentException("Cannot run two Fit() at the same time");
            }

            LearnSession.Instance = Instantiate(new LearnSession(this, optim, data, labels, epochs, batch_size, verbose));
        }

        private class LearnSession : MonoBehaviour
        {
            public static LearnSession Instance;

            Model<ModelType, IOType> model;
            Optimizer optim;
            Tensor[] data;
            Tensor[] labels;
            int epochs;
            int batch_size;
            bool verbose;

            List<Tensor[]> data_batches;
            List<Tensor[]> labels_batches;

            int current_epoch = 0;
            int current_batch = 0;

            public LearnSession(Model<ModelType, Tensor> model, Optimizer optim, Tensor[] data, Tensor[] labels, int epohcs, int batch_size, bool verbose)
            {
                this.model = model;
                this.data = data;
                this.labels = labels;
                this.epochs = epohcs;
                this.batch_size = batch_size;
                this.verbose = verbose;

                PrepareNextEpoch();
            }

            private void Update()
            {
                if (current_batch == data_batches.Count - 1)
                {
                    current_batch = 0;

                    if (verbose)
                    {
                        print($"Epoch {current_epoch}");
                    }
                    PrepareNextEpoch();

                    

                    if (current_epoch > epochs)
                    {
                        Finalize();
                        return;
                    }
                }
                Tensor input = Tensor.Concat(null, data_batches[current_batch]);
                Tensor target = Tensor.Concat(null, labels_batches[current_batch]);
                Tensor prediction = model.Forward(input);
                Loss loss = Loss.MSE(prediction, target);
                optim.ZeroGrad();
                model.Backward(loss.Gradient);
                optim.Step();

                current_batch++;
            }

            void PrepareNextEpoch()
            {

                for (int i = data.Length - 1; i > 0; i--)
                {
                    int r = Utils.Random.Range(0, i + 1);
                    var temp = data[i];
                    data[i] = data[r];
                    data[r] = temp;

                    temp = labels[i];
                    labels[i] = labels[r];
                    labels[r] = temp;
                }

                data_batches = Utils.Split(data, batch_size);
                labels_batches = Utils.Split(labels, batch_size);

                current_epoch++;

                model.Save();
            }

            void Finalize()
            {
                Instance = null;
            }
        }*/
    }

    

}



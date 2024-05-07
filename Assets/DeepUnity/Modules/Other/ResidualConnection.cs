using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

/// 
///  ------>-[Main Path]-------->      
///      |                  |          
///      +->--[modules]--->-|          
///   

namespace DeepUnity.Modules
{
    /// <summary>
    /// How to create a residual connection while defining your model: <br></br>
    /// <br></br>
    /// new <see cref="Fork"/>(*arg), <br></br>
    /// yatta(), yatta(), yatta() ... <br></br>
    /// new <see cref="Join"/>(), <br></br>
    /// </summary>
    [Serializable]
    public static class ResidualConnection
    {
        [Serializable]
        public class Fork : IModule, ISerializationCallbackReceiver
        {            
            public static Lazy<Stack<Fork>> UnjoinedForksOnCreate = new Lazy<Stack<Fork>>(); // this is used on creating them, the last fork added matched the first join created.
            public static Lazy<ConcurrentDictionary<int, Fork>> UnjoinedForksWaitingRoom = new(); // this is used on deserializing, they find themselves by their own residualConnectionHas;

            [SerializeField] private int residualConnectionHash;
            public Tensor ConnectionGrad { private get; set; }
            public Tensor Identity { get; private set; }

            // SO fork doesn't allow recursive deserialization (check on before serialization), so as compromise we use nothing for now.
            // [SerializeField] private IModule[] modules;
            // [SerializeField] private IModuleWrapper[] serializedModules;


            /// <summary>
            /// Forks a residual connection from the main path. Make sure to create a <see cref="Join"/>. If the input <see cref="Fork"/> doesn't match the shape of the <see cref="Join"/>, insert a linear projection as arg.
            /// </summary>
            /// <param name="modules">Modules attached on this residual connection</param>
            // public Fork(IModule[] modules)
            // {
            //     UnjoinedForksOnCreate.Value.Push(this);
            //     this.modules = modules;
            // }

            /// <summary>
            /// Forks a residual connection from the main path. Make sure to create a <see cref="Join"/>.
            /// </summary>
            public Fork()
            {
                UnjoinedForksOnCreate.Value.Push(this);
            }

            public Tensor Backward(Tensor loss)
            {
                // if(modules == null)
                     return loss + ConnectionGrad;
                // else
                // {                  
                //     for (int i = modules.Length - 1; i >= 0; --i)
                //     {
                //         loss = modules[i].Backward(loss);
                //     }
                //     return loss + ConnectionGrad;
                // }
            }

            public object Clone()
            {
                throw new ArgumentException("Skip connections cannot be cloned");
            }

            public Tensor Forward(Tensor input)
            {
                Identity = input.Clone() as Tensor;
                return input;
            }

            public Tensor Predict(Tensor input)
            {
                Identity = input.Clone() as Tensor;
                return input;
            }

            public void OnBeforeSerialize()
            {
                residualConnectionHash = GetHashCode();

                // try
                // {
                //     if (serializedModules != null && serializedModules.Length > 0)
                //         serializedModules = modules.Select(x => IModuleWrapper.Wrap(x)).ToArray();
                // }
                // catch { }
            }
            public void OnAfterDeserialize()
            {
                if(!UnjoinedForksWaitingRoom.Value.ContainsKey(residualConnectionHash))
                    UnjoinedForksWaitingRoom.Value.TryAdd(residualConnectionHash, this);

                // try
                // {
                //     if (modules != null && modules.Length > 0)
                //         modules = serializedModules.Select(x => IModuleWrapper.Unwrap(x)).ToArray();
                // }
                // catch { }
              
            }
        }

        [Serializable]
        public class Join : IModule, ISerializationCallbackReceiver
        {
            [SerializeField] private int residualConnectionHash;
            private Fork forkSource;

            /// <summary>
            /// Joins the last created <see cref="Fork"/> to the main path.
            /// </summary>
            public Join()
            {
                try
                {
                    forkSource = Fork.UnjoinedForksOnCreate.Value.Pop();
                }
                catch { }
            }
            

            public Tensor Backward(Tensor loss)
            {
                this.forkSource.ConnectionGrad = loss.Clone() as Tensor;
                return loss;
            }

            public object Clone()
            {
                throw new ArgumentException("Skip connections cannot be cloned");
            }

            public Tensor Forward(Tensor input)
            {
                return input + forkSource.Identity;
            }

            public Tensor Predict(Tensor input)
            {
                return input + forkSource.Identity;
            }

            public void OnBeforeSerialize()
            {
                try
                {
                    residualConnectionHash = forkSource.GetHashCode();
                }
                catch { }
            }
            public void OnAfterDeserialize()
            {
                bool remove = false;
                foreach (var item in Fork.UnjoinedForksWaitingRoom.Value)
                {
                    if(item.Key == residualConnectionHash)
                    {
                        this.forkSource = item.Value;
                        remove = true;
                        break;
                    }
                }
                if(remove)
                    Fork.UnjoinedForksWaitingRoom.Value.TryRemove(residualConnectionHash, out _);
            }
        }
    }
    
}




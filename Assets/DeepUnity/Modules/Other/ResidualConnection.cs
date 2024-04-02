using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity.Modules
{
    /// <summary>
    /// How to create a residual connection while defining your model: <br></br>
    /// <br></br>
    /// new <see cref="ResidualConnection.Fork"/>(), <br></br>
    /// yatta(), yatta(), yatta() ... <br></br>
    /// new <see cref="ResidualConnection.Join"/>(), <br></br>
    /// </summary>
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

            [SerializeField] private Tensor linearProjection;

            public Fork()
            {
                UnjoinedForksOnCreate.Value.Push(this);
            }

            // /// <summary>
            // /// Applies a linear projection to the identity to match the result
            // /// </summary>
            // /// <param name="linearProjection"></param>
            // public Fork(Tensor linearProjection)
            // {
            //     UnjoinedForks.Value.Push(this);
            //     this.linearProjection = linearProjection;
            // }

            public Tensor Backward(Tensor loss)
            {
                return loss + ConnectionGrad;
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
            }
            public void OnAfterDeserialize()
            {
                if(!UnjoinedForksWaitingRoom.Value.ContainsKey(residualConnectionHash))
                    UnjoinedForksWaitingRoom.Value.TryAdd(residualConnectionHash, this);
            }
        }

        [Serializable]
        public class Join : IModule, ISerializationCallbackReceiver
        {
            [SerializeField] private int residualConnectionHash;
            private Fork forkSource;

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




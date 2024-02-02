using DeepUnity;
using System.Collections.Generic;
using System;
using UnityEngine;
using System.Linq;

namespace DeepUnity
{
    /// <summary>
    /// A simple way to serialize a bunch of tensors inside scriptable objects. This is because 
    /// Unity serialization system cannot serialize directly a List of Tensors, instead can serialize
    /// a wrapping object of this type along with it's fields.
    /// </summary>
    [Serializable]
    public class TensorCollection
    {
        [SerializeField] private List<Tensor> tensors = new List<Tensor>();
        public int Count { get { return tensors.Count; } }

        public TensorCollection() { }
        public TensorCollection(IEnumerable<Tensor> tensors) => this.tensors = tensors.ToList();


        public void Add(Tensor tensor) => tensors.Add(tensor);
        public void InsertAt(int index, Tensor tensor) => tensors.Insert(index, tensor);
        public void RemoveAt(int index) => tensors.RemoveAt(index);
        public void Clear() => tensors.Clear(); 
        public List<Tensor> ToList() => tensors.ToList();
        public Tensor[] ToArray() => tensors.ToArray();
    }
}


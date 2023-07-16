using DeepUnity;
using System.Collections.Generic;
using System;
using UnityEngine;
using System.Collections.ObjectModel;
using System.Linq;

namespace DeepUnity
{
    /// <summary>
    /// A simple way to serialize a bunch of tensors.
    /// </summary>
    [Serializable]
    public class TensorCollection : Collection<Tensor>
    {
        [SerializeField] private List<Tensor> tensors = new List<Tensor>();

        public TensorCollection() { }
        public TensorCollection(IEnumerable<Tensor> tensors) => this.tensors = tensors.ToList();
        public void Add(Tensor tensor) => tensors.Add(tensor);
        public List<Tensor> ToList() => tensors;
    }
}


using System;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/HyperParameters"), Serializable]
    public class HyperParameters : MonoBehaviour
    {
        public bool normalize = false;
    }
}


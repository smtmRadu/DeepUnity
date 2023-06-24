using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/HyperParameters"), Serializable]
    public class HyperParameters : MonoBehaviour
    {
        [Header("Trainer Configurations")]

        [Tooltip("How many steps of experience to collect per-agent before adding it to the experience buffer.")]
        public int timeHorizon = 64;

        [Tooltip("Total number of steps (i.e., observation collected and action taken) that must be taken in the environment (or across all environments if using multiple in parallel) before ending the training process.")]
        public int maxSteps = 500000;

        [Tooltip("Initial learning rate for gradient descent.")]
        public float learningRate = 3e-4f;

        [Tooltip("This should always be multiple times smaller than bufferSize. Typical range: (Continuous 512 - 5120) (Discrete 32 - 512)")]
        public int batchSize = 512;

        [Tooltip("Typical range 2048 - 409600")]
        public int bufferSize = 4096;

        [Tooltip("Number of units in the hidden layers of the neural network.")]
        public int hiddenUnits = 128;

        [Tooltip("Number of hidden layers in the neural network.")]
        public int numLayers = 2;

        [Tooltip("Applies linear decay of the learning rate until reaching maxSteps")]
        public bool learningRateScheduler = false;

        [Tooltip("Apply normalization to observation inputs.")]
        public bool normalize = true;

        [Header("PPO-specific Configurations")]

        public float beta = 5e-3f;

        public float epsilon = 0.2f;

        public float lambda = 0.95f;

        public int numEpoch = 3;

        public bool betaScheduler = false;

        public bool epsilonScheduler = false;


    }
    [CustomEditor(typeof(HyperParameters), true), CanEditMultipleObjects]
    class ScriptlessHP : Editor
    {
        public override void OnInspectorGUI()
        {
            List<string> dontDrawMe = new List<string>() { "m_Script" };

            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
}


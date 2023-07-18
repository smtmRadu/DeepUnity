using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/HyperParameters"), Serializable]
    public class HyperParameters : MonoBehaviour
    {
        [Header("Simulation Configurations")]

        [Tooltip("Device used for computing high parallelizable operations.")]
        public Device device = Device.CPU;


        [Header("Trainer Configurations")]

        [ReadOnly, Tooltip("How many steps of experience to collect per-agent before adding it to the experience buffer.")]
        public int timeHorizon = 64;

        [Tooltip("The max step value determines the maximum length of an agent's episodes/trajectories. Set to a positive integer to limit the episode length to that many steps. Set to 0 for unlimited episode length.")]
        public int maxStep = 65536;

        [Tooltip("Initial learning rate for stochastic gradient descent.")]
        public float learningRate = 3e-4f;

        [Tooltip("This should always be multiple times smaller than bufferSize. Typical range: (Continuous 512 - 5120) (Discrete 32 - 512)")]
        public int batchSize = 256;

        [ReadOnly, Tooltip("Typical range 2048 - 409600")]
        public int bufferSize = 2048;

        [Tooltip("Number of units in the hidden layers of the neural network.")]
        public int hiddenUnits = 64;

        [ReadOnly, Tooltip("Number of hidden layers in the neural network.")]
        public int numLayers = 2;

        [Tooltip("Applies linear decay on learning rate (default step_size: 10, default decay: 0.99f).")]
        public bool learningRateSchedule = false;

        [Tooltip("Apply normalization to observation inputs and rewards, as well for the advantages.")]
        public bool normalize = true;

        [ReadOnly, Tooltip("Display statistics of each episode in the Console.")]
        public bool verbose = false;

        [Header("PPO-specific Configurations")]

        [Tooltip("Entropy regularization.")]
        public float beta = 5e-3f;

        [Tooltip("Clip factor.")]
        public float epsilon = 0.2f;

        [Tooltip("Discount factor.")]
        public float gamma = 0.99f;

        [Tooltip("GAE factor.")]
        public float lambda = 0.95f;

        [Tooltip("Number of epochs per buffer.")]
        public int numEpoch = 3;

        [ReadOnly, Tooltip("Applies linear decay on beta.")]
        public bool betaScheduler = false;

        [ReadOnly, Tooltip("Applies linear decay on epsilon.")]
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


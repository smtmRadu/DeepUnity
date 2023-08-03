using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/HyperParameters")]
    public class HyperParameters : MonoBehaviour
    { 
        [Header("Training Configurations")]

        [Tooltip("The framerate at which the simulation is runned (FixedUpdate() calls per second).")]
        [Min(30)] public int targetFPS = 50;

        [Tooltip("Tthe maximum length of an agent's episodes/trajectories. Set to a positive integer to limit the episode length to that many steps. Set to 0 for unlimited episode length.")]
        [Min(0)] public int maxSteps = 1000;

        [Space]
        [Tooltip("Initial learning rate for Adam optimizer.")]
        [Min(0)] public float learningRate = 3e-4f;

        [Tooltip("Number of epochs per episode trajectory.")]
        [Min(3)] public int numEpoch = 10;

        //[Tooltip("This should always be multiple times smaller than bufferSize. Typical range: (Continuous 512 - 5120) (Discrete 32 - 512)")]
        [Tooltip("Typical range (MaxSteps/16, MaxSteps/4), considering the batch will be most of the time less than the trajectory length.")]
        [Min(32)] public int batchSize = 256;

        // [ReadOnly, Tooltip("Typical range 2048 - 409600")]
        // [Min(1024)] public int bufferSize = 10240;

        [Tooltip("Applies linear decay on learning rate (default step_size: 10, default decay: 0.99f).")]
        public bool learningRateSchedule = false;

        [Tooltip("Apply normalization to observation inputs and rewards.")]
        public bool normalize = false;

        [Tooltip("Debug all timesteps in an output file.")]
        public bool debug = false;

        [Header("PPO-specific Configurations")]
        [ReadOnly, Tooltip("How many steps of experience to collect per-agent before adding it to the experience buffer.")]
        [Min(1)] public int timeHorizon = 64;

        [Tooltip("Entropy regularization.")]
        [Min(0f)] public float beta = 5e-3f;

        [Tooltip("Clip factor.")]
        [Min(0.1f)] public float epsilon = 0.2f;

        [Tooltip("Discount factor.")]
        [Min(0)] public float gamma = 0.99f;

        [ReadOnly, Tooltip("GAE factor.")]
        [Min(0)] public float lambda = 0.95f;

        

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


using System;
using UnityEditor;
using UnityEngine;
using System.Collections.Generic;
namespace DeepUnity
{
    [Serializable]
    public class Hyperparameters : ScriptableObject
    { 
        [Header("Training Configurations")]

        [Tooltip("The framerate at which the simulation is runned (FixedUpdate() calls per second).")]
        [Min(30)] public int targetFPS = 50;

        [Tooltip("The maximum length of an agent's episode. Set to a positive integer to limit the episode length to that many steps. Set to 0 for unlimited episode length.")]
        [Min(0)] public int maxSteps = 1000;

        [Tooltip("Debug all timesteps in an output file.")]
        public bool debug = false;


        [Space]
        [Tooltip("Initial learning rate for Adam optimizer.")]
        [Min(0)] public float learningRate = 3e-4f;

        [Tooltip("Number of epochs per episode trajectory.")]
        [Min(3)] public int numEpoch = 8;

        [Tooltip("Number of experiences in each iteration of gradient descent. This should always be multiple times smaller than buffer_size")]
        [Min(32)] public int batchSize = 128;

        [Tooltip("Number of experiences to collect before updating the policy model. Corresponds to how many experiences should be collected before we do any learning or updating of the model. This should be multiple times larger than batch_size. Typically a larger buffer_size corresponds to more stable training updates.")]
        [Min(64)] 
        public int bufferSize = 2048;

        [Tooltip("Auto-normalize the observation inputs.")]
        public bool normalizeObservations = false;

        [Tooltip("Apply normalization to advantages over the memory buffer.")]
        public bool normalizeAdvantages = false;

        [Tooltip("Applies linear decay on learning rate (default step_size: 10, default decay: 0.99f).")]
        public bool learningRateSchedule = false;

        
      
        [Header("PPO-specific Configurations")]
        [Tooltip("How many steps of experience to collect per-agent before adding it to the experience buffer.")]
        [Min(1)] public int horizon = 64;

        [Tooltip("Entropy regularization.")]
        [Min(0f)] public float beta = 5e-3f;

        [Tooltip("Clip factor.")]
        [Min(0.1f)] public float epsilon = 0.2f;

        [Tooltip("Discount factor.")]
        [Min(0)] public float gamma = 0.99f;

        [Tooltip("GAE factor.")]
        [Min(0)] public float lambda = 0.95f;

        [ReadOnly, Tooltip("Applies linear decay on beta.")]
        public bool betaScheduler = false;

        [ReadOnly, Tooltip("Applies linear decay on epsilon.")]
        public bool epsilonScheduler = false;

        /// <summary>
        /// Creates a new Hyperparameters asset in the <em>behaviour</em> folder.
        /// </summary>
        /// <param name="behaviourName"></param>
        /// <returns></returns>
        public static Hyperparameters CreateOrLoadAsset(string behaviourName)
        {
            var instance = AssetDatabase.LoadAssetAtPath<Hyperparameters>($"Assets/{behaviourName}/Config.asset");
            
            if(instance != null)
                return instance;

            Hyperparameters hp = new Hyperparameters();

            AssetDatabase.CreateAsset(hp, $"Assets/{behaviourName}/Config.asset");
            AssetDatabase.SaveAssets();
            Debug.Log($"<color=#0ef0bf>[<b>{behaviourName}/Config</b> <i>Hyperparameters</i> asset created]</color>");
            return hp;
        }
    }
    [CustomEditor(typeof(Hyperparameters), true), CanEditMultipleObjects]
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


using System;
using UnityEditor;
using UnityEngine;
using System.Collections.Generic;
namespace DeepUnity
{
    [Serializable]
    public class Hyperparameters : ScriptableObject
    {
        [Header("Training Configuration")]

        [Tooltip("Initial learning rate for Adam optimizer.")]
        [Min(1e-8f)] public float learningRate = 3e-4f;

        [Tooltip("Global Gradient Clipping max norm value. Set to 0 to turn off.")]
        [Min(0)] public float gradClipNorm = 0.5f;

        [Tooltip("Number of epochs per episode trajectory.")]
        [Min(3)] public int numEpoch = 8;

        [Tooltip("Number of experiences in each iteration of gradient descent. This should always be multiple times smaller than buffer_size")]
        [Min(32)] public int batchSize = 64;

        [Tooltip("Number of experiences to collect before updating the policy model. Corresponds to how many experiences should be collected before we do any learning or updating of the model. This should be multiple times larger than batch_size. Typically a larger buffer_size corresponds to more stable training updates.")]
        [Min(64)] public int bufferSize = 1024;

        [Tooltip("Apply normalization to advantages over the memory buffer.")]
        public bool normalizeAdvantages = true;


        [Tooltip("Applies linear decay on learning rate. Step occurs on each [Num Epoch * Parallel Agents] times per policy iteration.")]
        [SerializeField] public bool learningRateSchedule = false;
        [SerializeField] public int schedulerStepSize = 8;
        [SerializeField] public float schedulerDecay = 0.99f;




        [Header("PPO-specific Configuration")]
        [Tooltip("How many steps of experience to collect per-agent before adding it to the experience buffer.")]
        [Min(1)] public int horizon = 64;

        [Tooltip("Entropy regularization.")]
        [Min(0f)] public float beta = 5e-3f;

        [Tooltip("Clip factor.")]
        [Min(0.1f)] public float epsilon = 0.2f;

        [Tooltip("Discount factor.")]
        [Min(0)] public float gamma = 0.99f;

        [Tooltip("GAE factor.")]
        [ReadOnly, Min(0)] public float lambda = 0.95f;

        [ReadOnly, Tooltip("Use early stopping")]
        public bool earlyStopping = false;

        [Tooltip("Kullback-Leibler divergence target value")]
        [Min(0)] public float targetKL = 0.015f;

        [ReadOnly, Tooltip("Applies linear decay on beta.")]
        public bool betaScheduler = false;

        [ReadOnly, Tooltip("Applies linear decay on epsilon.")]
        public bool epsilonScheduler = false;

        [Space]
        [Tooltip("Debug all timesteps in an output file.")]
        public bool debug = false;


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
            serializedObject.Update();
            List<string> dontDrawMe = new List<string>() { "m_Script" };

            SerializedProperty lrs = serializedObject.FindProperty("learningRateSchedule");
            if(!lrs.boolValue)
            {
                dontDrawMe.Add("schedulerStepSize");
                dontDrawMe.Add("schedulerDecay");
            }

            SerializedProperty es = serializedObject.FindProperty("earlyStopping");
            if (!es.boolValue)
                dontDrawMe.Add("targetKL");

            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
}


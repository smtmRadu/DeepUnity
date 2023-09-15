using System;
using UnityEditor;
using UnityEngine;
using System.Collections.Generic;
using DeepUnityTutorials;

namespace DeepUnity
{
    [Serializable]
    public class Hyperparameters : ScriptableObject
    {
        [Header("Training Configuration")]

        [Tooltip("[Typical range: 1e-5 - 1e-3] Initial learning rate for Adam optimizer.")]
        [Min(1e-8f)] public float learningRate = 3e-4f;

        [Tooltip("[Typical range: 0 - 1] Global Gradient Clipping max norm value. Set to 0 to turn off.")]
        [Min(0)] public float gradClipNorm = 0.5f;

        [Tooltip("[Typical range: 3 - 10] Number of epochs per buffer.")]
        [Min(3)] public int numEpoch = 8;

        [Tooltip("[Typical range: (Continuous) 512 - 5120, (Discrete) 32 - 512] Number of experiences in each iteration of gradient descent. This should always be multiple times smaller than buffer size.")]
        [Min(32)] public int batchSize = 512;

        [Tooltip("[Typical range: 2048 - 409600] Number of experiences to collect before updating the policy model. Corresponds to how many experiences should be collected before we do any learning or updating of the model. This should be multiple times larger than batch size. Typically a larger buffer size corresponds to more stable training updates.")]
        [Min(512)] public int bufferSize = 10240;

        [Tooltip("Apply normalization to advantages over the memory buffer.")]
        public bool normalizeAdvantages = true;

        [Tooltip("Shuffle the training data each epoch. Increases generalization and convergence and avoid batch effects (in BatchNorm). Decreases policy update time.")]
        public bool shuffleTrainingData = true;

        [Tooltip("Applies linear decay on learning rate.")]
        [SerializeField] public bool learningRateSchedule = false;
        [SerializeField] public int schedulerStepSize = 8;
        [SerializeField] public float schedulerDecay = 0.99f;




        [Header("PPO-specific Configuration")]
        [Tooltip("[Typical range: 32 - 2048] How many steps of experience to collect per-agent before adding it to the experience buffer.")]
        [Min(1)] public int horizon = 64;

        [Tooltip("[Typical range: 1e-4 - 1e-2] Entropy regularization.")]
        [Min(0f)] public float beta = 5e-3f;

        [Tooltip("[Typical range: 0.1 - 0.3] Clip factor.")]
        [Min(0.1f)] public float epsilon = 0.2f;

        [Tooltip("[Typical range: 0.96 - 0.99] Discount factor.")]
        [Min(0)] public float gamma = 0.99f;

        [Tooltip("[Typical range: 0.9 - 0.95] GAE factor.")]
        [ReadOnly, Min(0)] public float lambda = 0.95f;

        [ReadOnly, Tooltip("Use early stopping")]
        public bool earlyStopping = false;

        [Tooltip("Kullback-Leibler divergence target value")]
        [Min(0)] public float targetKL = 0.015f;

        // [ReadOnly, Tooltip("Applies linear decay on beta.")]
        // public bool betaScheduler = false;
        // 
        // [ReadOnly, Tooltip("Applies linear decay on epsilon.")]
        // public bool epsilonScheduler = false;

        [Space(50)]
        [Tooltip("Debug the train_data into a file.")]
        public bool debug = false;


        private void Awake()
        {
            if (bufferSize % batchSize != 0)
                ConsoleMessage.Warning("Buffer size must be multiple of batch size");
        }
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


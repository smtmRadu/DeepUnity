using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// Default settings are taken from https://unity-technologies.github.io/ml-agents/Training-Configuration-File/
    /// </summary>
    [Serializable]
    public class Hyperparameters : ScriptableObject
    {
        [Header("Training Configuration")]

        [Tooltip("Algorithm used in training the agent. Note that defaults are for PPO.")]
        public TrainerType trainer = TrainerType.PPO;

        [Tooltip("[Typical range: 1e5 - 1e7] The maximum length in steps of this training session.")]
        [Min(10_000f)] public long maxSteps = 2_000_000_000;

        [Tooltip("[Typical range: 1e-4 - 1e-3] Initial learning rate for Actor's Adam optimizer.")]
        [MinMax(5e-6f, 1f)] public float actorLearningRate = 3e-4f;

        [Tooltip("[Typical range: 1e-4 - 1e-3] Initial learning rate for Critic's Adam optimizer")]
        [MinMax(5e-6f, 1f)] public float criticLearningRate = 3e-4f;

        [Tooltip("[Typical range: 0.9 - 0.9997] Discount factor.")]
        [MinMax(0.001f, 1f)] public float gamma = 0.99f;

        [Tooltip("Applies linear decay on learning rate with respect to the maxSteps. When maxSteps is reached, lr will be 0.")]
        [SerializeField] public bool LRSchedule = false;

        // ========================================================================================================================================================================

        [Header("Specific Configuration")] // https://github.com/yosider/ml-agents-1/blob/master/docs/Training-PPO.md

        [Tooltip("[Typical range: (Continuous) 512 - 5120, (Discrete) 32 - 512] Number of experiences in each iteration of gradient descent. This should always be multiple times smaller than buffer size.")]
        [MinMax(64, 5120)] public int batchSize = 512;

        [Tooltip("[Typical range: 2048 - 409600] Number of experiences to collect before updating the policy model. Corresponds to how many experiences should be collected before we do any learning or updating of the model. This should be multiple times larger than batch size. Typically a larger buffer size corresponds to more stable training updates.")]
        [MinMax(2048, 409600)] public int bufferSize = 10240; // Do not exagerate with this, keep it at a max of 1M.

        [Tooltip("[Typical range: 64 - 2048] How many steps of experience to collect per-agent before adding it to the experience buffer.")]
        [MinMax(32, 4096)] public int horizon = 256;

        [Tooltip("[Typical range: 3 - 10] Number of epochs per buffer.")]
        [MinMax(3, 20)] public int numEpoch = 8;

        [Tooltip("[Typical range: 1e-4 - 1e-2] Entropy regularization for trainable standard deviation. Also used for Shannon entropy in discrete action space.")]
        [MinMax(0f, 0.01f)] public float beta = 5e-3f;

        [Tooltip("[Typical range: 0.1 - 0.3] Clip factor.")]
        [MinMax(0.1f, 0.3f)] public float epsilon = 0.2f;

        [Tooltip("[Typical range: 0.92 - 0.98] GAE factor.")]
        [MinMax(0.9f, 1f)] public float lambda = 0.96f;

        [Tooltip("[Typical range: 0 - 0.5] Global Gradient Clipping max norm value. Set to 0 to turn off.")]
        [MinMax(0f, 0.5f)] public float maxNorm = 0.5f;

        [Tooltip("[Typical range: 0.5 - 1] Value loss function coefficient")]
        [MinMax(0.5f, 1f)] public float valueCoeff = 0.5f;

        [Tooltip("Use of KLE")]
        public KLType KLDivergence = KLType.Off;

        [Tooltip("Kullback-Leibler divergence target value")]
        [MinMax(0.015f, 0.15f)] public float targetKL = 0.015f;

        [Tooltip("Normalize the advantages at minibatch level. Might improve convergence for relative large minibatches, but might cause harm when minibatches are small.")]
        public bool normalizeAdvantages = true;

        // ========================================================================================================================================================================

        [Header("Specific Configuration")]

        [Tooltip("[Typical range: 50000 - 1000000] The maximum capacity to hold experices. When getting fullfilled, the old experiences are removed to enable new space.")]
        [Min(50000)] public int replayBufferSize = 1_000_000;

        [Tooltip("[Typical range: 32 - 1024] The number of samples in the minibatch when performing a policy or Q update.")]
        [MinMax(32, 1024)] public int minibatchSize = 64;

        [Tooltip("[Typicall range: 1 - 128] Number of steps taken before updating the policy. Doesn't count for parallel agents, this considers only 1 decision.")]
        [MinMax(1, 128)] public int updateInterval = 64;

        [Tooltip("[Typicall range > Batch Size] Number of steps collected before updating the policy. In this timeframe, the actions will be purely random.")]
        [Min(64)] public int updateAfter = 1024;

        [Tooltip("[Typical range: 1 - 8] Number of mini-batches sampled on policy model update. This can be increased while also increasing updateEvery.")]
        [MinMax(1, 8)] public int updatesNum = 1;

        [Tooltip("[Typicall range: (Continuous) 0.5 - 1.0 ~(Discrete) 0.05 - 0.5] Entropy tradeoff coefficient. Lower alpha means low exploration.")]
        [MinMax(1e-8f, 0.5f)] public float alpha = 0.2f;

        [Tooltip("[Typicall range: 0.001 - 0.01] How aggresively to update the target network used for boostraping value estimation.")]
        [MinMax(0.001f, 0.01f)] public float tau = 0.005f;

        // ========================================================================================================================================================================

        // TD3/DDPG specific

        [Tooltip("The clip applied to the active noise.")]
        public float noiseClip = 0.5f;

        [Tooltip("Policy will only be updated once every policy_delay times for each update of the Q-networks.")]
        [MinMax(0, 10)] public int policyDelay = 2;
        
        [HideInInspector]
        [Tooltip("Debug the train_data into a file.")]
        [Space(30)]
        public bool debug = false;

        [Space(50)]
        [Tooltip("How does the timescale modifies during training.")]
        public TimescaleAdjustmentType timescaleAdjustment = TimescaleAdjustmentType.Constant;
        [Tooltip("Timescale of the training session.")]
        [Min(0.1f)] public float timescale = 1f;


        private void Awake()
        {
            if (bufferSize % batchSize != 0)
            {
                int old_buff_size = bufferSize;
                int lowbound = bufferSize / batchSize;
                bufferSize = batchSize * lowbound;
                ConsoleMessage.Info($"Buffer size must be multiple of batch size. Old value ({old_buff_size}) was changed to {bufferSize}.");

            }

        }
        /// <summary>
        /// Creates a new Hyperparameters asset in the <em>behaviour</em> folder.
        /// </summary>
        /// <param name="behaviourName"></param>
        /// <returns></returns>
        public static Hyperparameters CreateOrLoadAsset(string behaviourName)
        {
            Hyperparameters hp = new Hyperparameters();
#if UNITY_EDITOR
            var instance = AssetDatabase.LoadAssetAtPath<Hyperparameters>($"Assets/{behaviourName}/Config.asset");

            if (instance != null)
                return instance;

           

            AssetDatabase.CreateAsset(hp, $"Assets/{behaviourName}/Config.asset");
            AssetDatabase.SaveAssets();
            Debug.Log($"<color=#0ef0bf>[<b>{behaviourName}/Config</b> <i>Hyperparameters</i> asset created]</color>");
#endif
            return hp;
        }
    }



#if UNITY_EDITOR
    [CustomEditor(typeof(Hyperparameters), true), CanEditMultipleObjects]
    class ScriptlessHP : Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            List<string> dontDrawMe = new List<string>() { "m_Script" };

            Hyperparameters script = (Hyperparameters)target;

            if (script.trainer == TrainerType.PPO)
            {
                if (script.KLDivergence == (int)KLType.Off)
                    dontDrawMe.Add("targetKL");

                dontDrawMe.Add("replayBufferSize");
                dontDrawMe.Add("minibatchSize");
                dontDrawMe.Add("updateInterval");
                dontDrawMe.Add("updateAfter");
                dontDrawMe.Add("updatesNum");
                dontDrawMe.Add("alpha");
                dontDrawMe.Add("saveReplayBuffer");
                dontDrawMe.Add("tau");

                dontDrawMe.Add("activeNoise");
                dontDrawMe.Add("noiseClip");
                dontDrawMe.Add("policyDelay");
            }
            else if (script.trainer == TrainerType.SAC)
            {
                dontDrawMe.Add("batchSize");
                dontDrawMe.Add("bufferSize");
                dontDrawMe.Add("horizon");
                dontDrawMe.Add("numEpoch");
                dontDrawMe.Add("beta");
                dontDrawMe.Add("epsilon");
                dontDrawMe.Add("lambda");
                dontDrawMe.Add("normalizeAdvantages");
                dontDrawMe.Add("KLDivergence");
                dontDrawMe.Add("targetKL");
                dontDrawMe.Add("maxNorm");
                dontDrawMe.Add("valueCoeff");
                dontDrawMe.Add("normalizeAdvantages");

                dontDrawMe.Add("noiseClip");
                dontDrawMe.Add("policyDelay");
            }
            else if (script.trainer == TrainerType.TD3)
            {
                dontDrawMe.Add("batchSize");
                dontDrawMe.Add("bufferSize");
                dontDrawMe.Add("horizon");
                dontDrawMe.Add("numEpoch");
                dontDrawMe.Add("beta");
                dontDrawMe.Add("epsilon");
                dontDrawMe.Add("lambda");
                dontDrawMe.Add("normalizeAdvantages");
                dontDrawMe.Add("KLDivergence");
                dontDrawMe.Add("targetKL");
                dontDrawMe.Add("maxNorm");
                dontDrawMe.Add("valueCoeff");
                dontDrawMe.Add("normalizeAdvantages");

                dontDrawMe.Add("alpha");

            }
            else if (script.trainer == TrainerType.DDPG)
            {
                dontDrawMe.Add("batchSize");
                dontDrawMe.Add("bufferSize");
                dontDrawMe.Add("horizon");
                dontDrawMe.Add("numEpoch");
                dontDrawMe.Add("beta");
                dontDrawMe.Add("epsilon");
                dontDrawMe.Add("lambda");
                dontDrawMe.Add("normalizeAdvantages");
                dontDrawMe.Add("KLDivergence");
                dontDrawMe.Add("targetKL");
                dontDrawMe.Add("maxNorm");
                dontDrawMe.Add("valueCoeff");
                dontDrawMe.Add("normalizeAdvantages");

                dontDrawMe.Add("alpha");
                dontDrawMe.Add("noiseClip");
                dontDrawMe.Add("policyDelay");
            }
            else
                throw new NotImplementedException("Unhandled trainer type");

            dontDrawMe.Add("timescale");



            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());

            if (script.timescaleAdjustment == TimescaleAdjustmentType.Dynamic)
            {

                // Add a read-only property drawer for the "timescale" field
                EditorGUI.BeginDisabledGroup(true);
                EditorGUILayout.PropertyField(serializedObject.FindProperty("timescale"), true);
                EditorGUI.EndDisabledGroup();
            }
            else
            {
                EditorGUILayout.PropertyField(serializedObject.FindProperty("timescale"), true);
            }



            // if (EditorApplication.isPlaying)
            //     EditorGUILayout.HelpBox("Hyperparameters values can be modified at runtime. Config file has no effect but when the agent is learning.", MessageType.Info);
            // DO not modify the values at runtime (lr will not change, may appear bugs when changing the buffer size to a smaller size when is already filled).


            serializedObject.ApplyModifiedProperties();
        }
    }
#endif

}


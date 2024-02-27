using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
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


        [Tooltip("[Typical range: (PPO) 5e-6 - 3e-3, (SAC) 1e-5 - 1e-3] Initial learning rate for Adam optimizer (both all networks).")]
        [Min(5e-6f)] public float learningRate = 3e-4f;

        [Tooltip("[Typical range: 0.8 - 0.9997] Discount factor.")]
        [Min(0.001f)] public float gamma = 0.99f;

        [Tooltip("[Typical range: (Continuous - PPO) 512 - 5120, (Continuous - SAC) 128 - 1024, (Discrete - PPO) 32 - 512] Number of experiences in each iteration of gradient descent. This should always be multiple times smaller than buffer size.")]
        [Min(32)] public int batchSize = 512;

        [Tooltip("[Typical range: (PPO) 2048 - 409600, (SAC) 50000 - 1000000] Number of experiences to collect before updating the policy model. Corresponds to how many experiences should be collected before we do any learning or updating of the model. This should be multiple times larger than batch size. Typically a larger buffer size corresponds to more stable training updates.")]
        [Min(512)] public int bufferSize = 10240; // Do not exagerate with this, keep it at a max of 1M.

        [Tooltip("Applies logarithmic decay on learning rate with respect to the maxSteps. When maxSteps is reached, lr will be 0.")]
        [SerializeField] public bool LRSchedule = false;

        // ========================================================================================================================================================================

        [Header("PPO specific Configuration")] // https://github.com/yosider/ml-agents-1/blob/master/docs/Training-PPO.md

        [Tooltip("[Typical range: 64 - 2048] How many steps of experience to collect per-agent before adding it to the experience buffer.")]
        [Min(32)] public int horizon = 256; 

        [Tooltip("[Typical range: 3 - 10] Number of epochs per buffer.")]
        [Min(3)] public int numEpoch = 8;

        [Tooltip("[Typical range: 1e-4 - 1e-2] Entropy regularization for trainable standard deviation. Also used for Shannon entropy in discrete action space.")]
        [Min(0f)] public float beta = 5e-3f;

        [Tooltip("[Typical range: 0.1 - 0.3] Clip factor.")]
        [Min(0.1f)] public float epsilon = 0.2f;

        [Tooltip("[Typical range: 0.92 - 0.98] GAE factor.")]
        [Min(0.001f)] public float lambda = 0.96f;

        [Tooltip("[Typical range: 0 - 1] Global Gradient Clipping max norm value. Set to 0 to turn off.")]
        [Min(0)] public float gradClipNorm = 0.5f;

        [Tooltip("Use of KLE")]
        public KLType KLDivergence = KLType.Off;

        [Tooltip("Kullback-Leibler divergence target value")]
        [Min(0.015f)] public float targetKL = 0.015f;

        [Tooltip("Normalize the advantages at minibatch level. Might improve convergence for relative large minibatches, but might cause harm when minibatches are small.")]
        public bool normalizeAdvantages = true;

        // ========================================================================================================================================================================

        [Header("SAC specific Configuration")] // https://github.com/yosider/ml-agents-1/blob/master/docs/Training-SAC.md
        [Tooltip("[Typicall range: 1 - 128] Number of steps taken before updating the policy. Doesn't count for parallel agents, this considers only 1 decision.")]
        [Min(1)] public int updateEvery = 64;

        [Tooltip("[Typicall range > Batch Size] Number of steps collected before updating the policy.")]
        [Min(64)] public int updateAfter = 1024;

        [Tooltip("[Typical range: 1 - 8] Number of mini-batches sampled on policy model update. This can be increased while also increasing updateEvery.")]
        [Min(1)] public int updatesNum = 1;

        [Tooltip("[Typicall range: (Continuous) 0.5 - 1.0 ~(Discrete) 0.05 - 0.5] Entropy tradeoff coefficient. Lower alpha means low exploration.")]
        [Min(1e-8f)] public float alpha = 0.2f;

        [Tooltip("[Typicall range: 0.005 - 0.01] Inversed Polyak. How aggresively to update the target network used for boostraping value estimation.")]
        [Min(0.005f)] public float tau = 0.005f;

        // ========================================================================================================================================================================

        [Header("GAIL specific Configuration")]
        [Tooltip("Whether or not to save the record buffer")]
        public bool saveRecordBuffer = false;

        // ========================================================================================================================================================================

        [HideInInspector]
        [Tooltip("Debug the train_data into a file.")]
        [Space(30)]
        public bool debug = false;

        [Space(50)]
        [Tooltip("How does the timescale is adjusted during training.")]
        public TimescaleAdjustmentType timescaleAdjustment = TimescaleAdjustmentType.Dynamic;
        [Tooltip("Timescale of the training session.")]
        [Min(0.01f)] public float timescale = 1f;
        

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

                dontDrawMe.Add("updateEvery");
                dontDrawMe.Add("updateAfter");
                dontDrawMe.Add("updatesNum");
                dontDrawMe.Add("alpha");
                dontDrawMe.Add("saveReplayBuffer");
                dontDrawMe.Add("tau");


                dontDrawMe.Add("saveRecordBuffer");
            }
            else if (script.trainer == TrainerType.SAC)
            {
                dontDrawMe.Add("horizon");
                dontDrawMe.Add("numEpoch");
                dontDrawMe.Add("beta");
                dontDrawMe.Add("epsilon");
                dontDrawMe.Add("lambda");
                dontDrawMe.Add("normalizeAdvantages");
                dontDrawMe.Add("KLDivergence");
                dontDrawMe.Add("targetKL");
                dontDrawMe.Add("gradClipNorm");
                dontDrawMe.Add("normalizeAdvantages");

                dontDrawMe.Add("saveRecordBuffer");

                EditorGUILayout.HelpBox("SAC is not released yet!", MessageType.Warning);
                EditorGUILayout.HelpBox("SAC defaults: lr - 1e-3, batch_size - 128, buffer_size - 1M", MessageType.Info);
            }
            // else if (script.trainer == TrainerType.GAIL)
            // {
            //     dontDrawMe.Add("updateEvery");
            //     dontDrawMe.Add("updateAfter");
            //     dontDrawMe.Add("updatesNum");
            //     dontDrawMe.Add("alpha");
            //     dontDrawMe.Add("saveReplayBuffer");
            //     dontDrawMe.Add("tau");
            // 
            //     dontDrawMe.Add("horizon");
            //     dontDrawMe.Add("numEpoch");
            //     dontDrawMe.Add("beta");
            //     dontDrawMe.Add("epsilon");
            //     dontDrawMe.Add("lambda");
            //     dontDrawMe.Add("normalizeAdvantages");
            //     dontDrawMe.Add("KLDivergence");
            //     dontDrawMe.Add("targetKL");
            //     dontDrawMe.Add("gradClipNorm");
            // 
            //     EditorGUILayout.HelpBox("GAIL is not released yet!", MessageType.Warning);
            // }

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


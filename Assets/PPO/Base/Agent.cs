using DeepUnity.NeuroForge;
using System.Collections.Generic;
using System.Text;
using UnityEditor;
using UnityEngine;

/*
 * Logic
 * 
 * OnTrigger()/OnCollision() => AddReward? / EndEpisode?
 * Update() => Action / Store 
 * LateUpdate() => ResetEpisode?
 */
namespace DeepUnity
{
    [AddComponentMenu("DeepUnity/Agent"), DisallowMultipleComponent, RequireComponent(typeof(HyperParameters))]
    public class Agent : MonoBehaviour
    {
        public string behaviourName = "Behaviour#1";
        public ActorCritic network;
        public BehaviourType behaviour = BehaviourType.Inference;
        public OnEpisodeEndType onEpisodeEnd = OnEpisodeEndType.ResetEnvironment;
        
        [Header("Observations & Actions")]
        public int spaceSize = 2;
        public int continuousActions = 2;
        public int[] discreteBranches = new int[] { };

        
        public MemoryBuffer Memory { get; private set; }
        public HyperParameters Hp { get; private set; }
        public SensorBuffer Observations { get; private set; }
        public ActionBuffer Actions { get; private set; }
        public int CompletedEpisodes { get; private set; }
        public int StepCount { get; private set; }
        public float TimestepReward { get; private set; }
        public float CumulativeReward { get; private set; }

        private List<ISensor> Sensors { get; set; }
        private TransformReseter PersonalEnvironment { get; set; }
        private bool IsEpisodeEnd { get; set; }

        private void Awake()
        {
            Application.targetFrameRate = 60;

            Hp = GetComponent<HyperParameters>();
            Sensors = new List<ISensor>();

            InitNetwork();
            InitBuffers();
            InitSensors(transform);

            PersonalEnvironment = onEpisodeEnd == OnEpisodeEndType.ResetAgent ?
                new TransformReseter(transform) :
                new TransformReseter(transform.parent);

            CompletedEpisodes = 0;
            StepCount = 0;
            TimestepReward = 0;
            CumulativeReward = 0;

            OnAwake();
        }
        private void Start()
        {
            if (behaviour == BehaviourType.Inference)
                Trainer.Subscribe(this);

            OnStart();
        }
        private void Update()
        {
            // AddReward(-1e-5f);

            switch (behaviour)
            {
                case BehaviourType.Inference:
                    InferenceBehavior();
                    break;

                case BehaviourType.Inactive:
                    break;

                case BehaviourType.Active:
                    ActiveBehavior();
                    break;

                case BehaviourType.Manual:
                    ManualBehavior();
                    break;

                case BehaviourType.Test:
                    ActiveBehavior();
                    break;
            }

            StepCount++;          
            TimestepReward = 0;

            if (StepCount == Hp.maxSteps)
                EndEpisode();

            if (behaviour == BehaviourType.Inference && Memory.IsFull())
                Trainer.Ready();

            OnUpdate();          
        }
        private void LateUpdate()
        {
            if (IsEpisodeEnd)
                ResetEpisode();

            OnLateUpdate();
        }


        // Setup
        private void InitNetwork()
        {
            if (network != null)
            {
                continuousActions = network.continuousDim;
                discreteBranches = network.discreteBranches;
            }
            else
            {
                network = new ActorCritic(spaceSize, continuousActions, discreteBranches, Hp, behaviourName);
            }
        }
        private void InitBuffers()
        {
            Memory = new MemoryBuffer(Hp.bufferSize);
            Observations = new SensorBuffer(network.observationSize);
            Actions = new ActionBuffer(network.continuousDim);
        }
        private void InitSensors(Transform parent)
        {
            ISensor sensor = parent.GetComponent<ISensor>();

            if (sensor != null)
                Sensors.Add(sensor);

            foreach (Transform child in parent)
            {
                InitSensors(child);
            }
        }
        private void ResetEpisode()
        {
            if (behaviour == BehaviourType.Active || behaviour == BehaviourType.Inactive)
                return;

            if (Hp.verbose)
            {
                StringBuilder statistic = new StringBuilder();
                statistic.Append("<color=#0c74eb>");
                statistic.Append("Episode: ");
                statistic.Append(CompletedEpisodes + 1);
                statistic.Append(" | Steps: ");
                statistic.Append(StepCount);
                statistic.Append("s | Cumulated Reward: ");
                statistic.Append(CumulativeReward);
                statistic.Append("</color>");
                Debug.Log(statistic.ToString());
            }
            PersonalEnvironment?.Reset();

            IsEpisodeEnd = false;
            CumulativeReward = 0;
            StepCount = 0;
            CompletedEpisodes++;

            OnEpisodeBegin();
        }

        // Loop
        private void ActiveBehavior()
        {

        }
        private void ManualBehavior()
        {
            Actions.Clear();
            Heuristic(Actions);
            OnActionReceived(Actions);
        }
        private void InferenceBehavior()
        {
            Observations.Clear();
            Actions.Clear();

            CollectObservations(Observations);
            Sensors.ForEach(x => Observations.AddObservation(x.GetObservations()));


            // Store all timestep info
            Tensor state = Tensor.Constant(Observations.values);
            Tensor reward = Tensor.Constant(TimestepReward);

            if (Hp.normalize)
            {
                state = network.stateStandardizer.Standardise(state);
                reward = network.rewardStadardizer.Standardise(reward);
            }
               
            Tensor continuous_log_probs;
            Tensor discrete_log_probs;

            Tensor contiunousAction = network.ContinuousAction(state, out continuous_log_probs, out _, out _);
            Tensor discreteAction = network.DiscreteAction(state, out discrete_log_probs);

            Tensor value = network.Value(state);
            Tensor done = Tensor.Constant(IsEpisodeEnd == true ? 1 : 0);

            Memory.Store(state, contiunousAction, discreteAction, continuous_log_probs, discrete_log_probs, value, reward, done);


            // Run agent's actions
            Actions.ContinuousActions = contiunousAction?.ToArray();
            Actions.DiscreteActions = discreteAction?.ToArray();
            OnActionReceived(Actions);
        }



        // User call
        public virtual void OnAwake() { }
        public virtual void OnStart() { }
        public virtual void OnUpdate() { }
        public virtual void OnLateUpdate() { }
        public virtual void OnEpisodeBegin() { }
        public virtual void CollectObservations(SensorBuffer sensorBuffer) { }
        public virtual void OnActionReceived(ActionBuffer actionBuffer) { } 
        public virtual void Heuristic(ActionBuffer actionBuffer) { }
        public void EndEpisode()
        {
            IsEpisodeEnd = true;
        }
        public void AddReward(float reward)
        {
            TimestepReward += reward;
            CumulativeReward += reward;
        }
    }

    public enum BehaviourType
    {
        [Tooltip("Complete inactive.")]
        Inactive,
        [Tooltip("Active behavior. No learning. No scene reset.")]
        Active,
        [Tooltip("Learning. Scene resets.")]
        Inference,
        [Tooltip("Manual control. No learning. Scene resets.")]
        Manual,
        [Tooltip("Active behavior. No Learning. Scene resets.")]
        Test
    }

    public enum OnEpisodeEndType
    {
        ResetAgent,
        ResetEnvironment
    }

    [CustomEditor(typeof(Agent), true), CanEditMultipleObjects]
    sealed class ScriptlessPPOAgent : Editor
    {
        public override void OnInspectorGUI()
        {
            var script = target as Agent;
            string[] dontDrawMe = new string[] { "m_Script" };

            DrawPropertiesExcluding(serializedObject, dontDrawMe);
            serializedObject.ApplyModifiedProperties();
        }
    }
}


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
        public new string name = "Behaviour#1";
        public AgentBehaviour model;

        [Space]
        public int spaceSize = 2;
        [Min(0)]public int continuousActions = 2;
        public int[] discreteBranches = new int[0];

        [Space]      
        public BehaviourType behaviourType = BehaviourType.Inference;
        public OnEpisodeEndType onEpisodeEnd = OnEpisodeEndType.ResetEnvironment;


        public TrajectoryBuffer Trajectory { get; private set; }
        public TimeStep Timestep { get; private set; }
        public HyperParameters Hp { get; private set; }
        public SensorBuffer Observations { get; private set; }
        public ActionBuffer Actions { get; private set; }
        public int CompletedEpisodes { get; private set; }
        public int StepCount { get; private set; }
        public float TimestepReward { get; private set; }
        public float CumulativeReward { get; private set; }

        private List<ISensor> Sensors { get; set; }
        private StateResetter PositionReseter { get; set; }
        private bool IsEpisodeEnd { get; set; } = false;
        private bool FixedUpdateOccured { get; set; } = false;

        private void Awake()
        {
            Hp = GetComponent<HyperParameters>();

            Sensors = new List<ISensor>();

            InitNetwork();
            InitBuffers();
            InitSensors(transform);

            PositionReseter = onEpisodeEnd == OnEpisodeEndType.ResetAgent ?
                new StateResetter(transform) :
                new StateResetter(transform.parent);

            CompletedEpisodes = 0;
            StepCount = 0;
            TimestepReward = 0;
            CumulativeReward = 0;

            OnAfterAwake();

            model.Save();
        }
        private void Start()
        {
            if (behaviourType == BehaviourType.Inference)
                Trainer.Subscribe(this);

            OnAfterStart();
        }
        private void FixedUpdate()
        {
            FixedUpdateOccured = true;

            switch (behaviourType)
            {
                case BehaviourType.Inference:
                    InferenceBehavior();
                    break;

                case BehaviourType.Inactive:
                    break;

                case BehaviourType.Active:
                    ActiveBehavior();
                    break;

                case BehaviourType.Heuristic:
                    ManualBehavior();
                    break;

                case BehaviourType.Test:
                    ActiveBehavior();
                    break;
            }

            OnAfterFixedUpdate();
        }
        private void Update()
        {
            if(FixedUpdateOccured)
            {
                FixedUpdateOccured = false;
                StepCount++;
                

                if (StepCount == Hp.maxStep && !IsEpisodeEnd)
                {
                    EndEpisode();
                    Trajectory.reachedTerminalState = false;                  
                }

                if(IsEpisodeEnd && behaviourType == BehaviourType.Inference)
                {
                    Trainer.Ready(this);
                }

                Timestep.reward = Tensor.Constant(TimestepReward);
                if (Hp.normalize)
                    Timestep.reward = model.rewardStadardizer.Standardise(Timestep.reward);

                Trajectory.Remember(Timestep);

                TimestepReward = 0; // reward[t+1] = 0 -> reset
            }

            OnAfterUpdate();
        }
        private void LateUpdate()
        {
            if (IsEpisodeEnd)
                ResetEpisode();

            OnAfterLateUpdate();
        }


        // Setup
        private void InitNetwork()
        {
            if (model != null)
            {
                continuousActions = model.continuousDim;
                discreteBranches = model.discreteBranches;
            }
            else
            {
                model = new AgentBehaviour(spaceSize, continuousActions, discreteBranches, Hp, name);
            }
        }
        private void InitBuffers()
        {
            Trajectory = new TrajectoryBuffer();
            Timestep = new TimeStep();

            Observations = new SensorBuffer(model.observationSize);
            Actions = new ActionBuffer(model.continuousDim, model.discreteBranches);
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
            if (behaviourType == BehaviourType.Active || behaviourType == BehaviourType.Inactive)
                return;

            if (Hp.verbose)
            {
                StringBuilder statistic = new StringBuilder();
                statistic.Append("<color=#0c74eb>");
                statistic.Append($"Agent {GetInstanceID()} | ");
                statistic.Append($"Episode: {CompletedEpisodes + 1} | ");
                statistic.Append($"Steps: {StepCount}s | ");
                statistic.Append($"Cumulated Reward: {CumulativeReward}");
                statistic.Append("</color>");
                Debug.Log(statistic.ToString());
            }
            PositionReseter?.Reset();

            IsEpisodeEnd = false;
            CumulativeReward = 0;
            StepCount = 0;
            CompletedEpisodes++;

            OnEpisodeBegin();
        }

        // Loop
        private void ActiveBehavior()
        {
            // Observations.Clear();
            // Actions.Clear();
            // 
            // CollectObservations(Observations);
            // Sensors.ForEach(x => Observations.AddObservation(x.GetObservations()));
            // 
            // Tensor state = Tensor.Constant(Observations.values);
            // 
            // if (Hp.normalize)
            // {
            //     state = model.stateStandardizer.Standardise(state);
            // }
            // 
            // Actions.ContinuousActions = model.ContinuousPredict(state, out _)?.ToArray();
            // Actions.DiscreteActions = null;
            // 
            // 
            // OnActionReceived(Actions);
        }
        private void ManualBehavior()
        {
            Actions.Clear();
            Heuristic(Actions);
            OnActionReceived(Actions);
        }
        private void InferenceBehavior()
        {
            // Collect new observations
            Observations.Clear();
            Actions.Clear();
            CollectObservations(Observations);
            Sensors.ForEach(x => Observations.AddObservation(x.GetObservations()));

            // Set state[t], action[t], reward[t]
            Timestep.state = Tensor.Constant(Observations.Observations);

            if (Hp.normalize)
                Timestep.state = model.stateStandardizer.Standardise(Timestep.state);


            Timestep.value = model.Value(Timestep.state);

            Timestep.continuous_action = model.ContinuousPredict(Timestep.state, out Timestep.continuous_log_prob);
            Timestep.discrete_action = model.DiscretePredict(Timestep.state, out Timestep.discrete_log_prob);
            
            // Run agent's actions
            Actions.ContinuousActions = Timestep.continuous_action?.ToArray();
            Actions.DiscreteActions = null; // need to convert afterwards from tensor of logits [branch, logits] to argmax int[]
            OnActionReceived(Actions);
        }


        // User call
        public virtual void OnAfterAwake() { }
        public virtual void OnAfterStart() { }
        public virtual void OnAfterFixedUpdate() { }
        public virtual void OnAfterUpdate() { }
        public virtual void OnAfterLateUpdate() { }
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
        Heuristic,
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


using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq.Expressions;
using System.Text;
using Unity.VisualScripting;
using UnityEditor;
using UnityEditor.EditorTools;
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
        public Model model;
        [SerializeField] private BehaviourType behaviourType = BehaviourType.Inference;

        [Space]
        [SerializeField] private int spaceSize = 2;
        [SerializeField, Min(0)] private int continuousActions = 2;
        [SerializeField] private int[] discreteBranches = new int[0];

        [Space]         
        [SerializeField] private OnEpisodeEndType onEpisodeEnd = OnEpisodeEndType.ResetEnvironment;
        [SerializeField, Tooltip("Request an action at each fixed timestep/frame.")] private bool autoRequestAction = true;
        [SerializeField, Tooltip("Auto use the sensors information attached to this agent.")] private bool useSensors = true;
        
        public HyperParameters Hp { get; private set; }
        private List<ISensor> Sensors { get; set; }
        private StateResetter PositionReseter { get; set; }
        public TrajectoryBuffer Trajectory { get; private set; }
        public TimeStep Timestep { get; private set; }    
        public SensorBuffer Observations { get; private set; }
        public ActionBuffer Actions { get; private set; }
        public int CompletedEpisodes { get; private set; }
        public int StepCount { get; private set; }
        public float TimestepReward { get; private set; }
        public float CumulativeReward { get; private set; }       
        private bool IsEpisodeEnd { get; set; } = false;
        private bool FixedUpdateOccured { get; set; } = false;
        private bool ActionRequested { get; set; } = false;

        public virtual void Awake()
        {
            Hp = GetComponent<HyperParameters>();
            Time.fixedDeltaTime = 1f / Hp.targetFPS;
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

            if (behaviourType == BehaviourType.Inference)
                Trainer.Subscribe(this);

            model.Save();
        }
        public virtual void FixedUpdate()
        {
            FixedUpdateOccured = true;
            if (autoRequestAction)
                RequestAction();
        }
        public virtual void Update()
        {
            if (!FixedUpdateOccured)
                return;

            if (!ActionRequested)
                return;



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



            StepCount++;

            if (StepCount == Hp.maxEpisodeSteps && !IsEpisodeEnd)
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
            Timestep = new TimeStep();
        }
        public virtual void LateUpdate()
        {
            if(FixedUpdateOccured && ActionRequested)
            {
                FixedUpdateOccured = false;             
                ActionRequested = false;

                if(IsEpisodeEnd)
                {
                    IsEpisodeEnd = false;

                    ResetEpisode();
                    OnEpisodeBegin();
                    TimestepReward = 0; // reward[t+1] = 0 -> reset
                }
               
            }          
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
                string[] modelFoundGUID = AssetDatabase.FindAssets(GetType().Name);

                if(modelFoundGUID.Length == 0)
                {
                    model = new Model(spaceSize, continuousActions, discreteBranches, Hp, GetType().Name);
                    return;
                }
                string modelFoundPath = AssetDatabase.GUIDToAssetPath(modelFoundGUID[0]);

                if (modelFoundPath.EndsWith(".cs"))
                {
                    model = new Model(spaceSize, continuousActions, discreteBranches, Hp, GetType().Name);
                    return;
                }
                

                model = AssetDatabase.LoadAssetAtPath<Model>(modelFoundPath);
                Debug.Log($"<b>{GetType().Name}<b/> model auto-loaded from project Assets.");

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
       
        // Loop
        private void ActiveBehavior()
        {
            throw new NotImplementedException();
            // Observations.Clear();
            // Actions.Clear();
            // 
            // CollectObservations(Observations);
            // if(useSensors) Sensors.ForEach(x => Observations.AddObservation(x.GetObservations()));
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
            if(useSensors) Sensors.ForEach(x => Observations.AddObservation(x.GetObservations()));

            // Set state[t], action[t], reward[t]
            Timestep.state = Tensor.Identity(Observations.Observations);

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
        public virtual void OnEpisodeBegin() { }
        public virtual void CollectObservations(SensorBuffer sensorBuffer) { }
        public virtual void OnActionReceived(ActionBuffer actionBuffer) { } 
        public virtual void Heuristic(ActionBuffer actionBuffer) { }
        /// <summary>
        /// Called only on <b>FixedUpdate()</b>, <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b>.
        /// </summary>
        public void RequestAction() => ActionRequested = true;
        /// <summary>
        /// Called only on <b>FixedUpdate()</b>, <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b>.
        /// </summary>
        public void EndEpisode() => IsEpisodeEnd = true;
        /// <summary>
        /// Called only on <b>FixedUpdate()</b>, <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b>.
        /// </summary>
        public void AddReward(float reward)
        {
            TimestepReward += reward;
            CumulativeReward += reward;
        }

        // Inner logic
        private void ResetEpisode()
        {
            if (behaviourType == BehaviourType.Active || behaviourType == BehaviourType.Inactive)
                return;

            if (Hp.verbose)
            {
                StringBuilder statistic = new StringBuilder();
                int id = GetInstanceID();
                if (CumulativeReward > 0)
                    statistic.Append("<color=#0c74eb>");
                else if (CumulativeReward < 0)
                    statistic.Append("<color=#ff4000>");
                else 
                    statistic.Append("<color=#696969>");

                statistic.Append($"Agent [#{id}] | ");
                statistic.Append($"Episode: {CompletedEpisodes + 1} | ");
                statistic.Append($"Steps: {StepCount} | ");
                statistic.Append($"Cumulated Reward: {CumulativeReward}");
                statistic.Append("</color>");
                Debug.Log(statistic.ToString());
            }
            PositionReseter?.Reset();


            CumulativeReward = 0;
            StepCount = 0;
            CompletedEpisodes++;
        }
    }

    public enum BehaviourType
    {
        [Tooltip("Latent behaviour. Learning: NO. Scene resets: NO.")]
        Inactive,
        [Tooltip("Active behaviour. Learning: NO. Scene resets: NO.")]
        Active,
        [Tooltip("Exploring behaviour. Learning: YES. Scene resets: YES.")]
        Inference,
        [Tooltip("Manual control. Learning: NO. Scene resets: YES.")]
        Heuristic,
        [Tooltip("Active behavior. Learning: NO. Scene resets: YES.")]
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
            string[] dontDrawMe = new string[] { "m_Script" };

            DrawPropertiesExcluding(serializedObject, dontDrawMe);
            serializedObject.ApplyModifiedProperties();
        }
    }
}


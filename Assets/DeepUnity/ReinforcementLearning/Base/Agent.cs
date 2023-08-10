using System;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
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
    [AddComponentMenu("DeepUnity/Agent"), DisallowMultipleComponent, RequireComponent(typeof(HyperParameters)), RequireComponent(typeof(DecisionRequester))]
    public class Agent : MonoBehaviour
    {
        [SerializeField] public AgentBehaviour model;
        [SerializeField] private BehaviourType behaviourType = BehaviourType.Learn;

        [Space]
        [SerializeField] private int spaceSize = 2;
        [SerializeField, Min(0)] private int continuousActions = 2;
        [SerializeField] private int[] discreteBranches = new int[0];

        [Space]         
        [SerializeField] 
        private OnEpisodeEndType onEpisodeEnd = OnEpisodeEndType.ResetEnvironment;
        [SerializeField, Tooltip("Collect sensors' observations attached to this agent automatically. ")] 
        private bool useSensors = true;


        [HideInInspector] public AgentPerformanceTracker PerformanceTracker;
        public HyperParameters Hp { get; private set; }
        public DecisionRequester DecisionRequester { get; private set; }
        private List<ISensor> Sensors { get; set; }
        private StateResetter PositionReseter { get; set; }
        public TrajectoryBuffer Trajectory { get; private set; }
        public TimeStep Timestep { get; private set; }    
        public SensorBuffer Observations { get; private set; }
        public ActionBuffer Actions { get; private set; }
        public int StepCount { get; private set; } = 1;
        public float TimestepReward { get; private set; } = 0f;    
        private bool IsEpisodeEnd { get; set; } = false;
        private bool ActionOccured { get; set; } = false;

        public virtual void Awake()
        {
            Hp = GetComponent<HyperParameters>();
            DecisionRequester = GetComponent<DecisionRequester>();
            TryGetComponent(out PerformanceTracker);
            Time.fixedDeltaTime = 1f / Hp.targetFPS;
            Sensors = new List<ISensor>();

            InitNetwork();
            InitBuffers();
            InitSensors(transform);

            PositionReseter = onEpisodeEnd == OnEpisodeEndType.ResetAgent ?
                new StateResetter(transform) :
                new StateResetter(transform.parent);
        }
        public virtual void Start()
        {
            if (behaviourType == BehaviourType.Learn)
                Trainer.Subscribe(this);
        }
        public virtual void FixedUpdate()
        {
            if (!DecisionRequester.DoITakeActionThisFrame())
                return;

            ActionOccured = true;

            switch (behaviourType)
            {
                case BehaviourType.Inactive:
                    break;

                case BehaviourType.Learn:
                    LearnBehaviour();
                    break;

                case BehaviourType.Active:
                    ActiveBehavior();
                    break;

                case BehaviourType.Manual:
                    ManualBehaviour();
                    break;

                default: throw new NotImplementedException("Unhandled behaviour type!");
            }
        }
        public virtual void Update()
        {
            if (!ActionOccured)
                return;

            // If the agent reached max steps without reaching the terminal state
            if (StepCount == Hp.maxSteps && !IsEpisodeEnd) 
            {
                EndEpisode();
                Trajectory.reachedTerminalState = false;                  
            }

            if(IsEpisodeEnd && behaviourType == BehaviourType.Learn)
            {
                Trainer.Ready(this);
            }


            // Norm the reward
            Timestep.reward = Tensor.Constant(TimestepReward);
            if (model.rewardNormalizer != null)
                Timestep.reward = model.rewardNormalizer.Normalize(Timestep.reward);

            // Remember the timestep
            Trajectory.Remember(Timestep);
        }
        public virtual void LateUpdate()
        {
            if (!ActionOccured)
                return;

            ActionOccured = false;

            TimestepReward = 0; // reward[t+1] = 0 -> reset
            Timestep = new TimeStep();
            

            if (IsEpisodeEnd)
            {
                IsEpisodeEnd = false;             
                PositionReseter?.Reset();
                PerformanceTracker.episodesCompleted++;
                PerformanceTracker.episodeLength.Append(StepCount);
                StepCount = 0;
                

                OnEpisodeBegin();          
            }

            StepCount++;
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
                    model = new AgentBehaviour(spaceSize, continuousActions, discreteBranches).CreateAsset(GetType().Name);
                    return;
                }
                string modelFoundPath = AssetDatabase.GUIDToAssetPath(modelFoundGUID[0]);

                if (modelFoundPath.EndsWith(".cs"))
                {
                    model = new AgentBehaviour(spaceSize, continuousActions, discreteBranches).CreateAsset(GetType().Name);
                    return;
                }
                

                model = AssetDatabase.LoadAssetAtPath<AgentBehaviour>(modelFoundPath);
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
        private void LearnBehaviour()
        {
            // Collect new observations
            Observations.Clear();
            Actions.Clear();
            CollectObservations(Observations);
            if (useSensors) Sensors.ForEach(x => Observations.AddObservation(x.GetObservations()));

            // Set state[t], action[t], reward[t]
            Timestep.state = Tensor.Identity(Observations.Observations);

            if (model.stateStandardizer != null)
                Timestep.state = model.stateStandardizer.Normalize(Timestep.state);


            Timestep.value = model.Value(Timestep.state);

            Timestep.continuous_action = model.ContinuousPredict(Timestep.state, out Timestep.continuous_log_prob);
            Timestep.discrete_action = model.DiscretePredict(Timestep.state, out Timestep.discrete_log_prob);

            // Run agent's actions
            Actions.ContinuousActions = Timestep.continuous_action?.ToArray();
            Actions.DiscreteActions = null; // need to convert afterwards from tensor of logits [branch, logits] to argmax int[]

            if(DecisionRequester.randomAction)
            {
                if(Actions.ContinuousActions != null)
                    Actions.ContinuousActions =  Actions.ContinuousActions.Select(x => Utils.Random.Range(-1f, 1f)).ToArray();
            
                if (Actions.DiscreteActions != null)
                    throw new NotImplementedException();
            
            }

            OnActionReceived(Actions);
        }
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
        private void ManualBehaviour()
        {
            Actions.Clear();
            Heuristic(Actions);
            OnActionReceived(Actions);
        }


        // User call
        public virtual void OnEpisodeBegin() { }
        public virtual void CollectObservations(SensorBuffer sensorBuffer) { }
        public virtual void OnActionReceived(ActionBuffer actionBuffer) { } 
        public virtual void Heuristic(ActionBuffer actionBuffer) { }

        /// <summary>
        /// Called only on <b>FixedUpdate()</b>, <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b>. <br></br>
        /// Ensures the agent will perform an action this frame.
        /// </summary>
        public void RequestAction() => DecisionRequester.TakeActionThisFrame = true;
        /// <summary>
        /// Called only on <b>FixedUpdate()</b>, and <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b>. <br></br>
        /// Ensures the episode will end this frame. Does not require the agent to take action this frame.
        /// </summary>
        public void EndEpisode() => IsEpisodeEnd = true;
        /// <summary>
        /// Called only on <b>FixedUpdate()</b>, and <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b>. <br></br>
        /// Modifies the reward of the current time step. Does not require the agent to take action this frame.
        /// </summary>
        /// <param name="reward">positive or negative</param>
        public void AddReward(float reward) => TimestepReward += reward;
    }

   

    [CustomEditor(typeof(Agent), true), CanEditMultipleObjects]
    sealed class CustomAgentEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            List<string> dontDrawMe = new(){ "m_Script" };

            SerializedProperty mod = serializedObject.FindProperty("model");

            if(mod.objectReferenceValue != null)
            {
                dontDrawMe.Add("spaceSize");
                dontDrawMe.Add("continuousActions");
                dontDrawMe.Add("discreteBranches");
            }
            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
}


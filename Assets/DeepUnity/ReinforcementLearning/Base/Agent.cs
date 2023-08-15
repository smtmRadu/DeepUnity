using System;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [AddComponentMenu("DeepUnity/Agent"), DisallowMultipleComponent, RequireComponent(typeof(DecisionRequester))]
    public class Agent : MonoBehaviour
    {
        [SerializeField] public AgentBehaviour model;
        [SerializeField] public Hyperparameters hp;
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


        public TrainingStatistics PerformanceTrack { get; private set; }
        public Trajectory Trajectory { get; set; }
        private TimeStep Timestep { get; set; }
        private DecisionRequester DecisionRequester { get; set; }
        private List<ISensor> Sensors { get; set; }
        private StateResetter PositionReseter { get; set; }
        private SensorBuffer Observations { get; set; }
        private ActionBuffer Actions { get; set; }
        private int StepCount { get; set; } = 1;
        private bool IsEpisodeEnd { get; set; } = false;
        private bool ActionOccured { get; set; } = false;


        public virtual void Awake()
        {
            DecisionRequester = GetComponent<DecisionRequester>();
            Sensors = new List<ISensor>();
            InitSensors(transform);
            InitBuffers();                    
        }
        public virtual void Start()
        {
            if (behaviourType != BehaviourType.Learn)
                return;

            Time.fixedDeltaTime = 1f / hp.targetFPS;
            TrainingStatistics pf;
            TryGetComponent(out pf);
            PerformanceTrack = pf;
            PositionReseter = onEpisodeEnd == OnEpisodeEndType.ResetAgent ?
                new StateResetter(transform) :
                new StateResetter(transform.parent);


            Trainer.Subscribe(this);
        }
        public virtual void FixedUpdate()
        {
            if (!DecisionRequester.DoITakeActionThisFrame())
                return;

            ActionOccured = true;

            if (behaviourType == BehaviourType.Inactive)
            {
                return;

            }

            if (behaviourType == BehaviourType.Manual)
            {
                Actions.Clear();
                Heuristic(Actions);
                OnActionReceived(Actions);
            }

            if (behaviourType == BehaviourType.Learn)
            {
                // Collect new observations
                Observations.Clear();
                Actions.Clear();
                CollectObservations(Observations);
                if (useSensors) Sensors.ForEach(x => Observations.AddObservation(x.GetObservations()));

                // Normalize the observations if requested
                if (hp.normalize)
                {
                    model.stateNormalizer.Update(Observations.Observations);
                    Observations.Observations = model.stateNormalizer.Normalize(Observations.Observations);
                }

                // Set state[t], action[t] & pi[t]
                Timestep.state = Tensor.Identity(Observations.Observations);
                model.ContinuousPredict(Timestep.state, out Timestep.action_continuous, out Timestep.piold_continuous);
                model.DiscretePredict(Timestep.state, out Timestep.action_discrete, out Timestep.piold_discrete);

                // Run agent's actions and clip them
                Actions.ContinuousActions = DecisionRequester.randomAction ? Tensor.RandomRange((-1f, 1f), continuousActions).ToArray() : Timestep.action_continuous.Clip(-1f, 1f).ToArray();
                Actions.DiscreteActions = null; // need to convert afterwards from tensor of logits [branch, logits] to argmax int[]

                OnActionReceived(Actions);
            }
        }
        public virtual void Update()
        {
            if (behaviourType != BehaviourType.Learn)
                return;

            if (!ActionOccured)
                return;

            // reward[t] already set
            Trajectory.Add(Timestep);

            // CHECK MAX STEPS: If the agent reached max steps without reaching the terminal state
            if (StepCount <= hp.maxSteps && !IsEpisodeEnd) 
            {
                EndEpisode();
                Trajectory.reachedTerminalState = false;                  
            }

            if (IsEpisodeEnd)
            {
                // Set advantage[t] & v_target
                Trajectory.ComputeAdvantagesAndVTargets(hp.gamma, hp.lambda, model.critic);
                // trajectory.NormAdvantages();
                Trainer.Ready(this);
            }

            
        }
        public virtual void LateUpdate()
        {
            if (behaviourType != BehaviourType.Learn)
                return;

            if (!ActionOccured)
                return;

            ActionOccured = false;
            StepCount++;
            Timestep = new TimeStep();
            

            if (IsEpisodeEnd)
            {
                IsEpisodeEnd = false;             
                PositionReseter?.Reset();
                if(PerformanceTrack != null)
                {
                    PerformanceTrack.episodesCompleted++;
                    PerformanceTrack.episodeLength.Append(StepCount - 1);
                }             
                StepCount = 0;
                OnEpisodeBegin();          
            }           
        }

        // Setup
        public void BakeModel()
        {
            model = AgentBehaviour.CreateOrLoadAsset(GetType().Name, spaceSize, continuousActions, discreteBranches);

            continuousActions = model.continuousDim;
            discreteBranches = model.discreteBranches;
        }
        public void BakeHyperParamters()
        {
            if (hp != null)
                return;
            hp = Hyperparameters.CreateOrLoadAsset(GetType().Name);

        }
        private void InitBuffers()
        {
            Trajectory = new Trajectory();
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


        // User call
        public virtual void OnEpisodeBegin() { }
        public virtual void CollectObservations(SensorBuffer sensorBuffer) { }
        public virtual void OnActionReceived(ActionBuffer actionBuffer) { } 
        public virtual void Heuristic(ActionBuffer actionBuffer) { }
        /// <summary>
        /// Called only on <b>FixedUpdate()</b>, <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b>. <br></br>
        /// Ensures the agent will perform an action this frame.
        /// </summary>
        public void RequestAction() => DecisionRequester.decisionWasRequested = true;
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
        public void AddReward(float reward)
        {
            if(Timestep.reward == null)
                Timestep.reward = Tensor.Constant(reward);    
            else
                Timestep.reward += reward;
        }
    }

    [CustomEditor(typeof(Agent), true), CanEditMultipleObjects]
    sealed class CustomAgentEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            List<string> dontDrawMe = new(){ "m_Script" };

            SerializedProperty modelProperty = serializedObject.FindProperty("model");
            SerializedProperty hpProperty = serializedObject.FindProperty("hp");
            var script = (Agent)target;
            
            if(modelProperty.objectReferenceValue == null || hpProperty.objectReferenceValue == null)
                if (GUILayout.Button("Bake/Load model and hyperparameters"))
                {
                    script.BakeModel();
                    script.BakeHyperParamters();
                }

            if (modelProperty.objectReferenceValue != null)
            {
                dontDrawMe.Add("spaceSize");
                dontDrawMe.Add("continuousActions");
                dontDrawMe.Add("discreteBranches");
            }

            SerializedProperty beh = serializedObject.FindProperty("behaviourType");

            if(beh.enumValueIndex != (int)BehaviourType.Learn)
            {
                dontDrawMe.Add("Hp");
            }

            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
}


using System;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;


/// <summary>
///  https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Learning-Environment-Design-Agents.md
/// </summary>
/// 
namespace DeepUnity
{
    [DisallowMultipleComponent, RequireComponent(typeof(DecisionRequester))]
    public abstract class Agent : MonoBehaviour
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
        [SerializeField, Tooltip("Collect automatically the [Compressed] Observation Vector of attached sensors to this GameObject, or any child GameObject of any degree of it. Consider the number of Observation Vector's float values when defining the Space Size.")]
        private UseSensorsType useSensors = UseSensorsType.ObservationsVector;

        public TrainingStatistics PerformanceTrack { get; set; }
        public ExperienceBuffer Memory { get; set; }
        private TimestepBuffer Timestep { get; set; }
        private DecisionRequester DecisionRequester { get; set; }
        private List<ISensor> Sensors { get; set; }
        private StateResetter PositionReseter { get; set; }
        private SensorBuffer Observations { get; set; }
        private ActionBuffer Actions { get; set; }
        public bool ActionOccured { get; private set; } = false;
        private int EpisodeStepCount { get; set; } = 1;
        private float EpsiodeCumulativeReward { get; set; } = 0f;      
        private bool FixedUpdateOccured { get; set; } = false;
        private bool UpdateOccured { get; set; } = false;
        private bool LateUpdateOccured { get; set; } = false;

        public virtual void Awake()
        {
            DecisionRequester = GetComponent<DecisionRequester>();
            Sensors = new List<ISensor>();
            InitSensors(transform);
            InitBuffers();
            switch (onEpisodeEnd)
            {
                case OnEpisodeEndType.Nothing:
                    PositionReseter = null;
                    break;
                case OnEpisodeEndType.ResetAgent:
                    PositionReseter = new StateResetter(transform);
                    break;
                case OnEpisodeEndType.ResetEnvironment:
                    PositionReseter = new StateResetter(transform.parent);
                    break;
            }
        }
        public virtual void Start()
        {
            if (behaviourType != BehaviourType.Learn)
                return;

            TrainingStatistics pf;
            TryGetComponent(out pf);
            PerformanceTrack = pf;

            Trainer.Subscribe(this);
            OnEpisodeBegin();
        }
        public virtual void FixedUpdate()
        {
            if (FixedUpdateOccured)
                return;

            FixedUpdateOccured = true;
            UpdateOccured = false;

            // ----------------------------------------------------------------------------------


            if (behaviourType == BehaviourType.Inactive)
                return;

            ActionOccured = true;
            Timestep.done = Tensor.Constant(0);
            Timestep.reward = Tensor.Constant(0);

            if (behaviourType == BehaviourType.Learn && DecisionRequester.DoITakeActionThisFrame())
            {
                // Collect new observations
                Observations.Clear();
                Actions.Clear();
                CollectObservations(Observations);
                if (useSensors == UseSensorsType.ObservationsVector) 
                    Sensors.ForEach(x => Observations.AddObservation(x.GetObservationsVector()));
                else
                    Sensors.ForEach(x => Observations.AddObservation(x.GetCompressedObservationsVector()));


                // Normalize the observations if neccesary
                if (model.normalizeObservations)
                {
                    model.normalizer.Update(Observations.Observations);
                    Observations.Observations = model.normalizer.Normalize(Observations.Observations);
                }

                // Set state[t], action[t] & pi[t]
                Timestep.state = Tensor.Identity(Observations.Observations);
                model.ContinuousPredict(Timestep.state, out Timestep.action_continuous, out Timestep.prob_continuous);
                model.DiscretePredict(Timestep.state, out Timestep.action_discrete, out Timestep.prob_discrete);

                // Run agent's actions and clip them
                Actions.ContinuousActions = model.IsUsingContinuousActions ? Timestep.action_continuous.Clip(-1f, 1f).ToArray() : null;
                Actions.DiscreteActions = null; // need to convert afterwards from tensor of logits [branch, logits] to argmax int[]

                OnActionReceived(Actions);
            }
            else if (behaviourType == BehaviourType.Active && DecisionRequester.DoITakeActionThisFrame())
            {
                // Collect new observations
                Observations.Clear();
                Actions.Clear();
                CollectObservations(Observations);
                if (useSensors == UseSensorsType.ObservationsVector)
                    Sensors.ForEach(x => Observations.AddObservation(x.GetObservationsVector()));
                else
                    Sensors.ForEach(x => Observations.AddObservation(x.GetCompressedObservationsVector()));


                // Normalize the observations if neccesary
                if (model.normalizeObservations)
                    Observations.Observations = model.normalizer.Normalize(Observations.Observations);

                // Set state[t], action[t] & pi[t]
                Timestep.state = Tensor.Identity(Observations.Observations);
                model.ContinuousPredict(Timestep.state, out Timestep.action_continuous, out Timestep.prob_continuous);
                model.DiscretePredict(Timestep.state, out Timestep.action_discrete, out Timestep.prob_discrete);

                // Run agent's actions and clip them
                Actions.ContinuousActions = model.IsUsingContinuousActions ? Timestep.action_continuous.Clip(-1f, 1f).ToArray() : null;
                Actions.DiscreteActions = null; // need to convert afterwards from tensor of logits [branch, logits] to argmax int[]

                OnActionReceived(Actions);
            }
            else if(behaviourType == BehaviourType.Manual)
            {
                Actions.Clear();
                Heuristic(Actions);
                OnActionReceived(Actions);
            }
        }
        public virtual void Update()
        {
            if (UpdateOccured)
                return;

            if (!ActionOccured)
                return;

            UpdateOccured = true;
            LateUpdateOccured = false;

            // ----------------------------------------------------------------------------------

            if (behaviourType == BehaviourType.Learn)
            {
                Timestep.index = EpisodeStepCount;

                // reward[t] object already set to 0 in FixedUpdate()
                Memory.Add(Timestep);

                // CHECK MAX STEPS: If the agent reached max steps without reaching the terminal state
                if (EpisodeStepCount == DecisionRequester.maxStep)
                    EndEpisode();

                if (Memory.IsFull())
                    Trainer.ReadyToTrain();
            }            
        }
        public virtual void LateUpdate()
        {
            if (LateUpdateOccured)
                return;

            if (!ActionOccured)
                return;

            LateUpdateOccured = true;
            FixedUpdateOccured = false;
            ActionOccured = false;

            // ----------------------------------------------------------------------------------
            if (behaviourType == BehaviourType.Inactive)
                return;

            if (behaviourType == BehaviourType.Learn)
            {
                EpisodeStepCount++;

                if (Timestep.done[0] == 1)
                {                  
                    PerformanceTrack?.episodeLength.Append(EpisodeStepCount);
                    PerformanceTrack?.cumulativeReward.Append(EpsiodeCumulativeReward);
                    EpisodeStepCount = 1;
                    EpsiodeCumulativeReward = 0f;

                    PositionReseter?.Reset();
                    OnEpisodeBegin();
                }
            }
            else if(behaviourType == BehaviourType.Active)
            {
                if (Timestep.done[0] == 1)
                {
                    PositionReseter?.Reset();
                    OnEpisodeBegin();
                }
                   
            }
            else if (behaviourType == BehaviourType.Manual)
            {
                if (Timestep.done[0] == 1)
                {
                    PositionReseter?.Reset();
                    OnEpisodeBegin();
                }
                    
            }

            Timestep = new TimestepBuffer(); 
        }
  
        public void BakeModel()
        {
            model = AgentBehaviour.CreateOrLoadAsset(GetType().Name, spaceSize, continuousActions, discreteBranches);

            continuousActions = model.continuousDim;
            discreteBranches = model.discreteBranches;
        }
        public void BakeHyperparamters()
        {
            hp = Hyperparameters.CreateOrLoadAsset(GetType().Name);
        }
        private void InitBuffers()
        {
            Memory = new ExperienceBuffer(hp.bufferSize);
            Timestep = new TimestepBuffer();

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
        /// <summary>
        /// Reinitializes the environment stochastically. If one of the parallel agents reaches a terminal state, 
        /// all environments are reinitialized also.
        /// </summary>
        public virtual void OnEpisodeBegin() { }
        public abstract void CollectObservations(SensorBuffer sensorBuffer);
        public abstract void OnActionReceived(ActionBuffer actionBuffer);
        public virtual void Heuristic(ActionBuffer actionBuffer) { }
        /// <summary>
        /// Called from <b>anywhere</b>, but if the action requires to happen in the same frame, needs to be called before base.<b>FixedUpdate()</b>. <br></br>
        /// Ensures the agent will perform an action in the next frame*.
        /// </summary>
        public void RequestDecision()
        {
            DecisionRequester.decisionWasRequested = true;
        }
        /// <summary>
        /// Called only inside <b>OnActionReceived()</b>, and <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b>. <br></br>
        /// Ensures the episode will end this frame for the current agent. (Theoretically all parallel environments must reinitialized to the initial state, but not here)
        /// </summary>
        public void EndEpisode()
        {
            Timestep.done = Tensor.Constant(1);
        }
        /// <summary>
        /// Called only inside <b>OnActionReceived()</b>, and <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b> <br></br>
        /// Modifies the reward of the current time step.
        /// </summary>
        /// <param name="reward">positive or negative</param>
        public void AddReward(float reward)
        {
            if(Timestep.reward == null)
            {
                ConsoleMessage.Warning($"Cannot add reward {reward} before taking an action");
                return;
            }

            Timestep.reward += reward;
            EpsiodeCumulativeReward += reward;
        }
        /// <summary>
        /// Called only inside <b>OnActionReceived()</b>, and <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b> <br></br>
        /// Modifies the reward of the current time step.
        /// </summary>
        /// <param name="reward">positive or negative</param>
        public void SetReward(float reward)
        {
            if (Timestep.reward == null)
            {
                ConsoleMessage.Warning($"Cannot set reward {reward} before taking an action");
                return;
            }

            Timestep.reward[0] = reward;
            EpsiodeCumulativeReward += reward;
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
                    script.BakeHyperparamters();
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


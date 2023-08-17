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
    [AddComponentMenu("DeepUnity/Agent"), DisallowMultipleComponent, RequireComponent(typeof(DecisionRequester))]
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
        [SerializeField, Tooltip("Collect sensors' observations attached to this agent automatically. ")]
        private bool useSensors = true;

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

        private void Awake()
        {
            DecisionRequester = GetComponent<DecisionRequester>();
            Sensors = new List<ISensor>();
            InitSensors(transform);
            InitBuffers();
        }
        private void Start()
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
            OnEpisodeBegin();
        }
        private void FixedUpdate()
        {
            if (FixedUpdateOccured)
                return;

            FixedUpdateOccured = true;
            UpdateOccured = false;

            if(DecisionRequester.DoITakeActionThisFrame())
            {
                ActionOccured = true;

                switch (behaviourType)
                {
                    case BehaviourType.Inactive:
                        break;
                    case BehaviourType.Manual:
                        Actions.Clear();
                        Heuristic(Actions);
                        OnActionReceived(Actions);
                        break;
                    case BehaviourType.Learn:

                        // Collect new observations
                        Observations.Clear();
                        Actions.Clear();
                        CollectObservations(Observations);
                        if (useSensors) Sensors.ForEach(x => Observations.AddObservation(x.GetObservations()));

                        // Normalize the observations if neccesary
                        if (hp.normalizeObservations)
                        {
                            model.stateNormalizer.Update(Observations.Observations);
                            Observations.Observations = model.stateNormalizer.Normalize(Observations.Observations);
                        }

                        // Set state[t], action[t] & pi[t]
                        Timestep.done = Tensor.Constant(0);
                        Timestep.state = Tensor.Identity(Observations.Observations);
                        model.ContinuousPredict(Timestep.state, out Timestep.action_continuous, out Timestep.log_probs_continuous);
                        model.DiscretePredict(Timestep.state, out Timestep.action_discrete, out Timestep.log_probs_discrete);

                        // Run agent's actions and clip them
                        Actions.ContinuousActions = DecisionRequester.randomAction ? Tensor.RandomRange((-1f, 1f), continuousActions).ToArray() : Timestep.action_continuous.Clip(-1f, 1f).ToArray();
                        Actions.DiscreteActions = null; // need to convert afterwards from tensor of logits [branch, logits] to argmax int[]

                        OnActionReceived(Actions);
                        break;
                    default: throw new NotImplementedException("Unhandled behaviour type");
                }
            }
        }
        private void Update()
        {
            if (behaviourType != BehaviourType.Learn)
                return;

            if (UpdateOccured)
                return;

            if (!ActionOccured)
                return;

            UpdateOccured = true;
            LateUpdateOccured = false;
            

            Timestep.index = EpisodeStepCount;

            // reward[t] already set
            Memory.Add(Timestep);

            // CHECK MAX STEPS: If the agent reached max steps without reaching the terminal state
            if (EpisodeStepCount == hp.maxSteps)
                EndEpisode();

            if (Memory.IsFull())
                Trainer.ReadyToTrain();
        }
        private void LateUpdate()
        {
            if (behaviourType != BehaviourType.Learn)
                return;

            if (LateUpdateOccured)
                return;

            if (!ActionOccured)
                return;


            LateUpdateOccured = true;
            FixedUpdateOccured = false;
            ActionOccured = false;





            EpisodeStepCount++;

            if (Timestep.done[0] == 1)
            {
                PositionReseter?.Reset();
                PerformanceTrack?.episodeLength.Append(EpisodeStepCount);
                PerformanceTrack?.cumulativeReward.Append(EpsiodeCumulativeReward);
                EpisodeStepCount = 1;
                EpsiodeCumulativeReward = 0f;
                OnEpisodeBegin();
            }

            Timestep = new TimestepBuffer();
        }

   
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
                Timestep.reward = Tensor.Constant(reward);    
            else
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
                Timestep.reward = Tensor.Constant(reward);
            else
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


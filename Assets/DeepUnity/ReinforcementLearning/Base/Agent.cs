using System;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

/// <summary>
///  https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Learning-Environment-Design-Agents.md
/// </summary> 
namespace DeepUnity
{
    [DisallowMultipleComponent, RequireComponent(typeof(DecisionRequester))]
    public abstract class Agent : MonoBehaviour
    {
        [Tooltip("The number of observations received at a specific timestep.")]
        [SerializeField, Min(1), HideInInspector] private int spaceSize = 0;
        [Tooltip("The inputs are enqueued in a queue buffer. t1 = [0, 0, x1] -> t2 = [0, x1, x2] -> t3 = [x1, x2, x3] -> t4 = [x2, x3, x4] -> .... For now they are disabled.")]
        [SerializeField, Range(1, 32), HideInInspector] private int stackedInputs = 1;
        [Tooltip("Number of Continuous Actions in continuous actions space. Values in range [-1, 1].")]
        [SerializeField, Min(0), HideInInspector] private int continuousActions = 0;
        [Tooltip("Number of discrete actions in discrete action space. Actions indexes [0, 1, .. n - 1]. Consider to include a 'NO_ACTION' index also (especially when working heuristically).")]
        [SerializeField, Min(0), HideInInspector] private int discreteActions = 0;
        [Tooltip("Number of hidden layers")]
        [SerializeField, Min(1), HideInInspector] private int numLayers = 2;
        [Tooltip("Number of units in a hidden layer in the model.")]
        [SerializeField, Min(32), HideInInspector] private int hidUnits = 64;

        [SerializeField, HideInInspector] public AgentBehaviour model;
        [SerializeField, HideInInspector] public BehaviourType behaviourType = BehaviourType.Learn;

        [Space]
        [SerializeField, HideInInspector, Tooltip("What happens after the episode ends? You can reset the agent's/environment's position and rigidbody automatically using this. OnEpisodeBegin() is called afterwards in any situation.")]
        private OnEpisodeEndType onEpisodeEnd = OnEpisodeEndType.ResetEnvironment;
        [SerializeField, HideInInspector, Tooltip("Collect automatically the [Compressed] Observation Vector of attached sensors (if any) to this GameObject, and any child GameObject of any degree. Consider the number of Observation Vector's float values when defining the Space Size.")]
        private UseSensorsType useSensors = UseSensorsType.ObservationsVector;

        public TrainingStatistics PerformanceTrack { get; set; }
        public MemoryBuffer Memory { get; set; }
        public DecisionRequester DecisionRequester { get; private set; }
        private TimestepTuple Timestep { get; set; }    
        private List<ISensor> Sensors { get; set; }
        private StateResetter PositionReseter { get; set; }
        private StateVector StatesBuffer { get; set; }
        private ActionBuffer ActionsBuffer { get; set; }       
        private Tensor lastState { get; set; }
        public int EpisodeStepCount { get; private set; } = 0;
        public float EpsiodeCumulativeReward { get; private set; } = 0f;
        private int EpisodeFixedFramesCount { get; set; } = -1;


        // Draw agent's memory.
        private LinkedList<Vector3> lastKWorldPositions; // keeps track of the last K last positions in order to draw them

        private const float observationsClip = 3f;

        public virtual void Awake()
        {
            if (!enabled)
                return;

            // Check if model is ok
            if(model == null)
            {
                ConsoleMessage.Error($"Please bake/load an agent model before using the <i>{GetType().Name}</i> behaviour");
                EditorApplication.isPlaying = false;
                return;
            }

            List<string> missingComponents = model.CheckForMissingAssets();
            if(missingComponents.Count > 0)
            {
                ConsoleMessage.Error($"Agent behaviour is missing the following assets {string.Join(", ", missingComponents)}");
                EditorApplication.isPlaying = false;
                return;
            }

            if (model.targetFPS < 30 || model.targetFPS > 100)
            {
                ConsoleMessage.Warning($"Behaviour's TargetFPS ({model.targetFPS}) allowed range is [30, 100].");
                EditorApplication.isPlaying = false;
                return;
            }


            // Check Decision Requester
            DecisionRequester = GetComponent<DecisionRequester>();

            if(behaviourType == BehaviourType.Manual)
                DecisionRequester.decisionPeriod = 1;

            // Setup 
            Time.fixedDeltaTime = 1f / model.targetFPS;
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
                    try
                    {
                        PositionReseter = new StateResetter(transform.parent);
                    }
                    catch
                    {
                        ConsoleMessage.Error("Cannot <b>Reset Environment</b> on episode reset because the agent was not introduced inside an Environment");
                        EditorApplication.isPlaying = false;
                    }
                    break;
            }

            // Other
            lastKWorldPositions = new LinkedList<Vector3>();
        }
        public virtual void Start()
        {
            if (!enabled)
                return;

            if (behaviourType == BehaviourType.Learn)
            {
                TrainingStatistics pf;
                TryGetComponent(out pf);
                PerformanceTrack = pf;
                OnEpisodeBegin();
                DeepUnityTrainer.Subscribe(this, model.config.trainer);                
            }
        }
        public virtual void FixedUpdate()
        {
            EpisodeFixedFramesCount++;

            PostTimestep();

            switch(DecisionRequester.RequestEvent(EpisodeFixedFramesCount))
            {
                case DecisionRequester.AgentEvent.None:
                    break;
                case DecisionRequester.AgentEvent.Action:
                    if (behaviourType != BehaviourType.Off)
                        OnActionReceived(ActionsBuffer);
                    break;
                case DecisionRequester.AgentEvent.DecisionAndAction:
                    PerformDecision();
                    if(behaviourType != BehaviourType.Off)
                        OnActionReceived(ActionsBuffer);
                    break;
            }
        }
        public virtual void OnDrawGizmos()
        {
            if (lastKWorldPositions == null)
                return;

            const float radiusRelativeToScale = 0.2f;
            float alpha = 1f / lastKWorldPositions.Count;

            float index = 1f;
            MeshRenderer meshRenderer = GetComponent<MeshRenderer>();
            Color c = meshRenderer == null ? Color.white : meshRenderer.sharedMaterial.color;
            foreach (var item in lastKWorldPositions)
            {
                Gizmos.color = new Color(c.r, c.g, c.b, alpha);
                Gizmos.DrawSphere(item, transform.localScale.x * radiusRelativeToScale);
                alpha = (index++) / lastKWorldPositions.Count;
            }
        }
       
        // Init
        public void BakeModel()
        {
            if(spaceSize == 0)
            {
                ConsoleMessage.Info("SensorBuffer vector size was set to 0. ObservationsTensor must be modified directly.");
            }
            if(continuousActions == 0 && discreteActions == 0)
            {
                ConsoleMessage.Warning("Cannot bake model with 0 actions");
                return;
            }

            model = AgentBehaviour.CreateOrLoadAsset(GetType().Name, spaceSize, stackedInputs, continuousActions, discreteActions, numLayers, hidUnits);

            continuousActions = model.continuousDim;
            discreteActions = model.discreteDim;
        }
        private void InitBuffers()
        {
            Memory = new MemoryBuffer();
            Timestep = new TimestepTuple(EpisodeStepCount);

            StatesBuffer = new StateVector(model.observationSize, model.stackedInputs);
            ActionsBuffer = new ActionBuffer(model.continuousDim, model.discreteDim);
        }
        private void InitSensors(Transform parent)
        {
            ISensor[] sensors = parent.GetComponents<ISensor>();

            if(sensors != null && sensors.Length > 0)
            {
                foreach (var s in sensors)
                {
                    Sensors.Add(s);
                }
            }

            foreach (Transform child in parent)
            {
                InitSensors(child);
            }
        }


        // Loop
        private void PostTimestep()
        {
            // Generally saves the previous timestep
            if (behaviourType == BehaviourType.Off)
                return;

            if (EpisodeFixedFramesCount == 0)
                return;

            if (!DecisionRequester.IsFrameBeforeDecisionFrame(EpisodeFixedFramesCount))
                return;

            if (behaviourType == BehaviourType.Learn)
            {
                // Observe s'
                lastState = GetState();
                Timestep.nextState = lastState.Clone() as Tensor;
              
                Memory.Add(Timestep);

                // CHECK MAX STEPS: If the agent reached max steps without reaching the terminal state (maxStep == 0 means unlimited steps per episode)
                if (EpisodeStepCount == DecisionRequester.maxStep && DecisionRequester.maxStep != 0)
                    EndEpisode();
            }

            // These checkup applies also for Manual and Inference..
            if (Timestep?.done[0] == 1)
            {
                if (PerformanceTrack) // These are mainly for Learning
                {
                    PerformanceTrack.episodeCount++;
                    PerformanceTrack.episodeLength.Append(EpisodeStepCount);
                    PerformanceTrack.cumulativeReward.Append(EpsiodeCumulativeReward);
                }

                StatesBuffer.ResetToZero(); // For Stacked inputs reset to 0 all sequence
                lastState = null;// On Next episode there was no last state

                EpisodeStepCount = 0;
                EpsiodeCumulativeReward = 0f;
                EpisodeFixedFramesCount = 0; // Used to make the agent take a decision in the first step of the new epiode

                lastKWorldPositions.Clear();
                PositionReseter?.Reset();
                OnEpisodeBegin();
            }

            // Reset timestep
            Timestep = new TimestepTuple(++EpisodeStepCount);

            if (PerformanceTrack) PerformanceTrack.stepCount++;
        }
        private void PerformDecision()
        {
            if (behaviourType == BehaviourType.Off)
                return;

            if (behaviourType == BehaviourType.Manual)
            {
                Heuristic(ActionsBuffer);
                return;
            }

            lastKWorldPositions.AddLast(transform.position);
            if (lastKWorldPositions.Count > model.stackedInputs)
                lastKWorldPositions.RemoveFirst();

            // OBSERVATION PROCESS ------------------------------------------------------------------
            if (lastState == null)
                Timestep.state = GetState();
            else
                Timestep.state = lastState;
            // OBSERVATION PROCESS ------------------------------------------------------------------



            // ACTION PROCESS -----------------------------------------------------------------------
            // Set state[t], action[t] & pi[t]
            ActionsBuffer.Clear();
            model.ContinuousPredict(Timestep.state, out Timestep.action_continuous, out Timestep.prob_continuous);
            model.DiscretePredict(Timestep.state, out Timestep.action_discrete, out Timestep.prob_discrete);

            // Run agent's actions and clip them
            ActionsBuffer.ContinuousActions = model.IsUsingContinuousActions ? new Tanh().Forward(Timestep.action_continuous).ToArray() : null; // values also can be clipped in [-1,1] directly
            ActionsBuffer.DiscreteAction = model.IsUsingDiscreteActions ? (int)Timestep.action_discrete.ArgMax(-1)[0] : -1;
            // ACTION PROCESS -----------------------------------------------------------------------
        }
        private Tensor GetState()
        {
            Tensor state = null;
            CollectObservations(out state); // custom state assignement

            if(state == null) // if no custom state is used, check the StateVector
            {
                if (useSensors == UseSensorsType.ObservationsVector)
                    Sensors.ForEach(x => StatesBuffer.AddObservationRange(x.GetObservationsVector()));
                else if (useSensors == UseSensorsType.CompressedObservationsVector)
                    Sensors.ForEach(x => StatesBuffer.AddObservationRange(x.GetCompressedObservationsVector()));
                CollectObservations(StatesBuffer);

                // Check StateVector is fullfilled
                int ok_sbuff = StatesBuffer.IsOk();
                if (ok_sbuff != 0)
                {
                    ConsoleMessage.Warning($"Make sure you added exactly {model.observationSize} (difference of {ok_sbuff}).");
                    EditorApplication.isPlaying = false;
                    return null;
                }
                state = StatesBuffer.State.Clone() as Tensor;
            }
           

                   
            if (model.normalize)
            {
                model.observationsNormalizer.Update(state);
                state = model.observationsNormalizer.Normalize(state);
            }

            state = state.Clip(-observationsClip, observationsClip);

            return state;
        }


        // User call
        /// <summary>
        /// Reinitializes the current environment stochastically. <br></br>
        /// <br></br>
        /// <em>The paper recommends all environments to be reinitialized when one agent reaches terminal state, if necesarry you can have a reference 
        /// to the trainer to access all agents and end their episodes too.</em>
        /// </summary>
        public virtual void OnEpisodeBegin() { }
        /// <summary>
        /// Fulfill <see cref="StateVector"/>  <paramref name="stateVector"/> argument with [normalized] observations using <em>AddObservation()</em> method.
        /// Sequence is automatically generated when <b>stacked inputs</b> > 1.
        /// <br></br>
        /// <br></br>
        /// <em>Example: <br></br>
        /// <paramref name="stateVector"/>.AddObservation(transform.position.normalized) <br></br>
        /// <paramref name="stateVector"/>.AddObservation(transform.rotation.x / 360f) <br></br>
        /// <paramref name="stateVector"/>.AddObservation(rb.angularVelocity.normalized) <br></br>
        /// </em>
        /// 
        /// </summary>
        /// <param name="stateVector"></param>
        public virtual void CollectObservations(StateVector stateVector) { }
        /// <summary>
        /// Set a custom shaped state by setting up the state <see cref="Tensor"/>. Note that cannot be used in parallel with the StateVector arg method.
        /// </summary>
        /// <param name="stateTensor">The <see cref="Tensor"/> observation input.</param>
        public virtual void CollectObservations(out Tensor stateTensor) { stateTensor = null; }
        /// <summary>
        /// Assign an action for each <em>Continuous</em> or <em>Discrete</em> value inside <b>ActionBuffer</b>'s arrays.
        /// <br></br>
        /// <br></br>
        /// <em>Example: <br></br>
        /// // Access <paramref name="actionBuffer"/>.ContinuousActions and <paramref name="actionBuffer"/>.DiscreteAction <br></br>
        /// transform.position = new Vector3(<paramref name="actionBuffer"/>.ContinuousActions[0], <paramref name="actionBuffer"/>.ContinuousActions[1], <paramref name="actionBuffer"/>.ContinuousActions[2]) <br></br>
        /// rb.AddForce(<paramref name="actionBuffer"/>.DiscreteAction == 0 ? -1f : 1f, 0, 0)
        /// </em>
        /// </summary>
        /// <param name="actionBuffer"></param>
        public virtual void OnActionReceived(ActionBuffer actionBuffer) { }
        /// <summary>
        /// Manually introduce actions controlled using user inputs inside <b>ActionBuffer</b>'s <em>Continuous</em> or <em>Discrete</em> arrays.
        /// <br></br>
        /// <br></br>
        /// <em>Example: <br></br>
        /// // Access <paramref name="actionBuffer"/>.ContinuousActions and <paramref name="actionBuffer"/>.DiscreteActions <br></br>
        /// <paramref name="actionOut"/>.ContinuousActions[0] = Input.GetAxis("Horizontal") <br></br>
        /// <paramref name="actionOut"/>.DiscreteAction = Input.GetKey(KeyCode.E) ? 1 : 0;
        /// </em>
        /// </summary>
        /// <param name="actionOut"></param>
        public virtual void Heuristic(ActionBuffer actionOut) { }
        /// <summary>
        /// Called only inside <b>OnActionReceived()</b>, and <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b>. <br></br>
        /// Ensures the episode will end this frame for the current agent.
        /// </summary>
        public void EndEpisode()
        {
            Timestep.done = Tensor.Constant(1);

        }
        /// <summary>
        /// Called only inside <b>OnActionReceived()</b>, and <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b>. <br></br>
        /// Modifies the reward of the current time step.
        /// </summary>
        /// <param name="reward">positive or negative</param>
        public void AddReward(float reward)
        {
            Timestep.reward += reward;
            EpsiodeCumulativeReward += reward;
        }
        /// <summary>
        /// Called only inside <b>OnActionReceived()</b>, and <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b>. <br></br>
        /// Modifies the reward of the current time step.
        /// </summary>
        /// <param name="reward">positive or negative</param>
        public void SetReward(float reward)
        {
            Timestep.reward[0] = reward;
            EpsiodeCumulativeReward += reward;
        }
    }

    
}


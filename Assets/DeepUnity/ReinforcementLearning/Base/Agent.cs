using DeepUnity.Activations;
using DeepUnity.Sensors;
using System;
using System.Collections.Generic;
using System.Text;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

/// <summary>
///  https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Learning-Environment-Design-Agents.md
/// </summary> 
namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// Note that the Agent Learning behavior is disabled on build.
    /// </summary>
    [DisallowMultipleComponent, RequireComponent(typeof(DecisionRequester))]
    public abstract class Agent : MonoBehaviour
    {
        [Tooltip("The number of observations received at a specific timestep.")]
        [SerializeField, Min(1), HideInInspector] private int spaceSize = 0;
        [Tooltip("Height of the visual observation")]
        [SerializeField, Min(9), HideInInspector] private int spaceHeight = 144;
        [Tooltip("Width of the visual observation")]
        [SerializeField, Min(16), HideInInspector] private int spaceWidth = 256;
        [Tooltip("Channels of the visual observation")]
        [SerializeField, Min(1), HideInInspector] private int spaceChannels = 3;
        [Tooltip("The inputs are enqueued in a queue buffer. t1 = [0, 0, x1] -> t2 = [0, x1, x2] -> t3 = [x1, x2, x3] -> t4 = [x2, x3, x4] -> .... For now they are disabled.")]
        [SerializeField, Range(1, 64), HideInInspector] private int stackedInputs = 1;

        [Tooltip("Number of Continuous Actions in continuous actions space. Values in range [-1, 1].")]
        [SerializeField, Min(0), HideInInspector] private int continuousActions = 0;
        [Tooltip("Number of discrete actions in discrete action space. Actions indexes [0, 1, .. n - 1]. Consider to include a 'NO_ACTION' index also (especially when working heuristically).")]
        [SerializeField, Min(0), HideInInspector] private int discreteActions = 0;

        [Tooltip("Architecture type")]
        [SerializeField, HideInInspector] private ArchitectureType archType = ArchitectureType.MLP;
        [Tooltip("Number of hidden layers")]
        [SerializeField, Min(1), HideInInspector] private int numLayers = 2;
        [Tooltip("Number of units in a hidden layer in the model.")]
        [SerializeField, Min(32), HideInInspector] private int hidUnits = 64;
        [Tooltip("The hidden activation used. Tanh yields a more stable policy (less prone to NaN appeareance), but ReLU is more efficient.")]
        [SerializeField, HideInInspector] private NonLinearity activation = NonLinearity.Tanh;

        [SerializeField, HideInInspector] public AgentBehaviour model;
        [SerializeField, HideInInspector] public BehaviourType behaviourType = BehaviourType.Learn;

        [Space]
        [SerializeField, HideInInspector, Tooltip("What happens after the episode ends? You can reset the agent's/environment's position and rigidbody automatically using this. OnEpisodeBegin() is called afterwards in any situation.")]
        private OnEpisodeEndType onEpisodeEnd = OnEpisodeEndType.ResetEnvironment;
        [SerializeField, HideInInspector, Tooltip("Collect automatically the [Compressed] Observation Vector of attached sensors (if any) to this GameObject, and any child GameObject of any degree. Consider the number of Observation Vector's float values when defining the Space Size.")]
        private UseSensorsType useSensors = UseSensorsType.ObservationsVector;

        public MemoryBuffer Memory { get; private set; }
        public DecisionRequester DecisionRequester { get; private set; }
        public TimestepTuple Timestep { get; private set; } // well this must be private completely but I had to make it public to get in Training Statistics the correct Cumulated Reward
        private List<ISensor> Sensors { get; set; }
        private PoseReseter PositionReseter { get; set; }
        private StateVector StatesBuffer { get; set; }
        private ActionBuffer ActionsBuffer { get; set; }
        public Tensor LastState { get; private set; }
        /// <summary>
        /// The number of decisions taken in the current episode.
        /// </summary>
        public int EpisodeStepCount { get; private set; } = 0;
        /// <summary>
        /// The cumulated reward in the current episode.
        /// </summary>
        public float EpisodeCumulativeReward { get; private set; } = 0f;
        private int EpisodeFixedFramesCount { get; set; } = -1;

        public virtual void Awake()
        {
            if (!enabled)
                return;

            // Check if model is ok
            if (model == null)
            {
                ConsoleMessage.Error($"<b>Bake/Load</b> model before using the <i>{GetType().Name}</i> behaviour");
#if UNITY_EDITOR
                EditorApplication.isPlaying = false;
#endif
                return;
            }

            List<string> missingComponents = model.CheckForMissingAssets();
            if (missingComponents.Count > 0)
            {
                ConsoleMessage.Error($"<i>{GetType().Name}</i> behaviour is missing the {string.Join(", ", missingComponents)} assets");
#if UNITY_EDITOR
                EditorApplication.isPlaying = false;
#endif
                return;
            }

            if (model.targetFPS < 30 || model.targetFPS > 100)
            {
                ConsoleMessage.Warning($"Behaviour's TargetFPS ({model.targetFPS}) allowed range is [30, 100].");
#if UNITY_EDITOR
                EditorApplication.isPlaying = false;
#endif
                return;
            }


            // Check Decision Requester
            DecisionRequester = GetComponent<DecisionRequester>();

            if (behaviourType == BehaviourType.Manual)
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
                    PositionReseter = new PoseReseter(transform);
                    break;
                case OnEpisodeEndType.ResetEnvironment:
                    try
                    {
                        PositionReseter = new PoseReseter(transform.parent);
                    }
                    catch
                    {
                        ConsoleMessage.Error("Cannot <b>Reset Environment</b> on episode reset because the agent was not introduced inside an Environment");
#if UNITY_EDITOR
                        EditorApplication.isPlaying = false;
#endif
                    }
                    break;
            }
        }
        public virtual void Start()
        {
            if (!enabled)
                return;

            if (model == null)
                return;

            // if (!Application.isEditor && behaviourType == BehaviourType.Learn)
            //     behaviourType = BehaviourType.Inference;
            // 
            if (behaviourType == BehaviourType.Learn)
            {
                TrainingStatistics pf;
                TryGetComponent(out pf);
                DeepUnityTrainer.Subscribe(this, model.config.trainer);
            }

            OnEpisodeBegin();
        }
        public virtual void FixedUpdate()
        {
            if (model == null)
                return;

            EpisodeFixedFramesCount++;

            PostTimestep();

            switch (DecisionRequester.RequestEvent(EpisodeFixedFramesCount))
            {
                case DecisionRequester.AgentEvent.None:
                    break;
                case DecisionRequester.AgentEvent.Action:
                    if (behaviourType != BehaviourType.Off)
                        OnActionReceived(ActionsBuffer);
                    break;
                case DecisionRequester.AgentEvent.DecisionAndAction:
                    PerformDecision();
                    if (behaviourType != BehaviourType.Off)
                        OnActionReceived(ActionsBuffer);
                    break;
            }
        }

        // Init
        public void BakeModel()
        {
            if (spaceSize == 0)
            {
                ConsoleMessage.Info("StateBuffer vector size is 0. Make sure to use CollectObservations(Tensor state) method to set your state.");
            }
            if (continuousActions == 0 && discreteActions == 0)
            {
                ConsoleMessage.Warning("Cannot bake model with 0 actions");
                return;
            }

            model = AgentBehaviour.CreateOrLoadAsset(GetType().Name, spaceSize, stackedInputs, spaceWidth, spaceHeight, spaceChannels, continuousActions, discreteActions, numLayers, hidUnits, archType, activation);
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

            if (sensors != null && sensors.Length > 0)
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
            // Generally this method saves the previous timestep

            if (behaviourType == BehaviourType.Off)
                return;

            if (EpisodeFixedFramesCount == 0)
                return;

            if (!DecisionRequester.IsFrameBeforeDecisionFrame(EpisodeFixedFramesCount))
                return;

            // CHECK MAX STEPS: If the agent reached max steps without reaching the terminal state (maxStep == 0 means unlimited steps per episode)
            if (EpisodeStepCount == DecisionRequester.maxStep && DecisionRequester.maxStep != 0)
                EndEpisode();

            if (Timestep?.done[0] == 1)
                OnEpisodeEnd?.Invoke(this, EventArgs.Empty);

            EpisodeCumulativeReward += Timestep.reward[0];

            // Observe s'
            LastState = GetState();

            if (behaviourType == BehaviourType.Learn)
            {            
                Timestep.nextState = LastState.Clone() as Tensor;

                // Scale and clip reward - there was a problem with the rewards normalizer (idk why), but anyways they should not be normalized online because we can use off-policy alogorithms like SAC. Just use a constant bro to scale it
                // Timestep.reward[0] = model.rewardsNormalizer.ScaleReward(Timestep.reward[0]);
                Memory.Add(Timestep);              
            }

            // These checkups applies also for Manual and Inference..
            if (Timestep?.done[0] == 1)
            {
                StatesBuffer.ResetToZero(); // For Stacked inputs reset to 0 all sequence
                LastState = null;// On Next episode (on timestep 0) there will be no last state

                EpisodeStepCount = 0;
                EpisodeCumulativeReward = 0f;
                EpisodeFixedFramesCount = 0; // Used to make the agent take a decision in the first step of the new epiode
                PositionReseter?.Reset();
                OnEpisodeBegin();
            }

            // Reset timestep
            Timestep = new TimestepTuple(++EpisodeStepCount);
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


            // ACTION PROCESS -----------------------------------------------------------------------
            // Set state[t], action[t] & pi[t]

            // This is a huge update really, the inference speed is amazing.
            if (behaviourType == BehaviourType.Learn && DeepUnityTrainer.Instance.parallelAgents.Count > 1 && DecisionRequester.decisionPeriod == 1)
            {
                DeepUnityTrainer.Instance.ParallelInference(this, DeepUnityTrainer.Instance.FixedFrameCount);
            } 
            else
            {
                // OBSERVATION PROCESS ------------------------------------------------------------------
                if (LastState == null)
                    Timestep.state = GetState();
                else
                    Timestep.state = LastState; // no need to clone because the timestep was already reset at this point
                // OBSERVATION PROCESS ------------------------------------------------------------------

                model.ContinuousPredict(Timestep.state, out Timestep.action_continuous, out Timestep.prob_continuous);
                model.DiscretePredict(Timestep.state, out Timestep.action_discrete, out Timestep.prob_discrete);
            }

            // Run agent's actions and clip them
            ActionsBuffer.Clear();
            if(model.IsUsingContinuousActions)
            {
                if (model.stochasticity == Stochasticity.FixedStandardDeviation || model.stochasticity == Stochasticity.TrainebleStandardDeviation)
                    ActionsBuffer.ContinuousActions = Timestep.action_continuous.Tanh().ToArray();
         
                else if (model.stochasticity == Stochasticity.ActiveNoise || model.stochasticity == Stochasticity.Random)
                    ActionsBuffer.ContinuousActions = Timestep.action_continuous.ToArray();
                
                else
                    throw new NotImplementedException("Unhandled stochasticity type");
                
            }
            else
                ActionsBuffer.ContinuousActions = null;

            ActionsBuffer.DiscreteAction = model.IsUsingDiscreteActions ? (int)Timestep.action_discrete.ArgMax(-1)[0] : -1;
            // ACTION PROCESS -----------------------------------------------------------------------
        }
        public Tensor GetState()
        {
            Tensor state = null;
            CollectObservations(out state); // custom state assignement

            if (state == null) // if no custom state is used, check the StateVector
            {
                if (useSensors == UseSensorsType.ObservationsVector)
                    Sensors.ForEach(x => StatesBuffer.AddObservation(x.GetObservationsVector()));
                else if (useSensors == UseSensorsType.CompressedObservationsVector)
                    Sensors.ForEach(x => StatesBuffer.AddObservation(x.GetCompressedObservationsVector()));
                CollectObservations(StatesBuffer);

                // Check StateVector does not exceed the max observations. I do not check btw if the minimum observations are added.
                if (StatesBuffer.GetOverflow() > 0)
                {
                    ConsoleMessage.Warning($"Make sure you added exactly {model.observationSize} ({StatesBuffer.GetOverflow()} extra observations added).");
#if UNITY_EDITOR
                    EditorApplication.isPlaying = false;
#endif
                    return null;
                }
                state = StatesBuffer.State.Clone() as Tensor;
            }

            if (model.normalize)
            {
                if(behaviourType == BehaviourType.Learn) 
                    model.observationsNormalizer.Update(state);

                state = model.observationsNormalizer.Normalize(state);
            }

            state = state.Clip(-model.clipping, model.clipping);

            return state;
        }

        // User call
        /// <summary>
        /// An event that is invoked when the agent enters in a terminal state.  <br></br>
        /// To subscribe to event, write in Start: <b>OnEpisodeEnd += (s, e) => { <em>// do whatever you want</em>};</b>
        /// </summary>
        public event EventHandler OnEpisodeEnd;
        /// <summary>
        /// Reinitializes the current environment [stochastically]. It is automatically called in <see cref="Start()"/> method and at the beginning of a new episode for any <see cref="BehaviourType"/> (except <see cref="BehaviourType.Off"/>).<br></br>
        /// <br></br>
        /// </summary>
        public virtual void OnEpisodeBegin() { }
        /// <summary>
        /// Fulfill <see cref="StateVector"/>  <paramref name="stateVector"/> argument with [normalized] observations using <em>AddObservation()</em> method.
        /// Sequence is automatically generated when <b>stacked inputs</b> > 1.
        /// <br></br>
        /// <br></br>
        /// <em>Example: <br></br>
        /// <paramref name="stateVector"/>.AddObservation(transform.position) <br></br>
        /// <paramref name="stateVector"/>.AddObservation(transform.rotation.x % 360 / 360f) <br></br>
        /// <paramref name="stateVector"/>.AddObservation(rb.angularVelocity.normalized) <br></br>
        /// </em>
        /// 
        /// </summary>
        /// <param name="stateVector"></param>
        public virtual void CollectObservations(StateVector stateVector) { }
        /// <summary>
        /// Set a custom shaped state by setting up the state <see cref="Tensor"/>. Note that cannot be used in parallel with the <see cref="StateVector"/> arg method, and it works for <see cref="TrainerType.PPO"/> only.
        /// </summary>
        /// <param name="stateTensor">The <see cref="Tensor"/> observation input.</param>
        public virtual void CollectObservations(out Tensor stateTensor) { stateTensor = null; }
        /// <summary>
        /// Assign an action for each <em>Continuous</em> or <em>Discrete</em> value inside <see cref="ActionBuffer"/>'s arrays.
        /// <br></br>
        /// <br></br>
        /// <em>Example: <br></br>
        /// // Access <paramref name="actionBuffer"/>.ContinuousActions and <paramref name="actionBuffer"/>.DiscreteAction <br></br>
        /// transform.position += new Vector3(<paramref name="actionBuffer"/>.ContinuousActions[0], <paramref name="actionBuffer"/>.ContinuousActions[1], <paramref name="actionBuffer"/>.ContinuousActions[2]) <br></br>
        /// rb.AddForce(<paramref name="actionBuffer"/>.DiscreteAction == 0 ? -1f : 1f, 0, 0)
        /// </em>
        /// </summary>
        /// <param name="actionBuffer"></param>
        public virtual void OnActionReceived(ActionBuffer actionBuffer) { }
        /// <summary>
        /// Manually introduce actions controlled using user inputs inside <see cref="ActionBuffer"/>'s <em>Continuous</em> or <em>Discrete</em> arrays. Note that for Discrete Actions, a "do-nothing" action must be considered.
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
        /// Marks the current state as terminal.
        /// </summary>
        public void EndEpisode()
        {
            Timestep.done[0] = 1f;
        }
        /// <summary>
        /// Called only inside <b>OnActionReceived()</b>, and <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b>. <br></br>
        /// Modifies the reward of the current time step.
        /// </summary>
        /// <param name="reward">positive or negative</param>
        public void AddReward(float reward)
        {
            Timestep.reward[0] += Math.Clamp(reward, -model.clipping, model.clipping);
        }
        /// <summary>
        /// Called only inside <b>OnActionReceived()</b>, and <b>OnTriggerXXX()</b> or <b>OnCollisionXXX()</b>. <br></br>
        /// Modifies the reward of the current time step.
        /// </summary>
        /// <param name="reward">positive or negative</param>
        public void SetReward(float reward)
        {
            Timestep.reward[0] = Math.Clamp(reward, -model.clipping, model.clipping);
        }
    }

#if UNITY_EDITOR
    [CustomEditor(typeof(Agent), true), CanEditMultipleObjects]
    sealed class CustomAgentEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            string[] drawNow = new string[] { "m_Script" };

            var script = (Agent)target;

            // Runtime Learn displays
            if (EditorApplication.isPlaying &&
                script.behaviourType == BehaviourType.Learn
                && script.enabled
                && script.model
                && DeepUnityTrainer.Instance)
            {
                // Draw Step
                int currentStep = script.EpisodeStepCount;
                string stepcount = $"Decisions [{currentStep}]";
                EditorGUILayout.HelpBox(stepcount, MessageType.None);

                // Draw Reward
                string cumReward = $"Cumulative Reward [{script.EpisodeCumulativeReward}]";
                EditorGUILayout.HelpBox(cumReward, MessageType.None);

                // Draw buffer           

                if (DeepUnityTrainer.Instance.GetType() == typeof(PPOTrainer))
                {
                    int buff_count = DeepUnityTrainer.MemoriesCount;
                    float bufferFillPercentage = buff_count / ((float)script.model.config.bufferSize) * 100f;
                    StringBuilder sb = new StringBuilder();
                    sb.Append("Buffer [");
                    sb.Append(buff_count);
                    sb.Append(" / ");
                    sb.Append(script.model.config.bufferSize);
                    sb.Append($"] \n[");
                    for (float i = 1.25f; i <= 100f; i += 1.25f)
                    {
                        if (i == 47.5f)
                            sb.Append($"{bufferFillPercentage.ToString("00.0")}%");
                        else if (i > 47.5f && i <= 53.75f)
                            continue;
                        else if (i <= bufferFillPercentage)
                            sb.Append("▮");
                        else
                            sb.Append("▯");
                    }
                    sb.Append("]");
                    EditorGUILayout.HelpBox(sb.ToString(), MessageType.None);
                    EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
                }
                else if (DeepUnityTrainer.Instance.GetType() == typeof(SACTrainer)
                    || DeepUnityTrainer.Instance.GetType() == typeof(TD3Trainer)
                    || DeepUnityTrainer.Instance.GetType() == typeof(DDPGTrainer))
                {
                    int collected_data_count = DeepUnityTrainer.Instance.train_data.Count;
                    float bufferFillPercentage = collected_data_count / ((float)script.model.config.replayBufferSize) * 100f;
                    StringBuilder sb = new StringBuilder();
                    sb.Append("Buffer [");
                    sb.Append(collected_data_count);
                    sb.Append(" / ");
                    sb.Append(script.model.config.replayBufferSize);
                    sb.Append($"] \n[");
                    for (float i = 1.25f; i <= 100f; i += 1.25f)
                    {
                        if (i == 47.5f)
                            sb.Append($"{bufferFillPercentage.ToString("00.0")}%");
                        else if (i > 47.5f && i <= 53.75f)
                            continue;
                        else if (i <= bufferFillPercentage)
                            sb.Append("▮");
                        else
                            sb.Append("▯");
                    }
                    sb.Append("]");
                    EditorGUILayout.HelpBox(sb.ToString(), MessageType.None);
                    EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
                }
                else
                    throw new NotImplementedException("Unhandled trainer type");


                // EditorGUILayout.HelpBox($"Reward [{script.EpsiodeCumulativeReward}]",
                //                         MessageType.None);
            }

            // On model create field draw
            if (serializedObject.FindProperty("model").objectReferenceValue == null)
            {
                EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);



                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button("Bake model", GUILayout.Width(EditorGUIUtility.labelWidth * 2.32f)))
                {
                    script.BakeModel();
                }

                // Create a Rect for the second field with a specific width
                Rect propertyFieldRect = GUILayoutUtility.GetRect(0, EditorGUIUtility.singleLineHeight);
                propertyFieldRect.width = 50; // Adjust the width as needed

                SerializedProperty typeProperty = serializedObject.FindProperty("archType");
                //EditorGUI.BeginDisabledGroup(true); // they prove very slow in computation.. better with mlp (also visual observations are not enough sometimes)
                EditorGUI.PropertyField(propertyFieldRect, typeProperty, GUIContent.none);
               // EditorGUI.EndDisabledGroup();
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.Space();

                EditorGUILayout.BeginHorizontal();
                GUILayout.Label("Num Layers", GUILayout.Width(EditorGUIUtility.labelWidth / 1.6f));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("numLayers"), GUIContent.none, GUILayout.Width(25f));
                GUILayout.Label("Hidden Units", GUILayout.Width(EditorGUIUtility.labelWidth / 1.5f));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("hidUnits"), GUIContent.none, GUILayout.Width(25f));
                GUILayout.Label("Activation", GUILayout.Width(EditorGUIUtility.labelWidth / 2f));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("activation"), GUIContent.none, GUILayout.Width(50f));
                EditorGUILayout.EndHorizontal();


                EditorGUILayout.Space();



                int arTp = serializedObject.FindProperty("archType").enumValueIndex;
                if (arTp == (int)ArchitectureType.MLP)
                {
                    EditorGUILayout.LabelField("Observations");

                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.Space(20);
                    EditorGUILayout.PrefixLabel("Space Size");
                    EditorGUILayout.PropertyField(serializedObject.FindProperty("spaceSize"), GUIContent.none);
                    EditorGUILayout.EndHorizontal();

                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.Space(20);
                    EditorGUILayout.PrefixLabel("Stacked Inputs");
                    EditorGUILayout.PropertyField(serializedObject.FindProperty("stackedInputs"), GUIContent.none);
                    EditorGUILayout.EndHorizontal();

                }
                else if (arTp == (int)ArchitectureType.CNN)
                {
                    EditorGUILayout.LabelField("Visual Observations");

                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.Space(20);
                    EditorGUILayout.PrefixLabel("Width");
                    EditorGUILayout.PropertyField(serializedObject.FindProperty("spaceWidth"), GUIContent.none);
                    EditorGUILayout.EndHorizontal();

                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.Space(20);
                    EditorGUILayout.PrefixLabel("Height");
                    EditorGUILayout.PropertyField(serializedObject.FindProperty("spaceHeight"), GUIContent.none);
                    EditorGUILayout.EndHorizontal();

                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.Space(20);
                    EditorGUILayout.PrefixLabel("Channels");
                    EditorGUILayout.PropertyField(serializedObject.FindProperty("spaceChannels"), GUIContent.none);
                    EditorGUILayout.EndHorizontal();
                }
                else if (arTp == (int)ArchitectureType.RNN || arTp == (int)ArchitectureType.ATT)
                {
                    EditorGUILayout.LabelField("Observations");

                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.Space(20);
                    EditorGUILayout.PrefixLabel("Space Size");
                    EditorGUILayout.PropertyField(serializedObject.FindProperty("spaceSize"), GUIContent.none);
                    EditorGUILayout.EndHorizontal();

                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.Space(20);
                    EditorGUILayout.PrefixLabel("Memory Size");
                    EditorGUILayout.PropertyField(serializedObject.FindProperty("stackedInputs"), GUIContent.none);
                    EditorGUILayout.EndHorizontal();
                }


                EditorGUILayout.Space(5);

                EditorGUILayout.LabelField("Actions");
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.Space(20);
                EditorGUILayout.PrefixLabel("Continuous Actions");
                EditorGUILayout.PropertyField(serializedObject.FindProperty("continuousActions"), GUIContent.none);
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.Space(20);
                EditorGUILayout.PrefixLabel("Discrete Actions");
                EditorGUILayout.PropertyField(serializedObject.FindProperty("discreteActions"), GUIContent.none);
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
            }


            EditorGUILayout.PropertyField(serializedObject.FindProperty("model"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("behaviourType"));
            if (script.behaviourType != BehaviourType.Off)
            {
                EditorGUILayout.PropertyField(serializedObject.FindProperty("onEpisodeEnd"));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("useSensors"));
            }

            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
            DrawPropertiesExcluding(serializedObject, drawNow);
            serializedObject.ApplyModifiedProperties();


        }
    }
#endif
}

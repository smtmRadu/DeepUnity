using System.Collections.Generic;
using System.Text;
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
        [SerializeField, Range(1, 32), HideInInspector, ReadOnly] private int stackedInputs = 1;
        [Tooltip("Number of Continuous Actions in continuous actions space. Values in range [-1, 1].")]
        [SerializeField, Min(0), HideInInspector] private int continuousActions = 0;
        [Tooltip("Number of discrete actions in discrete action space. Actions indexes [0, 1, .. n - 1]. Consider to include a 'NO_ACTION' index also (especially when working heuristically).")]
        [SerializeField, Min(0), HideInInspector] private int discreteActions = 0;
        [Tooltip("Arhitecture of the neural network")]
        [SerializeField, HideInInspector] private ModelType type = ModelType.NN;
        [Tooltip("Number of hidden layers")]
        [SerializeField, Min(1), HideInInspector] private int numLayers = 2;
        [Tooltip("Number of units in a hidden layer in the model.")]
        [SerializeField, Min(32), HideInInspector] private int hidUnits = 64;

        [SerializeField, HideInInspector] public AgentBehaviour model;
        [SerializeField, HideInInspector] public BehaviourType behaviourType = BehaviourType.Learn;
        [Tooltip("Learning rate of the imitation relative to the learning rate in config file.")]
        [SerializeField, HideInInspector, Range(0, 1)] public float imitationStrength = 1f;
     
        [Space]
        [SerializeField, HideInInspector, Tooltip("What happens after the episode ends? You can reset the agent's/environment's position and rigidbody automatically using this. OnEpisodeBegin() is called afterwards in any situation.")]
        private OnEpisodeEndType onEpisodeEnd = OnEpisodeEndType.ResetEnvironment;
        [SerializeField, HideInInspector, Tooltip("Collect automatically the [Compressed] Observation Vector of attached sensors (if any) to this GameObject, and any child GameObject of any degree. Consider the number of Observation Vector's float values when defining the Space Size.")]
        private UseSensorsType useSensors = UseSensorsType.ObservationsVector;

        public TrainingStatistics PerformanceTrack { get; set; }
        public MemoryBuffer Memory { get; set; }
        public DecisionRequester DecisionRequester { get; private set; }
        private TimestepBuffer Timestep { get; set; }    
        private List<ISensor> Sensors { get; set; }
        private StateResetter PositionReseter { get; set; }
        private SensorBuffer ObservationsBuffer { get; set; }
        private ActionBuffer ActionsBuffer { get; set; }
        public int EpisodeStepCount { get; private set; } = 0;
        public float EpsiodeCumulativeReward { get; private set; } = 0f;
        private int FixedFramesCount { get; set; } = -1;

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

            if (model.targetFPS < 30 || model.targetFPS > 100)
            {
                ConsoleMessage.Warning($"Behaviour's TargetFPS ({model.targetFPS}) allowed range is [30, 100].");
                EditorApplication.isPlaying = false;
                return;
            }

            DecisionRequester = GetComponent<DecisionRequester>();

            if(behaviourType == BehaviourType.Heuristic && DecisionRequester.decisionPeriod > 1)
            {
                ConsoleMessage.Warning("In Heuristic mode, Decision Period of the DecisionRequester must be always 1.");
                EditorApplication.isPlaying = false;
                return;
            }

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
                PPOTrainer.Subscribe(this);
            }

            if(behaviourType == BehaviourType.Heuristic)
            {
                TrainingStatistics pf;
                TryGetComponent(out pf);
                PerformanceTrack = pf;
                OnEpisodeBegin();
                HeuristicTrainer.Subscribe(this);
            }

        }
        public virtual void FixedUpdate()
        {
            FixedFramesCount++;

            PostTimestep();

            switch(DecisionRequester.RequestEvent(FixedFramesCount))
            {
                case DecisionRequester.AgentEvent.None:
                    break;
                case DecisionRequester.AgentEvent.Action:
                    PerformAction();
                    break;
                case DecisionRequester.AgentEvent.DecisionAndAction:
                    PerformDecision();
                    PerformAction();
                    break;
            }
        }
  
        private void PostTimestep()
        {
            if (behaviourType == BehaviourType.Off)
                return;

            if (FixedFramesCount == 0)
                return;

            if (!DecisionRequester.IsFrameBeforeDecisionFrame(FixedFramesCount))
                return;

            if(behaviourType == BehaviourType.Learn)
            {
                Memory.Add(Timestep);

                // CHECK MAX STEPS: If the agent reached max steps without reaching the terminal state (maxStep == 0 means unlimited steps per episode)
                if (EpisodeStepCount == DecisionRequester.maxStep && DecisionRequester.maxStep != 0)
                    EndEpisode();

                PPOTrainer.SendMemory(Memory);

                if (Timestep?.done[0] == 1)
                {
                    if (PerformanceTrack)
                    {
                        PerformanceTrack.episodeCount++;
                        PerformanceTrack.episodeLength.Append(EpisodeStepCount);
                        PerformanceTrack.cumulativeReward.Append(EpsiodeCumulativeReward);
                    }

                    EpisodeStepCount = 0;
                    EpsiodeCumulativeReward = 0f;


                    PositionReseter?.Reset();
                    // ObservationsBuffer.FlushStack();
                    OnEpisodeBegin();
                }
            }
            else if(behaviourType == BehaviourType.Heuristic)
            {
                Memory.Add(Timestep);

                // CHECK MAX STEPS: If the agent reached max steps without reaching the terminal state
                if (EpisodeStepCount == DecisionRequester.maxStep && DecisionRequester.maxStep != 0)
                    EndEpisode();

                HeuristicTrainer.SendMemory(Memory);

                if (Timestep?.done[0] == 1)
                {
                    EpisodeStepCount = 0;
                    EpsiodeCumulativeReward = 0f;


                    PositionReseter?.Reset();
                    OnEpisodeBegin();
                }
            }
            else if(Timestep?.done[0] == 1)
            {
                PositionReseter?.Reset();
                OnEpisodeBegin();
            }

            // Reset timestep
            Timestep = new TimestepBuffer(++EpisodeStepCount);
        }
        private void PerformAction()
        {
            if (behaviourType == BehaviourType.Off)
                return;

            if (behaviourType == BehaviourType.Heuristic)
            {
                // For now, in heuristic mode, only manual control is available
                ObservationsBuffer.Clear();
                ActionsBuffer.Clear();

                // Collect new observations
                if (useSensors == UseSensorsType.ObservationsVector) Sensors.ForEach(x => ObservationsBuffer.AddObservationRange(x.GetObservationsVector()));
                else if (useSensors == UseSensorsType.CompressedObservationsVector) Sensors.ForEach(x => ObservationsBuffer.AddObservationRange(x.GetCompressedObservationsVector()));
                CollectObservations(ObservationsBuffer);
                // ObservationsBuffer.PushToStack(ObservationsBuffer.TimestepObservation);
                

                // Check SensorBuffer is fullfilled
                int missing = 0;
                if (!ObservationsBuffer.IsFulfilled(out missing))
                {
                    ConsoleMessage.Warning($"SensorBuffer is missing {missing} observations. Please add {missing} more observations values or reduce the space size.");
                    EditorApplication.isPlaying = false;
                    return;
                }

                // Normalize the observations if neccesary
                if (model.normalizeObservations)
                    Timestep.state = model.normalizer.Normalize(ObservationsBuffer.Observations);
                else
                    Timestep.state = ObservationsBuffer.Observations.Clone() as Tensor;
                
                // Collect user input actions
                Heuristic(ActionsBuffer);
                OnActionReceived(ActionsBuffer);

                // Check if there was any input received in case for discrete actions, because we need a dedicated No_Action index.
                if (model.IsUsingDiscreteActions && ActionsBuffer.DiscreteAction == -1)
                {
                    ConsoleMessage.Warning("When using Discrete Actions for Heuristic Training, consider the 'NO_ACTION' case also when not introducing any input beside the already selected discrete actions");
                    EditorApplication.isPlaying = false;    
                    return;
                }

                
                if(model.IsUsingContinuousActions)
                    Timestep.action_continuous = Tensor.Constant(ActionsBuffer.ContinuousActions);

                if (model.IsUsingDiscreteActions)
                {
                    // Convert from action index to One Hot embedding vector
                    Timestep.action_discrete = Tensor.Zeros(model.discreteDim);
                    Timestep.action_discrete[ActionsBuffer.DiscreteAction] = 1f;
                }
            }
            else
            {
                OnActionReceived(ActionsBuffer);
            }

           
        }
        private void PerformDecision()
        {
            if (behaviourType == BehaviourType.Off)
                return;

            if (behaviourType == BehaviourType.Heuristic)
                return;

            
            ObservationsBuffer.Clear();
            ActionsBuffer.Clear();

            // Collect new observations
            if (useSensors == UseSensorsType.ObservationsVector) Sensors.ForEach(x => ObservationsBuffer.AddObservationRange(x.GetObservationsVector()));
            else if (useSensors == UseSensorsType.CompressedObservationsVector) Sensors.ForEach(x => ObservationsBuffer.AddObservationRange(x.GetCompressedObservationsVector()));
            CollectObservations(ObservationsBuffer);
            // ObservationsBuffer.PushToStack(ObservationsBuffer.TimestepObservation);

            // Check SensorBuffer is fullfilled
            int missing = 0;
            if (!ObservationsBuffer.IsFulfilled(out missing))
            {
                ConsoleMessage.Warning($"SensorBuffer is missing {missing} observations. Please add {missing} more observations values or reduce the space size.");
                EditorApplication.isPlaying = false;
                return;
            }

            // Normalize the observations if neccesary
            if (model.normalizeObservations)
            {
                Tensor st = ObservationsBuffer.Observations.Clone() as Tensor;

                if (behaviourType == BehaviourType.Learn)
                {
                    model.normalizer.Update(st);
                }

                Timestep.state = model.normalizer.Normalize(st);
                
            }
            else
            {
                Timestep.state = ObservationsBuffer.Observations.Clone() as Tensor;
            }

            // Set state[t], action[t] & pi[t]
            model.ContinuousPredict(Timestep.state, out Timestep.action_continuous, out Timestep.prob_continuous);
            model.DiscretePredict(Timestep.state, out Timestep.action_discrete, out Timestep.prob_discrete);

            // Run agent's actions and clip them
            ActionsBuffer.ContinuousActions = model.IsUsingContinuousActions ? Timestep.action_continuous.Clip(-1f, 1f).ToArray() : null;
            ActionsBuffer.DiscreteAction = model.IsUsingDiscreteActions ? (int)Timestep.action_discrete.ArgMax(-1)[0] : -1;
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

            model = AgentBehaviour.CreateOrLoadAsset(GetType().Name, spaceSize, stackedInputs, continuousActions, discreteActions, type, numLayers, hidUnits);

            continuousActions = model.continuousDim;
            discreteActions = model.discreteDim;
        }
        private void InitBuffers()
        {
            Memory = new MemoryBuffer();
            Timestep = new TimestepBuffer(EpisodeStepCount);

            ObservationsBuffer = new SensorBuffer(model.observationSize);
            ActionsBuffer = new ActionBuffer(model.continuousDim, model.discreteDim);
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
        /// Reinitializes the current environment stochastically. <br></br>
        /// <br></br>
        /// <em>The paper recommends all environments to be reinitialized when one agent reaches terminal state, if necesarry you can have a reference 
        /// to the trainer to access all agents and end their episodes too.</em>
        /// </summary>
        public virtual void OnEpisodeBegin() { }
        /// <summary>
        /// Fulfill <b>SensorBuffer</b>  <paramref name="sensorBuffer"/> argument with [normalized] observations using <em>AddObservation()</em> method.
        /// <br></br>
        /// <br></br>
        /// <em>Example: <br></br>
        /// <paramref name="sensorBuffer"/>.AddObservation(transform.position.normalized) <br></br>
        /// <paramref name="sensorBuffer"/>.AddObservation(transform.rotation.x / 360f) <br></br>
        /// <paramref name="sensorBuffer"/>.AddObservation(rb.angularVelocity.normalized) <br></br>
        /// </em>
        /// 
        /// </summary>
        /// <param name="sensorBuffer"></param>
        public virtual void CollectObservations(SensorBuffer sensorBuffer) { }
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

    [CustomEditor(typeof(Agent), true), CanEditMultipleObjects]
    sealed class CustomAgentEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            string[] drawNow = new string[]{ "m_Script" };

            var script = (Agent)target;

            // Runtime inspector things
            if (EditorApplication.isPlaying &&        
                script.behaviourType == BehaviourType.Learn
                && script.enabled 
                && script.model)
            {
                // Draw Step
                int currentStep = script.EpisodeStepCount;
                string stepcount = $"Decisions [{currentStep}]";
                EditorGUILayout.HelpBox(stepcount, MessageType.None);

                // Draw Reward
                string cumReward = $"Cumulative Reward [{script.EpsiodeCumulativeReward}]";
                EditorGUILayout.HelpBox(cumReward, MessageType.None);

                // Draw buffer 
                float bufferFillPercentage = script.Memory.Count * PPOTrainer.ParallelAgentsCount / ((float)script.model.config.bufferSize) * 100f;
                StringBuilder sb = new StringBuilder();
                sb.Append("Buffer [");
                sb.Append(script.Memory.Count * PPOTrainer.ParallelAgentsCount);
                sb.Append(" / ");
                sb.Append(script.model.config.bufferSize);
                sb.Append($"] \n[");
                for (float i = 1.25f; i <= 100f; i+=1.25f)
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

                
                // EditorGUILayout.HelpBox($"Reward [{script.EpsiodeCumulativeReward}]",
                //                         MessageType.None);
            }

            if (EditorApplication.isPlaying &&
                script.behaviourType == BehaviourType.Heuristic
                && script.enabled
                && script.model)
            {
                EditorGUILayout.HelpBox("Control the agent behaviour until the buffer is fullfiled.", MessageType.None);
                // Draw buffer 
                float bufferFillPercentage = script.Memory.Count / ((float)script.model.config.bufferSize) * 100f;
                StringBuilder sb = new StringBuilder();
                sb.Append("Buffer [");
                sb.Append(script.Memory.Count);
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

            }


            // All fields drawn
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

                SerializedProperty typeProperty = serializedObject.FindProperty("type");
                EditorGUI.BeginDisabledGroup(true);
                EditorGUI.PropertyField(propertyFieldRect, typeProperty, GUIContent.none);
                EditorGUI.EndDisabledGroup();
                EditorGUILayout.EndHorizontal();




                EditorGUILayout.BeginHorizontal();
                GUILayout.Label("Num Layers", GUILayout.Width(EditorGUIUtility.labelWidth / 1.08f));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("numLayers"), GUIContent.none, GUILayout.Width(50f));
                GUILayout.Label("Hidden Units", GUILayout.Width(EditorGUIUtility.labelWidth / 1.08f));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("hidUnits"), GUIContent.none, GUILayout.Width(50f));
                EditorGUILayout.EndHorizontal();


                EditorGUILayout.Space();

                EditorGUILayout.LabelField("Observations");
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.Space(20);
                EditorGUILayout.PrefixLabel("Space Size");
                EditorGUILayout.PropertyField(serializedObject.FindProperty("spaceSize"), GUIContent.none);
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.Space(20);
                if (serializedObject.FindProperty("type").enumValueIndex == (int)ModelType.RNN)
                    EditorGUILayout.PrefixLabel("Sequence Length");
                else
                    EditorGUILayout.PrefixLabel("Stacked Inputs");
                EditorGUI.BeginDisabledGroup(true);
                EditorGUILayout.PropertyField(serializedObject.FindProperty("stackedInputs"), GUIContent.none);
                EditorGUI.EndDisabledGroup();
                EditorGUILayout.EndHorizontal();

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

            if (script.behaviourType == BehaviourType.Heuristic)
                EditorGUILayout.PropertyField(serializedObject.FindProperty("imitationStrength"));

            EditorGUILayout.PropertyField(serializedObject.FindProperty("onEpisodeEnd"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("useSensors"));
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);


           
            DrawPropertiesExcluding(serializedObject, drawNow);

            serializedObject.ApplyModifiedProperties();

           
        }
    }
}


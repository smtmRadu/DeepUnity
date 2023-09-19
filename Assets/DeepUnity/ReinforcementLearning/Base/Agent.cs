using System.Collections.Generic;
using System.Text;
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
        [SerializeField, Min(1)] private int spaceSize = 1;
        [SerializeField, Min(0)] private int continuousActions = 0;
        [SerializeField, Min(0)] private int discreteActions = 0;

        [Space(5)]
        [SerializeField] public AgentBehaviour model;
        [SerializeField] public Hyperparameters hp;
        [SerializeField] private BehaviourType behaviourType = BehaviourType.Learn;
     
        [Space]
        [SerializeField, Tooltip("What happens after the episode ends? You can reset the agent's/environment's position and rigidbody automatically using this. OnEpisodeBegin() is called afterwards in any situation.")]
        private OnEpisodeEndType onEpisodeEnd = OnEpisodeEndType.ResetEnvironment;
        [SerializeField, Tooltip("Collect automatically the [Compressed] Observation Vector of attached sensors (if any) to this GameObject, and any child GameObject of any degree. Consider the number of Observation Vector's float values when defining the Space Size.")]
        private UseSensorsType useSensors = UseSensorsType.ObservationsVector;

        public TrainingStatistics PerformanceTrack { get; set; }
        public MemoryBuffer Memory { get; set; }
        public DecisionRequester DecisionRequester { get; private set; }
        private TimestepBuffer Timestep { get; set; }    
        private List<ISensor> Sensors { get; set; }
        private StateResetter PositionReseter { get; set; }
        private SensorBuffer Observations { get; set; }
        private ActionBuffer Actions { get; set; }
        public int EpisodeStepCount { get; private set; } = 0;
        public float EpsiodeCumulativeReward { get; private set; } = 0f;
        private int FixedFramesCount { get; set; } = -1;

        public virtual void Awake()
        {
            if (!this.enabled)
                return;

            if(this.model == null || this.hp == null)
            {
                ConsoleMessage.Error("Please bake/load an agent model & hyperparameters before using the behaviour in learning or heuristic mode.");
                EditorApplication.isPlaying = false;
            }

            if(model.targetFPS < 30 || model.targetFPS > 100)
            {
                model.targetFPS = 50;
                ConsoleMessage.Warning("Behaviour's targetFPS not in range 30 - 100. It was automatically set on 50 by default");
            }

            Time.fixedDeltaTime = 1f / model.targetFPS;
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
            if (!this.enabled)
                return;

            if (behaviourType == BehaviourType.Learn)
            {
                TrainingStatistics pf;
                TryGetComponent(out pf);
                PerformanceTrack = pf;

                Trainer.Subscribe(this);
            }

            if(behaviourType == BehaviourType.Learn || behaviourType == BehaviourType.Heuristic)
                OnEpisodeBegin();

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

            if (FixedFramesCount <= 0)
                return;

            if (!DecisionRequester.IsFrameBeforeDecisionFrame(FixedFramesCount))
                return;

            if(behaviourType == BehaviourType.Learn)
            {
                Memory.Add(Timestep);

                // CHECK MAX STEPS: If the agent reached max steps without reaching the terminal state
                if (EpisodeStepCount == DecisionRequester.maxStep)
                    EndEpisode();

                Trainer.SendMemory(Memory);

                if (Timestep?.done[0] == 1)
                {

                    if (PerformanceTrack)
                    {
                        PerformanceTrack.episodeCount++;
                        PerformanceTrack.episodeLength.Append(EpisodeStepCount);
                        PerformanceTrack.episodeReward.Append(EpsiodeCumulativeReward);
                    }

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
                Actions.Clear();
                Heuristic(Actions);
                OnActionReceived(Actions);
            }
            else
            {
                OnActionReceived(Actions);
            }

           
        }
        private void PerformDecision()
        {
            Observations.Clear();
            Actions.Clear();

            // Collect new observations
            if (useSensors == UseSensorsType.ObservationsVector) Sensors.ForEach(x => Observations.AddObservationRange(x.GetObservationsVector()));
            else if (useSensors == UseSensorsType.CompressedObservationsVector) Sensors.ForEach(x => Observations.AddObservationRange(x.GetCompressedObservationsVector()));
            CollectObservations(Observations);

            // Check SensorBuffer is fullfilled
            int missing = 0;
            if (!Observations.IsFulfilled(out missing))
            {
                ConsoleMessage.Warning($"SensorBuffer is missing {missing} observations. Please add {missing} more observations values or reduce the space size.");
                EditorApplication.isPlaying = false;
            }

            // Normalize the observations if neccesary
            if (model.normalizeObservations)
            {
                if (behaviourType == BehaviourType.Learn)
                    model.normalizer.Update(Observations.ObservationTensor);

                Observations.ObservationTensor = model.normalizer.Normalize(Observations.ObservationTensor);
            }

            // Set state[t], action[t] & pi[t]
            Timestep.state = Tensor.Identity(Observations.ObservationTensor);
            model.ContinuousPredict(Timestep.state, out Timestep.action_continuous, out Timestep.prob_continuous);
            model.DiscretePredict(Timestep.state, out Timestep.action_discrete, out Timestep.prob_discrete);

            // Run agent's actions and clip them
            Actions.ContinuousActions = model.IsUsingContinuousActions ? Timestep.action_continuous.Clip(-1f, 1f).ToArray() : null;
            Actions.DiscreteAction = model.IsUsingDiscreteActions ? (int)Timestep.action_discrete[0] : -1;
        }
        // Init
        public void BakeModel()
        {
            model = AgentBehaviour.CreateOrLoadAsset(GetType().Name, spaceSize, continuousActions, discreteActions);

            continuousActions = model.continuousDim;
            discreteActions = model.discreteDim;
        }
        public void BakeHyperparamters()
        {
            hp = Hyperparameters.CreateOrLoadAsset(GetType().Name);
        }
        private void InitBuffers()
        {
            Memory = new MemoryBuffer();
            Timestep = new TimestepBuffer(EpisodeStepCount);

            Observations = new SensorBuffer(model.observationSize);
            Actions = new ActionBuffer(model.continuousDim, model.discreteDim);
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
        /// // Access <paramref name="actionBuffer"/>.ContinuousActions and <paramref name="actionBuffer"/>.DiscreteActions <br></br>
        /// transform.position = new Vector3(<paramref name="actionBuffer"/>.ContinuousActions[0], <paramref name="actionBuffer"/>.ContinuousActions[1], <paramref name="actionBuffer"/>.ContinuousActions[2]) <br></br>
        /// rb.AddForce(<paramref name="actionBuffer"/>.DiscreteActions[0] == 0 ? -1f : 1f, 0, 0)
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
        /// <paramref name="actionOut"/>.DiscreteActions[0] = Input.GetKey(KeyCode.E) ? 1 : 0;
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
            List<string> drawNow = new(){ "m_Script" };

            SerializedProperty modelProperty = serializedObject.FindProperty("model");
            SerializedProperty hpProperty = serializedObject.FindProperty("hp");
            SerializedProperty beh = serializedObject.FindProperty("behaviourType");
            var script = (Agent)target;

            if (EditorApplication.isPlaying && beh.enumValueIndex == (int)BehaviourType.Learn && script.enabled)
            {
                // Draw Step
                int currentStep = script.EpisodeStepCount;
                string stepcount = $"Decisions [{currentStep}]";
                EditorGUILayout.HelpBox(stepcount, MessageType.None);

                // Draw Reward
                string cumReward = $"Reward [{script.EpsiodeCumulativeReward}]";
                EditorGUILayout.HelpBox(cumReward, MessageType.None);

                // Draw buffer 
                float bufferFillPercentage = script.Memory.Count * Trainer.ParallelAgentsCount / ((float)script.hp.bufferSize) * 100f;
                StringBuilder sb = new StringBuilder();
                sb.Append("Buffer [");
                sb.Append(script.Memory.Count * Trainer.ParallelAgentsCount);
                sb.Append(" / ");
                sb.Append(script.hp.bufferSize);
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

            // If no model or hp draw button
            if (modelProperty.objectReferenceValue == null || hpProperty.objectReferenceValue == null)
            {
                if (GUILayout.Button("Bake/Load model and hyperparameters"))
                {
                    script.BakeModel();
                    script.BakeHyperparamters();
                }
            }
                
            if (modelProperty.objectReferenceValue != null)
            {
                drawNow.Add("hiddenUnits");
                drawNow.Add("layers");
                drawNow.Add("spaceSize");
                drawNow.Add("continuousActions");
                drawNow.Add("discreteActions");
            }



            // Need to draw everything and then draw the line.. i will let this for the future
            // EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);


          

            if(beh.enumValueIndex != (int)BehaviourType.Learn)
            {
                drawNow.Add("Hp");
            }

           
            DrawPropertiesExcluding(serializedObject, drawNow.ToArray());

            serializedObject.ApplyModifiedProperties();

           
        }
    }
}


using DeepUnity.NeuroForge;
using System.Collections.Generic;
using System.ComponentModel.Design;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [AddComponentMenu("DeepUnity/Agent"), DisallowMultipleComponent, RequireComponent(typeof(HyperParameters))]
    public class Agent : MonoBehaviour
    {
        public ActorCritic network;
        public BehaviourType behaviour = BehaviourType.Inference;
        public OnEpisodeEndType onEpisodeEnd = OnEpisodeEndType.ResetEnvironment;
        
        [Space]
        public int observationSize = 2;
        public ActionType actionSpace = ActionType.Continuous;
        public int continuousDim = 2;
        public int[] discreteBranches = new int[] { 2 };

        
        [HideInInspector] public MemoryBuffer memory;
        private HyperParameters hp;
        private List<ISensor> sensors;
        private TransformReseter personalEnvironment;


        public SensorBuffer Observations { get; private set; }
        public ActionBuffer Actions { get; private set; }
        public int CompletedEpisodes { get; private set; }
        public int StepCount { get; private set; }
        public float TimestepReward { get; private set; }
        public float CumulativeReward { get; private set; }
        private bool IsEpisodeEnd { get; set; }

        private void Awake()
        {
            hp = GetComponent<HyperParameters>();

            InitNetwork();
            InitBuffers();
            InitSensors(transform);

            personalEnvironment = onEpisodeEnd == OnEpisodeEndType.ResetAgent ?
                new TransformReseter(transform) :
                new TransformReseter(transform.parent);

            CompletedEpisodes = 0;
            StepCount = 0;
            TimestepReward = 0;
            CumulativeReward = 0;

            OnAwake();
        }
        private void Start()
        {
            if (behaviour == BehaviourType.Inference)
                Trainer.Subscribe(this);

            OnStart();
        }
        private void FixedUpdate()
        {
            switch(behaviour)
            {
                case BehaviourType.Inference:
                    InferenceBehavior();
                    break;

                case BehaviourType.Inactive:
                    break;

                case BehaviourType.Active:
                    ActiveBehavior();
                    break;

                case BehaviourType.Manual:
                    ManualBehavior();
                    break;

                case BehaviourType.Test:
                    ActiveBehavior();
                    break;
            }

            IsEpisodeEnd = false;
            StepCount++;

            if (StepCount == hp.maxSteps)
                EndEpisode();


            if (behaviour == BehaviourType.Inference && memory.IsFull())
                Trainer.Ready();

            OnFixedUpdate();
        }

        // Setup
        private void InitNetwork()
        {
            if (network != null)
            {
                observationSize = network.observationSize;
                actionSpace = network.actionSpace;
                continuousDim = network.continuousDim;
                discreteBranches = network.discreteBranches;
            }
            else
            {
                network = actionSpace == ActionType.Continuous ?
                    new ActorCritic(observationSize, continuousDim, hp, NetworkNameGenerator) :
                    new ActorCritic(observationSize, discreteBranches, hp, NetworkNameGenerator);
            }
        }
        private void InitBuffers()
        {
            memory = new MemoryBuffer(hp.bufferSize);
            Observations = new SensorBuffer(network.observationSize);
            Actions = new ActionBuffer(network.continuousDim);
        }
        private void InitSensors(Transform parent)
        {
            ISensor sensor = parent.GetComponent<ISensor>();

            if (sensor != null)
                sensors.Add(sensor);

            foreach (Transform child in parent)
            {
                InitSensors(child);
            }
        }

        // Loop
        private void ActiveBehavior()
        {

        }
        private void ManualBehavior()
        {

        }
        private void InferenceBehavior()
        {
            Observations.Clear();
            Actions.Clear();

            CollectObservations(Observations);
            CollectSensorsObservations(Observations);


            if(actionSpace == ActionType.Continuous)
            {
                Tensor state = Tensor.Constant(Observations.values);
                Tensor log_prob;
                Tensor action = network.ContinuousAction(state, out log_prob);
                Tensor value = network.Value(state);
                Tensor reward = network.rewardStadardizer.Standardise(Tensor.Constant(TimestepReward));
                Tensor done = Tensor.Constant(IsEpisodeEnd == true ? 1 : 0);

                Actions.ContinuousActions = action.ToArray();
                memory.StoreContinuous(state, value, log_prob, value, reward, done);
            }
            else
            {
                throw new System.NotImplementedException();
            }
        }
        private void CollectSensorsObservations(SensorBuffer buffer)
        {
            foreach (var item in sensors)
            {
                buffer.AddObservation(item.GetObservations());

            }
        }


        // User call
        public virtual void OnAwake() { }
        public virtual void OnStart() { }
        public virtual void OnFixedUpdate() { }
        public virtual void OnEpisodeBegin() { }
        public virtual void CollectObservations(SensorBuffer sensorBuffer) { }
        public virtual void OnActionReceived(ActionBuffer actionBuffer) { }    
        public void EndEpisode(bool verbose = false)
        {
            if (behaviour == BehaviourType.Active || behaviour == BehaviourType.Inactive)
                return;

            

            if(verbose)
            {
                StringBuilder statistic = new StringBuilder();
                statistic.Append("<color=#0c74eb>");
                statistic.Append("Episode: ");
                statistic.Append(CompletedEpisodes + 1);
                statistic.Append(" | Steps: ");
                statistic.Append(StepCount);
                statistic.Append("s | Cumulated Reward: ");
                statistic.Append(CumulativeReward);
                statistic.Append("</color>");
                Debug.Log(statistic.ToString());
            }

            IsEpisodeEnd = true;
            personalEnvironment?.Reset();
            TimestepReward = 0;
            CumulativeReward = 0;
            StepCount = 0;
            CompletedEpisodes++;
        }
        public void AddReward(float reward)
        {
            TimestepReward = reward;
            CumulativeReward += reward;
        }



        private static string NetworkNameGenerator
        {
            get
            {
                short id = 1;
                while (AssetDatabase.LoadAssetAtPath<ActorCritic>("Assets/Actor#" + id + ".asset") != null)
                    id++;
                return "PPONetwork#" + id;
            }
        }
    }

    public enum BehaviourType
    {
        Inactive,
        Active,
        Inference,
        Manual,
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
            List<string> dontDrawMe = new List<string>();
            dontDrawMe.Add("m_Script");

            // Hide action space
            SerializedProperty actType = serializedObject.FindProperty("actionSpace");
            if (actType.enumValueIndex == (int)ActionType.Continuous)
                dontDrawMe.Add("discreteBranches");
            else
                dontDrawMe.Add("continuousDim");

            // Hide networks
            SerializedProperty beh = serializedObject.FindProperty("behaviour");
            if (beh.enumValueIndex == (int)BehaviourType.Manual)
            {
                dontDrawMe.Add("network");
            }


            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
}


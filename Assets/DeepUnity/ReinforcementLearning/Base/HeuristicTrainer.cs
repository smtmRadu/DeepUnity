using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Heuristic Training Works like this.
    /// The user controls the agent (with DecisionRequester period set on 1).
    /// When the memory reaches the size of a batch, One Update is made.
    /// </summary>
    public class HeuristicTrainer : MonoBehaviour
    {
        private static HeuristicTrainer Instance { get; set; }

        Agent ag;
        AgentBehaviour ac;
        Hyperparameters hp;       
        TrainingStatistics trainingStatistics;

        private float autosaveSecondsElapsed = 0f;
        [SerializeField, Min(1)] private int autosave = 1;
        private readonly DateTime timeWhenTheTrainingStarted = DateTime.Now;

        ExperienceBuffer train_data;
        List<Tensor> states_batches;
        List<Tensor> cont_act_batches;
        List<Tensor> disc_act_batches;

        private bool TrainFlag { get; set; } = false;
        private int batch_index = 0;
        private int current_epoch = 0;

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(this);
            }
            else
            {
                Instance = this;
            }

        }
        public static void Subscribe(Agent agent)
        {
            if(Instance == null)
            {
                EditorApplication.playModeStateChanged += Autosave1;
                EditorApplication.pauseStateChanged += Autosave2;
                GameObject go = new GameObject("[DeepUnity] Trainer - Imitation");
                go.AddComponent<HeuristicTrainer>();
                Instance.ag = agent;
                Instance.ac = agent.model;
                Instance.hp = agent.model.config;
                Instance.trainingStatistics = agent.PerformanceTrack;
                Instance.ac.InitOptimisers(Instance.hp, agent.imitationStrength);
                Instance.ac.InitSchedulers(Instance.hp);
                Instance.train_data = new ExperienceBuffer();
            }
            else
            {
                ConsoleMessage.Warning("More than one agent instances with Heuristic behaviour are enabled in the same time. Only one is required to run in Heuristic Mode");
                EditorApplication.isPlaying = false;
                return;
            }
        }

        private void FixedUpdate()
        {
            if(Instance.autosaveSecondsElapsed >= Instance.autosave * 60f)
            {
                Instance.autosaveSecondsElapsed = 0f;
                Instance.ac.Save();
            }


            if(TrainFlag)
            {
               
                if(batch_index == Instance.states_batches.Count - 1)
                {
                    batch_index = 0;
                    current_epoch++;
                }

                if(current_epoch >= Instance.hp.numEpoch)
                {
                    TrainFlag = false;
                    Instance.ag.enabled = true;
                    current_epoch = 0;
                    return;
                }


                TrainOnBatch(Instance.batch_index);

                batch_index++;
            }
        }
        private static void Autosave1(PlayModeStateChange state)
        {
            Instance.ac.Save();
        }
        private static void Autosave2(PauseState state)
        {
            Instance.ac.Save();
        }

        public static void SendMemory(in MemoryBuffer agent_memory)
        {
            if(agent_memory.Count == Instance.hp.bufferSize)
            {
                Instance.train_data.Add(agent_memory, Instance.hp.bufferSize);
                if (Instance.hp.debug) Utils.DebugInFile(agent_memory.ToString());
                agent_memory.Clear();

                Instance.states_batches = Utils.Split(Instance.train_data.States, Instance.hp.batchSize).Select(x => Tensor.Cat(null, x)).ToList();

                if(Instance.ac.IsUsingContinuousActions) 
                    Instance.cont_act_batches = Utils.Split(Instance.train_data.ContinuousActions, Instance.hp.batchSize).Select(x => Tensor.Cat(null, x)).ToList();
                
                if(Instance.ac.IsUsingDiscreteActions)
                    Instance.disc_act_batches = Utils.Split(Instance.train_data.DiscreteActions, Instance.hp.batchSize).Select(x => Tensor.Cat(null, x)).ToList();

                Instance.TrainFlag = true;
                Instance.ag.enabled = false;

            }
        }


        private static void TrainOnBatch(int batch_index)
        {
            Tensor states_batch = Instance.states_batches[batch_index];
                         
             // here we train our bad boys

            if (Instance.ac.IsUsingContinuousActions)
            {
                Tensor cont_act_batch_real = Instance.cont_act_batches[batch_index];

                // Train Discriminator
                Tensor cont_act_batch_fake;
                Instance.ac.ContinuousForward(states_batch, out cont_act_batch_fake, out _);

                Instance.ac.discriminatorContinuousOptimizer.ZeroGrad();
                
                var prediction_real = Instance.ac.discriminatorContinuous.Forward(cont_act_batch_real);
                var loss_real = Loss.BinaryCrossEntropy(prediction_real, DiscriminatorRealTarget(Instance.hp.batchSize));
                Instance.ac.discriminatorContinuous.Backward(loss_real.Derivative);

                var prediction_fake = Instance.ac.discriminatorContinuous.Forward(cont_act_batch_fake);
                var loss_fake = Loss.BinaryCrossEntropy(prediction_fake, DiscriminatorFakeTarget(Instance.hp.batchSize));
                Instance.ac.discriminatorContinuous.Backward(loss_fake.Derivative);

                Instance.ac.discriminatorContinuousOptimizer.ClipGradNorm(Instance.hp.gradClipNorm);
                Instance.ac.discriminatorContinuousOptimizer.Step();

                Instance.trainingStatistics?.valueLoss.Append(loss_fake.Item + loss_real.Item);
                


                // Train Generator
                Instance.ac.actorMuOptimizer.ZeroGrad();
                Tensor Gz;
                Instance.ac.ContinuousForward(states_batch, out Gz, out _);
                Tensor DGz = Instance.ac.discriminatorContinuous.Forward(states_batch);
                Loss loss = Loss.BinaryCrossEntropy(DGz, DiscriminatorRealTarget(Instance.hp.batchSize));
                var generatorLossDiff = Instance.ac.discriminatorContinuous.Backward(loss.Derivative);
                Instance.ac.actorContinuousMu.Backward(generatorLossDiff);
                Instance.ac.actorMuOptimizer.Step();

                Instance.trainingStatistics?.policyLoss.Append(loss.Item);
            }

            
            if(Instance.ac.IsUsingDiscreteActions)
            {
                Tensor disc_act_batch_real = Instance.disc_act_batches[batch_index];

                Debug.Log(states_batch);
                Tensor disc_act_batch_fake;
                Instance.ac.DiscreteForward(states_batch, out disc_act_batch_fake);

                // Train discriminator
                Instance.ac.discriminatorDiscreteOptimizer.ZeroGrad();

                var prediction_real = Instance.ac.discriminatorDiscrete.Forward(disc_act_batch_real);
                var loss_real = Loss.BinaryCrossEntropy(prediction_real, DiscriminatorRealTarget(Instance.hp.batchSize));
                Instance.ac.discriminatorDiscrete.Backward(loss_real.Derivative);

                var prediction_fake = Instance.ac.discriminatorDiscrete.Forward(disc_act_batch_fake);
                var loss_fake = Loss.BinaryCrossEntropy(prediction_fake, DiscriminatorFakeTarget(Instance.hp.batchSize));
                Instance.ac.discriminatorDiscrete.Backward(loss_fake.Derivative);

                Instance.ac.discriminatorDiscreteOptimizer.ClipGradNorm(Instance.hp.gradClipNorm);
                Instance.ac.discriminatorDiscreteOptimizer.Step();

                Instance.trainingStatistics?.valueLoss.Append(loss_fake.Item + loss_real.Item);



                // Train Generator
                Instance.ac.actorDiscreteOptimizer.ZeroGrad();
                Tensor Gz;
                Instance.ac.DiscreteForward(states_batch, out Gz);
                Tensor DGz = Instance.ac.discriminatorDiscrete.Forward(Gz);
                Loss loss = Loss.BinaryCrossEntropy(DGz, DiscriminatorRealTarget(Instance.hp.batchSize));
                var generatorLossDiff = Instance.ac.discriminatorDiscrete.Backward(loss.Derivative);
                Instance.ac.actorDiscrete.Backward(generatorLossDiff);
                Instance.ac.actorDiscreteOptimizer.Step();
                Instance.trainingStatistics?.policyLoss.Append(loss.Item);
            }

        }

        private static Tensor DiscriminatorRealTarget(int batch_size)
        {
            return Tensor.Ones(batch_size, 1);
        }
        private static Tensor DiscriminatorFakeTarget(int batch_size)
        {
            return Tensor.Zeros(batch_size, 1);
        }
        // Basically i'm planning to create a list
        // (int, Batch). First elem is e decremental integer. When reaches 0 the batch is eradicated.
        // On Each frame, one batch is backpropagated. An optimisation is made after exactly 16 backprops.
        // Every batch_size frames one new batch is added to the list.

        // Another option is to basically use a batch only once and that's all.
    }



}


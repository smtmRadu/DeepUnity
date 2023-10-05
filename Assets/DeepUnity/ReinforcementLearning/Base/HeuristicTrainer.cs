using System;
using System.Collections.Generic;
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

        AgentBehaviour ac;
        Hyperparameters hp;
        ExperienceBuffer train_batch;

        private float autosaveSecondsElapsed = 0f;
        [SerializeField, Min(1)] private int autosave = 1;
        private readonly DateTime timeWhenTheTrainingStarted = DateTime.Now;

        public static void Subscribe(Agent agent)
        {
            if(Instance == null)
            {
                EditorApplication.playModeStateChanged += Autosave1;
                EditorApplication.pauseStateChanged += Autosave2;
                GameObject go = new GameObject("[DeepUnity] Trainer");
                Instance.ac = agent.model;
                Instance.hp = agent.model.config;
                Instance.ac.InitOptimisers(Instance.hp, agent.imitationStrength);
                Instance.ac.InitSchedulers(Instance.hp);
                Instance.train_batch = new ExperienceBuffer();
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
        }
        private static void Autosave1(PlayModeStateChange state)
        {
            Instance.ac.Save();
        }
        private static void Autosave2(PauseState state)
        {
            Instance.ac.Save();
        }

        
        public static void TrainOnMemoryBatch(in MemoryBuffer agent_memory)
        {
            if(agent_memory.Count == Instance.hp.batchSize)
            {
                Instance.train_batch.Add(agent_memory, Instance.hp.batchSize);
                if (Instance.hp.debug) Utils.DebugInFile(agent_memory.ToString());


                ExperienceBuffer train_data = new ExperienceBuffer();
                train_data.Add(agent_memory, Instance.hp.batchSize);


                Tensor states_batch = Tensor.Cat(null, train_data.States);
                Tensor cont_act_batch_real = Tensor.Cat(null, train_data.ContinuousActions);
                Tensor disc_act_batch_real = Tensor.Cat(null, train_data.DiscreteActions);


                // here we train our bad boys

                if (Instance.ac.IsUsingContinuousActions)
                {
                    // Train Discriminator
                    Tensor cont_act_batch_fake;
                    Instance.ac.ContinuousPredict(states_batch, out cont_act_batch_fake, out _);

                    Instance.ac.discriminatorContinuousOptimizer.ZeroGrad();
                    
                    var prediction_real = Instance.ac.discriminatorContinuous.Forward(cont_act_batch_real);
                    var loss_real = Loss.BinaryCrossEntropy(prediction_real, DiscriminatorRealTarget(Instance.hp.batchSize));
                    Instance.ac.discriminatorContinuous.Backward(loss_real.Derivative);

                    var prediction_fake = Instance.ac.discriminatorContinuous.Forward(cont_act_batch_fake);
                    var loss_fake = Loss.BinaryCrossEntropy(prediction_fake, DiscriminatorFakeTarget(Instance.hp.batchSize));
                    Instance.ac.discriminatorContinuous.Backward(loss_fake.Derivative);

                    Instance.ac.discriminatorContinuousOptimizer.ClipGradNorm(Instance.hp.gradClipNorm);
                    Instance.ac.discriminatorContinuousOptimizer.Step();

                    float error = loss_fake.Item + loss_real.Item;

                    // add error to training statistics

                }

                
                if(Instance.ac.IsUsingDiscreteActions)
                {

                }





                agent_memory.Clear();
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


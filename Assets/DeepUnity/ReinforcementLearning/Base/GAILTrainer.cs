using System.Collections.Generic;
using System.Linq;
using UnityEditor;


namespace DeepUnity
{
    /// <summary>
    /// Generative Adversial Imitation Learning (GAIL)
    /// Heuristic Training Works like this.
    /// The user controls the agent (with DecisionRequester period set on 1).
    /// When the memory reaches the size of a batch, One Update is made.
    /// </summary>
    public class GAILTrainer : DeepUnityTrainer
    {
        List<Tensor> states_batches;
        List<Tensor> cont_act_batches;
        List<Tensor> disc_act_batches;

        private bool TrainFlag { get; set; } = false;
        private int batch_index = 0;

        private float epoch_generator_loss = 0f;
        private float epoch_discriminator_loss = 0f;

        protected override void FixedUpdate()
        {
            if(TrainFlag)
            {
                if (batch_index == states_batches.Count - 1)
                {
                    track?.policyLoss.Append(epoch_generator_loss / batch_index);
                    track?.valueLoss.Append(epoch_discriminator_loss / batch_index);
                    epoch_generator_loss = 0f;
                    epoch_discriminator_loss = 0f;

                    batch_index = 0;
                    track.iterations++;
                    train_data.Shuffle();
                }

                TrainOnBatch(batch_index);

                batch_index++;
            }

            if(MemoriesCount == hp.bufferSize)
            {
                train_data.TryAppend(parallelAgents[0].Memory, hp.bufferSize);
                if (hp.debug) Utils.DebugInFile(parallelAgents[0].Memory.ToString());
                parallelAgents[0].Memory.Clear();
                
                states_batches = Utils.Split(train_data.States, hp.batchSize).Select(x => Tensor.Concat(null, x)).ToList();
                
                if(model.IsUsingContinuousActions) 
                    cont_act_batches = Utils.Split(train_data.ContinuousActions, hp.batchSize).Select(x => Tensor.Concat(null, x)).ToList();
                
                if(model.IsUsingDiscreteActions)
                    disc_act_batches = Utils.Split(train_data.DiscreteActions, hp.batchSize).Select(x => Tensor.Concat(null, x)).ToList();
                
                TrainFlag = true;
                parallelAgents[0].behaviourType = BehaviourType.Off;

                model.discContNetwork?.SetDevice(Device.GPU);
                model.discDiscNetwork?.SetDevice(Device.GPU);
                model.muNetwork?.SetDevice(Device.GPU);
                model.sigmaNetwork?.SetDevice(Device.GPU);
                model.discreteNetwork?.SetDevice(Device.GPU);
            }


            base.FixedUpdate();   
        }

        private void TrainOnBatch(int batch_index)
        {
            Tensor states_batch = states_batches[batch_index];
                         
             // here we train our bad boys

            if (model.IsUsingContinuousActions)
            {
                Tensor cont_act_batch_real = cont_act_batches[batch_index];

                // Train Discriminator
                Tensor cont_act_batch_fake;
                model.ContinuousForward(states_batch, out cont_act_batch_fake, out _);

                model.dContOptimizer.ZeroGrad();
                
                var prediction_real = model.discContNetwork.Forward(cont_act_batch_real);
                var loss_real = Loss.BCE(prediction_real, DiscriminatorRealTarget(hp.batchSize));
                model.discContNetwork.Backward(loss_real.Derivative);

                var prediction_fake = model.discContNetwork.Forward(cont_act_batch_fake);
                var loss_fake = Loss.BCE(prediction_fake, DiscriminatorFakeTarget(hp.batchSize));
                model.discContNetwork.Backward(loss_fake.Derivative);

                model.dContOptimizer.ClipGradNorm(hp.gradClipNorm);
                model.dContOptimizer.Step();
                epoch_discriminator_loss += loss_fake.Item + loss_real.Item;
                
                


                // Train Generator
                model.muOptimizer.ZeroGrad();
                Tensor Gz;
                model.ContinuousForward(states_batch, out Gz, out _);
                Tensor DGz = model.discContNetwork.Forward(states_batch);
                Loss loss = Loss.MSE(DGz, DiscriminatorRealTarget(hp.batchSize));
                var generatorLossDiff = model.discContNetwork.Backward(loss.Derivative);
                model.muNetwork.Backward(generatorLossDiff);

                model.muOptimizer.ClipGradNorm(hp.gradClipNorm);
                model.muOptimizer.Step();
                epoch_generator_loss += loss.Item;
            }

            
            if(model.IsUsingDiscreteActions)
            {
                Tensor disc_act_batch_real = disc_act_batches[batch_index];

                Tensor disc_act_batch_fake;
                model.DiscreteForward(states_batch, out disc_act_batch_fake);

                // Train discriminator
                model.dDiscOptimizer.ZeroGrad();

                var prediction_real = model.discDiscNetwork.Forward(disc_act_batch_real);
                var loss_real = Loss.BCE(prediction_real, DiscriminatorRealTarget(hp.batchSize));
                model.discDiscNetwork.Backward(loss_real.Derivative);

                var prediction_fake = model.discDiscNetwork.Forward(disc_act_batch_fake);
                var loss_fake = Loss.BCE(prediction_fake, DiscriminatorFakeTarget(hp.batchSize));
                model.discDiscNetwork.Backward(loss_fake.Derivative);

                model.dDiscOptimizer.ClipGradNorm(hp.gradClipNorm);
                model.dDiscOptimizer.Step();
                epoch_discriminator_loss += loss_fake.Item + loss_real.Item;



                // Train Generator
                model.discreteOptimizer.ZeroGrad();
                Tensor Gz;
                model.DiscreteForward(states_batch, out Gz);
                Tensor DGz = model.discDiscNetwork.Forward(Gz);
                Loss loss = Loss.BCE(DGz, DiscriminatorRealTarget(hp.batchSize));
                var generatorLossDiff = model.discDiscNetwork.Backward(loss.Derivative);
                model.discreteNetwork.Backward(generatorLossDiff);

                model.discreteOptimizer.ClipGradNorm(hp.gradClipNorm);
                model.discreteOptimizer.Step();
                epoch_generator_loss += loss.Item;
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


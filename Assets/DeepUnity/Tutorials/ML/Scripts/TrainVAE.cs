using DeepUnity;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class TrainVAE : MonoBehaviour
{
    public float lr = 1e-3f;
    public int batchSize = 32;


    NeuralNetwork encoder;
    NeuralNetwork decoder;
    NeuralNetwork mu;
    NeuralNetwork logvar;

    Optimizer optim;

    List<(Tensor, Tensor)> train = new();
    List<(Tensor, Tensor)[]> train_batches;

    int batch_index = 0;

    private void Start()
    {
        Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out train, out _, DatasetSettings.LoadTrainOnly);
        Utils.Shuffle(train);
        train_batches = Utils.Split(train, batchSize);

        encoder = new NeuralNetwork(
            new Dense(784, 256),
            new ReLU(),
            new Dense(256, 8),
            new ReLU()).CreateAsset("encoder");

        mu = new NeuralNetwork(
            new Dense(8, 8)).CreateAsset("mu");

        logvar = new NeuralNetwork(
            new Dense(8, 8)).CreateAsset("log_var");

        decoder = new NeuralNetwork(
            new Dense(8, 256),
            new ReLU(),
            new Dense(256, 784),
            new ReLU()).CreateAsset("decoder");

        Parameter[] parameters = encoder.Parameters();
        parameters = parameters.Concat(mu.Parameters()).ToArray();
        parameters = parameters.Concat(logvar.Parameters()).ToArray();
        parameters = parameters.Concat(decoder.Parameters()).ToArray();

        optim = new Adam(parameters, lr);
    }


    private void Update()
    {
        if (batch_index % 50 == 0)
        {
            encoder.Save();
            decoder.Save();
            mu.Save();
            logvar.Save();
        }

        // Case when epoch finished
        if (batch_index == train_batches.Count - 1)
        {
            batch_index = 0;
            Utils.Shuffle(train);
        }

        float loss_value = 0f;

        var batch = train_batches[batch_index];
        Tensor input = Tensor.Concat(null, batch.Select(x => x.Item1).ToArray());

        Tensor encoded, mean, log_variance, eps;
        Tensor decoded = Forward(input, out encoded, out mean, out log_variance, out eps);

        // Backpropagate the MSE loss
        Loss mse = Loss.MSE(decoded, input); loss_value += mse.Item;
        Tensor dMSEdDecoded = mse.Derivative;

        // Backprop MSE  through decoder
        Tensor dMSE_dMu_SigmaEps = decoder.Backward(dMSEdDecoded); // derivative of the loss with respect to z = mu * sigma * std;

        // Backprop MSE  through mu
        Tensor dMSE_dMu = dMSE_dMu_SigmaEps; // dMu/dEnc = 1
        Tensor dMSE_dEncoder = mu.Backward(dMSE_dMu);

        // Backprop MSE  through sigma
        Tensor dMSE_dLogVar = dMSE_dMu_SigmaEps * eps; // dSigma/dEnc = eps
        dMSE_dEncoder += logvar.Backward(dMSE_dLogVar);

        // Backprop MSE  through encoder
        encoder.Backward(dMSE_dEncoder);

        Tensor kld = -0.5f * (1f + log_variance - mean.Pow(2f) - log_variance.Exp()); loss_value += kld.ToArray().Average();
        // Compute gradients for mu
        Tensor dKLD_dMu = mean;
        Tensor dMu_dEncoded = mu.Backward(dKLD_dMu);
        // Compute gradients for sigma
        Tensor dKLD_dLogVar = 0.5f + 0.5f * log_variance.Exp();
        Tensor dLogVar_dEncoded = logvar.Backward(dKLD_dLogVar);

        var kldDerivative = dMu_dEncoded + dLogVar_dEncoded;
        // Compute gradients for encoder
        encoder.Backward(3 * kldDerivative);



        print($"Batch: {batch_index} | Loss: {loss_value}");

        batch_index++;
    }
    private Tensor Reparametrize(Tensor mu, Tensor log_var, out Tensor eps)
    {
        var std = Tensor.Exp(0.5f * log_var);
        eps = Tensor.RandomNormal(log_var.Shape);
        return mu + std * eps;
    }

    private Tensor Forward(Tensor input, out Tensor encoded, out Tensor mu_v, out Tensor logvar_v, out Tensor eps)
    {
        encoded = encoder.Forward(input);

        mu_v = mu.Forward(encoded);
        logvar_v = logvar.Forward(encoded);
        

        var z = Reparametrize(mu_v, logvar_v, out eps);

        var decoded = decoder.Forward(z);
        return decoded;
    }

}



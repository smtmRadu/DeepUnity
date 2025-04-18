using UnityEngine;
using DeepUnity.Optimizers;
using DeepUnity.Activations;
using DeepUnity.Modules;
using DeepUnity.Models;

namespace DeepUnity.Tutorials
{
    public class Tutorial : MonoBehaviour
    {
        [SerializeField] private Sequential network;
        private Optimizer optim;
        private Tensor x;
        private Tensor y;

        public void Start()
        {
            network = new Sequential(
                new Dense(512, 64),
                new ReLU6(),
                new Dropout(0.1f),
                new Dense(64, 64, device: Device.GPU),
                new RMSNorm(64),
                new ReLU(),
                new Dense(64, 32)).CreateAsset("TutorialModel");
            
            optim = new AdamW(network.Parameters(), amsgrad: true);
            x = Tensor.RandomNormal(64, 512);
            y = Tensor.RandomNormal(64, 32);
        }

        public void Update()
        {
            Tensor yHat = network.Forward(x);
            Loss loss = Loss.MSE(yHat, y);
            
            optim.ZeroGrad();
            network.Backward(loss.Grad);
            optim.Step();

            print($"Epoch: {Time.frameCount} - Train Loss: {loss.Item}");
            network.Save();
        }
    }
}
using DeepUnity;
using DeepUnity.Optimizers;
using System.Linq;
using UnityEngine;
using DeepUnity.Activations;
using DeepUnity.Layers;
using DeepUnity.Models;

namespace DeepUnityTutorials
{
    public class UnitTest : MonoBehaviour
    {
        public Sequential net;
        Optimizer optim;
        [SerializeField] PerformanceGraph graph = new PerformanceGraph();

        Tensor input = Tensor.RandomNormal(64, 6, 10);
        Tensor target = Tensor.Random01(64, 1);

        public VariationalAutoencoder vae;


        private Sequential q1Network
            ;
        private Sequential Qtarg1;
        private void Start()
        {
            q1Network = new Sequential(
                new Dense(10, 100),
                new Tanh(),
                new Dense(100, 10));
            q1Network = new Sequential(
               new Dense(10, 100),
               new Tanh(),
               new Dense(100, 10));
        }


        private void Update()
        {

            Tensor[] phi1 = q1Network.Parameters().Select(x => x.theta).ToArray();

            Tensor[] phi_targ1 = Qtarg1.Parameters().Select(x => x.theta).ToArray();

            // We update the target q functions softly...
            // OpenAI algorithm uses polyak = 0.995, the same thing with using τ = 0.005, inverse the logic duhh. 
            // φtarg,i <- (1 - τ)φtarg,i + τφi     for i = 1,2

            for (int i = 0; i < phi1.Length; i++)
            {
                Tensor.CopyTo((1f - 0.001f) * phi_targ1[i] + 0.001f * phi1[i], phi_targ1[i]);
            }
        }
    }

}



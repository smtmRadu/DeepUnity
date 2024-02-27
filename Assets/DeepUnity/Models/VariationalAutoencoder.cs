using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    public class VariationalAutoencoder : Model<VariationalAutoencoder, Tensor>, ISerializationCallbackReceiver
    {
        [NonSerialized] private IModule[] encoder;
        [NonSerialized] private Dense mu;
        [NonSerialized] private Dense log_var;
        [NonSerialized] private IModule[] decoder;
        [SerializeField] private float kld_weight;

        [SerializeField] private IModuleWrapper[] serializedEncoder;
        [SerializeField] private IModuleWrapper serializedMu;
        [SerializeField] private IModuleWrapper serializedLogVar;
        [SerializeField] private IModuleWrapper[] serializedDecoder;

      
        Tensor mu_v;
        Tensor logvar_v;
        Tensor ksi;

        /// <summary>
        /// A Variational AutoEncoder (VAE) with inner KLDivergence loss computation.
        /// </summary>
        /// <param name="encoder"></param>
        /// <param name="latent_space">Note that the output of the encoder and the input of the decoder must match the latent_space.</param>
        /// <param name="decoder"></param>
        /// <param name="kld_weight">The inner KL divergence loss weighting factor.</param>
        /// <exception cref="ArgumentException"></exception>
        public VariationalAutoencoder(
            IModule[] encoder, 
            int latent_space,
            IModule[] decoder,
            float kld_weight = 3f)
        {
            if (latent_space < 1)
                throw new ArgumentException("Latent space must be > 0");

            if (encoder == null || encoder.Length == 0)
                throw new ArgumentException("Encoder was not initialized");

            if (decoder == null || decoder.Length == 0)
                throw new ArgumentException("Decoder was not initialized");

            if (kld_weight <= 0f)
                throw new ArgumentException("KLD loss weight must be > 0");

            this.encoder = encoder;
            mu = new Dense(latent_space, latent_space);
            log_var = new Dense(latent_space, latent_space);
            this.decoder = decoder;
            this.kld_weight = kld_weight;
        }

        public override Tensor Predict(Tensor input)
        {
            Tensor encoded = encoder[0].Predict(input);
            for (int i = 1; i < encoder.Length; i++)
            {
                encoded = encoder[i].Predict(encoded);
            }

            mu_v = mu.Predict(encoded);
            logvar_v = log_var.Predict(encoded);

            Tensor z = Reparametrize(mu_v, logvar_v, out ksi);

            Tensor decoded = decoder[0].Predict(z);
            for (int i = 1; i < decoder.Length; i++)
            {
                decoded = decoder[i].Predict(decoded);
            }

            return decoded;
        }
        public override Tensor Forward(Tensor input)
        {
            Tensor encoded = encoder[0].Forward(input);
            for (int i = 1; i < encoder.Length; i++)
            {
                encoded = encoder[i].Forward(encoded);
            }

            mu_v = mu.Forward(encoded);
            logvar_v = log_var.Forward(encoded);

            Tensor z = Reparametrize(mu_v, logvar_v, out ksi);

            Tensor decoded = decoder[0].Forward(z);
            for (int i = 1; i < decoder.Length; i++)
            {
                decoded = decoder[i].Forward(decoded);
            }

            return decoded;
        }
        public override Tensor Backward(Tensor lossGradient)
        {
            Tensor dLoss_dZ = decoder[decoder.Length - 1].Backward(lossGradient);
            for (int i = decoder.Length - 2; i >= 0; i--)
            {
                dLoss_dZ = decoder[i].Backward(dLoss_dZ);
            }

            Tensor dLoss_dMu = dLoss_dZ;
            Tensor dLoss_dEncoder = mu.Backward(dLoss_dMu);

            Tensor dLoss_dLogVar = dLoss_dZ * ksi;
            dLoss_dEncoder += log_var.Backward(dLoss_dLogVar);

            Tensor dLoss_dInput = encoder[encoder.Length - 1].Backward(dLoss_dEncoder);
            for (int i = encoder.Length - 2; i >= 0; i--)
            {
                dLoss_dInput = encoder[i].Backward(dLoss_dInput);
            }

            // Compute inner kld
            // Tensor kld = -0.5f * (1f + logvar_v - mu_v.Pow(2f) - logvar_v.Exp());

            Tensor dKLD_dMu = mu_v;
            Tensor dMu_dEncoded = mu.Backward(dKLD_dMu * kld_weight);
            Tensor dKLD_dLogvar = 0.5f * (logvar_v.Exp() - 1f);
            Tensor dLogvar_dEncoded = log_var.Backward(dKLD_dLogvar * kld_weight);

            Tensor dZ_dInput = encoder[encoder.Length - 1].Backward(dMu_dEncoded + dLogvar_dEncoded);
            for (int i = encoder.Length - 2; i >= 0; i--)
            {
                dZ_dInput = encoder[i].Backward(dZ_dInput);
            }

            return dLoss_dInput + dZ_dInput;
        }

        private Tensor Reparametrize(Tensor mu, Tensor log_var, out Tensor ksi)
        {
            var std = Tensor.Exp(0.5f * log_var);
            ksi = Tensor.RandomNormal(log_var.Shape);
            return mu + std * ksi;
        }


        public override Parameter[] Parameters()
        {
            List<Parameter> param = new();
            foreach (var item in encoder.OfType<ILearnable>())
            {
                param.AddRange(item.Parameters());
            }
            param.AddRange(mu.Parameters());
            param.AddRange(log_var.Parameters());
            foreach (var item in decoder.OfType<ILearnable>())
            {
                param.AddRange(item.Parameters());
            }
            return param.ToArray();

        }
        public override void SetDevice(Device device)
        {
            mu.SetDevice(device);
            log_var.SetDevice(device);
            foreach (var item in encoder.OfType<ILearnable>())
            {
                item.SetDevice(device);
            }
            foreach (var item in decoder.OfType<ILearnable>())
            {
                item.SetDevice(device);
            }
        }

        public string Summary()
        {
            StringBuilder stringBuilder = new StringBuilder();

            stringBuilder.AppendLine($"Name: {name}");
            stringBuilder.AppendLine($"Type: {GetType().Name}");
            stringBuilder.AppendLine($"Parameters: " +
                $"{encoder.Where(x => x is ILearnable).Select(x => (ILearnable)x).Sum(x => x.ParametersCount()) + decoder.Where(x => x is ILearnable).Select(x => (ILearnable)x).Sum(x => x.ParametersCount()) + mu.ParametersCount() + log_var.ParametersCount()}");
            return stringBuilder.ToString();
        }
        public override object Clone()
        {
            var cloned_encoder = encoder.Select(x => (IModule)x.Clone()).ToArray();
            var cloned_mu = mu.Clone() as Dense;
            var cloned_logvar = log_var.Clone() as Dense;
            var cloned_decoder = decoder.Select(x => (IModule)(x.Clone())).ToArray();

            var vae = new VariationalAutoencoder(cloned_encoder, 1, cloned_decoder);
            vae.mu = cloned_mu;
            vae.log_var = cloned_logvar;
            return vae;
        }

        public void OnBeforeSerialize()
        {
            serializedEncoder = encoder.Select(x => IModuleWrapper.Wrap(x)).ToArray();
            serializedMu = IModuleWrapper.Wrap(mu);
            serializedLogVar = IModuleWrapper.Wrap(log_var);
            serializedDecoder = decoder.Select(x => IModuleWrapper.Wrap(x)).ToArray();
        }
        public void OnAfterDeserialize()
        {
            encoder = serializedEncoder.Select(x => IModuleWrapper.Unwrap(x)).ToArray();
            decoder = serializedDecoder.Select(x => IModuleWrapper.Unwrap(x)).ToArray();
            mu = (Dense) IModuleWrapper.Unwrap(serializedMu);
            log_var = (Dense)IModuleWrapper.Unwrap(serializedLogVar);
        }
    }
}




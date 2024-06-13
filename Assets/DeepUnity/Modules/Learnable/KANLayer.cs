using DeepUnity.Activations;
using System;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity.Modules
{
    /// <summary>
    /// <b>A Kolmogorov-Arnold layer that can be stacked into a KAN.</b> <br></br><br></br>
    /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
    /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
    /// where  B = batch_size and H = in_features.<br />
    /// </summary>
    [Serializable]
    public sealed class KANLayer : ILearnable, IModule
    {
        [SerializeField] public Device Device { get; set; } = Device.CPU;
        [SerializeField] public bool RequiresGrad { get; set; } = true;
        /// <summary>
        /// The basis activation
        /// </summary>
        [SerializeField] private IActivation b { get; set; } = new Swish();

        /// <summary>
        /// x Cache.
        /// </summary>
        private Tensor InputCache { get; set; }
        /// <summary>
        /// Residual cache.
        /// </summary>
        private Tensor WbB_xCache { get; set; }
        /// <summary>
        /// Splines cache.
        /// </summary>
        private Tensor WsSpline_xCache { get; set; }




        /// Serializable.

        [SerializeField] private int splineOrder;
        [SerializeField] private string basis_activation;

        [SerializeField] private Tensor weight_base;
        [SerializeField] private Tensor weight_base_grad;
        [SerializeField] private Tensor weight_spline;
        [SerializeField] private Tensor weight_spline_grad;

        /// <summary>
        /// B-Spline coeff.
        /// </summary>
        [SerializeField] private Tensor c;
        [SerializeField] private Tensor c_grad;

        /// <summary>
        /// <b>A Kolmogorov-Arnold layer that can be stacked into a KAN.</b> <br></br><br></br>
        /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
        /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
        /// where  B = batch_size and H = in_features.<br />
        /// 
        /// <br></br><i>If <paramref name="activation"/> = null, basis activation <em>b(x)</em> will be <see cref="Swish"/>.</i>
        /// </summary>
        /// <param name="in_features"></param>
        /// <param name="out_features"></param>
        /// <param name="base_activation">If null, <see cref="Swish"/> is used.</param>
        /// <param name="spline_order">the order of piecewise polynomial</param>
        /// <param name="scale_base">The scale of the </param>
        public KANLayer(int in_features, int out_features, int grid_size = 5, int spline_order = 3, IActivation activation = null, Device device = default)
        {
            if (spline_order < 2)
                throw new ArgumentException("Spline order cannot be less than 2");

            this.Device = device;
            activation = activation ?? new Swish();
            b = activation.Clone() as IActivation;
            basis_activation = b.GetType().Name;


            this.splineOrder = spline_order;

         
            // Initialization scales. Each activation function is initialized to have ws = 1 and spline(x) ≈ 0^2.
            // wb is initialized according to the Xavier initialization, which has been used to initialize linear layers in MLPs.

            weight_base = Parameter.Create(new int[] { out_features, in_features }, in_features, out_features, InitType.Xavier_Uniform);
            weight_spline = Parameter.Create(new int[] { out_features, in_features }, in_features, out_features, InitType.Ones);

            // 2This is done by drawing B-spline coefficients ci ∼ N (0, σ**2) with a small σ, typically we set σ = 0.1.
            c = Parameter.Create(new int[] { grid_size }, grid_size, grid_size, InitType.Normal0_1);
        }
        public Tensor Predict(Tensor x)
        {
            // phi(x) = w_b * b(x) + w_s * spline(x)
            // b(x) = silu(x)
            // spline(x) = sum(c_i * B_i(x))

            Tensor phi = Linear(b.Predict(x), weight_base) + Linear(Spline(x), weight_spline);

            return phi;
        }
        public Tensor Forward(Tensor x)
        {
            WbB_xCache = Linear(b.Predict(x), weight_base);
            WsSpline_xCache = Linear(Spline(x), weight_spline);

            return WbB_xCache + WsSpline_xCache;
        }
        private Tensor Linear(Tensor x, Tensor weights)
        {
            bool isBatched = x.Rank == 2;
            int B_size = isBatched ? x.Size(0) : 1;
            // x = (B, H_in) or (H_in)
            // W = (H_out, H_in)
            // B = (H_out)
            int H_out = weights.Size(-2);
            int H_in = weights.Size(-1);

            Tensor y = isBatched ? Tensor.Zeros(B_size, H_out) : Tensor.Zeros(H_out);

            //   (B, H_in) * (H_in, H_out)
            //  (n, m) * (m, p) = (n, p)
            if (isBatched)
            {
                Parallel.For(0, B_size, b =>
                {
                    for (int hout = 0; hout < H_out; hout++)
                    {
                        float sum = 0f;
                        for (int hin = 0; hin < H_in; hin++)
                        {
                            sum += x[b, hin] * weights[hout, hin];
                        }
                        y[b, hout] = sum;
                    }
                });
            }
            else
            {
                //Tests show that even only on 64 hid units it might perform better without multithread, so keep it like this indefinetely..
                Parallel.For(0, H_out, hout =>
                {
                    float sum = 0f;
                    for (int hin = 0; hin < H_in; hin++)
                    {
                        sum += x[hin] * weights[hout, hin];
                    }
                    y[hout] = sum;
                });
            }

            return y;
        }
        private Tensor Spline(Tensor x)
        {
            // w_s * spline(x) = Linear(((B, H_in), (H_out, H_in))
            // spline(x) =  (B, GRID_SIZE) * (GRID_SIZE, H_in)       (B, H_in)
            return Tensor.MatMul(c, BSpline(x), Device);
        }

        private Tensor BSpline(Tensor x)
        {
            // x = (B, H_in)
            // return: 
            throw null;
        }
        public Tensor Backward(Tensor dLdY)
        {
            throw new NotImplementedException();
        }

        public Parameter[] Parameters()
        {
            return null;
        }

        public object Clone()
        {
            return null;
        }

       

        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.

            if (weight_base.Shape == null)
                return;

            if (weight_base.Shape.Length == 0)
                return;

            // do not check if gamma is != null...
            weight_base_grad = Tensor.Zeros(weight_base.Shape);
            weight_spline_grad = Tensor.Zeros(weight_spline.Shape);
            c_grad = Tensor.Zeros(c.Shape);

            b = Activator.CreateInstance(Type.GetType(basis_activation)) as IActivation;
        }
    }
}




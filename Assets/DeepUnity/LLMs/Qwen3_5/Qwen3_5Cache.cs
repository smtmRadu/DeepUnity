using System;
using UnityEngine;

namespace DeepUnity
{
    namespace Qwen3_5Modeling
    {
        // Hybrid cache:
        //   - Full-attention layers store standard K/V buffers (per token, per kv-head, per head_dim).
        //   - Linear-attention (Gated DeltaNet) layers store SSM state:
        //         conv_state      [conv_dim * (kernel_size - 1)]   FP32
        //         recurrent_state [num_v_heads * head_k_dim * head_v_dim] FP32
        //
        // v1: in-memory only. No disk persistence.
        public class Qwen3_5Cache : IDisposable
        {
            public ComputeBuffer[] kCaches; // length numLayers; null on linear layers
            public ComputeBuffer[] vCaches; // length numLayers; null on linear layers

            public ComputeBuffer[] convStates;      // length numLayers; null on full layers
            public ComputeBuffer[] recurrentStates; // length numLayers; null on full layers

            public int CachedTokenCount { get; set; }

            readonly int numLayers;
            readonly int capacity;

            public Qwen3_5Cache(
                int capacity,
                Qwen3_5LayerType[] layerTypes,
                int headsKV, int headDim,
                int convDim, int convKernelSize,
                int numVHeads, int headKDim, int headVDim)
            {
                this.numLayers = layerTypes.Length;
                this.capacity = capacity;

                kCaches = new ComputeBuffer[numLayers];
                vCaches = new ComputeBuffer[numLayers];
                convStates = new ComputeBuffer[numLayers];
                recurrentStates = new ComputeBuffer[numLayers];

                int kvFloats = capacity * headsKV * headDim;
                int convFloats = convDim * (convKernelSize - 1);
                int recFloats = numVHeads * headKDim * headVDim;

                for (int i = 0; i < numLayers; i++)
                {
                    if (layerTypes[i] == Qwen3_5LayerType.FullAttention)
                    {
                        kCaches[i] = new ComputeBuffer(kvFloats, 4, ComputeBufferType.Structured);
                        vCaches[i] = new ComputeBuffer(kvFloats, 4, ComputeBufferType.Structured);
                    }
                    else // LinearAttention
                    {
                        convStates[i]      = new ComputeBuffer(convFloats, 4, ComputeBufferType.Structured);
                        recurrentStates[i] = new ComputeBuffer(recFloats, 4, ComputeBufferType.Structured);
                    }
                }

                CachedTokenCount = 0;
            }

            // Resets the logical token count only. The SSM state zero-fill is done GPU-side by
            // Qwen3_5Model.ResetCache (ZeroBuffer kernel) — the old CPU path allocated ~19 MB of
            // managed zero arrays and SetData'd them on the main thread on every reset.
            public void Reset()
            {
                CachedTokenCount = 0;
            }

            public void Dispose()
            {
                for (int i = 0; i < numLayers; i++)
                {
                    kCaches[i]?.Release();
                    vCaches[i]?.Release();
                    convStates[i]?.Release();
                    recurrentStates[i]?.Release();
                }
            }
        }
    }
}

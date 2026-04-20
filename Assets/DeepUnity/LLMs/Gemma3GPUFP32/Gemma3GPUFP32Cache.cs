using System;
using UnityEngine;

namespace DeepUnity
{
    namespace Gemma3GPUFP32Modeling
    {
        public class Gemma3GPUFP32Cache : IDisposable
        {
            public ComputeBuffer[] kCaches;
            public ComputeBuffer[] vCaches;
            public int CachedTokenCount { get; set; }

            readonly int numLayers;
            readonly int capacity;

            public Gemma3GPUFP32Cache(int numLayers, int capacity, int headsKV, int headDim)
            {
                this.numLayers = numLayers;
                this.capacity = capacity;

                kCaches = new ComputeBuffer[numLayers];
                vCaches = new ComputeBuffer[numLayers];

                int bufSize = capacity * headsKV * headDim;
                for (int i = 0; i < numLayers; i++)
                {
                    kCaches[i] = new ComputeBuffer(bufSize, 4, ComputeBufferType.Structured);
                    vCaches[i] = new ComputeBuffer(bufSize, 4, ComputeBufferType.Structured);
                }

                CachedTokenCount = 0;
            }

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
                }
            }
        }
    }
}

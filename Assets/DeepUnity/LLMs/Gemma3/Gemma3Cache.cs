using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace DeepUnity
{
    namespace Gemma3Modeling
    {
        public class Gemma3Cache : IDisposable
        {
            public ComputeBuffer[] kCaches;
            public ComputeBuffer[] vCaches;
            public int CachedTokenCount { get; set; }

            readonly int numLayers;
            readonly int headsKV;
            readonly int headDim;
            readonly int capacity;

            const int FILE_MAGIC = 0x47334B56; // "G3KV"
            const int FILE_VERSION = 1;

            public Gemma3Cache(int numLayers, int capacity, int headsKV, int headDim)
            {
                this.numLayers = numLayers;
                this.headsKV = headsKV;
                this.headDim = headDim;
                this.capacity = capacity;

                kCaches = new ComputeBuffer[numLayers];
                vCaches = new ComputeBuffer[numLayers];

                // FP32 activations — stride 4, full count
                int bufSize = capacity * headsKV * headDim;
                for (int i = 0; i < numLayers; i++)
                {
                    kCaches[i] = new ComputeBuffer(bufSize, 4, ComputeBufferType.Structured);
                    vCaches[i] = new ComputeBuffer(bufSize, 4, ComputeBufferType.Structured);
                }

                CachedTokenCount = 0;
            }

            public void Reset() => CachedTokenCount = 0;

            // Persist the populated slice of K/V caches to a folder.
            // Layout:
            //   {folder}/meta.bin                  -- header (magic, version, dims, tokenCount)
            //   {folder}/k_cache_layer_{i}.bin     -- packed FP16, perLayerFloats halves
            //   {folder}/v_cache_layer_{i}.bin     -- packed FP16, perLayerFloats halves
            //
            // Coroutine: each layer is read back via AsyncGPUReadback and yields a frame
            // between layers. Avoids the back-to-back synchronous GetData storm that
            // tripped GPU TDR on lower-end cards.
            public IEnumerator SaveAsync(string folder)
            {
                int tokens = CachedTokenCount;
                if (tokens <= 0) yield break;

                int perLayerFloats = tokens * headsKV * headDim;
                int perLayerBytes = perLayerFloats * 4;
                Directory.CreateDirectory(folder);

                using (FileStream fs = new FileStream(Path.Combine(folder, "meta.bin"), FileMode.Create, FileAccess.Write))
                using (BinaryWriter bw = new BinaryWriter(fs))
                {
                    bw.Write(FILE_MAGIC);
                    bw.Write(FILE_VERSION);
                    bw.Write(numLayers);
                    bw.Write(headsKV);
                    bw.Write(headDim);
                    bw.Write(tokens);
                }

                ushort[] scratchH = new ushort[perLayerFloats];
                byte[] bytes = new byte[perLayerFloats * 2];

                for (int i = 0; i < numLayers; i++)
                {
                    AsyncGPUReadbackRequest reqK = AsyncGPUReadback.Request(kCaches[i], perLayerBytes, 0);
                    while (!reqK.done) yield return null;
                    if (reqK.hasError)
                    {
                        ConsoleMessage.Info($"Gemma3 cache save: readback error on K layer {i}.");
                        yield break;
                    }
                    var dataK = reqK.GetData<float>();
                    for (int j = 0; j < perLayerFloats; j++) scratchH[j] = Mathf.FloatToHalf(dataK[j]);
                    Buffer.BlockCopy(scratchH, 0, bytes, 0, bytes.Length);
                    File.WriteAllBytes(Path.Combine(folder, $"k_cache_layer_{i}.bin"), bytes);

                    yield return null;

                    AsyncGPUReadbackRequest reqV = AsyncGPUReadback.Request(vCaches[i], perLayerBytes, 0);
                    while (!reqV.done) yield return null;
                    if (reqV.hasError)
                    {
                        ConsoleMessage.Info($"Gemma3 cache save: readback error on V layer {i}.");
                        yield break;
                    }
                    var dataV = reqV.GetData<float>();
                    for (int j = 0; j < perLayerFloats; j++) scratchH[j] = Mathf.FloatToHalf(dataV[j]);
                    Buffer.BlockCopy(scratchH, 0, bytes, 0, bytes.Length);
                    File.WriteAllBytes(Path.Combine(folder, $"v_cache_layer_{i}.bin"), bytes);

                    yield return null;
                }
            }

            // Attempt to load a previously persisted KV cache from a folder.
            // Coroutine: yields a frame between layers to avoid stalling on SetData
            // and the FP16->FP32 conversion. Result delivered via the onComplete callback
            // (true = loaded successfully, false = missing/mismatch/IO error).
            public IEnumerator TryLoadAsync(string folder, Action<bool> onComplete)
            {
                string metaPath = Path.Combine(folder, "meta.bin");
                if (!File.Exists(metaPath)) { onComplete?.Invoke(false); yield break; }

                int magic, version, fNumLayers, fHeadsKV, fHeadDim, fTokens;
                using (FileStream fs = new FileStream(metaPath, FileMode.Open, FileAccess.Read))
                using (BinaryReader br = new BinaryReader(fs))
                {
                    if (fs.Length < 24) { onComplete?.Invoke(false); yield break; }
                    magic = br.ReadInt32();
                    version = br.ReadInt32();
                    fNumLayers = br.ReadInt32();
                    fHeadsKV = br.ReadInt32();
                    fHeadDim = br.ReadInt32();
                    fTokens = br.ReadInt32();
                }

                if (magic != FILE_MAGIC || version != FILE_VERSION ||
                    fNumLayers != numLayers || fHeadsKV != headsKV || fHeadDim != headDim ||
                    fTokens <= 0 || fTokens > capacity)
                {
                    onComplete?.Invoke(false); yield break;
                }

                int perLayerFloats = fTokens * headsKV * headDim;
                int expectedBytes = perLayerFloats * 2;

                float[] scratchF = new float[perLayerFloats];
                ushort[] scratchH = new ushort[perLayerFloats];

                for (int i = 0; i < numLayers; i++)
                {
                    string kPath = Path.Combine(folder, $"k_cache_layer_{i}.bin");
                    string vPath = Path.Combine(folder, $"v_cache_layer_{i}.bin");
                    if (!File.Exists(kPath) || !File.Exists(vPath)) { onComplete?.Invoke(false); yield break; }

                    byte[] kBytes = File.ReadAllBytes(kPath);
                    if (kBytes.Length != expectedBytes) { onComplete?.Invoke(false); yield break; }
                    Buffer.BlockCopy(kBytes, 0, scratchH, 0, expectedBytes);
                    for (int j = 0; j < perLayerFloats; j++) scratchF[j] = Mathf.HalfToFloat(scratchH[j]);
                    kCaches[i].SetData(scratchF, 0, 0, perLayerFloats);

                    yield return null;

                    byte[] vBytes = File.ReadAllBytes(vPath);
                    if (vBytes.Length != expectedBytes) { onComplete?.Invoke(false); yield break; }
                    Buffer.BlockCopy(vBytes, 0, scratchH, 0, expectedBytes);
                    for (int j = 0; j < perLayerFloats; j++) scratchF[j] = Mathf.HalfToFloat(scratchH[j]);
                    vCaches[i].SetData(scratchF, 0, 0, perLayerFloats);

                    yield return null;
                }

                CachedTokenCount = fTokens;
                onComplete?.Invoke(true);
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

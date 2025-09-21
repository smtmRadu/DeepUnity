
using DeepUnity.Activations;
using DeepUnity.Models;
using DeepUnity.Modules;
using DeepUnity.Optimizers;
using DeepUnity.Qwen3Modeling;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity.Tutorials
{
    public class UnitTest : MonoBehaviour
    {
        public Device device = Device.CPU;
        public bool online_sm = false;
        public Sequential net;
        public GameObject canvas;
        public Optimizer optim;
        public int batch_size = 10;
        private List<RawImage> displays;
        public PerformanceGraph performanceGraph = new PerformanceGraph();
        public PerformanceGraph performanceGraph2 = new PerformanceGraph();

        private Tensor x;
        public float mu = 0, sigma = 1f, theta = 0.2f, dt = 0.02f, x0 = 2f;

        private void TestConcat()
        {
            Tensor x = Tensor.Random01(2, 3);
            Tensor y = Tensor.Random01(2, 3);
            Tensor z = Tensor.Random01(2, 3);

            print(x);
            print(y);
            print(z);
            print(Tensor.Concat(null, x, y, z));

        }
        Gemma3ForCausalLM gemma_model;
        GemmaTokenizerFast gemma_tokenizer;

        private void Start()
        {
            Tensor x = Tensor.RandomNormal(100, 100);
            Softmax sm = new Softmax();
            Benckmark.Start();
            sm.Predict(x);
            Benckmark.Stop();
            
            //print(gemma_model.ParameterCount());


            // string input = "What's the capital of France?";
            // 
            // gemma_model.Generate(input, gemma_tokenizer);

            //TestGemma();
            //TestGemmaTokenizer();
            //TestSwiGLU();
            //TestEmbedding();
            //TestRMSNorm();
            //TestSwiGLU();
            //TestGQACache();
            //TestQwen();
            // TestGQA();
            return;

            // TestTokenizer();
            
            var conversation = new List<Dictionary<string, string>>()
                {
                    new Dictionary<string, string>
                    {
                        { "role", "system" },
                        {"content", "You are a helpfull assistant" }
                    },
                    new Dictionary<string, string>
                    {
                        { "role", "user" },
                        { "content", "How's the day today?" }
                    },
                };
            
            Benckmark.Start();
            var tokenizer = new Qwen2TokenizerFast("Assets/DeepUnity/Tokenizers/Qwen2TokenizerFast.json");
            
            string inpu = tokenizer.ApplyChatTemplate(conversation);
            print(inpu);
            // print(tokenizer.Encode(new List<string> { "Hi sanclsnac", " csaciauh ncsknacasbiucn jcknas cijas cianckjasbi", " csaciauh ncsknacasbiucn jcknas cijas cianckjasbi soa hciucb kjasch iausc aksjch aoshc assixhaos xakhsbx aisxg asib ib" }));
            Benckmark.Stop();
        }

        private void TestGemma()
        {
            var gemma_model = new Gemma3ForCausalLM();
            print(gemma_model.ParameterCount());

            Tensor input_ids = Tensor.Ones(1, 1);
            Tensor attention_mask = null;// Tensor.Ones(1, 1);
            gemma_model.Predict(input_ids, attention_mask);
        }
        private void TestEmbedding()
        {
            Embedding emb = new Embedding(30, 4, pad_idx:0);

            print(emb.embeddings);

            Tensor x = Tensor.Ones(1);

            Tensor x2 = Tensor.Constant(new float[] { 0, 3 });

            print(emb.Predict(x));
            print(emb.Predict(x2));
        }
        private void TestRMSNorm()
        {
            RMSNorm rms = new RMSNorm(103, 1e-6f);
            Qwen3RMSNorm rms2 = new Qwen3RMSNorm(103);

            rms.gamma = Tensor.LinSpace(-0.5f, 0.7f, 103);
            rms2.gamma = rms.gamma.ToArray();

            print(rms.gamma.ToArray().ToCommaSeparatedString());
            print(rms2.gamma.ToArray().ToCommaSeparatedString());
            Tensor x = Tensor.LinSpace(-1, 1, 206).Reshape(2, 103);

            print(rms.Predict(x));
            print(rms2.Predict(x));
        }
        private void TestSwiGLU()
        {
            // works.
            // the kernel of qwen3 mlp fails.

           // GatedLinearUnit swiglu = new GatedLinearUnit(1024, 3072, 1024, device:Device.GPU);
           // Qwen3MLP swiglu2 = new Qwen3MLP(1024, 3072);
           // float[] w = swiglu.gate_proj.weights.ToArray().Concat(swiglu.up_proj.weights.ToArray()).ToArray();
           // w = w.Concat(swiglu.down_proj.weights.ToArray()).ToArray();
// 
           // swiglu2.weights.Dispose();
           // swiglu2.weights = TensorGPU.Constant(w);
// 
           // Tensor x = Tensor.RandomNormal(1, 1, 1024);
           // TensorGPU xgpu = TensorGPU.Identity(x);
           // //print(swiglu.gate_proj.weights);
           // //print(swiglu2.weights.ToCommaSeparatedString());
           // Benckmark.Start();
           // print(swiglu.Predict(x));
           //  Benckmark.Stop();
// 
           //  Benckmark.Start();
           //  print(swiglu2.Predict(xgpu));
           //  Benckmark.Stop();
// 
// 
           //  xgpu.Dispose();
           //  // print(swiglu2.Predict(x));
        }
        private void TestQwen()
        {
            var qwen_model = new Qwen3ForCausalLM();
            print(qwen_model.ParameterCount());

            Tensor input_ids = Tensor.Ones(1, 1);
            Tensor attention_mask = null;// Tensor.Ones(1, 1);
            qwen_model.Predict(input_ids, attention_mask);
        }
        private void TestTokenizer()
        {
            var tokenizer = new Qwen2TokenizerFast();
            var enc = tokenizer.Encode("Hello there martin, here's your special delight zone!");
            print(enc);
            print(tokenizer.Decode(enc.Item1)[0]);
        }
        private void TestGemmaTokenizer()
        {
            string str = "Hello there martin, here's your special delight zone!";
            var tokenizer = new GemmaTokenizerFast();
            var enc = tokenizer.Encode(str);
           
            // print(tokenizer.token2id.ElementAt(993));
            // print(str);
            // print(enc);
            // print(tokenizer.Decode(enc.Item1)[0]);
        }
        private void TestGQA()
        {
            x = Tensor.LinSpace(-0.01f, 0.01f, steps: 81_920).Reshape(batch_size, 80, 1024);
            int expansion_factor = 2;
            GroupedQueryAttention mha = new GroupedQueryAttention(1024, 
                num_heads_q : 16,
                num_heads_kv: 8, 
                expansion_factor: expansion_factor, 
                is_causal: true, 
                device:device, 
                use_rope:true, 
                qk_norm:true,
                rope_theta:1_000_000,
                rope_max_seq_len:32_768);

            mha.W_QKV.weights = Tensor.LinSpace(-0.02f, 0.01f, 2048 * expansion_factor * 1024).Reshape(2048 * expansion_factor, 1024);
            mha.W_O.weights = Tensor.LinSpace(-0.01f, 0.02f, 1024 * 1024 * expansion_factor).Reshape(1024, 1024 * expansion_factor);

            print(x);

            Benckmark.Start();
          
            print(mha.Predict(x));
            Benckmark.Stop();
        }

        private void TestGQACache()
        {
            Tensor x = Tensor.RandomNormal(3, 16);
            GroupedQueryAttention mha = new GroupedQueryAttention(
                16,
                8, 4, 2, is_causal:true, 0, true, use_rope: true, device: Device.CPU);

            mha.BuildKVCache = true;
            Benckmark.Start();
            print(mha.Predict(x));
            Benckmark.Stop();

            Tensor x2 = Tensor.RandomNormal(1, 16);
            Benckmark.Start();
            print(mha.Predict(x2));
            Benckmark.Stop();

            Tensor x3 = Tensor.Concat(-2, x.Split(-2, 1).Concat(new Tensor[] { x2}).ToArray());
            mha.BuildKVCache = false;
            Benckmark.Start();
            print(mha.Predict(x3));
            Benckmark.Stop();



        }
    }

}



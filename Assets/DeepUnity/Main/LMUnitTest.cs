using DeepUnity.Gemma3Modeling;
using DeepUnity.Modules;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

namespace DeepUnity.Tutorials
{
    public class LMUnitTest : MonoBehaviour
    {
        [SerializeField] Device device = Device.CPU;
        [SerializeField] int batch_size = 1;
        Gemma3ForCausalLM gemma_model;
        GemmaTokenizerFast gemma_tokenizer;

        bool output_once = false;
        private void Update()
        {
            if (!gemma_model.IsReady || !gemma_tokenizer.IsReady)
                return;
        
            if (!output_once)
            {
                string input = "Hi Gemma!";
                var x = gemma_tokenizer.Encode(input);
        
                print(x.Item1);
                Benckmark.Start();
                print(gemma_model.Predict(x.Item1, x.Item2));
                Benckmark.Stop();
                output_once = true;
            }
            
        }
        private void Start()
        {
            Benckmark.Start();
            gemma_model = new Gemma3ForCausalLM();
            gemma_tokenizer = new GemmaTokenizerFast();
            Benckmark.Stop("gemma model init");
        }
        // private void Start()
        // {
        //     Utils.Random.Seed = 42;
        //     GatedLinearUnit glu = new GatedLinearUnit(63, 208, 63, activation:"gelu");
        //     Gemma3MLP mlp = new Gemma3MLP(63, 208, null);
        // 
        //     var weights = glu.gate_proj.weights.ToArray();
        //     weights = weights.Concat(glu.up_proj.weights.ToArray()).ToArray();
        //     weights = weights.Concat(glu.down_proj.weights.ToArray()).ToArray();
        //     mlp.weights.SetData(weights);
        // 
        //     Tensor x = Tensor.RandomNormal(4, 63);
        //     print(glu.Predict(x));
        //     print(mlp.Predict(x));
        // 
        //     mlp.weights.Dispose();
        // }
        // private void Start()
        // {
        //     Tensor x = Tensor.LinSpace(-0.01f, 0.01f, steps: 51_200).Reshape(batch_size, 80, 640);
        //     float expansion_factor = 1.6f;
        //     GroupedQueryAttention mha = new GroupedQueryAttention(640,
        //         num_heads_q: 4,
        //         num_heads_kv: 1,
        //         expansion_factor: expansion_factor,
        //         is_causal: true,
        //         device: device,
        //         use_rope: true,
        //         qk_norm: true,
        //         rope_theta: 1_000_000,
        //         rope_max_seq_len: 32_768);
        // 
        //     Gemma3GQA gqa = new Gemma3GQA(
        //         640,
        //         4,
        //         1,
        //         expansion_factor: expansion_factor,
        //         qk_norm_eps: 1e-6f,
        //         sliding_window: -1,
        //         query_pre_attention_scalar: 256,
        //         rope: mha.rope);
        // 
        //     mha.W_QKV.weights = Tensor.LinSpace(-0.02f, 0.01f, 983_040).Reshape(1536, 640);
        //     mha.W_O.weights = Tensor.LinSpace(-0.01f, 0.02f, 655_360).Reshape(640, 1024);
        //     gqa.W_QKV.SetData(mha.W_QKV.weights.ToArray());
        //     gqa.W_O.SetData(mha.W_O.weights.ToArray());
        //     gqa.q_norm.gamma = mha.q_rmsn.gamma.ToArray();
        //     gqa.k_norm.gamma = mha.k_rmsn.gamma.ToArray();
        //     //print(mha.W_QKV.weights);
        //     //print(Tensor.Constant(gqa.W_QKV, 1536, 640));
        // 
        //     print("Base GQA (out):" + mha.Predict(x));
        //     var gqa_out = gqa.Predict(x);
        //     print("Gemma GQA (out):" + gqa_out);
        // 
        //     gqa.W_QKV.Release();
        //     gqa.W_O.Release();
        // }
    }
}

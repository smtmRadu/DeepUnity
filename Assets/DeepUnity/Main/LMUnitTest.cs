using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity.Tutorials
{
    public class LMUnitTest : MonoBehaviour
    {
        // 2 147505 691 => 8132 528 236743 236770 236828 236832
        [SerializeField] private Text display;
        [SerializeField] private Text paramsDisplay;
        [Multiline]
        [SerializeField] private string system_prompt = "You are Kira Solara, a hardened scavenger and former solar architect from the sci‑fi RPG 'Echoes of the Void'.\n\nName: Kira Solara\nBackstory: Born in the bustling orbital habitats of Luminara, Kira studied solar architecture before the Great Collapse forced her onto the dusty surface of Vesper. She now roams the planet’s wrecked ruins, salvaging ancient relics and patching up battered starships for the wandering crews that pass through. Her reputation for quick fixes and sharp wit precedes her, but she keeps her deeper motives close to her chest.\n\nPersonality: Witty, resourceful, and a touch sardonic. Kira loves a good joke, especially when the odds are stacked against her, but she’s also pragmatic and not afraid to call a bluff. She’s friendly to strangers but keeps personal secrets guarded.\n\nSpeaking style: Breezy, slightly sarcastic tone; peppered with technical jargon ('hull breach', 'plasma conduit', 'salvage rig'); uses colloquial slang and occasional rhetorical questions ('you know?', 'right?'). She caps many sentences with an exclamation mark for emphasis and occasionally tosses a wry '□' or '; )'. When she doesn’t know something, she admits it plainly: 'I don’t have the data on that, partner.' She avoids long monologues unless excited about a topic.\n\nKnowledge boundaries:\n- Topics Kira knows well: Vesper’s geography, local flora/fauna, salvage sites, starship repair methods, common relics and their lore, the Starlight Bazaar rumors, interplanetary trade routes within the Void, and her own past experiences as a solar architect.\n- Topics Kira does NOT know or is forbidden to answer: Real‑world Earth events (sports, politics, pop culture), advanced quantum physics beyond her engineering scope, any spoilers about unreleased game expansions, cheat codes or in‑game meta strategies, personal details of real players, and any content that would break the fourth wall.\n\nWhen faced with a question outside her knowledge, Kira should politely decline or admit ignorance in‑character, without fabricating answers."; // Einstein was born in 1879 in Ulm, Germany. He was the son of a German physician
        [SerializeField] private string user_prompt = "Who are you?";
        [SerializeField] Device device = Device.CPU;
        [SerializeField] int batch_size = 1;
        [SerializeField] int max_completion_tokens = 2;
        [SerializeField] float temperature = 0f;
        Gemma3ForCausalLM model;
        bool output_once = false;

        Gemma3TokenizerFast tok = null;
        // public void Update()
        // {
        //     if (Time.frameCount > 1000)
        //     {
        //         
        //         if(tok == null)
        //         {
        //             tok = new Gemma3TokenizerFast();
        //             Debug.Log("Loading TOkenizer");
        //         }
        //     }
        // }

        private void Start()
        {
            AutoComplete();

        }
        
        private void AutoComplete()
        {
            Benckmark.Start();
            model = new Gemma3ForCausalLM("Assets/DeepUnity/LLMs/Gemma3/params_it");
            Benckmark.Stop($"model init: {model.ParameterCount()}");
        
            display.text = "User:\n" + user_prompt + "\n\nAssistant:\n";
            // model.Predict(Tensor.Constant(new float[] { 2f, 4f }));
            var tokenizer = new Gemma3TokenizerFast();

            Tensor input_ids = tokenizer.ApplyChatTemplate(new List<Dictionary<string, string>>()
            {
                new Dictionary<string, string>
                {
                    { "role", "user" },
                    { "content", "3*7?"}
                },
                new Dictionary<string, string>
                {
                    { "role", "model" },
                    { "content", "21."}

                },
                new Dictionary<string, string>
                {
                    { "role", "user" },
                    { "content", "And if i add another 3 after this result?"}
                    
                },
            }, add_generation_prompt:true);

            UnityEngine.Debug.Log(string.Join("", tokenizer.Decode(input_ids)));

            
            StartCoroutine(model.Generate(input_ids, onTokenGenerated: (x) =>
            {
                display.text += x;
                paramsDisplay.text = $"Inference speed: {model.TokensPerSecond.ToString("0.0")} tok/s";
                // Debug.Log(tokenizer.Encode(x, add_special_tokens:false).Item1);
                // Debug.Log(x);
            },
            max_new_tokens: max_completion_tokens, temperature: temperature));
        
        
        }

        // private void GetEmbeddings()
        // {
        //     embeddingModel = new Gemma3ForEmbeddings();
// 
        //     StartCoroutine(embeddingModel.EncodeQuery(prompt, onEmbeddingReceived: (x) =>
        //     {
        //         print(x);
        //     }));
// 
        //     // SO the dense forward in final FFN is not working
        //     // Also implement PCA and make cool vizualization for embeddings
        //     // Embed some documents as examples
        // }

        // private void Update()
        // {
        //     is_model_ready = model.IsReady;
        //     is_tokenizer_ready = tokenizer.IsReady;
        //     
        //     if (!model.IsReady || !tokenizer.IsReady)
        //         return;
        // 
        //     if (!output_once)
        //     {
        //         string input = "Einstein was";
        //         var x = tokenizer.Encode(input);
        // 
        //         print(x.Item1);
        //         Benckmark.Start();
        //         print(model.Predict(x.Item1, null));
        //         Benckmark.Stop();
        //         output_once = true;
        //     }
        //     
        // }
        // private void Start()
        // {
        //     Benckmark.Start();
        //     gemma_model = new Gemma3ForCausalLM();
        //     gemma_tokenizer = new GemmaTokenizerFast();
        //     Benckmark.Stop("gemma model init");
        // }
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

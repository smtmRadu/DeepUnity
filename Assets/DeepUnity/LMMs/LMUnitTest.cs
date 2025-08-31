using UnityEngine;

namespace DeepUnity.Tutorials
{
    public class LMUnitTest : MonoBehaviour
    {
        Gemma3ForCausalLM gemma_model;
        GemmaTokenizerFast gemma_tokenizer;
        private void Update()
        {
            if (!gemma_model.IsReady || !gemma_tokenizer.IsReady)
                return;

            // print(gemma_model.model.embed_tokens.Predict(Tensor.Constant(new float[] { 0, 1 })));

            string input = "Hi Gemma!";
            var x = gemma_tokenizer.Encode(input);
            
            print(x.Item1);
            Benckmark.Start();
            print(gemma_model.Predict(x.Item1, x.Item2));
            Benckmark.Stop();
        }
        private void Start()
        {
            Benckmark.Start();
            gemma_model = new Gemma3ForCausalLM();
            gemma_tokenizer = new GemmaTokenizerFast();
            Benckmark.Stop("gemma model init");
        }
    }
}

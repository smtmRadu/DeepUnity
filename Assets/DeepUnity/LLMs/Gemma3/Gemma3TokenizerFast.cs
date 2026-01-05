using System;
using System.Collections.Generic;

namespace DeepUnity
{
    public class Gemma3TokenizerFast : BPETokenizer
    {
        
        public static readonly int EOS_TOKEN_ID = 1;
        public static readonly int BOS_TOKEN_ID = 2;
        public static readonly int UNK_TOKEN_ID = 3;
        public static readonly int MASK_TOKEN_ID = 4;
        public static readonly int START_OF_TURN_TOKEN_ID = 105;
        public static readonly int END_OF_TURN_TOKEN_ID = 106;

        // Assets/DeepUnity/LMMs/Gemma3/GemmaTokenizerFast.json
        public Gemma3TokenizerFast(string path_to_vocab_file = "Assets/DeepUnity/LLMs/Gemma3/Gemma3TokenizerFast.json", bool load_async = true) : base(path_to_vocab_file, load_async)
        {

        }

        /// <inheritdoc/>
        public override (Tensor, Tensor) Encode(string input, bool add_special_tokens = true, bool truncation = false, int max_length = 512)
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (!IsReady)
            {
                throw new ArgumentException("Tokenizer loaded asynchronously and not yet initialized. Check 'tokenizer.IsReady' before using the tokenizer.");
            }

            int n = input.Length;
            Span<char> text = n <= 32_768 ? stackalloc char[n] : new char[n];
            for (int i = 0; i < n; i++)
                text[i] = input[i] == ' ' ? 'Ġ' : input[i]; // note \u2581 was replaced by Ġ in the gemma tokenizer otherwise it would have never worked... maybe other \u2581 like this should be replaced into the future....

            int capacity = truncation ? Math.Min(n, max_length) : n;
            int[] input_ids_buffer = new int[capacity];
            int idCount = 0;
            int pos = 0;

            while (pos < n && (!truncation || idCount < max_length))
            {
                TrieNode cur = token2id_trie;
                int bestId = PAD_TOKEN_ID;
                int bestLen = 1;
                int p = pos;

                while (p < n &&
                       (p - pos) < _maxTokenLength &&
                       cur.Next.TryGetValue(text[p], out TrieNode nxt))
                {
                    cur = nxt;
                    p++;

                    if (cur.TokenId is int tokId)
                    {
                        bestId = tokId;
                        bestLen = p - pos;
                    }
                }

                input_ids_buffer[idCount++] = bestId;
                pos += bestLen;
            }

            float[] input_ids = null;
            if(add_special_tokens)
            {
                input_ids = new float[idCount + 1];
                input_ids[0] = 2f; // <bos> tokens. Never append special tokens as strings '<bos>' because tokenization might go wrong. It will split a special token in more.
                for (int i = 0; i < idCount; i++) input_ids[i + 1] = input_ids_buffer[i];
            }
            else
            {
                input_ids = new float[idCount];
                for (int i = 0; i < idCount; i++) input_ids[i] = input_ids_buffer[i];
            }

            

            Tensor inputTensor = Tensor.Constant(input_ids);
            Tensor maskTensor = Tensor.Ones(input_ids.Length);

            return (inputTensor, maskTensor);
        }


        /// <inheritdoc/>
        public override string ApplyChatTemplate(List<Dictionary<string, string>> input, bool add_generation_prompt = true)
        {
            throw new NotImplementedException();
            // template is like so:
            /*
              <bos><start_of_turn>user
              [SYSTEM PROMPT]

              [USER PROMPT]<end_of_turn>
              <start_of_turn>model
              [ASSISTANT]<end_of_turn>
              ...etc
             
             
             */
        }

    }

}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Unity.VisualScripting;

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
        public static readonly int USER_TOKEN_ID = 2364;
        public static readonly int MODEL_TOKEN_ID = 4368;
        public static readonly int NEWLINE_TOKEN_ID = 107;
        public static readonly int DOUBLE_NEWLINE_TOKEN_ID = 108;

        // Assets/DeepUnity/LMMs/Gemma3/GemmaTokenizerFast.json
        public Gemma3TokenizerFast(string path_to_vocab_file = "Assets/DeepUnity/LLMs/Gemma3/Gemma3TokenizerFast.json", bool load_async = false) : base(path_to_vocab_file, load_async)
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
        public override Tensor ApplyChatTemplate(List<Dictionary<string, string>> input, bool add_generation_prompt = false)
        {
            // template is like so:
            /*
              <bos><start_of_turn>user\n
              [SYSTEM PROMPT]

              [USER PROMPT]<end_of_turn>\n|TILLHERE|
              <start_of_turn>model\n
              [ASSISTANT]<end_of_turn>\n|TILLHERE|
              ...etc
             
             
             */

            List<float> input_ids = new ();
            input_ids.Add(BOS_TOKEN_ID);
            

            if(input[0]["role"] == "system")
            {
                input_ids.Add(START_OF_TURN_TOKEN_ID);
                input_ids.Add(USER_TOKEN_ID);
                input_ids.Add(NEWLINE_TOKEN_ID);
                input_ids.AddRange(this.Encode(input[0]["content"], add_special_tokens:false).Item1.ToArray());
            
                for(int i = 1; i < input.Count; i++)
                {
                    if(i%2 == 1 && input[i]["role"] == "user")
                    {
                        if(i == 1)
                        {
                            input_ids.Add(DOUBLE_NEWLINE_TOKEN_ID);
                            input_ids.AddRange(this.Encode(input[i]["content"], add_special_tokens:false).Item1.ToArray());
                            input_ids.Add(END_OF_TURN_TOKEN_ID);
                            input_ids.Add(NEWLINE_TOKEN_ID);
                        }
                        else
                        {
                            
                            input_ids.Add(START_OF_TURN_TOKEN_ID);
                            input_ids.Add(USER_TOKEN_ID);
                            input_ids.Add(NEWLINE_TOKEN_ID);
                            input_ids.AddRange(this.Encode(input[i]["content"], add_special_tokens:false).Item1.ToArray());
                            input_ids.Add(END_OF_TURN_TOKEN_ID);
                            input_ids.Add(NEWLINE_TOKEN_ID);
                        }
                    }
                    else if (i%2 == 0 && input[i]["role"] == "model")
                    {
                        input_ids.Add(START_OF_TURN_TOKEN_ID);
                        input_ids.Add(MODEL_TOKEN_ID);
                        input_ids.Add(NEWLINE_TOKEN_ID);
                        input_ids.AddRange(this.Encode(input[i]["content"], add_special_tokens:false).Item1.ToArray());
                        input_ids.Add(END_OF_TURN_TOKEN_ID);
                        input_ids.Add(NEWLINE_TOKEN_ID);
                    }
                    else
                        throw new ArgumentException("The input doesn t have alternating user and prompt messages");
                }       
            }
            else if(input[0]["role"] == "user")
            {
                for(int i = 0; i < input.Count; i++)
                {
                    if(i%2 == 0 && input[i]["role"] == "user")
                    {
                        input_ids.Add(START_OF_TURN_TOKEN_ID);
                            input_ids.Add(USER_TOKEN_ID);
                            input_ids.Add(NEWLINE_TOKEN_ID);
                            input_ids.AddRange(this.Encode(input[i]["content"], add_special_tokens:false).Item1.ToArray());
                            input_ids.Add(END_OF_TURN_TOKEN_ID);
                            input_ids.Add(NEWLINE_TOKEN_ID);
                    }
                    else if(i%2 == 1 && input[i]["role"] == "model")
                    {
                         input_ids.Add(START_OF_TURN_TOKEN_ID);
                        input_ids.Add(MODEL_TOKEN_ID);
                        input_ids.Add(NEWLINE_TOKEN_ID);
                        input_ids.AddRange(this.Encode(input[i]["content"], add_special_tokens:false).Item1.ToArray());
                        input_ids.Add(END_OF_TURN_TOKEN_ID);
                        input_ids.Add(NEWLINE_TOKEN_ID);
                    }
                    else
                        throw new ArgumentException("The input doesn t have alternating user and model messages");
                }
            }
            else
                throw new ArgumentException("The input conversation doesn't start with a system or user message.");
 
            
            if(add_generation_prompt)
            {
                input_ids.Add(START_OF_TURN_TOKEN_ID);
                input_ids.Add(MODEL_TOKEN_ID);
                input_ids.Add(NEWLINE_TOKEN_ID);
            }

            return Tensor.Constant(input_ids.ToArray());
        }

    }

}

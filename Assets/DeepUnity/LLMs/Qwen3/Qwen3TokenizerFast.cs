using System;
using System.Collections.Generic;
using System.Text;


namespace DeepUnity
{
    public class Qwen3TokenizerFast : BPETokenizer
    {
        public static readonly int EOS_TOKEN_ID = 151643;
        public new static readonly int PAD_TOKEN_ID = 151643;
        public Qwen3TokenizerFast(string path_to_vocab_file = "Assets/DeepUnity/LLMs/Qwen3TokenizerFast.json") : base(path_to_vocab_file)
        {

        }

        /// <inheritdoc/>
        public override (Tensor, Tensor) Encode(string input, bool add_special_tokens = true,bool truncation = false,int max_length = 512)
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (!IsReady)
            {
                throw new ArgumentException("Tokenizer loaded asynchronously and not yet initialized. Check 'tokenizer.IsInitialized' before using the tokenizer.");
            }
            int n = input.Length;
            Span<char> text = n <= 32_768 ? stackalloc char[n] : new char[n];
            for (int i = 0; i < n; i++)
                text[i] = input[i] == ' ' ? 'Ġ' : input[i];

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


            float[] input_ids = new float[idCount];
            for (int i = 0; i < idCount; i++) input_ids[i] = input_ids_buffer[i];

            Tensor inputTensor = Tensor.Constant(input_ids);
            Tensor maskTensor = Tensor.Ones(idCount);   

            return (inputTensor, maskTensor);
        }


        /// <inheritdoc/>
        public override string ApplyChatTemplate(List<Dictionary<string, string>> input, bool add_generation_prompt = true)
        {
            StringBuilder formattedChat = new StringBuilder();

            foreach (var message in input)
            {
                if (message.TryGetValue("role", out string role) && message.TryGetValue("content", out string content))
                {
                    formattedChat.AppendFormat("<|im_start|>{0}\n{1}<|im_end|>\n", role, content);
                }
                else
                {
                    throw new ArgumentException("Messages should contain the following keys: role and content");
                }
            }
            if(add_generation_prompt)
                formattedChat.Append("<|im_start|>assistant\n");
            return formattedChat.ToString();
        }
    }
}


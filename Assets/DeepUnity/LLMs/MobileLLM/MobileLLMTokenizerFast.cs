using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace DeepUnity
{
    public class MobileLLMTokenizerFast : BPETokenizer
    {

        public static readonly int BOS_TOKEN_ID = 1;
        public static readonly int EOS_TOKEN_ID = 2;
        public MobileLLMTokenizerFast(string path_to_vocab_file = "Assets/DeepUnity/LLMs/MobileLLM/MobileLLMTokenizerFast.json", bool load_async = true) : base(path_to_vocab_file, load_async)
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

            if (!input.StartsWith(" "))
                input = " " + input;

            int n = input.Length;
            Span<char> text = n <= 2_048 ? stackalloc char[n] : new char[n];
            for (int i = 0; i < n; i++)
                text[i] = input[i] == ' ' ? '▁' : input[i];

            int capacity = truncation ? Math.Min(n, max_length) : n;
            int[] input_ids_buffer = new int[capacity];
            int idCount = 0;
            int pos = 0;

            while (pos < n && (!truncation || idCount < max_length))
            {
                TrieNode cur = token2id_trie;
                int bestId = -1;
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


            float[] input_ids = new float[add_special_tokens? idCount + 1 : idCount];
            if (!add_special_tokens)
                for (int i = 0; i < idCount; i++) input_ids[i] = input_ids_buffer[i];
            else
            {
                input_ids[0] = BOS_TOKEN_ID;
                for (int i = 1; i < idCount + 1; i++) input_ids[i] = input_ids_buffer[i-1];
            }
            Tensor inputTensor = Tensor.Constant(input_ids);
            Tensor maskTensor = Tensor.Ones(add_special_tokens ? idCount + 1 : idCount);

            return (inputTensor, maskTensor);
        }

        public override string ApplyChatTemplate(List<Dictionary<string, string>> input, bool add_generation_prompt = true)
        {
            throw new Exception("MobileLLM doesn't support chat template.");
        }

    }

}

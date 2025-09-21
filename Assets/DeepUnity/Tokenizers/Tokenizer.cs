using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.VisualScripting;


namespace DeepUnity
{
    public sealed class TrieNode
    {
        public int? TokenId;
        public readonly Dictionary<char, TrieNode> Next = new();
    }
    public abstract class Tokenizer
    {
        protected int EOS_TOKEN_ID;
        protected int PAD_TOKEN_ID;

        protected TrieNode token2id_trie;
        
        public Dictionary<string, int> token2id = new();
        public Dictionary<int, string> id2token = new();
        protected int _maxTokenLength;

        public bool IsReady { get; private set; } = false;

        /// <summary>
        /// Base class for a BinaryPairEncoding (BPE) Tokenizer. 
        /// </summary>
        /// <param name="path_to_vocab_json"></param>
        /// <param name="load_async">Either to initialize the tokenizer async or not. Default: true</param>
        /// <exception cref="ArgumentException"></exception>
        public Tokenizer(string path_to_vocab_json, bool load_async = true)
        {
            if (!File.Exists(path_to_vocab_json))
                throw new ArgumentException(nameof(path_to_vocab_json));

            if (load_async)
            {
                _ = LoadAsync(path_to_vocab_json);
                return;
            }
            else
            {
                // Sync loading
                string content = File.ReadAllText(path_to_vocab_json);
                ProcessContent(content);
            }      
        }

        public async Task LoadAsync(string path_to_vocab_json)
        {
            string content = await Task.Run(() => File.ReadAllText(path_to_vocab_json));
            await Task.Run(() => ProcessContent(content));
            ConsoleMessage.Info("Tokenizer loaded async");
        }

        private void ProcessContent(string content)
        {
            ReadOnlySpan<char> span = content.AsSpan();

            while (!span.IsEmpty)
            {
                int lineEnd = span.IndexOf('\n');
                ReadOnlySpan<char> line = lineEnd == -1 ? span : span.Slice(0, lineEnd);
                int colonIndex = line.IndexOf(':');
                if (colonIndex > 0)
                {
                    ReadOnlySpan<char> tokenPart = line.Slice(0, colonIndex);
                    ReadOnlySpan<char> valuePart = line.Slice(colonIndex + 1);

                    int firstQuote = tokenPart.IndexOf('"');
                    int lastQuote = tokenPart.LastIndexOf('"');
                    if (firstQuote >= 0 && lastQuote > firstQuote)
                    {
                        ReadOnlySpan<char> tokenSpan = tokenPart.Slice(firstQuote + 1, lastQuote - firstQuote - 1);

                        valuePart = valuePart.Trim();
                        if (valuePart.EndsWith(","))
                            valuePart = valuePart.Slice(0, valuePart.Length - 1);
                        valuePart = valuePart.Trim();

                        if (int.TryParse(valuePart, out int tokenId))
                        {
                            string token = tokenSpan.ToString();
                            token2id[token] = tokenId;
                            id2token[tokenId] = token;
                        }
                    }
                }

                span = lineEnd == -1 ? ReadOnlySpan<char>.Empty : span.Slice(lineEnd + 1);
            }

            BuildTrie();
            
        }

        private void BuildTrie()
        {
            token2id_trie = new TrieNode();
            foreach (var kvp in token2id)
            {
                _maxTokenLength = Math.Max(_maxTokenLength, kvp.Key.Length);
                TrieNode root = token2id_trie;
                foreach (char ch in kvp.Key)
                {
                    if (!root.Next.TryGetValue(ch, out TrieNode nxt))
                        root.Next[ch] = nxt = new TrieNode();
                    root = nxt;
                }
                root.TokenId = kvp.Value;
            }
            IsReady = true;
        }


        /// <param name="input"></param>
        /// <param name="add_special_tokens"></param>
        /// <param name="padding"></param>
        /// <param name="truncation"></param>
        /// <param name="max_length"></param>
        /// <returns><b>input_ids</b> and <b>attention_mask</b> tensors</returns>
        public abstract (Tensor, Tensor) Encode(string input, bool add_special_tokens = true, bool truncation = false, int max_length = 512);

        /// <param name="inputs"></param>
        /// <param name="add_special_tokens"></param>
        /// <param name="padding"></param>
        /// <param name="truncation"></param>
        /// <param name="max_length"></param>
        /// <returns><b>input_ids</b> and <b>attention_mask</b> tensors</returns>
        /// <exception cref="NotImplementedException"></exception>
        public (Tensor, Tensor) Encode(List<string> inputs, bool add_special_tokens=true, bool truncation = false, int max_length = 512, string padding_side = "right")
        {
            if (!IsReady)
            {
                throw new ArgumentException("Tokenizer loaded asynchronously and not yet initialized. Check 'tokenizer.IsInitialized' before using the tokenizer.");
            }
            Tensor[] input_ids = new Tensor[inputs.Count];
            Tensor[] attn_masks = new Tensor[inputs.Count];

            max_length = -1;
            Parallel.For(0, inputs.Count, b =>
            {
                var enc = Encode(inputs[b], add_special_tokens, truncation, max_length);
                input_ids[b] = enc.Item1;
                attn_masks[b] = enc.Item2;
                max_length = Math.Max(max_length, enc.Item1.Size(-1));
            });

            Tensor input_ids_tensor = Tensor.Fill(value:PAD_TOKEN_ID, inputs.Count, max_length);
            Tensor attention_masks_tensor = Tensor.Zeros(inputs.Count, max_length);

            for (int b = 0; b < inputs.Count; b++)
            {
                int elem_length = input_ids[b].Size(-1);
                if (padding_side == "right")
                {
                    for (int l = 0; l < elem_length; l++)
                    {
                        input_ids_tensor[b, l] = input_ids[b][l];
                        attention_masks_tensor[b, l] = attn_masks[b][l];     
                    }
                }
                else if (padding_side == "left")
                {
                    int left_pad = max_length - elem_length;
                    for (int i = 0; i < max_length; i++)
                    {
                        input_ids_tensor[b, i + left_pad] = input_ids[b][i];
                        attention_masks_tensor[b, i + left_pad] = attn_masks[b][i];
                    }
                }    
                else
                    throw new NotImplementedException();
               
            }


            return (input_ids_tensor, attention_masks_tensor);
        }

        public List<string> Decode(Tensor input_ids)
        {
            if (input_ids.Rank == 1 || input_ids.Rank == 0)
            {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < input_ids.Size(-1); i++)
                {
                    if (id2token.ContainsKey(i))
                        sb.Append(id2token[(int)input_ids[i]]);     
                    else
                        sb.Append("<UNK>");
                }
                return new List<string> { sb.Replace("Ġ", " ").ToString() };
            }
            else if (input_ids.Rank == 2)
            {
                string[] outp = new string[input_ids.Size(-2)];
                Parallel.For(0, input_ids.Size(-2), b =>
                {
                    StringBuilder sb = new StringBuilder();
                    for (int l = 0; l < input_ids.Size(-1); l++)
                    {
                        if (id2token.ContainsKey(l))
                            sb.Append(id2token[(int)input_ids[b, l]]);
                        else
                            sb.Append("<UNK>");
                    }
                    outp[b] = sb.Replace("Ġ", " ").ToString();
                });
                return outp.ToList();
            }
            else
                throw new ArgumentException($"Decoding works only for tensors of shape (L) or (B, L) - received ({input_ids.Shape.ToCommaSeparatedString()})");
        }

        public void AddToken(string token)
        {
            if (!token2id.ContainsKey(token))
            {
                int idx = token2id.Count;
                token2id[token] = idx;
                id2token[idx] = token;

                TrieNode root = token2id_trie;

                foreach (char ch in token)
                {
                    if (!root.Next.TryGetValue(ch, out TrieNode nxt))
                        root.Next[ch] = nxt = new TrieNode();
                    root = nxt;
                }
                root.TokenId = idx;

            }
        }

        public virtual string ApplyChatTemplate(List<Dictionary<string, string>> input, bool add_generation_prompt = true)
        {
            throw new NotImplementedException("The current tokenizer does not implement the ApplyChatTemplate function.");
        }
        public List<string> ApplyChatTemplate(List<List<Dictionary<string, string>>> input, bool add_generation_prompt = true)
        {
            var output = new List<string>();

            foreach (var item in input)
            {
                output.Add(ApplyChatTemplate(item, add_generation_prompt: add_generation_prompt));
            }

            return output;
        }

    }
}

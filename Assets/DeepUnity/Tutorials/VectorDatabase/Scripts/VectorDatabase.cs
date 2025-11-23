using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;


namespace DeepUnity.Tutorials
{
    
    public class VectorDatabase : MonoBehaviour
    {
        [SerializeField] private ScrollView scrollView;
        [SerializeField] private GameObject docCard;
        [SerializeField] private List<TextAsset> documents;
        private List<VectorDatabaseNode> nodes;
        private Gemma3ForEmbeddings embedding_gemma;
        [Button("Retrieve")]
        [SerializeField] string query;
        [SerializeField] int top_k;


        private void Awake()
        {
            documents = documents.Where(x => x != null).ToList();
            StartCoroutine(Vectorize());
        }

        private void Start()
        {
            BuildNodes();
        }

        private IEnumerator Vectorize()
        {
# if UNITY_EDITOR
            /// Create the text files
            foreach (var document in documents)
            {
                // try to get embedding
                var emb_text_asset = Resources.Load<TextAsset>($"Cache/{document.name}_embedding");

                if (emb_text_asset == null)
                {
                    if (embedding_gemma == null)
                        embedding_gemma = new Gemma3ForEmbeddings();

                    Debug.Log($"Vectorizing {document.name}...");
                    yield return StartCoroutine(embedding_gemma.EncodeQuery(prompt: document.text, onEmbeddingReceived:
                        (x) =>
                        {
                            
                            File.WriteAllText($"Assets/Resources/Cache/{document.name}_embedding.txt", x.ToArray().ToCommaSeparatedString());
                            Debug.Log($"Document {document.name} was vectorized at path Assets/Resources/Cache/{document.name}_embedding.txt.");
                        }));
                    
                }   
            }
            AssetDatabase.Refresh();
#endif
        }
    
    
        private void BuildNodes()
        {
            nodes = new List<VectorDatabaseNode>();
            foreach (var doc in documents)
            {
                var emb_str = Resources.Load<TextAsset>($"Cache/{doc.name}_embedding").text;
                float[] emb_values = emb_str.Split(", ").Select(x => float.Parse(x)).ToArray();
                nodes.Add(new VectorDatabaseNode(doc.text, Tensor.Constant(emb_values)));

                Debug.Log(nodes.Last().Embedding);
            }

        }

        public void Retrieve()
        {
            StartCoroutine(Retrieve(query, top_k));
        }
        private IEnumerator Retrieve(string query, int top_k = 5)
        {
            if(embedding_gemma == null)
            {
                embedding_gemma = new Gemma3ForEmbeddings();
            }

            yield return embedding_gemma.EncodeQuery(query, onEmbeddingReceived:
            emb => {

                float[] similarities = nodes.Select(x => Tensor.CosineSimilarity(x.Embedding, emb)[0]).ToArray();
                List<(int, float)> indexed_similarities = new();
                for (int i = 0; i < similarities.Length; i++)
                {
                    indexed_similarities.Add((i,  similarities[i]));
                }

                List<VectorDatabaseNode> top_k_nodes = new();

                scrollView.Clear();

                
                
            });

            

        }
    }
    

}

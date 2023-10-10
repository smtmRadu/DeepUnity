using DeepUnity;
using UnityEngine;

namespace kbRadu
{
    public class MazeEnvironment : MonoBehaviour
    {
        [SerializeField] Agent agent;

        [SerializeField] GameObject spacePrefab;
        [SerializeField] GameObject wallPrefab;
        [SerializeField] GameObject coinPrefab;
        [SerializeField] GameObject agentPrefab;

        [Space]
        [Button("GenerateMaze")]
        [SerializeField] private int width = 30;
        [SerializeField] private int height = 20;
        [SerializeField] private float wallsPercentage = 0.2f;
        [SerializeField] private int coinsNumber = 3;

        private int coins_collected = 0;

        Vector2Int agent_position;
        
     
       
        GameObject[,] matrix;

        [Button("MoveUp")]    public int a;
        [Button("MoveDown")]  public int b;
        [Button("MoveLeft")]  public int c;
        [Button("MoveRight")] public int d;
        


        // Initialization
        public void GenerateMaze()
        {
            
            if (matrix != null)
            {
                foreach (var item in matrix)
                {
                    Destroy(item);
                }

            }
            matrix = new GameObject[height, width];
            FillEmpty();
            FillOuterWalls();
            FillInnerWalls();
            AddCoins();
            AddAgent();

            coins_collected = 0;
        }
        private void FillEmpty()
        {
            for (int j = 0; j < height; j++)
            {
                for (int i = 0; i < width; i++)
                {
                    matrix[j,i] = Instantiate(spacePrefab, (Vector2)transform.position + new Vector2(i,j), Quaternion.identity);
                }
            }
        }
        private void FillOuterWalls()
        {
            for (int j = 0; j < height; j++)
            {
                for (int i = 0; i < width; i++)
                {
                    if(i == 0 || j == 0 || (i == width - 1) || (j == height - 1))
                    {
                        Destroy(matrix[j, i]);
                        matrix[j, i] = Instantiate(wallPrefab, (Vector2)transform.position + new Vector2(i, j), Quaternion.identity);
                    }
                    
                }
            }
        }
        private void FillInnerWalls()
        {
            for (int j = 0; j < height; j++)
            {
                for (int i = 0; i < width; i++)
                {
                    if(Random.value < wallsPercentage)
                    {
                        Destroy(matrix[j, i]);
                        matrix[j, i] = Instantiate(wallPrefab, (Vector2)transform.position + new Vector2(i, j), Quaternion.identity);
                    }
                }
            }
            // this part may be unsafe.
            //return;
            /*bool placedOne = true;
            while(placedOne)
            {
                placedOne = false;
                // Cover the locked spots
                for (int j = 0; j < height; j++)
                {
                    for (int i = 0; i < width; i++)
                    {
                        int neighbourWalls = 0;
                        try
                        {
                            if (matrix[j - 1, i - 1].CompareTag("Wall"))
                                neighbourWalls++;
                        }
                        catch { }
                        try
                        {
                            if (matrix[j - 1, i + 1].CompareTag("Wall"))
                                neighbourWalls++;
                        }
                        catch { }
                        try
                        {

                            if (matrix[j + 1, i + 1].CompareTag("Wall"))
                                neighbourWalls++;
                        }
                        catch { }
                        try
                        {
                            if (matrix[j + 1, i - 1].CompareTag("Wall"))
                                neighbourWalls++;


                        }
                        catch { }
                       
                        
                 
                        if(neighbourWalls > 2)
                        {
                            Destroy(matrix[j, i]);
                            matrix[j, i] = Instantiate(wallPrefab, (Vector2)transform.position + new Vector2(i, j), Quaternion.identity);
                            placedOne = true;
                        }

                    }
                }
            }*/
            
        }
        private void AddCoins()
        {
            for (int i = 0; i < coinsNumber; i++)
            {
                int randx = Random.Range(1, width - 1);
                int randy = Random.Range(1, height - 1);

                Destroy(matrix[randy, randx]);
                matrix[randy, randx] = Instantiate(coinPrefab, (Vector2)transform.position + new Vector2(randx, randy), Quaternion.identity);
            }
        }
        private void AddAgent()
        {
            int randx = Random.Range(1, width - 1);
            int randy = Random.Range(1, height - 1);

            Destroy(matrix[randy, randx]);
            matrix[randy, randx] = Instantiate(agentPrefab, (Vector2)transform.position + new Vector2(randx, randy), Quaternion.identity);
            agent_position = new Vector2Int(randx, randy);
        }


        public void MoveUp()
        {
            if (matrix[agent_position.y + 1, agent_position.x].CompareTag("Wall"))
            {
                // Do nothing
            }

            else if (matrix[agent_position.y + 1, agent_position.x].CompareTag("Empty"))
            {
                // Replace
                Destroy(matrix[agent_position.y, agent_position.x]);
                Destroy(matrix[agent_position.y + 1, agent_position.x]);


                matrix[agent_position.y, agent_position.x] = Instantiate(spacePrefab, (Vector2)transform.position + new Vector2(agent_position.x, agent_position.y), Quaternion.identity);
                matrix[agent_position.y + 1, agent_position.x] =  Instantiate(agentPrefab, (Vector2)transform.position + new Vector2(agent_position.x, agent_position.y + 1f), Quaternion.identity);

                agent_position = new Vector2Int(agent_position.x, agent_position.y + 1);
            }
            else if (matrix[agent_position.y + 1, agent_position.x].CompareTag("Coin"))
            {
                // Replace
                Destroy(matrix[agent_position.y, agent_position.x]);
                Destroy(matrix[agent_position.y + 1, agent_position.x]);


                matrix[agent_position.y, agent_position.x] = Instantiate(spacePrefab, (Vector2)transform.position + new Vector2(agent_position.x, agent_position.y), Quaternion.identity);
                matrix[agent_position.y + 1, agent_position.x] = Instantiate(agentPrefab, (Vector2)transform.position + new Vector2(agent_position.x, agent_position.y + 1f), Quaternion.identity);

                agent_position = new Vector2Int(agent_position.x, agent_position.y + 1);



                agent?.AddReward(1f);
                coins_collected++;
                if (coins_collected == coinsNumber)
                    agent.EndEpisode();
            }

        }
        public void MoveDown()
        {
            if (matrix[agent_position.y - 1, agent_position.x].CompareTag("Wall"))
            {
                // Do nothing
            }

            else if (matrix[agent_position.y - 1, agent_position.x].CompareTag("Empty"))
            {
                // Replace
                Destroy(matrix[agent_position.y, agent_position.x]);
                Destroy(matrix[agent_position.y - 1, agent_position.x]);


                matrix[agent_position.y, agent_position.x] = Instantiate(spacePrefab, (Vector2)transform.position + new Vector2(agent_position.x, agent_position.y), Quaternion.identity);
                matrix[agent_position.y - 1, agent_position.x] = Instantiate(agentPrefab, (Vector2)transform.position + new Vector2(agent_position.x, agent_position.y - 1f), Quaternion.identity);

                agent_position = new Vector2Int(agent_position.x, agent_position.y - 1);
            }
            else if (matrix[agent_position.y - 1, agent_position.x].CompareTag("Coin"))
            {
                // Replace
                Destroy(matrix[agent_position.y, agent_position.x]);
                Destroy(matrix[agent_position.y - 1, agent_position.x]);


                matrix[agent_position.y, agent_position.x] = Instantiate(spacePrefab, (Vector2)transform.position + new Vector2(agent_position.x, agent_position.y), Quaternion.identity);
                matrix[agent_position.y - 1, agent_position.x] = Instantiate(agentPrefab, (Vector2)transform.position + new Vector2(agent_position.x, agent_position.y - 1f), Quaternion.identity);

                agent_position = new Vector2Int(agent_position.x, agent_position.y - 1);

                agent?.AddReward(1f);
                coins_collected++;
                if (coins_collected == coinsNumber)
                    agent.EndEpisode();
            }
        }
        public void MoveRight()
        {
            if (matrix[agent_position.y, agent_position.x + 1].CompareTag("Wall"))
            {
                // Do nothing
            }

            else if (matrix[agent_position.y, agent_position.x + 1].CompareTag("Empty"))
            {
                // Replace
                Destroy(matrix[agent_position.y, agent_position.x]);
                Destroy(matrix[agent_position.y, agent_position.x + 1]);


                matrix[agent_position.y, agent_position.x] = Instantiate(spacePrefab, (Vector2)transform.position + new Vector2(agent_position.x, agent_position.y), Quaternion.identity);
                matrix[agent_position.y, agent_position.x + 1] = Instantiate(agentPrefab, (Vector2)transform.position + new Vector2(agent_position.x + 1, agent_position.y), Quaternion.identity);

                agent_position = new Vector2Int(agent_position.x + 1, agent_position.y);
            }
            else if (matrix[agent_position.y , agent_position.x].CompareTag("Coin"))
            {
                // Replace
                Destroy(matrix[agent_position.y, agent_position.x]);
                Destroy(matrix[agent_position.y, agent_position.x + 1]);


                matrix[agent_position.y, agent_position.x] = Instantiate(spacePrefab, (Vector2)transform.position + new Vector2(agent_position.x, agent_position.y), Quaternion.identity);
                matrix[agent_position.y, agent_position.x + 1] = Instantiate(agentPrefab, (Vector2)transform.position + new Vector2(agent_position.x + 1, agent_position.y), Quaternion.identity);

                agent_position = new Vector2Int(agent_position.x + 1, agent_position.y);

                agent?.AddReward(1f);
                coins_collected++;
                if (coins_collected == coinsNumber)
                    agent.EndEpisode();
            }
        }
        public void MoveLeft()
        {
            if (matrix[agent_position.y, agent_position.x - 1].CompareTag("Wall"))
            {
                // Do nothing
            }

            else if (matrix[agent_position.y, agent_position.x - 1].CompareTag("Empty"))
            {
                // Replace
                Destroy(matrix[agent_position.y, agent_position.x]);
                Destroy(matrix[agent_position.y, agent_position.x - 1]);


                matrix[agent_position.y, agent_position.x] = Instantiate(spacePrefab, (Vector2)transform.position + new Vector2(agent_position.x, agent_position.y), Quaternion.identity);
                matrix[agent_position.y, agent_position.x - 1] = Instantiate(agentPrefab, (Vector2)transform.position + new Vector2(agent_position.x - 1, agent_position.y), Quaternion.identity);

                agent_position = new Vector2Int(agent_position.x - 1, agent_position.y);
            }
            else if (matrix[agent_position.y, agent_position.x].CompareTag("Coin"))
            {
                // Replace
                Destroy(matrix[agent_position.y, agent_position.x]);
                Destroy(matrix[agent_position.y, agent_position.x - 1]);


                matrix[agent_position.y, agent_position.x] = Instantiate(spacePrefab, (Vector2)transform.position + new Vector2(agent_position.x, agent_position.y), Quaternion.identity);
                matrix[agent_position.y, agent_position.x - 1] = Instantiate(agentPrefab, (Vector2)transform.position + new Vector2(agent_position.x - 1, agent_position.y), Quaternion.identity);

                agent_position = new Vector2Int(agent_position.x - 1, agent_position.y);

                agent?.AddReward(1f);
                coins_collected++;
                if (coins_collected == coinsNumber)
                    agent.EndEpisode();
            }
        }


        // Loop
        private void StepEnvironment()
        {

            // Utils.DebugInFile(GetState().ToString());
            // print(GetState());
            // agent.RequestAction();
        }

        /// <summary>
        /// Returns a Tensor(4, height, width).
        /// First channel is for free spaces.
        /// Second channel is for walls.
        /// Third channel is for coins.
        /// Fourth channel is for agent.
        /// </summary>
        /// <returns></returns>
        public Tensor GetState()
        {
            Tensor mapToTensor = Tensor.Zeros(4, height, width);
            for (int j = 0; j < height; j++)
            {
                for (int i = 0; i < width; i++)
                {
                    int invertedJ = height - 1 - j; // Calculate the inverted row index

                    if (matrix[j, i].CompareTag("Empty"))
                        mapToTensor[0, invertedJ, i] = 1;
                    else if (matrix[j, i].CompareTag("Wall"))
                        mapToTensor[1, invertedJ, i] = 1;
                    else if (matrix[j, i].CompareTag("Coin"))
                        mapToTensor[2, invertedJ, i] = 1;
                    else if (matrix[j, i].CompareTag("Agent"))
                        mapToTensor[3, invertedJ, i] = 1;
                }
            }
            return mapToTensor;
        }


        public void MoveAgent(ActionBuffer actions)
        {
            // consider having 4 continuous actions
            // we should actually have 4 discrete actions.

        }
    }
}


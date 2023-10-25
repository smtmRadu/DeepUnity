using DeepUnity;
using kbRadu;

namespace DeepUnityTutorials
{

    // This agent uses a convolutional neural network and works in discrete action space.
    public class CoinCollector : Agent
    {
        public MazeEnvironment env;

        public override void OnEpisodeBegin()
        {
            env.GenerateMaze();
        }
        public override void CollectObservations(StateBuffer sensorBuffer)
        {
            // sensorBuffer.TimestepObservation = env.GetState();
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            switch(actionBuffer.DiscreteAction)
            {
                case 0:
                    env.MoveUp();
                    break;
                case 1:
                    env.MoveDown();
                    break;
                case 2:
                    env.MoveLeft();
                    break;
                case 3:
                    env.MoveRight();
                    break;
            }
        }
    }

}



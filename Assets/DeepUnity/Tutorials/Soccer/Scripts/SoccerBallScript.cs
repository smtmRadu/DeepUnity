using UnityEngine;

namespace DeepUnityTutorials
{
    public class SoccerBallScript : MonoBehaviour
    {
        private SoccerEnvironmentScript env;

        private void Awake()
        {
            env = transform.parent.GetComponent<SoccerEnvironmentScript>();
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.name == "PinkGoal")
                env.BlueTeamScored();
            else if (other.name == "BlueGoal")
                env.PinkTeamScored();
            else if (other.name == "OutOfBounds")
                env.StartNewRound(true);
        }
    }
}




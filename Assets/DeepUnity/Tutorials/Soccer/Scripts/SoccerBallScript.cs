using UnityEngine;

namespace DeepUnity.Tutorials
{
    public class SoccerBallScript : MonoBehaviour
    {
        private SoccerEnvironmentScript env;

        private void Awake()
        {
            env = transform.parent.GetComponent<SoccerEnvironmentScript>();
        }

        private void OnCollisionEnter(Collision collision)
        {
            Collider other = collision.collider;
            if (other.CompareTag("PinkGoal"))
                env.BlueTeamScored();
            else if (other.CompareTag("BlueGoal"))
                env.PinkTeamScored();
            else if (other.name == "OutOfBounds")
                env.StartNewRound(true);
        }
    }
}




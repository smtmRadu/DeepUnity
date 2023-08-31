using DeepUnity;
using UnityEngine;

public class CheetahSensitivePart : MonoBehaviour
{
	[SerializeField] Agent agent;

    private void OnCollisionEnter2D(Collision2D collision)
    {
        agent.AddReward(-1f);
        agent.EndEpisode();
    }
}



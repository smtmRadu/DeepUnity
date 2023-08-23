using DeepUnity;
using UnityEngine;

public class BoxScript : MonoBehaviour
{
	public Agent partnerAgent;

    private void OnTriggerEnter(Collider other)
    {
        partnerAgent.AddReward(1f);
        partnerAgent.EndEpisode();
    }
}



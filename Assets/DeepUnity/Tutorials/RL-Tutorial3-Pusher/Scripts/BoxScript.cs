using DeepUnity;
using UnityEngine;

public class BoxScript : MonoBehaviour
{
	public Agent partnerAgent;

    private void OnTriggerEnter(Collider other)
    {
        if(other.CompareTag("Goal"))
            partnerAgent.AddReward(1f);
        else if(other.CompareTag("Wall"))
            partnerAgent.AddReward(-1f);

        partnerAgent.EndEpisode();
    }
}



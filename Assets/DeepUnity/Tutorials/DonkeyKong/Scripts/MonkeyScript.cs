using DeepUnity;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnityTutorials
{
    public class MonkeyScript : MonoBehaviour
    {
        public GameObject barrelPrefab;
        public Transform barrelSpawnPoint;
        private Animator animator;
        [Header("Recommended: 1.5f - 3.5f (to make the game _hard_)")]
        public Vector2 throwOnSecondsRange = new Vector2(2.5f, 3.5f);
        public float timeElapsedUntilNextThrow;
        public static LinkedList<GameObject> barrels = new();


        private void Awake()
        {
            timeElapsedUntilNextThrow = Utils.Random.Range(throwOnSecondsRange.x, throwOnSecondsRange.y);
            animator = GetComponent<Animator>();
        }



        private void FixedUpdate()
        {
            timeElapsedUntilNextThrow -= Time.fixedDeltaTime;

            if(timeElapsedUntilNextThrow <= 0f)
            {
                StartCoroutine("ThrowBarrel");
            }
        }

        IEnumerator ThrowBarrel()
        {
            animator.SetBool("isThrowingBarrel", true);
            timeElapsedUntilNextThrow = float.MaxValue;

            yield return new WaitForSeconds(1f);
            GameObject barrel = Instantiate(barrelPrefab, barrelSpawnPoint, true);
            barrels.AddLast(barrel);
            yield return new WaitForSeconds(0.5f);


            animator.SetBool("isThrowingBarrel", false);
            timeElapsedUntilNextThrow = Utils.Random.Range(throwOnSecondsRange.x, throwOnSecondsRange.y);

        }

    }

}



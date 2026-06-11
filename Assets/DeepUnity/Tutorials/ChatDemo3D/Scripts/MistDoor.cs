using System.Collections;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Souls-style fog wall sealing the boss room arch. The wall blocks passage; walking up
    /// to it shows a prompt, and pressing E plays the traversal sequence on the player
    /// (reach into the mist, push through) with a soft white flash as they cross the plane.
    /// Works from both sides, so the boss room can also be left the same way.
    /// </summary>
    public class MistDoor : MonoBehaviour
    {
        [SerializeField, ViewOnly] private SoulsPlayerController player;
        [SerializeField] private GameObject prompt;           // screen-space "Traverse the mist — [ E ]" box
        [SerializeField] private Image whiteFlash;            // full-screen image, alpha 0 when idle
        [SerializeField] private Renderer[] mistLayers;       // scrolling fog quads in the archway
        [SerializeField] private Collider blocker;            // solid collider, disabled during traversal
        [SerializeField] private BossController boss;         // woken when the player steps through
        [SerializeField] private float scrollSpeed = 0.05f;

        private Material[] mats;
        private Color[] baseTints;
        private bool playerNear, traversing;

        private void Start()
        {
            mats = new Material[mistLayers.Length];
            baseTints = new Color[mistLayers.Length];
            for (int i = 0; i < mistLayers.Length; i++)
            {
                mats[i] = mistLayers[i].material;   // instance: each layer scrolls independently
                baseTints[i] = mats[i].GetColor("_TintColor");
            }
            if (prompt != null) prompt.SetActive(false);
        }

        private void Update()
        {
            // opposing drift on the two layers + a slow breathing pulse sells the fog
            for (int i = 0; i < mats.Length; i++)
            {
                float dir = i % 2 == 0 ? 1f : -0.62f;
                mats[i].mainTextureOffset = new Vector2(Time.time * scrollSpeed * dir, 0f);
                // shallow pulse only — the wall must stay opaque, the life comes from the scroll
                Color c = baseTints[i];
                c.a *= 0.94f + 0.06f * Mathf.Sin(Time.time * 0.9f + i * 1.7f);
                mats[i].SetColor("_TintColor", c);
            }

            if (playerNear && !traversing && player != null && player.IsExploring && Input.GetKeyDown(KeyCode.E))
                StartCoroutine(Traverse());
        }

        private IEnumerator Traverse()
        {
            traversing = true;
            if (prompt != null) prompt.SetActive(false);
            blocker.enabled = false;

            // entry/exit points along the door normal, on whichever side the player stands
            Vector3 n = transform.forward;
            float side = Mathf.Sign(Vector3.Dot(player.transform.position - transform.position, n));
            Vector3 doorPoint = transform.position + n * (side * 1.15f);
            Vector3 exitPoint = transform.position - n * (side * 2.3f);

            StartCoroutine(FlashWhenCrossing(n, side));
            yield return player.TraverseMist(doorPoint, exitPoint);

            blocker.enabled = true;
            traversing = false;
            if (playerNear && prompt != null) prompt.SetActive(true);

            // stepping INTO the chamber (the door's +forward side) wakes the Sentinel
            if (boss != null && Vector3.Dot(player.transform.position - transform.position, transform.forward) > 0f)
                boss.Activate();
        }

        private IEnumerator FlashWhenCrossing(Vector3 n, float startSide)
        {
            while (traversing && Mathf.Sign(Vector3.Dot(player.transform.position - transform.position, n)) == startSide)
                yield return null;
            if (whiteFlash == null) yield break;

            const float duration = 0.9f;
            for (float t = 0f; t < duration; t += Time.deltaTime)
            {
                float a = Mathf.Sin(Mathf.Clamp01(t / duration) * Mathf.PI);
                whiteFlash.color = new Color(0.91f, 0.94f, 1f, a * 0.5f);
                yield return null;
            }
            whiteFlash.color = Color.clear;
        }

        private void OnTriggerEnter(Collider other)
        {
            if (!other.CompareTag("Player")) return;
            player = other.GetComponent<SoulsPlayerController>();
            playerNear = true;
            if (prompt != null && !traversing) prompt.SetActive(true);
        }

        private void OnTriggerExit(Collider other)
        {
            if (!other.CompareTag("Player")) return;
            playerNear = false;
            if (prompt != null) prompt.SetActive(false);
        }
    }
}

using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Souls-style full-screen verdicts: the dark red "YOU DIED" on player death and the
    /// golden "SENTINEL FELLED" when the boss goes down. Fade in, hold, fade out.
    /// </summary>
    public class DeathScreen : MonoBehaviour
    {
        [SerializeField] private Image dim;
        [SerializeField] private TMP_Text deathText;
        [SerializeField] private TMP_Text victoryText;
        [SerializeField] private AudioSource audioSource;
        [SerializeField] private AudioClip deathClip;   // the YOU DIED sting

        public void ShowDeath()
        {
            if (audioSource != null && deathClip != null)
                audioSource.PlayOneShot(deathClip, 0.85f);
            Show(deathText, 0.62f, 3.6f);
        }

        public void ShowVictory() => Show(victoryText, 0.35f, 4.2f);

        private void Show(TMP_Text text, float dimAlpha, float hold)
        {
            StopAllCoroutines();
            StartCoroutine(Run(text, dimAlpha, hold));
        }

        private IEnumerator Run(TMP_Text text, float dimAlpha, float hold)
        {
            if (deathText != null) deathText.gameObject.SetActive(false);
            if (victoryText != null) victoryText.gameObject.SetActive(false);
            text.gameObject.SetActive(true);

            Color textColor = text.color;
            for (float t = 0f; t < 1.1f; t += Time.deltaTime)
            {
                float a = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(t / 1.1f));
                SetAlpha(a, dimAlpha, text, textColor);
                yield return null;
            }
            yield return new WaitForSeconds(hold);
            for (float t = 0f; t < 1.4f; t += Time.deltaTime)
            {
                float a = 1f - Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(t / 1.4f));
                SetAlpha(a, dimAlpha, text, textColor);
                yield return null;
            }
            text.gameObject.SetActive(false);
        }

        private void SetAlpha(float a, float dimAlpha, TMP_Text text, Color baseColor)
        {
            if (dim != null) dim.color = new Color(0f, 0f, 0f, a * dimAlpha);
            text.color = new Color(baseColor.r, baseColor.g, baseColor.b, a);
        }
    }
}

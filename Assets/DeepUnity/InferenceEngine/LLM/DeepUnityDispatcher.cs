using System.Collections;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Minimal persistent MonoBehaviour for running main-thread coroutines from non-MonoBehaviour
    /// code (e.g. frame-budgeted GPU weight uploads at model boot). Created lazily on first use and
    /// kept alive across scene loads. Hidden from the hierarchy.
    /// </summary>
    internal sealed class DeepUnityDispatcher : MonoBehaviour
    {
        static DeepUnityDispatcher _instance;

        public static void Run(IEnumerator routine)
        {
            if (_instance == null)
            {
                var go = new GameObject("[DeepUnityDispatcher]");
                go.hideFlags = HideFlags.HideAndDontSave;
                DontDestroyOnLoad(go);
                _instance = go.AddComponent<DeepUnityDispatcher>();
            }
            _instance.StartCoroutine(routine);
        }
    }
}

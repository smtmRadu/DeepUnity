using System;
using System.Collections;
using System.Collections.Concurrent;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Minimal persistent MonoBehaviour for running main-thread coroutines from non-MonoBehaviour
    /// code (e.g. frame-budgeted GPU weight uploads at model boot). Created lazily on first use and
    /// kept alive across scene loads. Hidden from the hierarchy.
    /// Also pumps a cross-thread action queue: Unity graphics objects are MAIN-THREAD-only, so GC
    /// finalizers must marshal their ComputeBuffer.Release calls here instead of touching the GPU
    /// from the finalizer thread (editor: warning; player build: native crash in DestroyBuffer).
    /// </summary>
    internal sealed class DeepUnityDispatcher : MonoBehaviour
    {
        static DeepUnityDispatcher _instance;
        static readonly ConcurrentQueue<Action> mainThreadQueue = new ConcurrentQueue<Action>();

        public static void Run(IEnumerator routine)
        {
            EnsureExists();
            _instance.StartCoroutine(routine);
        }

        /// <summary>Queue an action for the next Update on the MAIN thread. Safe from ANY thread,
        /// including GC finalizers. Actions enqueued during app shutdown simply never run —
        /// that's the point (Unity reclaims the native memory itself on exit).</summary>
        public static void RunOnMainThread(Action action) => mainThreadQueue.Enqueue(action);

        /// <summary>Create the hidden pump object now (MAIN-THREAD only). Model constructors call
        /// this so a later cross-thread RunOnMainThread always has an Update to drain it.</summary>
        public static void EnsureExists()
        {
            if (_instance != null) return;
            var go = new GameObject("[DeepUnityDispatcher]");
            go.hideFlags = HideFlags.HideAndDontSave;
            if (Application.isPlaying) DontDestroyOnLoad(go);   // edit-mode probes: illegal outside play mode
            _instance = go.AddComponent<DeepUnityDispatcher>();
        }

        void Update()
        {
            while (mainThreadQueue.TryDequeue(out Action a))
            {
                try { a(); }
                catch (Exception e) { Debug.LogException(e); }
            }
        }
    }
}

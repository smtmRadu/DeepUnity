using UnityEngine;
using System.Reflection;
using System;

namespace DeepUnity
{
    internal static class ConsoleMessage
    {
        /// <summary>
        /// Display some event in the Debug Console.
        /// </summary>
        /// <param name="message"></param>
        public static void Info(string message)
        {
            Debug.Log($"<color=#03a9fc><b>[INFO] [{DateTime.Now}]</b> {message}.</color>");
        }
        /// <summary>
        /// Display some unexpected bugs in the Debug Console that should be repaired.
        /// </summary>
        /// <param name="message"></param>
        public static void Warning(string message)
        {
            Debug.Log($"<color=#fcba03><b>[WARNING] [{DateTime.Now}]</b> {message}.</color>");
        }
        /// <summary>
        /// Display errors in the Debug Console that must be repaired.
        /// </summary>
        /// <param name="message"></param>
        public static void Error(string message)
        {
            Debug.Log($"<color=red><b>[ERROR] [{DateTime.Now}]</b> {message}.</color>");
        }


        /// <summary>
        /// Clears all Debug Console messages (Editor-Only).
        /// </summary>
        public static void ClearLog()
        {
#if UNITY_EDITOR
            var assembly = Assembly.GetAssembly(typeof(UnityEditor.Editor));
            var type = assembly.GetType("UnityEditor.LogEntries");
            var method = type.GetMethod("Clear");
            method.Invoke(new object(), null);
#endif
        }
    }

}





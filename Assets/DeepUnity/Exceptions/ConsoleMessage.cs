using UnityEngine;

namespace DeepUnity
{
    public static class ConsoleMessage
    {
        /// <summary>
        /// Display some event in the Debug Console.
        /// </summary>
        /// <param name="message"></param>
        public static void Info(string message)
        {
            Debug.Log($"<color=#03a9fc><b>[INFO]</b> {message}.</color>");
        }
        /// <summary>
        /// Display some unexpected bugs in the Debug Console that should be repaired.
        /// </summary>
        /// <param name="message"></param>
        public static void Warning(string message)
        {
            Debug.Log($"<color=#fcba03><b>[WARNING]</b> {message}.</color>");
        }
        /// <summary>
        /// Display errors in the Debug Console that must be repaired.
        /// </summary>
        /// <param name="message"></param>
        public static void Error(string message)
        {
            Debug.Log($"<color=red><b>[ERROR]</b> {message}.</color>");
        }    
        
    }

}





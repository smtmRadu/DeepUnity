using UnityEngine;

namespace DeepUnity
{
    public static class ConsoleMessage
    {
        public static void Warning(string message)
        {
            Debug.Log($"<color=#fcba03><b>[WARNING]</b> {message}.</color>");
        }
        public static void Error(string message)
        {
            Debug.Log($"<color=red><b>[ERROR]</b> {message}.</color>");
        }     
        public static void Info(string message)
        {
            Debug.Log($"<color=#03a9fc><b>[INFO]</b> {message}.</color>");
        }
    }

}





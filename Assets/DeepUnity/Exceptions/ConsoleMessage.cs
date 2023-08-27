using UnityEngine;

namespace DeepUnity
{
    public static class ConsoleMessage
    {
        public static void Warning(string message)
        {
            Debug.Log($"<color=#fcba03>Warning! {message}.</color>");
        }
        public static void Error(string message)
        {
            Debug.Log($"<color=red>Error! {message}.</color>");
        }     
        public static void Info(string message)
        {
            Debug.Log($"<color=#03a9fc> {message}.</color>");
        }
    }

}





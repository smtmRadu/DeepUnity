using System.Collections.Generic;
using UnityEditor;

namespace DeepUnity
{
    [CustomEditor(typeof(Hyperparameters), true), CanEditMultipleObjects]
    class ScriptlessHP : Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            List<string> dontDrawMe = new List<string>() { "m_Script" };

            Hyperparameters script = (Hyperparameters)target;   

            if(script.trainer == TrainerType.PPO)
            {
                if (script.KLDivergence == (int)KLType.Off)
                    dontDrawMe.Add("targetKL");

                dontDrawMe.Add("updateEvery");
                dontDrawMe.Add("updateAfter");
                dontDrawMe.Add("updatesNum");
                dontDrawMe.Add("alpha");
                dontDrawMe.Add("saveReplayBuffer");
                dontDrawMe.Add("tau");


                dontDrawMe.Add("saveRecordBuffer");
            }
            else if(script.trainer == TrainerType.SAC)
            {
                dontDrawMe.Add("numEpoch");
                dontDrawMe.Add("beta");
                dontDrawMe.Add("epsilon");
                dontDrawMe.Add("lambda");
                dontDrawMe.Add("KLDivergence");
                dontDrawMe.Add("targetKL");

                dontDrawMe.Add("saveRecordBuffer");
            }
            else if(script.trainer == TrainerType.GAIL)
            {
                dontDrawMe.Add("updateEvery");
                dontDrawMe.Add("updateAfter");
                dontDrawMe.Add("updatesNum");
                dontDrawMe.Add("alpha");
                dontDrawMe.Add("saveReplayBuffer");
                dontDrawMe.Add("tau");


                dontDrawMe.Add("numEpoch");
                dontDrawMe.Add("beta");
                dontDrawMe.Add("epsilon");
                dontDrawMe.Add("lambda");
                dontDrawMe.Add("KLDivergence");
                dontDrawMe.Add("targetKL");
            }

            



            // if (EditorApplication.isPlaying)
            //     EditorGUILayout.HelpBox("Hyperparameters values can be modified at runtime. Config file has no effect but when the agent is learning.", MessageType.Info);
            // DO not modify the values at runtime (lr will not change, may appear bugs when changing the buffer size to a smaller size when is already filled).


            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
}

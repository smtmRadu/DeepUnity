using UnityEngine;

namespace kbRadu
{
    public class TestLogicInherited : TestLogic
    {
        protected override void Start()
        {
            base.Start();
            Debug.Log("INHERIT");
        }
    }
}


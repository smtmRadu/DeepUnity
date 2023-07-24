using UnityEngine;

namespace DeepUnity
{
    public interface IModel : ISerializationCallbackReceiver
    {
        /// <summary>
        /// Gets all <typeparamref name="Learnable"/> modules.
        /// </summary>
        /// <returns></returns>
        public Learnable[] Parameters();
        /// <summary>
        /// Save path: "Assets/". Creates/Overwrites model on the same path.
        /// For specific existing folder saving, <b><paramref name="name"/> = "folder_name/model_name"</b>
        /// </summary>
        public void Save(string name);

        public string Summary();
    }


}



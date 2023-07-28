using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor.XR;
using UnityEngine;

namespace DeepUnity
{
    public static class Datasets
    {
        /// <summary>
        /// Item1 = input: Tensor(1,28,28)<br />
        /// Item2 = target: Tensor(10) -> onehot encoding<br />
        /// Files used: <br></br>
        ///     "train_input.txt"<br />
        ///     "train_label.txt"<br />
        ///     "test_input.txt"<br />
        ///     "test_label.txt"<br />
        /// </summary>
        /// <param name="path">Example: C:\\Users\\Desktop</param>
        /// <param name="train"></param>
        /// <param name="test"></param>
        public static void MNIST(string path, out List<(Tensor, Tensor)> train, out List<(Tensor, Tensor)> test, DatasetSettings whatToLoad = DatasetSettings.LoadAll)
        {
            train = new();
            test = new();

            string json_train_image = null;
            string json_train_label = null;
            string json_test_image = null;
            string json_test_label = null;
            List<Tensor> collect_train_image = null;
            List<Tensor> collect_train_label = null;
            List<Tensor> collect_test_image = null;
            List<Tensor> collect_test_label = null;
            if (whatToLoad == DatasetSettings.LoadAll || whatToLoad == DatasetSettings.LoadTrainOnly) 
            {
                json_train_image = File.ReadAllText(path + "\\train_input.txt");
                json_train_label = File.ReadAllText(path + "\\train_target.txt");
                collect_train_image = JsonUtility.FromJson<TensorCollection>(json_train_image).ToList();
                collect_train_label = JsonUtility.FromJson<TensorCollection>(json_train_label).ToList();
                for (int i = 0; i < collect_train_image.Count; i++)
                {
                    train.Add((collect_train_image[i], collect_train_label[i]));
                }
            }
            if (whatToLoad == DatasetSettings.LoadAll || whatToLoad == DatasetSettings.LoadTestOnly)
            {
                json_test_image = File.ReadAllText(path + "\\test_input.txt");
                json_test_label = File.ReadAllText(path + "\\test_target.txt");
                collect_test_image = JsonUtility.FromJson<TensorCollection>(json_test_image).ToList();
                collect_test_label = JsonUtility.FromJson<TensorCollection>(json_test_label).ToList();
                for (int i = 0; i < collect_test_image.Count; i++)
                {
                    test.Add((collect_test_image[i], collect_test_label[i]));
                }
            }            
        }
        public static void SerializeMNIST()
        {
            TimerX.Start();
            string trainPath = "C:\\Users\\radup\\OneDrive\\Desktop\\TRAIN\\";
            string testPath = "C:\\Users\\radup\\OneDrive\\Desktop\\TEST\\";

           
            TensorCollection train_image = new();
            TensorCollection train_label = new();

            TensorCollection test_image = new();
            TensorCollection test_label = new();
        
            for (int i = 0; i < 10; i++)
            {
                string[] trainPaths = Directory.GetFiles(trainPath + i, "*.png", SearchOption.TopDirectoryOnly);
                string[] testPaths = Directory.GetFiles(testPath + i, "*.png", SearchOption.TopDirectoryOnly);

                foreach (var tp in trainPaths)
                {
                    Tensor image = Tensor.Constant(LoadTexture(tp), true);

                    float[] number = new float[10];
                    number[i] = 1;
                    Tensor label = Tensor.Constant(number);

                    train_image.Add(image);
                    train_label.Add(label);

                }
                foreach (var vp in testPaths)
                {
                    Tensor image = Tensor.Constant(LoadTexture(vp), true);


                    float[] number = new float[10];
                    number[i] = 1;
                    Tensor label = Tensor.Constant(number);

                    test_image.Add(image);
                    test_label.Add(label);
                }
            }

            string path_to_serialize = "C:\\Users\\radup\\OneDrive\\Desktop\\";

           
            string train_img = JsonUtility.ToJson(train_image);
            string train_lbl = JsonUtility.ToJson(train_label);
            string test_img = JsonUtility.ToJson(test_image);
            string test_lbl = JsonUtility.ToJson(test_label);

            FileStream train_i = File.Open(path_to_serialize + "train_input.txt", FileMode.OpenOrCreate);
            FileStream train_l = File.Open(path_to_serialize + "train_target.txt", FileMode.OpenOrCreate) ;
            FileStream test_i = File.Open(path_to_serialize + "test_input.txt", FileMode.OpenOrCreate);
            FileStream test_l = File.Open(path_to_serialize + "test_target.txt", FileMode.OpenOrCreate);

            
            StreamWriter train_1_sw = new StreamWriter (train_i);
            StreamWriter train_2_sw = new StreamWriter(train_l);
            StreamWriter test_1_sw = new StreamWriter(test_i);
            StreamWriter test_2_sw = new StreamWriter(test_l);

            train_1_sw.Write(train_img);
            train_2_sw.Write(train_lbl);
            test_1_sw.Write(test_img);
            test_2_sw.Write(test_lbl);

            train_1_sw.Flush();
            train_2_sw.Flush();
            test_1_sw .Flush();
            test_2_sw.Flush();


            train_i.Close();
            train_l.Close();
            test_i.Close();
            test_l.Close();

            Debug.Log("MNIST Serialized.");
            TimerX.Stop();
        }
        private static Texture2D LoadTexture(string filePath)
        {
            Texture2D tex = null;
            byte[] fileData;

            if (File.Exists(filePath))
            {
                fileData = File.ReadAllBytes(filePath);
                tex = new Texture2D(28, 28);
                tex.LoadImage(fileData);
            }
            return tex;
        }

        
    }

   
   
}


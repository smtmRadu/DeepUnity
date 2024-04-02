using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
            train = null;
            test = null;
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
                train = new(60000);
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
                test = new(10000);
                json_test_image = File.ReadAllText(path + "\\test_input.txt");
                json_test_label = File.ReadAllText(path + "\\test_target.txt");
                collect_test_image = JsonUtility.FromJson<TensorCollection>(json_test_image).ToList();
                collect_test_label = JsonUtility.FromJson<TensorCollection>(json_test_label).ToList();
                for (int i = 0; i < collect_test_image.Count; i++)
                {
                    test.Add((collect_test_image[i], collect_test_label[i]));
                }
            }

            json_train_image = null;
            json_train_label = null;
            json_test_image = null;
            json_test_label = null;
            collect_train_image?.Clear();
            collect_train_label?.Clear();
            collect_test_image?.Clear();
            collect_test_label?.Clear();
        }
        /// <summary>
        /// Item1 = input: Tensor(10)<br />
        /// Item2 = target: Tensor(2) -> onehot encoding<br /></summary>
        /// <param name="train"></param>
        public static void BinaryClassification(out List<(Tensor, Tensor)> train)
        {
            var csguid = UnityEditor.AssetDatabase.FindAssets("simple_classif_dataset")[0];
            var cspath = UnityEditor.AssetDatabase.GUIDToAssetPath(csguid);
            TextAsset file = UnityEditor.AssetDatabase.LoadAssetAtPath(cspath, typeof(TextAsset)) as TextAsset;

            string text = file.text;

            train = new();
            using (var reader = new StringReader(text))
            {
                // Read and skip the header
                string headerLine = reader.ReadLine();

                while (reader.Peek() != -1)
                {
                    string line = reader.ReadLine();
                    string[] fields = line.Split(',');

                    // Assuming your data is in the order of features followed by target_0 and target_1
                    float[] features = fields.Take(10).Select(float.Parse).ToArray();
                    float[] target = { float.Parse(fields[10]), float.Parse(fields[11]) };

                    train.Add((Tensor.Constant(features), Tensor.Constant(target)));
                }
            }
        }
        private static void SerializeMNIST()
        {
            BenchmarkClock.Start();
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
                    Tensor image = Tensor.Constant(GetTexturePixels(tp), (1,28,28));

                    Tensor label = Tensor.Zeros(10);
                    label[i] = 1;

                    train_image.Add(image);
                    train_label.Add(label);

                }
                foreach (var vp in testPaths)
                {
                    Tensor image = Tensor.Constant(GetTexturePixels(vp), (1, 28, 28));

                    Tensor label = Tensor.Zeros(10);
                    label[i] = 1;

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
            BenchmarkClock.Stop();
        }
        
        private static Color[] GetTexturePixels(string filePath)
        {
            if (File.Exists(filePath))
            {
                var fileData = File.ReadAllBytes(filePath);
                Texture2D tex = new Texture2D(28, 28);
                tex.LoadImage(fileData);
                Color[] pixels = tex.GetPixels();
                MonoBehaviour.Destroy(tex);
                return pixels;
               
            }
            else
            {
                throw new FileNotFoundException($"File at path {filePath} not found!");
            }
        }

        
        private static void SerializeFaces250()
        {
            BenchmarkClock.Start();
            string trainPath = "C:\\Users\\radup\\OneDrive\\Desktop\\faces\\";


            TensorCollection images = new();

            Directory.GetFiles(trainPath);

            string[] directories = Directory.GetDirectories(trainPath);

            foreach (var dir in directories)
            {
                string[] files = Directory.GetFiles(dir);

                foreach (var file in files)
                {
                    try
                    {
                        Tensor image = Tensor.Constant(GetTexturePixels(file), (3, 250, 250));
                        images.Add(image);
                    }
                    catch{ }
                }
            }

            string path_to_serialize = "C:\\Users\\radup\\OneDrive\\Desktop\\";


            string train_img = JsonUtility.ToJson(images);

            FileStream train = File.Open(path_to_serialize + "faces.txt", FileMode.OpenOrCreate);


            StreamWriter train_sw = new StreamWriter(train);

            train_sw.Write(train_img);
            train_sw.Flush();
            train.Close();

            Debug.Log($"Faces 250x250 serialized {images.Count}.");
            images.Clear();
            BenchmarkClock.Stop();
        }
    }

   
   
}


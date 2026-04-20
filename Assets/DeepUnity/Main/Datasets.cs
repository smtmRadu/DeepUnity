using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    public static class Datasets
    {
        
        /// <summary>
        /// Loads MNIST dataset from PNG image folders. Download from: https://www.kaggle.com/datasets/alexanderyyy/mnist-png <br />
        /// Item1 = input: Tensor(1,28,28)<br />
        /// Item2 = target: Tensor(10) -> onehot encoding<br />
        /// Expected folder structure: path/train/0-9/*.png and path/test/0-9/*.png <br />
        /// </summary>
        /// <param name="path">Root folder containing train/ and test/ subdirectories. Defaults to Desktop/mnist/mnist_png.</param>
        /// <param name="train"></param>
        /// <param name="test"></param>
        public static void MNIST(string path, out List<(Tensor, Tensor)> train, out List<(Tensor, Tensor)> test, DatasetSettings whatToLoad = DatasetSettings.LoadAll)
        {
            if (path == null)
                path = Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop), "mnist", "mnist_png");

            train = null;
            test = null;

            if (whatToLoad == DatasetSettings.LoadAll || whatToLoad == DatasetSettings.LoadTrainOnly)
            {
                train = new(60000);
                string trainPath = Path.Combine(path, "train");
                train.AddRange(LoadSplit(trainPath));
            }

            if (whatToLoad == DatasetSettings.LoadAll || whatToLoad == DatasetSettings.LoadTestOnly)
            {
                test = new(10000);
                string testPath = Path.Combine(path, "test");
                test.AddRange(LoadSplit(testPath));
            }

            System.GC.Collect();
        }
        private static List<(Tensor, Tensor)> LoadSplit(string splitPath)
        {
            // Phase 1: collect all file paths and labels
            var fileEntries = new List<(string path, int label)>();
            for (int i = 0; i < 10; i++)
            {
                string[] files = Directory.GetFiles(Path.Combine(splitPath, i.ToString()), "*.png", SearchOption.TopDirectoryOnly);
                foreach (var file in files)
                    fileEntries.Add((file, i));
            }

            // Phase 2: parallel read of file bytes from disk
            var bytesArray = new byte[fileEntries.Count][];
            Parallel.For(0, fileEntries.Count, i =>
            {
                bytesArray[i] = File.ReadAllBytes(fileEntries[i].path);
            });

            // Phase 3: sequential tensor creation (Texture2D requires main thread)
            var result = new List<(Tensor, Tensor)>(fileEntries.Count);
            for (int i = 0; i < fileEntries.Count; i++)
            {
                Texture2D tex = new Texture2D(28, 28);
                tex.LoadImage(bytesArray[i]);
                Color[] pixels = tex.GetPixels();
                Object.Destroy(tex);

                Tensor image = Tensor.Constant(pixels, (1, 28, 28));
                Tensor label = Tensor.Zeros(10);
                label[fileEntries[i].label] = 1;
                result.Add((image, label));

                bytesArray[i] = null;
            }

            return result;
        }

        private static void SerializeMNIST()
        {
            Benckmark.Start();
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
            Benckmark.Stop();
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
            Benckmark.Start();
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
            Benckmark.Stop();
        }
    }

   
   
}


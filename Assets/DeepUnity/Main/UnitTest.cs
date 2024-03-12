using UnityEngine;
using DeepUnity.Models;
using DeepUnity;
using System.Collections.Generic;
using UnityEngine.UI;
using DeepUnity.Modules;
using DeepUnity.Optimizers;
using DeepUnity.Activations;

namespace DeepUnityTutorials
{
    public class UnitTest : MonoBehaviour
    {
        public Device device = Device.CPU;
        public Sequential net;
        public GameObject canvas;

        private List<RawImage> displays;

        private void Start()
        {
            // Tensor input = Tensor.Random01(64, 3, 256, 256);
            // 
            // Conv2D conv = new Conv2D(3, 32, 3, Device.GPU);
            // 
            // BenchmarkClock.Start();
            // conv.Predict(input);
            // BenchmarkClock.Stop();

            int ic = 3, oc = 6, ih = 7, iw = 7;
            Conv2D conv = new Conv2D(ic, oc, 3, Device.CPU);
            
            Tensor input = Tensor.Arange(0, ic * ih * iw).Reshape(ic, ih, iw);
             input = Tensor.Random01(ic, ih, iw);

            print(input);
            var output = conv.Forward(input);
            print(output);
            var inputGrad = conv.Backward(output);
            print(inputGrad);
            print(conv.kernelsGrad);
            print(conv.biasesGrad);
            
            new Adam(conv.Parameters()).ZeroGrad();
            
            
            print("repeat");
            conv.Device = Device.CPU;
            
            print(input);
            output = conv.Forward(input);
            print(output);
            inputGrad = conv.Backward(output);
            print(inputGrad);
            print(conv.kernelsGrad);
            print(conv.biasesGrad);


            //  displays = new();
            //  for (int i = 0; i < canvas.transform.childCount; i++)
            //  {
            //      var child = canvas.transform.GetChild(i);
            //      var img = child.GetComponent<RawImage>();
            //      displays.Add(img);
            //  }
            //  List<(Tensor, Tensor)> tests;
            //  Datasets.MNIST("C:\\Users\\radup\\OneDrive\\Desktop", out _, out tests, DatasetSettings.LoadTestOnly);
            // 
            //  var layers = net.modules;
            // 
            // 
            //  // (C, H, W)
            // 
            //  Tensor input = Utils.Random.Sample(tests).Item1;
            //  displays[0].texture = new Texture2D(28, 28);
            //  Texture2D tex = displays[0].texture as Texture2D;
            //  
            //  tex.SetPixels(Utils.TensorToColorArray(input));
            //  tex.Apply();
            // 
            // 
            // 
            //  Tensor convolved = layers[0].Predict(input);
            //  convolved = layers[1].Predict(convolved);
            //  // convolved = layers[2].Predict(convolved);
            //  // convolved = layers[3].Predict(convolved);
            // 
            //  print(convolved);
            // 
            //  Tensor[] channels = Tensor.Split(convolved, 0, 1);
            // 
            // 
            // 
            //  for (int i = 1; i < 18; i++)
            //  {
            // 
            //      displays[i].texture = new Texture2D(13, 13);
            //      tex = displays[i].texture as Texture2D;
            //      tex.SetPixels(Utils.TensorToColorArray(channels[i]));
            //      tex.Apply();
            // 
            //  }


        }

    }

}



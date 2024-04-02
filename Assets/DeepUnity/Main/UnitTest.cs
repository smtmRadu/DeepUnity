using UnityEngine;
using DeepUnity.Models;
using DeepUnity;
using System.Collections.Generic;
using UnityEngine.UI;
using DeepUnity.Modules;
using DeepUnity.Activations;
using TMPro;
using System.Linq;
using System.Threading;

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
            int hidden = 16;
            net = new Sequential(
                         new Conv2D(1, hidden, 3),
                         new AvgPool2D(2),

                         new ResidualConnection.Fork(),
                         new Pad2D(1),
                         new Conv2D(hidden, hidden, 3),
                         new PReLU(),
                         new ResidualConnection.Join(),

                         new ResidualConnection.Fork(),
                         new Pad2D(1),
                         new Conv2D(hidden, hidden, 3),
                         new PReLU(),
                         new ResidualConnection.Join(),

                         new ResidualConnection.Fork(),
                         new Pad2D(1),
                         new Conv2D(hidden, hidden, 3),
                         new PReLU(),
                         new ResidualConnection.Join(),

                         new ResidualConnection.Fork(),
                         new Pad2D(1),
                         new Conv2D(hidden, hidden, 3),
                         new PReLU(),
                         new ResidualConnection.Join(),

                         new Flatten(),
                         new LazyDense(512),
                         new LayerNorm(),
                         new PReLU(),
                         new Dropout(0.2f),
                         new Dense(512, 512),
                         new LayerNorm(),
                         new PReLU(),
                         new Dense(512, 10),

                         new Softmax()).CreateAsset("MNIST_ResNet");

            net.Forward(Tensor.Random01(1, 28, 28));
            print(net.Parameters().Sum(x => x.theta.Count()));


            // net = new Sequential(
            //     new ResidualConnection.Fork(),
            //     new Attention(100, 100),
            //     new Dense(100, 100),
            //     new ResidualConnection.Join(),
            //     
            //     new ResidualConnection.Fork(),
            //     new RNNCell(100, 100),
            //     new Dense(100, 100),
            //     new ResidualConnection.Join(),
            //     
            //     new ResidualConnection.Fork(),
            //     new RNNCell(100, 100),
            //     new Dense(100, 100),
            //     new ResidualConnection.Join(),
            //     
            //     new ResidualConnection.Fork(),
            //     new RNNCell(100, 100),
            //     new Dense(100, 100),
            //     new ResidualConnection.Join(),
            //     
            //     new ResidualConnection.Fork(),
            //     new RNNCell(100, 100),
            //     new Dense(100, 100),
            //     new ResidualConnection.Join(),
            //     
            //     new RNNCell(100, 100, HiddenStates.ReturnLast),
            //     new Dense(100, 128),
            //     new GELU(),
            //     new Dense(128, 100),
            //     new Softmax()
            //     );
            // 
            // print(net.Forward(Tensor.Random01(64, 100)));
            // print(net.Backward(Tensor.Random01(100)));


            // Tensor input = Tensor.Random01(8, 100);
            // 
            // Attention att = new Attention(100, 100);
            // print(att.Forward(input));
            // print(att.Backward(Tensor.Random01(8, 100)));
            // 
            // print(att.Forward(input.Unsqueeze(0)));
            // print(att.Backward(Tensor.Random01(1, 8, 100)));
            // 
            // print(att.Forward((input.Unsqueeze(0).Expand(0, 5))));
            // print(att.Backward(Tensor.Random01(5, 8, 100)));

            // Tensor input = Tensor.Random01(64, 3, 256, 256);
            // 
            // Conv2D conv = new Conv2D(3, 32, 3, device: Device.GPU);
            // 
            // BenchmarkClock.Start();
            // conv.Predict(input);
            // BenchmarkClock.Stop();

            // Optimizer optimizer = new SGD(net.Parameters());
            // 
            // BenchmarkClock.Start();
            // optimizer.ZeroGrad();
            // BenchmarkClock.Stop();


            // int ic = 2, oc = 3, ih = 6, iw = 6;
            // Conv2D conv = new Conv2D(ic, oc, 3, InitType.Ones, InitType.Ones, device: Device.CPU);
            // 
            // Tensor input = Tensor.Arange(0, ic * ih * iw).Reshape(ic, ih, iw);
            // // input = Tensor.Random01(ic, ih, iw);
            // 
            // print(input);
            // var output = conv.Forward(input);
            // print(output);
            // var inputGrad = conv.Backward(output);
            // print(inputGrad);
            // print("kern_grad - " + conv.kernelsGrad);
            // print("bias_grad - " + conv.biasesGrad);
            // 
            // new Adam(conv.Parameters()).ZeroGrad();
            // 
            // 
            // print("repeat");
            // conv.Device = Device.GPU;
            // 
            // print(input);
            // output = conv.Forward(input);
            // print(output);
            // inputGrad = conv.Backward(output);
            // print(inputGrad);
            // print(conv.kernelsGrad);
            // print(conv.biasesGrad);


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



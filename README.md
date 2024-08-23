# DeepUnity (2022.3.43f1 lts)
![version](https://img.shields.io/badge/version-v0.9.9.9-blue)
[Reference Paper](https://github.com/smtmRadu/Policy-Gradient-Methods-Insights-and-Optimization-Strategies)

DeepUnity is an add-on framework that provides tensor computation [with GPU acceleration support] and deep neural networks, along with reinforcement learning tools that enable training for intelligent agents within Unity environments using state-of-the-art algorithms.

#### Run your first DeepUnity script
```csharp
using UnityEngine;
using DeepUnity;
using DeepUnity.Optimizers;
using DeepUnity.Activations;
using DeepUnity.Modules;
using DeepUnity.Models;

public class Tutorial : MonoBehaviour
{
    [SerializeField] private Sequential network;
    private Optimizer optim;
    private Tensor x;
    private Tensor y;

    public void Start()
    {
        network = new Sequential(
            new Dense(512, 256, device: Device.GPU),
            new ReLU(),
            new Dropout(0.1f),
            new Dense(256, 64, device: Device.GPU),
            new LayerNorm(64),
            new Swish(),
            new Dense(64, 32)).CreateAsset("TutorialModel");
        
        optim = new AdamW(network.Parameters());
        x = Tensor.RandomNormal(64, 512);
        y = Tensor.RandomNormal(64, 32);
    }

    public void Update()
    {
        Tensor yHat = network.Forward(x);
        Loss loss = Loss.MSE(yHat, y);

        optim.ZeroGrad();
        network.Backward(loss.Grad);
        optim.Step();

        print($"Epoch: {Time.frameCount} - Train Loss: {loss.Item}");
        network.Save();
    }
}
```
###### _Digits generated with a GAN trained on MNIST dataset._
![digits](https://github.com/smtmRadu/DeepUnity/blob/main/Assets/DeepUnity/Documentation/gan.gif?raw=true)

###### _Image reconstruction with a VAE trained on MNIST dataset. (original - first line, reconstructed - second line)_
![digits](https://github.com/smtmRadu/DeepUnity/blob/main/Assets/DeepUnity/Documentation/reconstruction.gif?raw=true)

###### _Digit recognition with a ConvNet trained on MNIST dataset (390k params)._
![digits](https://github.com/smtmRadu/DeepUnity/blob/main/Assets/DeepUnity/Documentation/digit.gif?raw=true)

## Reinforcement Learning
In order to work with Reinforcement Learning tools, you must create a 2D or 3D agent using Unity provided GameObjects and Components. The setup flow works similary to ML Agents, so you must create a new behaviour script (e.g. _ReachGoal_) that must inherit the **Agent** class. Attach the new behaviour script to the agent GameObject (automatically **DecisionRequester** script is attached too) [Optionally, a **TrainingStatistics** script can be attached]. Choose the space size and number of continuous/discrete actions, then override the following methods in the behavior script:
- _CollectObservations()_
- _OnActionReceived()_
- _Heuristic()_ [Optional]
- _OnEpisodeBegin()_ [Optional]

Also in order to decide the reward function and episode's terminal state, use the following calls:
-  _AddReward(*reward*)_
-  _SetReward(*reward*)_
-  _EndEpsiode()_ 

When the setup is ready, press the _Bake_ button; a behaviour along with all neural networks and hyperparameters assets are created inside a folder with the _behaviour's name_, located in _Assets/_ folder. From this point everything is ready to go. 

To get into advanced training, check out the following assets created:
- **Behaviour** can be set whether to use a fixed or trainable standard deviation for continuous actions. Inference and Training devices are also available to be set (set both on CPU if your machine lacks a graphics card). TargetFPS modifies the rate of physics update, being equal to _1 / Time.fixedDeltaTime (default: 50)_.
- **Config** provides all hyperparameters necesarry for a custom training session.

#### Behaviour script overriding example
```csharp
using UnityEngine;
using DeepUnity.ReinforcementLearning;

public class MoveToGoal : Agent
{
    public Transform apple;

    public override void OnEpisodeBegin()
    {
        float xrand = Random.Range(-8, 8);
        float zrand = Random.Range(-8, 8);
        apple.localPosition = new Vector3(xrand, 2.25f, zrand);
        
        xrand = Random.Range(-8, 8);
        zrand = Random.Range(-8, 8);
        transform.localPosition = new Vector3(xrand, 2.25f, zrand);
    }

    public override void CollectObservations(StateVector sensorBuffer)
    {
        sensorBuffer.AddObservation(transform.localPosition.x);
        sensorBuffer.AddObservation(transform.localPosition.z);
        sensorBuffer.AddObservation(apple.transform.localPosition.x);
        sensorBuffer.AddObservation(apple.transform.localPosition.z);
    }

    public override void OnActionReceived(ActionBuffer actionBuffer)
    {
        float xmov = actionBuffer.ContinuousActions[0];
        float zmov = actionBuffer.ContinuousActions[1];

        transform.position += new Vector3(xmov, 0, zmov) * Time.fixedDeltaTime * 10f;
        AddReward(-0.0025f); // Step penalty
    } 

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Apple"))
        {
            SetReward(1f);
            EndEpisode();
        }
        if (other.CompareTag("Wall"))
        {
            SetReward(-1f);
            EndEpisode();
        }
    }
}
```
###### _This example considers an agent (with 4 space size and 2 continuous actions) positioned in the middle of an arena that moves forward, backward, left or right, and must reach a randomly positioned goal (see GIF below). The agent is rewarded by 1 point if he touches the apple, and penalized by 1 point if he's falling of the floor, and in both situations the episode ends._

![reacher](https://github.com/smtmRadu/DeepUnity/blob/main/Assets/DeepUnity/Documentation/reacher.gif?raw=true)



### Tips 
- **Parallel training** is one option to use your device at maximum efficiency. After inserting your agent inside an Environment GameObject, you can duplicate that environment several times along the scene before starting the training session; this method is necessary for multi-agent co-op or adversarial training. Note that DeepUnity dynamically adapts the timescale of the simulation to get the maximum efficiency out of your machine.

- In order to properly get use of _AddReward()_ and _EndEpisode()_ consult the diagram below. These methods work well being called inside _OnTriggerXXX()_ or _OnCollisionXXX()_, as well as inside _OnActionReceived()_ rightafter actions are performed. 

- **Decision Period** high values increases overall performance of the training session, but lacks when it comes to agent inference accuracy. Typically, use a higher value for broader parallel environments, then decrease this value to 1 to fine-tune the agent.

- **Input Normalization** plays a huge role in policy convergence. To outcome this problem, observations can be auto-normalized by checking the corresponding box inside behaviour asset, but instead, is highly recommended to manually normalize all input values before adding them to the __SensorBuffer__. Scalar values can be normalized within [0, 1] or [-1, 1] ranges by using the formula **normalized_value = (value - min) / (max - min)**. Note that inputs are clipped for network scability (default [-5, 5]).

- The following MonoBehaviour methods: **Awake()**, **Start()**, **FixedUpdate()**, **Update()** and **LateUpdate()** are virtual. If neccesary, in order to override them, call the their **base** each time, respecting the logic of the diagram below.

### Training on built application for faster inference
- Training inside the Editor is a bit more cumbersome comparing to the built version. Building the application and open it up to start up the training enables faster inference, and the framework was adapted for this.

- Whenever you want to stop the training, close the .exe file. The trained behavior is automatically saved and serialized in .json format on your desktop. Go back in Unity and check your behavior asset, and press on the newly button to overwrite the editor behavior with the trained weights from .json.

- The previous built app, along with the trained weights in .json format are now disposable (remove them and replace the build with a new one).

###### _Base Agent class - order of execution for event functions_
![agentclass](https://github.com/smtmRadu/DeepUnity/blob/main/Assets/DeepUnity/Documentation/agentclass.png?raw=true)

All tutorial scripts are included inside _Assets/DeepUnity/Tutorials_ folder, containing all features provided by the framework and RL environments inspired from ML-Agents examples (note that not all of them have trained models attached).

###### _Sorter agent whose task is to visit the tiles in ascending order_
![sorter](https://github.com/smtmRadu/DeepUnity/blob/main/Assets/DeepUnity/Documentation/sorter.gif?raw=true)

###### _These crawlers are training to scrape over the internet_
![crawlers](https://github.com/smtmRadu/DeepUnity/blob/main/Assets/DeepUnity/Documentation/crawlers.gif?raw=true)

###### _Disney Robots are on the way_
![robot](https://github.com/smtmRadu/DeepUnity/blob/main/Assets/DeepUnity/Documentation/robot.gif?raw=true)





# DeepUnity

DeepUnity is an add-on framework that provides tensor computation [with GPU support] and deep neural networks, along with reinforcement learning tools.
#### Run your first DeepUnity script
```csharp
using UnityEngine;
using DeepUnity;

public class Tutorial : MonoBehaviour
{
    [Header("Learning z = x^2 + y^2.")]
    [SerializeField] private Sequential network;
    [SerializeField] private PerformanceGraph trainLossGraph = new PerformanceGraph();
    [SerializeField] private PerformanceGraph validLossGraph = new PerformanceGraph();
   
    private Optimizer optim;
    private LRScheduler scheduler;

    private Tensor train_inputs;
    private Tensor train_targets;
    private Tensor valid_inputs;
    private Tensor valid_targets;

    public void Start()
    {
        if (network == null)
        {
            network = new Sequential(
                new Dense(2, 64),
                new Tanh(),
                new Dense(64, 64, device: Device.GPU),
                new ReLU(),
                new Dense(64, 1)).CreateAsset("TutorialModel");
        }
        optim = new Adam(network.Parameters(), 0.001f);
        scheduler = new LRScheduler(optim, 30, 0.1f);

        // Generate training dataset
        int data_size = 1024;
        Tensor x = Tensor.RandomNormal(data_size, 1);
        Tensor y = Tensor.RandomNormal(data_size, 1);
        train_inputs = Tensor.Cat(1, x, y);
        train_targets = x * x + y * y;

        // Generate validation set
        int valid_size = 64;
        x = Tensor.RandomNormal(valid_size, 1);
        y = Tensor.RandomNormal(valid_size, 1);
        valid_inputs = Tensor.Cat(1, x, y);
        valid_targets = x * x + y * y;
    }

    public void Update()
    {
        // Training. Split the dataset into batches of 32.
        float train_loss = 0f;
        Tensor[] input_batches = train_inputs.Split(0, 32);
        Tensor[] target_batches = train_targets.Split(0, 32);
        for (int i = 0; i < input_batches.Length; i++)
        {
            Tensor prediction = network.Forward(input_batches[i]);
            Loss loss = Loss.MSE(prediction, target_batches[i]);

            optim.ZeroGrad();
            network.Backward(loss.Derivative);
            optim.ClipGradNorm(0.5f);
            optim.Step();

            train_loss += loss.Item;
        }
        train_loss /= input_batches.Length;
        trainLossGraph.Append(train_loss);

        // Validation
        Tensor valid_prediction = network.Predict(valid_inputs);
        float valid_loss = Metrics.MeanSquaredError(valid_prediction, valid_targets);
        validLossGraph.Append(valid_loss);

        print($"Epoch: {Time.frameCount} - Train Loss: {train_loss} - Valid Loss: {valid_loss}");

        scheduler.Step();
        network.Save();
    }
}
```
![rl](https://github.com/RaduTM-spec/DeepUnity/blob/main/Assets/DeepUnity/Documentation/tensors.png?raw=true)

### Reinforcement Learning [In Development]
In order to work with Reinforcement Learning tools, you must create a 2D or 3D agent using Unity provided GameObjects and Coomponents. The setup flow works similary to ML Agents (but with some restrictions described in the diagram below), so you must create a new behaviour script (e.g. _MoveToGoal_) that must inherit the **Agent** class. Attach the new behaviour script to the agent GameObject (automatically are attached 2 more scripts, **HyperParameters** and **DecisionRequester**) [Optionally, a **PerformanceTrack** script can be attached]. Choose the space size and number of actions, then override the following methods in the behavior script:
- _CollectObservations()_
- _OnActionReceived()_
- _Heuristic()_ [Optional]
- _OnEpisodeBegin()_ [Optional]

Also in order to decide the reward function and episode terminal state, use the following calls inside FixedUpdate(), OnTriggerXXX() or OnCollisionXXX():
-  _AddReward(*reward*)_
-  _EndEpsiode()_ 
-  _RequestAction()_ [Optional, if decision is requested manually]
#### Behaviour script overriding example
```csharp
using UnityEngine;
using DeepUnity;

public class MoveToGoal : Agent
{
    [Header("Properties")]
    public float speed = 10f;
    public Transform target;

    public override void FixedUpdate()
    {
        base.FixedUpdate();
        AddReward(-0.001f);
    }
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        sensorBuffer.AddObservation(transform.localPosition.x);
        sensorBuffer.AddObservation(transform.localPosition.z);
        sensorBuffer.AddObservation(target.transform.localPosition.x);
        sensorBuffer.AddObservation(target.transform.localPosition.z);
    }
    public override void OnActionReceived(ActionBuffer actionBuffer)
    {
        float xmov = actionBuffer.ContinuousActions[0];
        float zmov = actionBuffer.ContinuousActions[1];
        
        transform.position += new Vector3(xmov, 0, zmov) * Time.fixedDeltaTime * speed;
    }
    public override void Heuristic(ActionBuffer actionBuffer)
    {
        float xmov = 0;
        float zmov = 0;

        if (Input.GetKey(KeyCode.W))
            zmov = 1;
        else if(Input.GetKey(KeyCode.S))
            zmov = -1;

        if (Input.GetKey(KeyCode.D))
            xmov = 1;
        else if (Input.GetKey(KeyCode.A))
            xmov = -1;

        actionBuffer.ContinuousActions[0] = xmov;
        actionBuffer.ContinuousActions[1] = zmov;
    }
    
    public override void OnEpisodeBegin()
    {
        float xrand = Random.Range(-5, 5);
        float zrand = Random.Range(-5, 5);
        target.position = new Vector3(xrand, 0, zrand);
    }
    private void OnCollisionEnter(Collision collision)
    {
        if(collision.collider.CompareTag("Target"))
        {
            AddReward(1f);
            EndEpisode();
        }    
        else if(collision.collider.CompareTag("Wall"))
        {
            AddReward(-1f);
            EndEpisode();
        }
    }
}
```
_This example considers an agent (with 4 space size and 2 continuous actions) positioned in a middle of an arena that moves forward, backward, left or right (decision is requested each frame), and must reach a randomly positioned target. The agent is rewarded by 1 point if he touches the target, or penalized by 1 point if he hits a wall. The agent is penalized constantly by 0.001 points at each time step, to encourage the agent reaching the target as fast as possible._

![rl](https://github.com/RaduTM-spec/DeepUnity/blob/main/Assets/DeepUnity/Documentation/RL_schema.jpg?raw=true)

#### Notes
- The following MonoBehaviour methods: **Awake()**, **Start()**, **FixedUpdate()**, **Update()** and **LateUpdate()** are virtual. In order to override them, call the their **base** *[first]* each time.
- Call _AddReward()_, _EndEpisode()_ and _RequestAction()_ only inside **FixedUpdate()**, or any **OnTriggerXXX()**/**OnCollisionXXX()** methods.



# DeepUnity

DeepUnity is an add-on framework that provides tensor computation [with GPU support] and deep neural networks, along with reinforcement learning tools.

#### Run your first DeepUnity script
```csharp
using UnityEngine;
using DeepUnity;
using System.Collections.Generic;
using System.Linq;

public class FirstScript : MonoBehaviour
{
    [SerializeField] 
    private Sequential network;
    private Optimizer optim;
    private StepLR scheduler;

    private Tensor train_inputs;
    private Tensor train_targets;

    private Tensor valid_inputs;
    private Tensor valid_targets;

    public void Start()
    {
        if(network == null)
        {
            network = new Sequential(
                new Dense(2, 64),
                new Tanh(),
                new Dense(64, 64),                
                new ReLU(),
                new Dense(64, 1));
        }

        optim = new Adam(network.Parameters);
        scheduler = new StepLR(optim, 100);
	
	    // Learning z = x^2 + y^2 function.
        // Generate dataset
        int data_size = 1024;
        Tensor x = Tensor.RandomNormal((0, 0.5f), data_size, 1);
        Tensor y = Tensor.RandomNormal((0, 0.5f), data_size, 1);
        train_inputs = Tensor.Cat(1, x, y);
        train_targets = x.Zip(y, (a, b) => a * a + b * b);

        // Generate validation set
        int valid_size = 64;
        x = Tensor.RandomNormal((0, 0.5f), valid_size, 1);
        y = Tensor.RandomNormal((0, 0.5f), valid_size, 1);
        valid_inputs = Tensor.Cat(1, x, y);
        valid_targets = x.Zip(y, (a, b) => a * a + b * b);
    }

    public void Update()
    {
        List<float> epoch_train_accuracies = new List<float>();

        // Split dataset into batches
        int batch_size = 32;
        Tensor[] input_batches = Tensor.Split(train_inputs, 0, batch_size);
        Tensor[] target_batches = Tensor.Split(train_targets, 0, batch_size);

        // Update the network for each batch
        for (int i = 0; i < input_batches.Length; i++)
        {
            Tensor prediction = network.Forward(input_batches[i]);
            Tensor loss = Loss.MSEDerivative(prediction, target_batches[i]);

            optim.ZeroGrad();
            network.Backward(loss);
            optim.ClipGradNorm(0.5f);
            optim.Step();
            
            float train_acc = Metrics.Accuracy(prediction, target_batches[i]);
            epoch_train_accuracies.Add(train_acc);       
        }

        scheduler.Step();
        network.Save("tutorial_model");

        float valid_acc = Metrics.Accuracy(network.Predict(valid_inputs), valid_targets);
        print($"[Epoch {Time.frameCount} | Train Accuracy: {epoch_train_accuracies.Average() * 100f}% | Validation Accuracy: {valid_acc * 100f}%]");
    }
}

```
### Reinforcement Learning [In Development]
In order to work with Reinforcement Learning tools, you must create a 2D or 3D agent using Unity provided GameObjects and Coomponents. The setup flow works similary to ML Agents (but with some restrictions described in the diagram below), so you must create a new behaviour script (e.g. _MoveToGoal_) that must inherit the **Agent** class. Attach the new behaviour script to the agent (automatically is attached a **HyperParameters** script). Choose the space size and number of actions, then override the following methods in the behavior script:
- __CollectObservations()__
- __OnActionReceived()__
- _Heuristic()_ [Optional]
- _OnEpisodeBegin()_ [Optional]

Also in order to decide the reward function and episode terminal state, use the following calls:
-  __AddReward(*reward*)__ 
-  __EndEpsiode()__ 
#### Behaviour script overriding example
```csharp
using UnityEngine;
using DeepUnity;

public class MoveToGoal : Agent
{
    [Header("Properties")]
    public float speed = 10f;
    public Transform target; // referenced manually

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
_This example considers an agent (with 4 space size and 2 continuous actions) positioned in a middle of an arena that can move forward, backward, left or right, and must reach a randomly positioned target. The agent is rewarded by 1 point if it touches the target, or penalized by 1 point if it touches a wall. The agent is penalized constantly by 0.001 points at each time step, to encourage the agent reaching the target as fast as possible._

![rl]("Assets\DeepUnity\Documentation\RL_order_of_execution.png")

#### Notes
- The following MonoBehaviour methods: **Awake()**, **Start()**, **FixedUpdate()**, **Update()** and **LateUpdate()** are virtual. In order to override them, call the their **base** each time.
- Call _AddReward()_ and _EndEpisode()_ only inside **FixedUpdate()**, or any **OnTriggerXXX()**/**OnCollisionXXX()** methods.



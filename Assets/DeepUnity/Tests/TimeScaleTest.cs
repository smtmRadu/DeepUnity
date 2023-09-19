using UnityEngine;
using DeepUnity;
using System.Diagnostics;

public class TimeScaleTest : MonoBehaviour
{
	public float timescale;
    public float matmul;
	public int fixedCOunt = 0;
	public int updateCount = 0;
	public int lateupdateCOunt = 0;
    public int size = 129;

    public PerformanceGraph graph;
    public Stopwatch stopwatch;
    private void Awake()
    {
        stopwatch = new Stopwatch();
        stopwatch.Start();
        Time.timeScale = timescale;
    }
    private void FixedUpdate()
    {
        
        if(fixedCOunt == 50)
        {
            //Time.timeScale = 0f;
            graph.Append(Time.deltaTime);
            Tensor input = Tensor.Random01(size, size);
            Tensor input2 = Tensor.Random01(size, size);
            Tensor.MatMul(input, input2);
            //Time.timeScale = 1f;
        }
       
        fixedCOunt++;
    }

    private void Update()
    {
        updateCount++;
    }

    private void LateUpdate()
    {

        lateupdateCOunt++;
    }

    private void OnCollisionEnter(Collision collision)
    {
        stopwatch.Stop();
        print(stopwatch.Elapsed);
    }

}



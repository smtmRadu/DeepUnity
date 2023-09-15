using DeepUnity;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnityTutorials
{
    public class SoccerEnvironmentScript : MonoBehaviour
    {
        [SerializeField] private float time_per_round = 60f;
        private float current_round_time_left;
        [Space]
        [SerializeField] public Transform ball;
        [Space]
        [SerializeField] private Agent pink_striker;
        [SerializeField] private Agent pink_goalie;
        [Space]
        [SerializeField] private Agent blue_striker;
        [SerializeField] private Agent blue_goalie;
        [Space]

        [SerializeField] Text score_label;
        [SerializeField] Text time_label;

        [Space]
        [ReadOnly, SerializeField] private int pink_score = 0;
        [ReadOnly, SerializeField] private int blue_score = 0;

        private Vector3 ball_initial_pos;
        private Rigidbody ball_rb;
        private void Awake()
        {
            ball_initial_pos = ball.transform.localPosition;
            current_round_time_left = time_per_round;
            ball_rb = ball.GetComponent<Rigidbody>();   
        }

        private void Update()
        {

            current_round_time_left -= Time.deltaTime;
            if (current_round_time_left <= 0f)
                StartNewRound();
            time_label.text = $"{(int)(current_round_time_left/60)}:{(int)current_round_time_left%60}";
        }


        public void BlueTeamScored()
        {
            blue_score++;
            blue_striker.AddReward(1f);
            blue_goalie.AddReward(0.1f);
            pink_goalie.AddReward(-1f);
            pink_striker.AddReward(-0.1f);


            ChangeScoreLabel();
            StartNewRound();
        }
        public void PinkTeamScored()
        {
            pink_score++;
            pink_striker.AddReward(1f);
            pink_goalie.AddReward(0.1f);
            blue_goalie.AddReward(-1f);
            blue_striker.AddReward(-0.1f);


            ChangeScoreLabel();
            StartNewRound();
        }
        public List<float> GetBallInfo()
        {
            List<float> ball_info = new List<float>();
            Vector3 pos = ball.localPosition;
            ball_info.Add(pos.x);
            ball_info.Add(pos.y);
            ball_info.Add(pos.z);
            Vector3 vel = ball_rb.velocity.normalized;
            ball_info.Add(vel.x);
            ball_info.Add(vel.y);
            ball_info.Add(vel.z);
            Vector3 angVel = ball_rb.angularVelocity.normalized;
            ball_info.Add(angVel.x);
            ball_info.Add(angVel.y);
            ball_info.Add(angVel.z);
            return ball_info;
        }
        public void StartNewRound()
        {
            pink_striker.EndEpisode();
            pink_goalie.EndEpisode();
            blue_striker.EndEpisode();
            blue_goalie.EndEpisode();

            current_round_time_left = time_per_round;
            ball.localPosition = ball_initial_pos;
            ball_rb.velocity = Vector3.zero;
            ball_rb.angularVelocity = Vector3.zero;
        }
        private void ChangeScoreLabel() => score_label.text = $"{pink_score}:{blue_score}";
        
    }



}



using DeepUnity;
using DeepUnity.ReinforcementLearning;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity.Tutorials
{
    public class SoccerEnvironmentScript : MonoBehaviour
    {
        [Header("At least 1 striker and 1 goalie per team")]
        [SerializeField] private float time_per_round = 60f;
        private float current_round_time_left;
        [Space]
        [SerializeField] public Transform ball;
        [Space]
        [SerializeField] private Agent pink_goalie;
        [SerializeField] private Agent pink_striker;
        [SerializeField] private Agent pink_striker2;
        [Space]
        [SerializeField] private Agent blue_goalie;
        [SerializeField] private Agent blue_striker;
        [SerializeField] private Agent blue_striker2;
        [Space]

        [SerializeField] Text score_label;
        [SerializeField] Text time_label;

        [Space]
        [ViewOnly, SerializeField] private int pink_score = 0;
        [ViewOnly, SerializeField] private int blue_score = 0;

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
                StartNewRound(true);
            time_label.text = $"{(int)(current_round_time_left/60)}:{(int)current_round_time_left%60}";
        }


        public void BlueTeamScored()
        {
            float goal_reward = current_round_time_left / time_per_round;
            blue_score++;

            blue_striker.AddReward(goal_reward);
            blue_striker2?.AddReward(goal_reward);
            

            pink_goalie.AddReward(-1f);
            pink_striker.AddReward(-1f);
            pink_striker2?.AddReward(-1f);

            ChangeScoreLabel();
            StartNewRound();
        }
        public void PinkTeamScored()
        {
            float goal_reward = current_round_time_left / time_per_round;
            pink_score++;

            pink_striker.AddReward(goal_reward);
            pink_striker2?.AddReward(goal_reward);

            blue_goalie.AddReward(-1f);
            blue_striker.AddReward(-1f);
            blue_striker2?.AddReward(-1f);

            ChangeScoreLabel();
            StartNewRound();
        }
        public List<float> GetBallInfo()
        {
            List<float> ball_info = new List<float>();
            Vector3 pos = ball.localPosition;
            ball_info.Add(pos.x / 30f);
            ball_info.Add(pos.y);
            ball_info.Add(pos.z / 15f);
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
        public void StartNewRound(bool no_score = false)
        {
            if(no_score)
            {
                pink_striker.AddReward(-1);
                blue_striker.AddReward(-1);
                pink_striker2?.AddReward(-1);
                blue_striker2?.AddReward(-1);
            }

            pink_striker2?.EndEpisode();
            pink_striker.EndEpisode();
            pink_goalie.EndEpisode();
            blue_striker2?.EndEpisode();
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



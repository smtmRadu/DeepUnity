using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Walks the gossiping pair (plus the cow that tags along) around a closed waypoint loop
    /// through the village. One component owns the shared arc-length progress so the three
    /// bodies stay abreast forever — each member is placed at the loop point plus its own
    /// lateral offset, so the pair walks shoulder to shoulder and the cow keeps their flank.
    ///
    /// The whole group STOPS when either stroller enters a conversation: the pressed one is
    /// turned by the dialogue machinery itself, the partner is turned toward the player here,
    /// and the cow just stands (its gait idles into grazing). Walking resumes a moment after
    /// the chat closes.
    /// </summary>
    public class VillageStrollGroup : MonoBehaviour
    {
        [Tooltip("Closed loop, in order; Y is ignored (members snap to the ground by raycast).")]
        [SerializeField] private Vector3[] waypoints;
        [SerializeField] private float walkSpeed = 0.85f;
        [SerializeField] private float turnDegPerSec = 240f;
        [Tooltip("Seconds after the chat closes before the stroll resumes.")]
        [SerializeField] private float resumeDelay = 1.6f;

        [Header("Members (wired by the scene builder)")]
        [SerializeField] private VillageStroller strollerA;
        [SerializeField] private VillageStroller strollerB;
        [SerializeField] private Transform cow;
        [SerializeField] private QuadrupedGait cowGait;
        [Tooltip("Lateral offsets along the walk direction's RIGHT (+) / LEFT (-), in meters.")]
        [SerializeField] private float offsetA = 0.55f;
        [SerializeField] private float offsetB = -0.55f;
        [Tooltip("The cow walks BEHIND the pair, on the rope Odo holds — trailing his shoulder.")]
        [SerializeField] private float cowOffset = 0.55f;
        [SerializeField] private float cowTrail = 1.7f;
        [Tooltip("Loop progress at scene start, meters — set so the pair first appears walking OUT of the market square toward the spawn.")]
        [SerializeField] private float startDistance = 0f;

        [SerializeField] private VillageBanter banter;    // suspended while any dialogue is open

        float[] cumLen;      // cumulative polyline length per vertex (closed)
        float totalLen;
        float dist;          // shared progress along the loop, meters
        float resumeAt;      // Time.time gate after a conversation closes
        Transform playerT;
        VillageStroller facePlayer;   // the partner turning toward the player while paused

        public bool IsWalking { get; private set; }

        void Start()
        {
            var playerGO = GameObject.FindWithTag("Player");
            if (playerGO != null) playerT = playerGO.transform;

            cumLen = new float[waypoints.Length + 1];
            for (int i = 0; i < waypoints.Length; i++)
            {
                Vector3 a = waypoints[i], b = waypoints[(i + 1) % waypoints.Length];
                a.y = 0; b.y = 0;
                cumLen[i + 1] = cumLen[i] + Vector3.Distance(a, b);
            }
            totalLen = cumLen[waypoints.Length];
            dist = Mathf.Repeat(startDistance, totalLen);

            // drop everyone onto the loop immediately so frame 0 doesn't show them off-path
            PlaceMembers(1f);
        }

        void Update()
        {
            bool inDialogue = (strollerA != null && strollerA.InConversation)
                           || (strollerB != null && strollerB.InConversation);
            bool walk = !inDialogue && Time.time >= resumeAt && !PlayerBlocksPath();

            if (walk != IsWalking)
            {
                IsWalking = walk;
                strollerA?.PlayWalkAnim(walk);
                strollerB?.PlayWalkAnim(walk);
                if (walk) facePlayer = null;
            }

            if (cowGait != null) cowGait.Speed = IsWalking ? walkSpeed : 0f;
            if (!IsWalking)
            {
                // the partner keeps eye contact with the player for the whole conversation
                if (facePlayer != null && playerT != null && !facePlayer.InConversation)
                {
                    Vector3 to = playerT.position - facePlayer.transform.position;
                    to.y = 0f;
                    if (to.sqrMagnitude > 1e-4f)
                        facePlayer.transform.rotation = Quaternion.RotateTowards(
                            facePlayer.transform.rotation, Quaternion.LookRotation(to.normalized),
                            turnDegPerSec * Time.deltaTime);
                }
                return;
            }

            dist = (dist + walkSpeed * Time.deltaTime) % totalLen;
            PlaceMembers(turnDegPerSec * Time.deltaTime);
        }

        // The members move kinematically (no physics push), so without this they would walk
        // STRAIGHT THROUGH a player standing on the path. Standing in their way instead makes
        // the whole group stop and wait — which is also the natural moment to press E.
        bool PlayerBlocksPath()
        {
            if (playerT == null) return false;
            Sample(dist, out Vector3 p, out Vector3 tangent);
            Vector3 to = playerT.position - p;
            to.y = 0f;
            return to.sqrMagnitude < 1.5f * 1.5f && Vector3.Dot(tangent, to.normalized) > 0.15f;
        }

        void PlaceMembers(float maxTurnDeg)
        {
            Sample(dist, out Vector3 p, out Vector3 tangent);
            // Cross(up, fwd) is the walk direction's RIGHT. On the leg that comes down the main
            // street toward the spawn (heading -Z) that right is -X — the approaching player's
            // LEFT, which is exactly where the cow was asked to be.
            Vector3 right = Vector3.Cross(Vector3.up, tangent);
            Place(strollerA != null ? strollerA.transform : null, p + right * offsetA, tangent, maxTurnDeg);
            Place(strollerB != null ? strollerB.transform : null, p + right * offsetB, tangent, maxTurnDeg);
            Place(cow, p + right * cowOffset - tangent * cowTrail, tangent, maxTurnDeg);
        }

        void Place(Transform t, Vector3 target, Vector3 fwd, float maxTurnDeg)
        {
            if (t == null) return;
            target.y = GroundY(target);
            t.position = target;
            t.rotation = Quaternion.RotateTowards(t.rotation, Quaternion.LookRotation(fwd), maxTurnDeg);
        }

        float GroundY(Vector3 at)
        {
            return Physics.Raycast(at + Vector3.up * 3f, Vector3.down, out RaycastHit hit, 6f,
                                   Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore)
                 ? hit.point.y : at.y;
        }

        void Sample(float d, out Vector3 point, out Vector3 tangent)
        {
            // find the segment containing d (cumLen is monotonic; last segment closes the loop)
            int seg = waypoints.Length - 1;
            for (int i = 0; i < waypoints.Length; i++)
                if (d >= cumLen[i] && d <= cumLen[i + 1]) { seg = i; break; }

            Vector3 a = waypoints[seg], b = waypoints[(seg + 1) % waypoints.Length];
            a.y = 0; b.y = 0;
            float segLen = Mathf.Max(1e-4f, cumLen[seg + 1] - cumLen[seg]);
            float t = (d - cumLen[seg]) / segLen;
            point = Vector3.Lerp(a, b, t);
            tangent = (b - a).normalized;
        }

        public void OnMemberInteractionStarted(VillageStroller who)
        {
            resumeAt = float.MaxValue;   // pinned until the close hook fires
            facePlayer = who == strollerA ? strollerB : strollerA;
            strollerA?.CutAmbientSpeech();
            strollerB?.CutAmbientSpeech();
            banter?.Suspend();
        }

        public void OnMemberInteractionClosed(VillageStroller who)
        {
            bool stillTalking = (strollerA != null && strollerA.InConversation)
                             || (strollerB != null && strollerB.InConversation);
            if (!stillTalking)
            {
                resumeAt = Time.time + resumeDelay;
                banter?.Resume();
            }
        }
    }
}

using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// TradingVillage dialogue NPC: the souls NPC stack (NPCInteractor3D → NPCChatBase) rebound
    /// to E — the village key, matching the 2D farm demo — plus a nearest-candidate gate. The
    /// gate exists because this scene is the first where two talkable NPCs walk SIDE BY SIDE:
    /// the player standing between them is inside both talk triggers at once, and without the
    /// gate one press of E would open both conversations on the same frame.
    /// </summary>
    public class VillageInteractor : NPCInteractor3D
    {
        // horizontal reach within which another villager counts as a rival candidate — the talk
        // trigger the builder puts on every villager is a 2.2 m sphere at chest height
        const float CANDIDATE_RANGE = 2.6f;

        static readonly List<VillageInteractor> all = new List<VillageInteractor>();

        Transform playerT;

        protected override KeyCode InteractKey => KeyCode.E;

        protected override bool PlayerReady => base.PlayerReady && IsClosestCandidate();

        protected override void OnEnable()
        {
            base.OnEnable();
            if (!all.Contains(this)) all.Add(this);
        }

        protected override void OnDisable()
        {
            all.Remove(this);
            base.OnDisable();
        }

        protected override void Start()
        {
            base.Start();
            var playerGO = GameObject.FindWithTag("Player");
            if (playerGO != null) playerT = playerGO.transform;
        }

        /// <summary>In conversation right now (any phase — opening, waiting, or talking).</summary>
        public bool InConversation => state != NPCState.Idle;

        // Distance-based, deliberately NOT a virtual-property handshake: asking the rival for its
        // own PlayerReady would virtual-dispatch back into ITS gate and recurse between the two
        // strollers. A rival is simply another idle villager standing closer to the player than
        // this one while also within talk reach.
        bool IsClosestCandidate()
        {
            if (playerT == null) return true;
            Vector3 p = playerT.position;
            float mine = Flat(transform.position - p);
            foreach (var v in all)
            {
                if (v == this || v.state != NPCState.Idle) continue;
                float theirs = Flat(v.transform.position - p);
                if (theirs < CANDIDATE_RANGE * CANDIDATE_RANGE && theirs < mine - 0.01f)
                    return false;
            }
            return true;
        }

        static float Flat(Vector3 d)
        {
            d.y = 0f;
            return d.sqrMagnitude;
        }
    }
}

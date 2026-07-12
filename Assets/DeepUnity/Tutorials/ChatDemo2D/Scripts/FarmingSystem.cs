using System;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo2D
{
    /// <summary>
    /// The farm loop: hoe → plant → water → timed growth (water is consumed per stage) → harvest.
    /// Owns the crop table, the toolbar selection (1-6 / mouse wheel), the target-cell highlight
    /// and the growth tick for every FarmPlot. Farming input is ignored while a dialogue owns the
    /// player (chat typing must not swing the hoe).
    ///
    /// Tools: 0 Hoe · 1 Watering bucket · 2-4 Seeds (carrot/turnip/tomato) · 5 Harvest.
    /// Use with Space (or left click). Seeds are free — approved scope has no inventory/economy.
    /// </summary>
    public class FarmingSystem : MonoBehaviour
    {
        [Serializable]
        public class CropDef
        {
            public string name;
            [Tooltip("Sprout / growing / ready sprites (3 stages).")]
            public Sprite[] stageSprites;
            [Tooltip("Seconds of WATERED time to advance one stage (2 advances to ripen).")]
            public float secondsPerStage = 25f;
            [Tooltip("Coins an NPC pays for one of these (give-items flow).")]
            public int coinValue = 2;
        }

        [Header("Wired by the scene builder")]
        [SerializeField] private PlayerController2D player;
        [SerializeField] private FarmPlot[] plots;
        [SerializeField] private CropDef[] crops;
        [SerializeField] private SpriteRenderer highlight;
        [SerializeField] private FarmHud hud;
        [SerializeField] private AudioSource sfx;          // soft click on every successful action
        [SerializeField] private AudioClip actionClip;

        [Header("Tuning")]
        [Tooltip("Max distance from player to a workable plot.")]
        [SerializeField] private float reach = 1.9f;
        [Tooltip("How far ahead of the player the target probe sits.")]
        [SerializeField] private float probeDistance = 0.85f;

        public const int TOOL_HOE = 0, TOOL_WATER = 1, TOOL_SEED_0 = 2, TOOL_HARVEST = 5;
        private const int TOOL_COUNT = 6;

        private int tool;
        private readonly int[] harvested = new int[3];
        private FarmPlot target;

        /// <summary>Coins earned by giving produce to the villagers (HUD-displayed).</summary>
        public int Coins { get; private set; }

        /// <summary>True while at least one harvested vegetable sits in the basket.</summary>
        public bool HasAnyHarvest
        {
            get { foreach (int n in harvested) if (n > 0) return true; return false; }
        }

        /// <summary>Human-readable basket summary, e.g. "2 carrots, 1 tomato" (crop-table names).</summary>
        public string DescribeHarvest()
        {
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < harvested.Length; i++)
            {
                if (harvested[i] <= 0) continue;
                if (sb.Length > 0) sb.Append(", ");
                string n = crops[i].name.ToLowerInvariant();
                sb.Append(harvested[i]).Append(' ').Append(harvested[i] == 1 ? n : n + "s");
            }
            return sb.ToString();
        }

        /// <summary>Empties the basket and returns the per-crop counts taken (give-to-NPC flow).</summary>
        public int[] TakeAllHarvested()
        {
            var taken = (int[])harvested.Clone();
            for (int i = 0; i < harvested.Length; i++)
            {
                harvested[i] = 0;
                hud?.SetCount(i, 0);
            }
            return taken;
        }

        /// <summary>Coin value of a per-crop count array (per the crop table).</summary>
        public int HarvestValue(int[] counts)
        {
            int sum = 0;
            for (int i = 0; i < counts.Length && i < crops.Length; i++)
                sum += counts[i] * crops[i].coinValue;
            return sum;
        }

        public void AddCoins(int amount)
        {
            Coins += amount;
            hud?.SetCoins(Coins);
        }

        private void Start()
        {
            if (hud != null)
            {
                hud.SetSelected(tool);
                for (int i = 0; i < harvested.Length; i++) hud.SetCount(i, 0);
                hud.SetCoins(0);
            }
        }

        private void Update()
        {
            TickGrowth();

            if (player == null || player.IsBusy)
            {
                if (highlight != null) highlight.enabled = false;
                return;
            }

            ReadToolSelection();
            AcquireTarget();

            if (target != null && (Input.GetKeyDown(KeyCode.Space) || Input.GetMouseButtonDown(0)))
                ApplyTool(target);
        }

        // growth runs on real time, gated on water: a stage only ripens while its soil is wet,
        // and each advance dries the soil again — the Stardew rhythm, minutes instead of days
        private void TickGrowth()
        {
            foreach (var p in plots)
            {
                if (!p.HasCrop || p.IsReady || !p.IsWatered) continue;
                var def = crops[p.CropIndex];
                p.Growth += Time.deltaTime;
                if (p.Growth >= def.secondsPerStage)
                    p.AdvanceStage(def.stageSprites[p.Stage + 1]);
            }
        }

        private void ReadToolSelection()
        {
            for (int i = 0; i < TOOL_COUNT; i++)
                if (Input.GetKeyDown(KeyCode.Alpha1 + i))
                    SelectTool(i);

            float scroll = Input.mouseScrollDelta.y;
            if (scroll > 0.01f) SelectTool((tool + TOOL_COUNT - 1) % TOOL_COUNT);
            else if (scroll < -0.01f) SelectTool((tool + 1) % TOOL_COUNT);
        }

        private void SelectTool(int i)
        {
            tool = i;
            hud?.SetSelected(tool);
        }

        // target = the plot nearest to a probe point just ahead of the player's facing,
        // if it is close enough to the probe AND within arm's reach of the player
        private void AcquireTarget()
        {
            Vector2 probe = (Vector2)player.transform.position + player.Facing * probeDistance;
            target = null;
            float best = 0.95f;   // must be near the probe — no telefarming sideways
            foreach (var p in plots)
            {
                float dProbe = Vector2.Distance(probe, p.transform.position);
                if (dProbe > best) continue;
                if (Vector2.Distance(player.transform.position, p.transform.position) > reach) continue;
                best = dProbe;
                target = p;
            }

            if (highlight != null)
            {
                highlight.enabled = target != null;
                if (target != null)
                    highlight.transform.position = target.transform.position;
            }
        }

        private void ApplyTool(FarmPlot p)
        {
            bool acted = false;
            switch (tool)
            {
                case TOOL_HOE:
                    if (!p.IsTilled) { p.Till(); acted = true; }
                    break;
                case TOOL_WATER:
                    if (p.IsTilled && !p.IsWatered) { p.Water(); acted = true; }
                    break;
                case TOOL_HARVEST:
                    if (p.IsReady)
                    {
                        harvested[p.CropIndex]++;
                        hud?.SetCount(p.CropIndex, harvested[p.CropIndex]);
                        p.ClearCrop();
                        acted = true;
                    }
                    break;
                default:   // seeds
                    int cropIdx = tool - TOOL_SEED_0;
                    if (p.IsTilled && !p.HasCrop && cropIdx >= 0 && cropIdx < crops.Length)
                    {
                        p.Plant(cropIdx, crops[cropIdx].stageSprites[0]);
                        acted = true;
                    }
                    break;
            }

            if (acted && sfx != null && actionClip != null)
            {
                sfx.pitch = UnityEngine.Random.Range(0.94f, 1.08f);
                sfx.PlayOneShot(actionClip, 0.4f);
            }
        }
    }
}

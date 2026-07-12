using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo2D
{
    /// <summary>
    /// One farmable cell. Holds its soil/crop state and updates its two SpriteRenderers (flat
    /// soil below everything, crop sprite Y-sorted with the characters). All the rules — what a
    /// tool does, growth timing — live in FarmingSystem; the plot is state + view.
    ///
    /// Lifecycle: Wild (invisible, hoe-able) → Tilled dry → watered → crop stages 0→1→2
    /// (watering is consumed per stage; the soil dries after each advance) → harvest → Tilled dry.
    /// </summary>
    public class FarmPlot : MonoBehaviour
    {
        [Header("Wired by the scene builder")]
        [SerializeField] private SpriteRenderer soil;
        [SerializeField] private SpriteRenderer crop;
        [SerializeField] private Sprite soilDry;
        [SerializeField] private Sprite soilWet;

        // runtime state (owned/mutated via FarmingSystem)
        public bool IsTilled { get; private set; }
        public bool IsWatered { get; private set; }
        /// <summary>-1 = empty; otherwise index into FarmingSystem's crop table.</summary>
        public int CropIndex { get; private set; } = -1;
        /// <summary>0..2; 2 = ready to harvest.</summary>
        public int Stage { get; private set; }
        public float Growth;   // seconds accumulated toward the next stage (FarmingSystem ticks it)

        public bool HasCrop => CropIndex >= 0;
        public bool IsReady => HasCrop && Stage >= 2;

        public void Till()
        {
            IsTilled = true;
            IsWatered = false;
            RefreshSoil();
        }

        public void Water()
        {
            IsWatered = true;
            RefreshSoil();
        }

        public void Plant(int cropIndex, Sprite stage0)
        {
            CropIndex = cropIndex;
            Stage = 0;
            Growth = 0f;
            crop.sprite = stage0;
            crop.enabled = true;
        }

        /// <summary>Advance one growth stage; consumes the water (soil dries).</summary>
        public void AdvanceStage(Sprite stageSprite)
        {
            Stage++;
            Growth = 0f;
            IsWatered = false;
            crop.sprite = stageSprite;
            RefreshSoil();
        }

        public void ClearCrop()
        {
            CropIndex = -1;
            Stage = 0;
            Growth = 0f;
            crop.enabled = false;
            IsWatered = false;
            RefreshSoil();
        }

        private void RefreshSoil()
        {
            soil.enabled = IsTilled;
            soil.sprite = IsWatered ? soilWet : soilDry;
        }
    }
}

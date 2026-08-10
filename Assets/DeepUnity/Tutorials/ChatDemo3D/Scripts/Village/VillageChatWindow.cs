namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// The souls chat panel re-skinned by vocabulary alone: a market town deals in coins, not
    /// souls, so a GiveItem offer reads "Smoked trout  -  4 coins". Everything else — slide-in,
    /// streaming transcript, tool panels — is SoulsChatWindow verbatim.
    /// </summary>
    public class VillageChatWindow : SoulsChatWindow
    {
        protected override string GiveItemCurrency => "coins";
    }
}

# Reinforcement Learning — Bugs Found & Fixes

Log of every bug found during the SAC/off-policy investigation (March 2026) and the
follow-up session (June 2026) that built the FullGPU SAC trainer and closed the
BalanceBall case. Full investigation narrative: `../../AGENT_COLLAB.md`.

---

## TL;DR

- **PPO / PPOGPU** were always correct.
- **SAC** had real config bugs (all fixed below), but after fixing them it *still*
  plateaued on BalanceBall. A from-scratch GPU reimplementation (**SACGPU**) reproduced
  the exact same plateau — proving the remaining problem was the **environment setup**,
  not the trainer. With `decisionPeriod=5` + a `-1` fall penalty + `alpha=0.005`,
  SAC converges on BalanceBall (episodes hit the 10,000-step cap).
- **VPG** failed for its own reasons: a gradient-normalization hack that destroyed the
  gradient scale, plus copy-paste bugs. Fixed.
- **TD3 / DDPG** shared several off-policy bugs (scheduler, truncation, optimizer
  resume, static targets). Fixed.

---

## VPGTrainer.cs

### 1. Policy gradient destroyed by unit-norm hack (CRITICAL — why VPG never learned)
- **Bug:** the gradient was computed as `(-A/π) · (π·(a-μ)/σ²)`. The two π factors cancel
  *analytically*, but computing `-A/π` in floating point overflows when π→0 (densities of
  unlikely actions). The resulting explosions were masked by
  `dmObjective_dPi /= dmObjective_dPi.Norm()[0]` — normalizing the entire batch gradient to
  unit norm, which destroys the gradient scale and lets a single outlier sample dominate
  every minibatch.
- **Fix:** compute the analytic form directly — `∂-(logπ·A)/∂μ = -A(a-μ)/σ²` and
  `∂-(logπ·A)/∂σ = -A((a-μ)²-σ²)/σ³`. No division by π anywhere; norm hack removed.
  The displayed loss now uses a numerically stable closed-form `log π` instead of
  `Log(Probability(...))`.

### 2. Discrete optimizer trained the wrong network
- **Bug:** `optim_discrete = new StableAdamW(model.muNetwork.Parameters(), ...)` —
  copy-paste error; discrete VPG never updated `discreteNetwork`.
- **Fix:** constructed on `model.discreteNetwork.Parameters()`.

### 3. Always-true stochasticity guard
- **Bug:** `if (stoch != Fixed || stoch != Trainable)` is always true (`||` instead of
  `&&`), force-resetting user-selected stochasticity every run.
- **Fix:** `&&`.

### 4. KLE rollback cached the wrong network
- **Bug:** `disc_kle_cache = model.sigmaNetwork.Parameters()...` — the discrete rollback
  snapshot cached the sigma network.
- **Fix:** caches `discreteNetwork` parameters.

### 5. KLE old-probabilities never assigned (NullReference when earlyStopping ≠ Off)
- **Bug:** `cont_probs_old_kle` / `disc_probs_old_kle` were declared `null` and never set,
  so enabling early stopping crashed in `ComputeKLDivergence`.
- **Fix:** rollout-time probabilities are now split into minibatches (same as PPOTrainer)
  and assigned each minibatch iteration.

---

## SACTrainer.cs

### 1. Actor weight decay 0.01 (CRITICAL, fixed in March)
- **Bug:** `optim_mu` / `optim_sigma` were AdamW without `weight_decay:`, inheriting the
  0.01 default → actor parameters shrank ~26% per 100k steps. The critic correctly used 0.
- **Fix:** explicit `weight_decay: 0f` on both actor optimizers.

### 2. Hidden gradient clipping maxNorm=0.5 on the actor (CRITICAL, fixed in March)
- **Bug:** `ClipGradNorm(hp.maxNorm)` with the PPO default 0.5 was applied to mu/sigma —
  inherited from PPO defaults, hidden from the SAC inspector.
- **Fix:** removed; canonical SAC uses no gradient clipping.

### 3. LR scheduler total_iters counted env steps, not gradient steps (fixed in March)
- **Bug:** `total_iters = maxSteps` while `Scheduler.Step()` is called per gradient step →
  LR effectively never annealed.
- **Fix:** `total_iters = maxSteps * updatesNum / updateInterval` (matching DDPG).

### 4. Max-step timeout zeroed the bootstrap (CRITICAL for multi-step, fixed in March)
- **Bug:** maxStep truncation set `done=1`, so targets used `y = r` — teaching the critic
  that timeout states have zero future value.
- **Fix:** `truncated` flag added to `TimestepTuple`; targets use
  `bootstrap = 1 - done + truncated` (applied to SAC, TD3, DDPG).

### 5. Static target networks (fixed June 2026)
- **Bug:** `private static Sequential q1TargNetwork/q2TargNetwork` — shared across trainer
  instances; multiple SAC behaviours in one process would corrupt each other's targets.
- **Fix:** instance fields (TD3 fixed identically; DDPG was already correct).

### 6. Build-path save wrote wrong networks (fixed in March, AgentBehaviour.cs)
- **Bug:** Q1/Q2 saved from `vNetwork`, Sigma saved from `muNetwork` (`#if !UNITY_EDITOR`
  path only).
- **Fix:** correct networks serialized.

### Verified NOT bugs
All gradient formulas (tanh-Gaussian log-prob, reparameterized actor gradients, Q-target
math) were verified twice: finite-difference probes (March, cos ≈ 1.0) and an independent
GPU reimplementation (June) that behaves identically. The math is correct.

---

## TD3Trainer.cs

1. **Optimizer states ignored on resume** — `Initialize(optimizer_states)` always created
   fresh Adam optimizers, resetting momentum on every continued session.
   **Fix:** deserialize `[q1, q2, mu]` from disk when present (same pattern as SAC).
2. **Static target networks** — same class of bug as SAC. **Fix:** instance fields.
3. **Scheduler total_iters** — same env-steps/gradient-steps mismatch as SAC (fixed March).
4. **Critic not frozen during policy update** — the actor backward accumulated garbage
   weight-gradients in Q1 (harmless only because of ZeroGrad ordering).
   **Fix:** `q1Network.RequiresGrad = false` during `UpdatePolicy`, restored after.
5. **Truncation bootstrap** — fixed in March (see SAC #4).

---

## DDPGTrainer.cs

1. **Optimizer states ignored on resume** — same as TD3. **Fix:** deserialize `[q1, mu]`
   StableAdamW states when present.
2. **Truncation bootstrap** — fixed in March (see SAC #4).

---

## Shared infrastructure

1. **ParallelInference cache crash with >1 agent** (DeepUnityTrainer.cs, fixed in March) —
   warmup/noise modes return `probs = null`; the multi-agent batch path split them
   unconditionally and could leave the per-frame action cache half-built
   (`KeyNotFoundException`). Fixed with null handling + cache rebuild guards.
2. **Quaternion observation sign aliasing** (StateVector.cs, fixed in March) —
   `AddObservation(Quaternion)` stored raw components; the same physical orientation could
   appear as `q` and `-q` across episodes. Fixed by hemisphere canonicalization.
3. **SoftPlus deserialization** (Modules/Activations/SoftPlus.cs, fixed in March) — could
   reach `Predict()` with an invalid internal tensor after asset deserialization;
   hardened with state self-repair.
4. **Batch-runner stats lookup** (BalanceBallBatchPlayController.cs, June) — looked for
   `TrainingStatistics` only on its own GameObject while runners attach it to the agent;
   reward/loss CSVs were never written. Fixed with a scene-wide fallback.

---

## The BalanceBall "SAC doesn't work" mystery — final resolution (June 2026)

After all the code fixes above, SAC *still* plateaued at random-policy level (~85 frames
survival) on BalanceBall while PPO solved it in a minute. A from-scratch FullGPU SAC
(`FullGPU/SACGPUTrainer.cs` + `SACLossCS.compute`, selectable as `TrainerType.SACGPU`)
reproduced the plateau exactly — two independent implementations failing identically means
the trainer code was innocent. Ablations on the GPU trainer isolated three compounding
**setup** factors:

| Factor | Problem | Evidence |
|---|---|---|
| `alpha = 0.2` vs reward `0.025/step` | entropy term ~20× the task reward; the critic's value surface encodes entropy, not balance | J → 5.1 (> reward-only Q_max 2.5) at α=0.2; J sane at α=0.005 |
| no fall penalty | reward is `+0.025` even on the falling step; the only terminal signal is a missing bootstrap on ~1% of replay samples | fall penalty alone (dp=1) still flat |
| `decisionPeriod = 1` | each action tilts the platform ≤1°, so Q(s,a) is nearly flat in `a` — dQ/da is below the TD noise floor | **decisive**: dp=5 unlocked learning |

With `decisionPeriod=5`, a `-1` fall penalty (added to `BalanceBall.cs` as a diagnostic;
the field was later removed at the user's request to restore the original PPO-tuned reward —
note the verified converging runs all included it, so convergence WITHOUT the penalty at
dp=5 is plausible but untested)
and `alpha=0.005`: flat until ~30k decisions, then hockey stick — last-25 mean episode
length 1076 decisions (~5,400 physics frames), best episode hit the 10,000-decision
maxStep cap. Run: `ProbeLogs/balanceball_runtime_sacgpu_mlagents_parity_20260610_065406`.

The **CPU SACTrainer converges with the same config too** (run
`balanceball_runtime_saccpu_mlagents_parity`, 40k decisions): takeoff at ~30k decisions,
peak decile mean 559 decisions (~2,800 frames), best episode 7,731 decisions
(~38,600 frames), with typical post-takeoff SAC variance. Both implementations confirmed.

PPO never cared about any of the three factors because normalized advantages +
trajectory-level returns are insensitive to reward scale, terminal shaping, and
per-action effect size. This is why "PPO works, everything else doesn't" persisted for
months while the algorithms themselves were repeatedly (and correctly) cleared.

**Practical SAC/TD3/DDPG guidance for DeepUnity environments:**
- Scale `alpha` to the per-step reward magnitude (or scale rewards to ~1/step).
- Give survival tasks an explicit negative terminal reward.
- Use `decisionPeriod` ≥ 5 for fine-grained physics control (action repeat).
- Use `updateInterval = 1` (UTD ≈ 1); the old default of 50 undertrains 50×.
- Expect the hockey stick 10–50× later than PPO; don't judge a run at 5k steps.

---

## SACGPU (new, June 2026)

`TrainerType.SACGPU` — full-GPU SAC sibling of PPOGPU (forward, backward, AdamW, Polyak
and the whole SAC loss on compute shaders; MLP/LnMLP only). Verified on OneDimReach
(learns ≥ CPU SAC, ~3× faster wall-clock) and on BalanceBall (converges, see above).
Optimizer-state files are interchangeable with the CPU SACTrainer. Batch-mode runs need
`-batchmode` **without** `-nographics` (compute shaders require a graphics device).


## 2026-03-28 | Codex | Smoke-Bandit Root-Cause Update

### New evidence from DeepUnity-native smoke ablations
Used the real DeepUnity SAC stack on the repo's one-step contextual bandit:
- `Assets/DeepUnity/Tutorials/SacSmoke/Scripts/QuadraticBanditAgent.cs`
- runner: `Assets/DeepUnity/ReinforcementLearning/Tools/Editor/QuadraticBanditBatchRunner.cs`

Runs:
1. sparse baseline (`updateInterval=50`, `updatesNum=1`)
- report: `ProbeLogs/quadratic_bandit_baseline_20260328_164842/report.md`
- mean reward last 25: `0.5918`

2. dense updates (`updateInterval=1`, `updatesNum=1`)
- report: `ProbeLogs/quadratic_bandit_dense_updates_20260328_164930/report.md`
- mean reward last 25: `0.8772`

3. dense replay (`updateInterval=50`, `updatesNum=8`)
- report: `ProbeLogs/quadratic_bandit_dense_replay_20260328_165023/report.md`
- mean reward last 25: `0.6972`

### Interpretation
This is the strongest result so far.
- DeepUnity SAC is not globally broken. It can learn through the real Unity trainer stack.
- But the default off-policy update cadence is too sparse.
- Increasing replay reuse helps somewhat.
- Increasing update cadence helps a lot more.

For the tiny one-step bandit, the default SAC cadence is weak even on a trivial problem. That strongly supports the hypothesis that `updateInterval=50`, `updatesNum=1` is materially undertraining SAC on harder environments like BalanceBall.

### Current root-cause ranking after smoke ablations
1. off-policy update cadence / low effective UTD ratio
2. replay reuse policy and large chunk eviction amplifying weak cadence
3. scheduler/reporting issues as secondary diagnostics problems
4. local gradient math bugs: no longer primary suspect

### Additional confirmed trainer issues
- `SACTrainer` scheduler total-iters formula needed to be based on gradient steps, not raw env steps
- `TD3Trainer` had the same scheduler issue
- `updateIterations` under-reports actual optimization work when `updatesNum > 1` because it counts train cycles, not gradient epochs

### Practical implication
For SAC on BalanceBall, the first config to test is:
- `updateInterval = 1`
- or at minimum `updatesNum ˜ updateInterval`

Without that, the actor can plateau even while the critic fits.

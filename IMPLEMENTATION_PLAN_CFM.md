# Implementation Plan: Flow-Matching Latent Context Inference for CMDPs

This document maps the proposed algorithm in [cfmforcmdp.pdf](cfmforcmdp.pdf) (*"Latent Context Inference in Contextual Markov Decision Processes via Flow Matching"*, Jeong et al., 2026) onto the existing [VariBAD](1910.08348v2.pdf) codebase. The guiding principle is **additive**: every conflict with the current VariBAD pipeline is resolved by introducing a parallel module rather than editing existing code in place.

---

## 1. Algorithm Summary (Target)

The proposed method replaces VariBAD's recurrent variational encoder with a generative posterior over the latent context `c` built via **conditional flow matching (CFM)** and a **product-of-experts (PoE)** score composition.

Three learned components:

| Component | Symbol | Role |
| --- | --- | --- |
| Conditional flow network | `v_θ(c_τ, τ | x_i)` | Per-transition local-expert velocity field |
| Prior flow network | `v_φ(c_τ, τ)` | Unconditional marginal-prior velocity field |
| Context-conditioned policy | `π_ψ(a | s, ĉ)` | Standard actor-critic conditioned on a single sampled context `ĉ` |

Two interaction points with sampling:

- **Inference (rollout time)**: For each process, gather all transitions `x_{1:t}` of the current episode, convert each `v_θ(·| x_i)` to a per-transition score `s_θ(·| x_i)` via Eq. (3), and fuse via PoE (Eq. 5):
  `s_fused = Σ_i s_θ(·|x_i) − (t−1)·s_φ(·)`.
  Convert back to a velocity field and integrate one ODE from `c_0 ∼ N(0, I)` to obtain `ĉ ∼ p(c | x_{1:t})`. This `ĉ` replaces the `(latent_mean, latent_logvar)` that VariBAD currently passes to the policy.
- **Training (bootstrapped EM)**:
  - **E-step**: for each trajectory `k`, run the fused ODE once (gradients stopped) using all transitions to get a pseudo-target `ĉ^{(k)}`.
  - **M-step**: regress `v_θ(c_τ, τ | x_i^{(k)})` to `ĉ^{(k)} − c_0` via CFM loss (Eq. 6); regress `v_φ` to the marginal of `{ĉ^{(k)}}`.
  - Policy `π_ψ` is updated by RL on `ĉ`; gradients flow only through `ψ`.

---

## 2. High-Level Mapping to Repo

| Algorithm concept | Closest existing object | Strategy |
| --- | --- | --- |
| Recurrent encoder `q_φ(m | τ:t)` | [models/encoder.py](models/encoder.py) `RNNEncoder` | **Keep**. Add a parallel [models/flow_encoder.py](models/flow_encoder.py) for the new per-transition encoder. |
| VAE `compute_vae_loss` | [vae.py](vae.py) `VaribadVAE` | **Keep**. Add parallel [cfm.py](cfm.py) class `CFMContextInferer` mirroring its public surface. |
| `MetaLearner` orchestrator | [metalearner.py](metalearner.py) | **Keep**. Add parallel [cmdp_metalearner.py](cmdp_metalearner.py); wired in from `main.py` by a new `--env-type` suffix `_cfm`. |
| Latent passed to policy = `[μ; logvar]` (`2·latent_dim`) | [metalearner.py:110](metalearner.py#L110) | The flow model emits a single `ĉ` of `context_dim`. New metalearner sets `dim_latent=context_dim` when constructing the `Policy`. |
| OnlineStorage tracks `(latent_sample, latent_mean, latent_logvar)` | [algorithms/online_storage.py](algorithms/online_storage.py) | **Keep**. Add parallel [algorithms/online_storage_cmdp.py](algorithms/online_storage_cmdp.py) that stores `ĉ` only (and optionally the prior `c_0` and per-step transition cache). |
| `RolloutStorageVAE` (off-policy trajectory buffer) | [utils/storage_vae.py](utils/storage_vae.py) | **Reuse as-is** — already provides full `(prev_obs, next_obs, action, reward)` mini-batches of complete trajectories, which is exactly what the E-step needs. |
| `select_action`, `update_encoding`, `get_latent_for_policy` | [utils/helpers.py](utils/helpers.py) | **Keep**. Add parallel [utils/helpers_cmdp.py](utils/helpers_cmdp.py) with `update_context_via_flow`, `get_context_for_policy`. |
| Per-env config files | [config/](config/) | **Add** new `args_*_cfm.py` per benchmark (gridworld, pointrobot, cheetah_vel, ...). Reuse existing args where possible (`+= include`). |
| Environments | [environments/](environments/) | **Reuse** existing CMDP-shaped envs (`HalfCheetahVelEnv`, `AntDirEnv`, `Walker2DRandParamsEnv`, ...). **Add** a `rand_param`-style benchmark for explicit mass/friction/actuator-gain randomization where missing (see §7). |
| `main.py` entry | [main.py](main.py) | Append new `elif env == '<name>_cfm':` branches **at the end** of the existing chain. Do not delete existing ones. This is the *only* edit to a non-new file; see §8. |

---

## 3. New Files (Detailed)

### 3.1 `models/flow_encoder.py`

Per-transition feature encoder used inside both `v_θ` and the pseudo-target pipeline. Pure MLP — no recurrence.

```python
class TransitionEncoder(nn.Module):
    """Encodes a single transition x_i = (s_i, a_i, s_{i+1}, r_i) to a fixed vector."""
    def __init__(self, args, state_dim, action_dim, embed_dim):
        # Mirrors the FeatureExtractors used in models/encoder.py:36-38, but no GRU.
        # Output: (B, embed_dim)
    def forward(self, prev_state, action, next_state, reward) -> torch.Tensor: ...
```

Rationale: the proposed method drops the RNN. Keep this module local to flow inference so [models/encoder.py](models/encoder.py) is untouched.

### 3.2 `models/flow_network.py`

Two MLP classes plus utilities. All learned weights live here.

```python
def sinusoidal_time_embedding(tau: torch.Tensor, dim: int) -> torch.Tensor: ...

class ConditionalFlow(nn.Module):
    """v_θ(c_τ, τ | x). MLP. Input: cat(c_τ, time_emb(τ), x_enc). Output: velocity in R^context_dim."""
    def __init__(self, context_dim, transition_embed_dim, hidden_layers, time_embed_dim): ...
    def forward(self, c_tau, tau, x_enc) -> torch.Tensor: ...

class PriorFlow(nn.Module):
    """v_φ(c_τ, τ). Unconditional. Same shape signature but no x_enc."""
    def __init__(self, context_dim, hidden_layers, time_embed_dim): ...
    def forward(self, c_tau, tau) -> torch.Tensor: ...

# OT-interpolant score conversion (paper Eq. 3 and inverse).
def velocity_to_score(v, c_tau, tau): ...   # s = -(c_τ - τ·v) / (1 - τ)^2
def score_to_velocity(s, c_tau, tau): ...   # v = (c_τ + s·(1-τ)^2) / τ
```

### 3.3 `models/flow_inference.py`

The PoE composition and ODE integrator. Stateless — operates on the two networks above.

```python
def fused_velocity(c_tau, tau, x_enc_per_transition, conditional_flow, prior_flow):
    """Compute s_fused per Eq. 5 then convert back to a single velocity."""
    # x_enc_per_transition: shape (T, B, embed_dim) — T transitions per process

def integrate_ode(conditional_flow, prior_flow, x_enc, num_steps, device):
    """Euler/Heun integrator from τ=0 to τ=1.
    Returns ĉ of shape (B, context_dim). Used at inference and (with no_grad) for E-step pseudo-targets.
    """

def cfm_loss(conditional_flow, x_enc, c_target, num_tau_samples):
    """Eq. 6: regress v_θ(c_τ, τ | x_i) → (c_target − c_0). c_target is detached."""

def prior_cfm_loss(prior_flow, c_target, num_tau_samples):
    """Marginal CFM regression for v_φ."""
```

Notes:
- The integrator must support a configurable step count (`--cfm_ode_steps`) — this is one of the ablations in §4 of the proposal.
- A single ODE integration produces *one* sample `ĉ`. Multi-sample variants (for variance reduction or uncertainty visualization) are optional and can be added later as a flag.

### 3.4 `cfm.py` (parallel to `vae.py`)

Container class that owns the flow modules, their optimizers, the off-policy buffer hookup, and exposes a `compute_cfm_loss(update=True)` callable shaped like `VaribadVAE.compute_vae_loss`. This is the **single biggest new file** and is the natural drop-in for the M-step.

Suggested skeleton:

```python
class CFMContextInferer:
    def __init__(self, args, logger, get_iter_idx):
        self.transition_encoder = TransitionEncoder(...).to(device)
        self.conditional_flow  = ConditionalFlow(...).to(device)
        self.prior_flow        = PriorFlow(...).to(device)

        # Reuse RolloutStorageVAE — same trajectory-level buffer is what we need.
        self.rollout_storage = RolloutStorageVAE(...)

        self.optimiser_flow  = torch.optim.Adam(
            [*self.transition_encoder.parameters(),
             *self.conditional_flow.parameters()], lr=args.lr_cfm)
        self.optimiser_prior = torch.optim.Adam(self.prior_flow.parameters(), lr=args.lr_cfm_prior)

    @torch.no_grad()
    def infer_context(self, prev_obs, action, next_obs, reward):
        """Online posterior used at rollout time.
        prev_obs/...: (T, B, *) tensors covering the current episode so far.
        Returns ĉ of shape (B, context_dim).
        """
        x_enc = self.transition_encoder(prev_obs, action, next_obs, reward)
        return integrate_ode(self.conditional_flow, self.prior_flow, x_enc, ...)

    def compute_cfm_loss(self, update=True, pretrain_index=None):
        """M-step.
        1. Sample a mini-batch of full trajectories from rollout_storage.
        2. E-step: per trajectory, run integrate_ode with no_grad → ĉ^{(k)}.
        3. M-step:
              loss_flow  = cfm_loss(conditional_flow, x_enc, ĉ^{(k)})
              loss_prior = prior_cfm_loss(prior_flow, ĉ^{(k)})
              (loss_flow * α + loss_prior * α').backward()
        4. Log to tb_logger like VaribadVAE.log.
        """
```

Public surface intentionally matches `VaribadVAE` so that [algorithms/ppo.py](algorithms/ppo.py) and [algorithms/a2c.py](algorithms/a2c.py) can call it through the existing `compute_vae_loss` plumbing — see §6.

### 3.5 `cmdp_metalearner.py` (parallel to `metalearner.py`)

Drop-in orchestrator class `CMDPMetaLearner`. Differs from `MetaLearner` in three concrete places:

1. **Policy init dimension** — pass `dim_latent=args.context_dim` (not `latent_dim * 2`) when constructing `Policy`.
2. **Per-step encoding** — replace `utl.update_encoding(...)` with `utl_cmdp.update_context_via_flow(...)`. The new helper holds a *transition cache* per process and re-runs `infer_context` after each environment step.
3. **VAE call** — replace `self.vae = VaribadVAE(...)` with `self.cfm = CFMContextInferer(...)`. `policy.update(..., compute_vae_loss=self.cfm.compute_cfm_loss)` keeps the rest of the loop intact.

Keep the existing `log` / `save` plumbing; just rename the saved artifacts (`conditional_flow.pt`, `prior_flow.pt`, `transition_encoder.pt`) inside this file's `log()`.

### 3.6 `algorithms/online_storage_cmdp.py`

Mirror of [algorithms/online_storage.py](algorithms/online_storage.py) with the latent triple `(latent_sample, latent_mean, latent_logvar)` collapsed to a single `context_samples` tensor list. Also store a running cache of `(prev_state, action, next_state, reward)` per process, since the flow inference needs the *entire current episode* (not just a hidden state) to produce `ĉ` at each step.

Minimum-diff alternative: instead of a new file, keep `OnlineStorage` and dump `ĉ` into the `latent_sample` slot while leaving `latent_mean`/`latent_logvar` as zeros. This works but is fragile because PPO's minibatch path detaches and concatenates those tensors ([algorithms/ppo.py](algorithms/ppo.py) feed-forward generator). The cleanest solution is the parallel storage.

### 3.7 `utils/helpers_cmdp.py`

Two functions, no other changes:

```python
def update_context_via_flow(cfm, transition_cache, prev_state, action, next_state, reward, done):
    """Append the new transition to per-process caches, reset on done,
    then call cfm.infer_context(...) to produce the updated ĉ.
    Returns (context_sample, transition_cache)."""

def get_context_for_policy(args, context_sample):
    """Trivial passthrough (or with optional nonlinearity); mirrors
    helpers.get_latent_for_policy but no mean/logvar concatenation."""
```

### 3.8 Config files (one per benchmark)

Add files of the form `config/<env>/args_<env>_cfm.py`. Each file is largely a copy of the corresponding `args_<env>_varibad.py` with:

- **New args** (CMDP-specific):
  - `--context_dim` (replaces `--latent_dim`; default 5–8)
  - `--cfm_ode_steps` (default 20; ablate in {5, 10, 20, 50})
  - `--cfm_time_embed_dim` (default 32)
  - `--cfm_conditional_hidden` (list, default [128, 128, 128])
  - `--cfm_prior_hidden` (list, default [128, 128])
  - `--cfm_history_window` (max number of transitions composed by PoE; default = max episode length; ablate)
  - `--cfm_alpha`, `--cfm_alpha_prior` (loss weights `α`, `α'` from Eq. 7)
  - `--cfm_pseudo_target_detach` (bool, default `True`)
  - `--lr_cfm`, `--lr_cfm_prior`
  - `--num_cfm_updates` (analogue of `num_vae_updates`)
- **Removed/ignored args**: VAE-specific ones (`--decode_reward`, `--decode_state`, `--decode_task`, `--kl_weight`, `--rew_loss_coeff`, etc.) — leave the args defined for compatibility, but the new code path ignores them.

Recommended first benchmark: `config/gridworld/args_grid_cfm.py` (smallest, fastest iteration; matches the Bayes-optimal gridworld from VariBAD §4.1).

### 3.9 (Optional) `environments/rand_dynamics/`

The proposal's Stage 1 calls for "controlled CMDP benchmarks with hidden physical parameters such as mass, friction, or actuator gain." Existing repo coverage:

- **Already present**: `Walker2DRandParamsEnv`, `HopperRandParamsEnv` (under `environments/mujoco/rand_param_envs/`) randomize body mass, body inertia, dof damping, geom friction. These should be sufficient for the headline experiment.
- **Optional new env**: a half-cheetah variant with explicit (mass, friction, actuator gain) randomization for ablations on context dimensionality. If added, place it in `environments/rand_dynamics/half_cheetah_rand.py` and register it in [environments/__init__.py](environments/__init__.py) under a new id `HalfCheetahRandDyn-v0`. Do not modify existing envs.

### 3.10 (Optional) Diagnostic baseline files

For Stage 3 of the plan, a *supervised CFM* variant (same architecture, but trained against the true context label) is wanted as an upper-bound diagnostic. Add this as `cfm_supervised.py` (a thin subclass of `CFMContextInferer` that replaces `ĉ^{(k)}` with the ground-truth task from `RolloutStorageVAE` — which the storage already stashes in its `tasks` field; see [utils/storage_vae.py](utils/storage_vae.py)).

---

## 4. Existing Files — Minimal Edits

Only one existing file is touched, and the edits are purely additive:

### 4.1 `main.py`

Append (do not replace) new branches in the `elif` chain:

```python
elif env == 'gridworld_cfm':
    args = args_grid_cfm.get_args(rest_args)
# ... one per benchmark you intend to run
```

And dispatch to the new orchestrator instead of `MetaLearner` when the env name ends in `_cfm`:

```python
if env.endswith('_cfm'):
    learner = CMDPMetaLearner(args)
elif args.disable_metalearner:
    learner = Learner(args)
else:
    learner = MetaLearner(args)
```

This is the only edit required to existing code. `MetaLearner` / `Learner` paths remain bit-identical for users of the original `*_varibad` / `*_rl2` env types.

> **Everything else** — `vae.py`, `metalearner.py`, `models/encoder.py`, `models/decoder.py`, `models/policy.py`, `algorithms/{ppo,a2c,online_storage}.py`, `utils/helpers.py`, `utils/storage_vae.py` — **stays untouched**.

---

## 5. Bootstrapped EM Training Loop (Where It Lives)

The training loop fits inside `CMDPMetaLearner.train()` and reuses VariBAD's existing scaffolding:

```
for iter in num_updates:
    # --- rollout (E-step happens implicitly inside the rollout)
    for step in policy_num_steps:
        action ← π_ψ(state, ĉ_t)
        next_state, reward, done ← env.step(action)
        push (s,a,s',r) into RolloutStorageVAE
        ĉ_{t+1} ← cfm.infer_context(full episode so far)   ← E-step at rollout
        push ĉ_{t+1} into online_storage_cmdp

    # --- updates
    if precollect satisfied:
        for _ in num_cfm_updates:
            cfm.compute_cfm_loss(update=True)              ← M-step: v_θ + v_φ
        policy.update(online_storage, encoder=None,
                      rlloss_through_encoder=False,
                      compute_vae_loss=cfm.compute_cfm_loss)
```

Key design choices the code must respect:

- **`rlloss_through_encoder` must be `False`** for the CFM path. The flow network is not differentiable through the ODE integration in a numerically stable way at training scale, and the proposal explicitly states that `∇_θ L_RL = 0` (paper Eq. 7: gradients from `L_RL` only flow through `ψ`).
- **`compute_vae_loss=cfm.compute_cfm_loss`** allows reusing the A2C / PPO scheduling of VAE-style updates with zero changes to those files. PPO's path that calls it only when `rlloss_through_encoder=True` ([algorithms/ppo.py:147](algorithms/ppo.py#L147)) is irrelevant here; we use A2C's separate-call path ([algorithms/a2c.py:124](algorithms/a2c.py#L124)) or simply call `cfm.compute_cfm_loss` directly in the metalearner outer loop.
- **Pseudo-target stop-gradient**: implemented inside `cfm.compute_cfm_loss`'s E-step by wrapping the ODE integration in `torch.no_grad()` and `.detach()`-ing the result before it enters Eq. 6.

---

## 6. Interface Compatibility Notes

A few specific mismatches between the new module and the existing policy/storage path are worth calling out because they will cause silent shape errors if missed:

1. **Policy `dim_latent`**: [metalearner.py:110](metalearner.py#L110) currently passes `latent_dim * 2` because the VariBAD policy receives concatenated `(μ, logvar)`. The CFM policy receives a single sample `ĉ` of size `context_dim`, so the new metalearner must pass `dim_latent=args.context_dim`.
2. **`get_latent_for_policy`**: in [utils/helpers.py:108](utils/helpers.py#L108) this concatenates `(latent_mean, latent_logvar)` when `args.sample_embeddings` is false. `helpers_cmdp.get_context_for_policy` must *not* do this — it is a passthrough.
3. **`OnlineStorage.latent_*` lists**: PPO's minibatch generator reads three separate lists. If reusing `OnlineStorage`, the cleanest hack is to set `latent_mean = ĉ`, `latent_sample = ĉ`, `latent_logvar = zeros_like(ĉ)`, and patch `get_latent_for_policy` to detect zero logvar and skip the concat. Cleaner long-term: `online_storage_cmdp.py`.
4. **`encoder.reset_hidden(done)`**: the recurrent encoder zeros its hidden state on `done`. The flow path has no hidden state, but the *transition cache* must be reset on `done`. This is handled inside `update_context_via_flow`.
5. **VAE buffer reuse**: `RolloutStorageVAE` happily takes `task=None` ([utils/storage_vae.py:61](utils/storage_vae.py#L61)) and stores the trajectory regardless; no changes needed to use it as the source for the E-step.

---

## 7. Benchmarks (Stage 1 of the Plan)

Recommended sequence, smallest to largest:

| Order | Benchmark | Source | What it tests |
| --- | --- | --- | --- |
| 1 | `Gridworld-v0` (cell goal) | already registered | Posterior multimodality (multiple cells still plausible) — primary motivation for replacing Gaussian posteriors. |
| 2 | `PointEnvWind` / `PointEnv` (2D) | [environments/navigation/](environments/navigation/) | Continuous low-dim context, sanity check ODE integration. |
| 3 | `HalfCheetahVel-v0` | existing | Standard meta-RL benchmark for matching VariBAD baseline. |
| 4 | `Walker2DRandParams-v0` / `HopperRandParams-v0` | existing | Hidden mass/friction — the headline CMDP scenario from §1 of the proposal. |
| 5 | (optional) `HalfCheetahRandDyn-v0` | new, see §3.9 | Higher-dim hidden physical parameter vector for context-dim ablation. |

---

## 8. Final File-Change Manifest

**New files (add)**
- `models/flow_encoder.py`
- `models/flow_network.py`
- `models/flow_inference.py`
- `cfm.py`
- `cmdp_metalearner.py`
- `algorithms/online_storage_cmdp.py`
- `utils/helpers_cmdp.py`
- `config/gridworld/args_grid_cfm.py`
- `config/pointrobot/args_pointrobot_cfm.py`
- `config/mujoco/args_cheetah_vel_cfm.py`
- `config/mujoco/args_walker_cfm.py` *(and one per additional benchmark you run)*
- *(optional)* `cfm_supervised.py`
- *(optional)* `environments/rand_dynamics/half_cheetah_rand.py`

**Existing files (edit)**
- `main.py` — append new `elif` branches and route to `CMDPMetaLearner`. **No code is removed.**

**Existing files left untouched**
- `vae.py`, `metalearner.py`, `learner.py`
- `models/encoder.py`, `models/decoder.py`, `models/policy.py`
- `algorithms/a2c.py`, `algorithms/ppo.py`, `algorithms/online_storage.py`
- `utils/helpers.py`, `utils/storage_vae.py`, `utils/evaluation.py`, `utils/tb_logger.py`
- everything under `config/*_varibad.py`, `config/*_rl2.py`, etc.
- everything under `environments/` (unless adding the optional new env in §3.9)

---

## 9. Ablation Hooks (Stage 4)

The proposal's ablations are easiest if the following are wired as CLI flags from the start (all read inside `cfm.py` / `flow_inference.py`):

- `--cfm_disable_score_composition` — fall back to using only the most recent transition's expert (no PoE).
- `--cfm_disable_prior_correction` — drop the `−(t−1)·s_φ` term in Eq. 5; treat the marginal as uniform.
- `--cfm_history_window` — cap PoE to the last `k` transitions; ablate.
- `--cfm_ode_steps` — integrator step count.
- `--cfm_supervised` — use ground-truth task instead of pseudo-targets (diagnostic upper bound; routes through `cfm_supervised.py`).

Each maps to a single conditional inside `compute_cfm_loss` or `integrate_ode` — no further file additions required.

"""
T-MAZE environment sanity checks (TMAZE_guideline_v6.md §6-1).

Three required checks plus one bonus wrapper-compatibility check. Run with:

    python -m environments.navigation.tmaze_sanity_check

Exits with code 0 on all pass, non-zero otherwise. Verifies env semantics
only — does NOT exercise the training loop. Algorithm-side training is the
team lead's responsibility (guideline §2).
"""

import sys
import numpy as np

from environments.navigation.tmaze import TMazeEnv


# ---------------------------------------------------------------------------
# Test 1 — random policy return distribution (v6 §6-1.1)
# ---------------------------------------------------------------------------

def test_random_policy_return(corridor_length=10, num_episodes=100, seed=0):
    """Random uniform-over-4-actions policy should yield mean return strictly
    between worst (−1.0) and Markovian (0.5).

    Note: v6 §6-1 mentions "보통 −0.5 ~ 0 부근" as informal expectation, but
    a fully uniform 4-action policy on T-MAZE is harsh: the probability of
    staying on-pace at step t is (1/4)^t, so almost every step incurs the
    intermediate penalty. Empirically the mean lands close to −1.0, not −0.5.
    The hard assertion below uses the wider v6 bound (−1.0, 0.5).
    """
    env = TMazeEnv(corridor_length=corridor_length, mode='passive')
    env.seed(seed)

    returns = []
    for _ in range(num_episodes):
        env.reset_task()
        env.reset()
        done = False
        G = 0.0
        while not done:
            action = env.action_space.sample()
            _, r, done, _ = env.step(action)
            G += r
        returns.append(G)

    mean = float(np.mean(returns))
    mn, mx = float(np.min(returns)), float(np.max(returns))
    print(f"[1] random policy: mean={mean:+.3f}, min={mn:+.3f}, max={mx:+.3f} "
          f"(n={num_episodes}, L={corridor_length})")
    assert -1.0 <= mean < 0.5, \
        f"mean return {mean} not in (-1.0, 0.5) per v6 §6-1"
    return mean


# ---------------------------------------------------------------------------
# Test 2 — manual trajectories with hand-computed returns (v6 §6-1.2)
# ---------------------------------------------------------------------------

def test_manual_trajectories(corridor_length=10):
    """Two pre-defined action sequences against hand-calculated returns.

    Sequence A (optimal, goal_y=+1):
        actions = [right] * L + [up]
        every step on pace (r=0) until terminal step, where y=+1 == goal_y
        → return = +1.0

    Sequence B (one wall-bump at step 2, goal_y=+1):
        actions = [right, up, right * (L-2), up]    # total T = L+1 actions
        t=1: right → x=1, on-pace, r=0
        t=2: up at (1,0) → wall, stay at (1,0), x=1 < 2, r=-1/L
        t=3..L: right each, x lags by 1 → x<t, r=-1/L each (L-2 steps)
        t=T=L+1: up at (L-1, 0) → wall (y=±1 only at x=L), stay, y=0 != goal_y
        → return = 0 + (L-1) * (-1/L) + 0 = -(L-1)/L
        For L=10 → -0.9
    """
    L = corridor_length
    T = L + 1
    expected_A = 1.0
    expected_B = -(L - 1) / L

    env = TMazeEnv(corridor_length=L, mode='passive')
    env.seed(0)
    env.reset_task(task=np.array([+1.0]))  # force goal_y = +1

    # Sequence A
    env.reset()
    seq_A = [0] * L + [1]  # right ×L, up
    assert len(seq_A) == T
    G_A, rewards_A = 0.0, []
    for a in seq_A:
        _, r, done, _ = env.step(a)
        G_A += r
        rewards_A.append(r)
    print(f"[2A] optimal (L={L}): return={G_A:+.4f} (expected {expected_A:+.4f}), "
          f"rewards={['{:+.2f}'.format(r) for r in rewards_A]}")
    assert done, "sequence A did not terminate after T steps"
    assert abs(G_A - expected_A) < 1e-9, \
        f"sequence A return {G_A} != expected {expected_A}"

    # Sequence B
    env.reset_task(task=np.array([+1.0]))
    env.reset()
    seq_B = [0, 1] + [0] * (L - 2) + [1]  # right, up(wall), right ×(L-2), up(wall)
    assert len(seq_B) == T
    G_B, rewards_B = 0.0, []
    for a in seq_B:
        _, r, done, _ = env.step(a)
        G_B += r
        rewards_B.append(r)
    print(f"[2B] 1-step delay (L={L}): return={G_B:+.4f} (expected {expected_B:+.4f}), "
          f"rewards={['{:+.2f}'.format(r) for r in rewards_B]}")
    assert done, "sequence B did not terminate after T steps"
    assert abs(G_B - expected_B) < 1e-9, \
        f"sequence B return {G_B} != expected {expected_B}"


# ---------------------------------------------------------------------------
# Test 3 — context exposure differentiation (v6 §6-1.3)
# ---------------------------------------------------------------------------

def test_context_exposure(corridor_length=10):
    """Forcing goal_y=+1 vs goal_y=-1 must change position-0 observation but
    not change observations elsewhere in the corridor."""
    env = TMazeEnv(corridor_length=corridor_length, mode='passive')

    # Run UP
    env.seed(42)
    env.reset_task(task=np.array([+1.0]))
    obs_up_start = env.reset()
    obs_up_mid = []
    for a in [0, 0, 0]:  # right ×3 → x=3
        o, _, _, _ = env.step(a)
        obs_up_mid.append(o.copy())

    # Run DOWN with the same seed
    env.seed(42)
    env.reset_task(task=np.array([-1.0]))
    obs_dn_start = env.reset()
    obs_dn_mid = []
    for a in [0, 0, 0]:
        o, _, _, _ = env.step(a)
        obs_dn_mid.append(o.copy())

    print(f"[3] oracle obs UP={obs_up_start.tolist()} vs DOWN={obs_dn_start.tolist()}")
    print(f"    corridor (x=1..3) UP={[o.tolist() for o in obs_up_mid]}")
    print(f"    corridor (x=1..3) DN={[o.tolist() for o in obs_dn_mid]}")

    assert not np.allclose(obs_up_start, obs_dn_start), \
        f"oracle obs failed to differentiate context: UP={obs_up_start}, DN={obs_dn_start}"
    for i, (ou, od) in enumerate(zip(obs_up_mid, obs_dn_mid)):
        assert np.allclose(ou, od), \
            f"corridor obs at step {i+1} leaked context: UP={ou}, DN={od}"


# ---------------------------------------------------------------------------
# Test 4 (bonus) — VariBadWrapper compatibility
# ---------------------------------------------------------------------------

def test_wrapper_compatibility(corridor_length=10):
    """End-to-end smoke through gym.make + VariBadWrapper. Catches obs
    shape / attribute / done-handling bugs before they show up in training.
    """
    import gym
    import environments  # noqa: F401 — triggers gym register side effects
    from environments.wrappers import VariBadWrapper

    raw = gym.make('TMaze-passive-v0', corridor_length=corridor_length, mode='passive')
    env = VariBadWrapper(env=raw, episodes_per_task=1)

    state = env.reset()
    assert state.shape == (2,), f"unexpected wrapped obs shape {state.shape}"

    T = corridor_length + 1
    steps = 0
    for _ in range(T):
        state, r, done, info = env.step(env.action_space.sample())
        steps += 1
        if done:
            break
    print(f"[4] wrapper compat: ran {steps}/{T} steps, last obs shape={state.shape}, done={done}")
    assert done, "wrapper did not signal done within T steps"
    assert 'task' in info, "info missing 'task' key"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("=== T-MAZE passive sanity checks (guideline v6 §6-1) ===\n")
    failures = []
    for name, fn in [
        ("random_policy_return",    test_random_policy_return),
        ("manual_trajectories",     test_manual_trajectories),
        ("context_exposure",        test_context_exposure),
        ("wrapper_compatibility",   test_wrapper_compatibility),
    ]:
        try:
            fn()
            print(f"PASS: {name}\n")
        except AssertionError as e:
            print(f"FAIL: {name} — {e}\n")
            failures.append(name)
        except Exception as e:
            print(f"ERROR: {name} — {type(e).__name__}: {e}\n")
            failures.append(name)
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == '__main__':
    main()

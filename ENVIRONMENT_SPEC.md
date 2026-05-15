# T-MAZE Environment Specification

본 문서는 VariBAD repo와 PEARL repo에 동일하게 두는 T-MAZE 환경 사양 문서다. 두 repo의 환경 클래스가 따라야 할 수학적·인터페이스적 약속, 각 repo에서 다르게 가는 deviation, 동치성 검증 protocol을 정리한다. 가이드라인 출처: `TMAZE_guideline_v6.md` §7-5.

**상태**: Sprint 1 완료 (VariBAD passive). Sprint 2/3에서 추가 갱신 예정.

---

## 1. 환경 개요

Ni et al. 2023 (arXiv:2307.03864) Section 3.2의 T-MAZE를 그대로 차용. CMDP에서 latent context posterior 추론 기법의 sanity testbed.

- Hidden context `c` = `goal_y ∈ {-1, +1}` (binary, episode 시작 시 uniform sample).
- 단일 정보원: `c`는 position 0의 oracle observation에 노출.
- Long-horizon memory가 필요한 minimal 환경.

본 환경은 milestone report 단계의 **빠른 iteration용 testbed**. Main claim의 본격 검증은 후속 continuous-context (mass/friction 등 hidden physical parameter) 환경에서 별도 수행.

---

## 2. 환경 정의 (두 repo 공통, hard-coded)

### 2.1 인자 (외부 노출)

| 인자 | 타입 | Default | 비고 |
|---|---|---|---|
| `corridor_length` (L) | int >= 1 | 10 | Ablation 대상 |
| `mode` | str | `'passive'` | `'active'`는 Sprint 2 |
| `seed` | int 또는 None | None | gym 표준 `seed()` 메서드 |

### 2.2 격자 구조

- Corridor: y=0, x ∈ [0, L]. 총 L+1개 cell.
- Junction: (L, 0).
- Goal arms: (L, +1) (UP), (L, -1) (DOWN).
- 그 외 위치는 wall. 벽을 향한 action은 stay (position 변화 없음).

### 2.3 Horizon T

| Mode | T | 구성 |
|---|---|---|
| Passive | `T = L + 1` | L step 오른쪽 + 1 step up/down |
| Active | `T = L + 3` | 1 left (oracle) + 1 right (return) + L right + 1 up/down |

**Active T 결정 노트** — Ni et al. 2023 본문은 "L = T-2"로 적어 T=L+2를 시사하지만, Memory-RL 공식 코드는 `episode_length = corridor_length + 2*oracle_length + 1` = L+3 (oracle_length=1). 본 spec은 **Memory-RL 공식 코드를 진실의 소스로 채택**. 가이드라인 v5→v6 업데이트 시 결정. Reward 식 cross-check도 코드를 따라왔으므로 일관적.

### 2.4 Reward 함수 (Ni et al. 2023 §3.2)

매 step `t = 1, ..., T-1`:
```
R_t = (𝟙(x_{t+1} >= t) - 1) / (T - 1)
```
즉 `x_{t+1} >= t`면 0, 아니면 `-1/(T-1)`. `x_{t+1}`은 step `t`의 action 직후의 horizontal position.

마지막 step `t = T`:
```
R_T = 𝟙(correct goal arm 도달) = 𝟙(x_T = L AND y_T = goal_y)
```

**Return range**: `[-1, +1]`.
- Optimal: `+1.0` (매 step 오른쪽 + 마지막에 정답 arm).
- Markovian (oracle 기억 없음, 항상 오른쪽 + 끝에서 random up/down): `0.5`.
- Worst: `-1.0` (시작부터 페이스 잃고 회복 못 함).

### 2.5 Observation 인코딩 (Memory-RL `ambiguous_position=True` 채택)

`shape=(2,)`, `dtype=float32`, range `[-1, +1]`:

| 위치 | obs | 비고 |
|---|---|---|
| x=0 (oracle), 첫 방문 | `[0, goal_y]` | exposure 노출 |
| x=0, 두 번째 이후 방문 | `[0, 0]` | exposure 0 |
| 0 < x < L (corridor middle) | `[0, 0]` | uninformative |
| x = L, y = 0 (junction) | `[1, 0]` | |
| x = L, y = ±1 (goal arm) | `[1, ±1]` | |

`add_timestep=False`, oracle 첫 방문 추적은 `self.oracle_visited` flag로.

### 2.6 Seed semantics

두 repo 모두 gym 표준 `self.np_random = gym.utils.seeding.np_random(seed)` 사용. `goal_y` 결정은 `self.np_random.choice([-1, +1])`. 같은 외부 seed → 같은 `goal_y` 시퀀스 (의미적 동치성 검증의 기반).

---

## 3. Repo별 Deviation (가이드라인 v6 §7-1)

두 repo의 인터페이스 가정이 다르므로 강제로 동일 클래스로 만들지 않는다. 각 repo의 자연스러운 표현을 따라가고 deviation을 여기 명시한다.

### 3.1 Action interface

| Repo | Action space | 매핑 |
|---|---|---|
| **VariBAD** | `Discrete(4)` | 0=right, 1=up, 2=down, 3=left |
| **PEARL** (예정) | `Box(low=-1, high=1, shape=(4,))` | 내부에서 `idx = int(np.argmax(action))`로 위 4개 매핑 적용 |

PEARL은 `TanhGaussianPolicy` + `NormalizedBoxEnv` 사용이라 continuous action이 강제. 환경 내부 argmax discretization은 학습 dynamics 관점에서 부자연스럽지만 T-MAZE 단계에서 수용. Main claim 검증의 후속 continuous-context 환경에서는 모든 repo가 자연스러운 continuous action을 사용하므로 이 deviation은 사라진다.

**Action index 순서 주의**: 가이드라인 v6 §7-1 본문이 authoritative source. "Memory-RL과 동일 순서"라는 괄호는 인덱스 순서가 아닌 "stay 없이 4방향" 의미. Memory-RL 코드의 실제 매핑은 (right/up/left/down)으로 인덱스 2, 3이 swap돼 있음. 우리는 v6 본문 따름.

### 3.2 Task interface

| Repo | Reset signature | Task 표현 | 분포 |
|---|---|---|---|
| **VariBAD** | `reset_task(task=None)` | `np.array([goal_y], dtype=float32)`, `task_dim=1` | None이면 uniform sample, 명시 시 그 값 |
| **PEARL** (예정) | `reset_task(idx)` | `idx ∈ {0, 1}` → `goal_y = [-1, +1][idx]` | `__init__`에서 `goals=[-1, +1]` pre-generate, `n_tasks=2` 강제 (assert) |

`get_task()` (VariBAD) / `get_all_task_idx()` (PEARL) 등 repo별 관례는 각 클래스에서 따른다.

---

## 4. 운영상 제약 (Operational Constraints)

### 4.1 PEARL train/eval split의 의미 약화

PEARL은 원래 task 100개를 80:20으로 split해 generalization 측정. T-MAZE는 task 2개뿐이라 이 split이 의미를 잃음. T-MAZE는 milestone testbed이고 generalization 측정은 main claim 검증의 본격 환경에서 하므로 fatal하지 않다. PEARL config에서 어떻게 처리할지는 강동환(PEARL 알고리즘 담당) 영역. 본 환경 클래스는 두 idx 모두 받게만 보장.

### 4.2 Action discretization (PEARL)

PEARL의 Box(4) action을 환경 내부 argmax로 4개 discrete action으로 매핑. Continuous policy의 출력 분포 모양이 학습 신호로 작용하지만 환경 dynamics는 discrete로 양자화됨. T-MAZE에서만 발생하는 인공성으로, 후속 continuous-context 환경에서는 사라진다.

### 4.3 `max_rollouts_per_task` (VariBAD)

T-MAZE는 binary context가 1 episode에 완전히 노출되는 환경. `max_rollouts_per_task=1`이 본 환경의 본질에 부합 (BAMDP의 multi-rollout 의미가 약함). VariBAD config default를 1로 둠. 알고리즘 담당자가 다른 값을 원하면 변경 가능.

### 4.4 L ablation 노출 방식 (VariBAD)

VariBAD `metalearner.py`가 `make_vec_envs`에 env-specific kwargs를 통과시키지 않음. 본 환경은 단일 id `TMaze-passive-v0`로 register하고, `main.py`의 `tmaze_varibad` 분기에서 `args.corridor_length`를 사용해 register kwargs를 **re-register**로 갱신. 이 패턴이 gym 0.21에서 동작 확인됨 (gym 0.22+에서 `registry.env_specs` API가 변경됐으니 의존성 업그레이드 시 갱신 필요).

---

## 5. 동치성 검증 (Sprint 3)

**byte-level 일치는 요구하지 않음**. 의미적 동치성만 검증.

### 5.1 검증 형태

같은 seed로 두 repo의 환경을 생성하고, 다음을 비교:

1. `goal_y` 시퀀스가 같은지 (seed → context 결정성).
2. 같은 "semantic action sequence" (예: `[right, right, ..., up]`)에 대해 같은 reward sequence가 나오는지.
3. 같은 done sequence가 나오는지.

VariBAD에서는 semantic action을 Discrete index 그대로, PEARL에서는 그 index에서 argmax가 나오는 one-hot vector (예: `right`은 `[1, 0, 0, 0]`)로 변환해 입력한다.

### 5.2 통과 기준

100 episode 정도 돌려서 모든 episode에서 위 3개가 일치. Observation은 dtype/shape이 wrapper에 의해 미묘하게 다를 수 있어 직접 비교에서 제외.

### 5.3 검증 스크립트 위치

- VariBAD: [environments/navigation/tmaze_sanity_check.py](environments/navigation/tmaze_sanity_check.py) (single-repo 검증).
- Cross-repo equivalence script: Sprint 3에서 두 repo 모두 접근 가능한 위치에 추가 예정.

---

## 6. Sprint 1 검증 결과 (VariBAD passive)

[environments/navigation/tmaze_sanity_check.py](environments/navigation/tmaze_sanity_check.py)에 4개 test:

1. **Random policy return** (v6 §6-1.1): L=10, 100 episode 평균 return이 `(-1.0, 0.5)` 범위 내. 실측 mean ≈ -0.967 (uniform 4-action 정책은 worst에 가까움 — v6의 "보통 -0.5~0" 표현은 더 똑똑한 정책 가정인 듯).
2. **Manual trajectories** (v6 §6-1.2): 두 sequence의 reward sequence + 누적 return이 손계산과 일치. Optimal = +1.0, 1-step delay = -0.9 (L=10).
3. **Context exposure** (v6 §6-1.3): `goal_y=+1` vs `-1`에서 oracle 위치 obs는 다르고 (각각 `[0, +1]`, `[0, -1]`), corridor 중간 obs는 동일 (`[0, 0]`).
4. **Wrapper compatibility** (bonus): `gym.make()` + `VariBadWrapper(episodes_per_task=1)` 끝까지 step 가능, obs shape 유지, `done` 정상 signal.

Sprint 1 통과.

---

## 7. Roadmap

| Sprint | 범위 | 상태 |
|---|---|---|
| Sprint 1 | VariBAD repo, passive mode | 완료 |
| Sprint 2 | VariBAD repo, active mode 추가 (T = L+3) | 미시작 |
| Sprint 3 | PEARL repo로 이식 + 두 repo 동치성 검증 | 미시작 |

### Sprint 2 진입 시 처리할 항목 (체크리스트)

- `TMazeEnv.__init__`의 `mode == 'active'` 분기에서 `NotImplementedError` 제거하고 active 로직 구현.
- `oracle_length = 1` (active). Start at x=1, oracle at x=0, junction at x=L+1.
- `_max_episode_steps = corridor_length + 3` (active).
- Reward 식의 시점 정의는 동일하게 적용되지만 oracle visit 페이스가 다름 → Memory-RL 공식 코드 cross-check 필수.
- `'TMaze-active-v0'` 등록.
- Sanity check에 active 시나리오 추가.

### Sprint 3 진입 시 처리할 항목

- PEARL `rlkit/envs/tmaze.py` 작성 + `@register_env` 데코레이터.
- 핵심 로직(grid, transition, reward) 재사용 (별도 helper 모듈로 분리하거나 두 클래스에서 동일 함수 복붙).
- Cross-repo equivalence script 작성.
- 본 SPEC을 PEARL repo 루트에도 동일하게 복사.

### 후속 단계 (main claim 검증)

T-MAZE 이후 continuous-context 환경 (mass/friction/payload 등 hidden physical parameter)으로 main claim 검증. 그 단계에서는 두 repo 동치성을 엄격하게 가져가야 알고리즘 비교가 fair해진다 (T-MAZE 단계의 의미적 동치성보다 강한 기준).

---

## 부록 A. 참고 자료

- 가이드라인: `TMAZE_guideline_v6.md`
- 원 논문: Ni et al. 2023, "When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment", arXiv:2307.03864 (`./docs/tmaze_paper_ni2023.pdf`)
- 참고 코드: Memory-RL `/PublicSSD/shnoh/Memory-RL/envs/tmaze.py` (수정 X)
- VariBAD 환경: [environments/navigation/tmaze.py](environments/navigation/tmaze.py)
- PEARL 환경: (Sprint 3에서 추가)

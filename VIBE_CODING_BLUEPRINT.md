# Vibe Coding Blueprint: HGC-RL System Implementation

> **Role**: AI Architect / Lead Developer
> **Goal**: Implement the "Hierarchical Guided Curriculum RL" (HGC-RL) framework for UAV swarms.
> **Philosophy**: Focus on interfaces, data flow, and mechanism design. Let the coding agent handle the implementation details.

---

## 1. System Architecture Overview

The system is decoupled into two asynchronous loops:
*   **High-Frequency Loop (100Hz)**: `Control Layer` (LADRC + TSA-RL). Handles stability and disturbance rejection.
*   **Low-Frequency Loop (10Hz)**: `Guidance Layer` (Curriculum MARL + PHM). Handles planning and coordination.

### Directory Structure Plan
```text
lc/
├── Gym_env/
│   ├── LADRC_Controller.py       # [New] Linear Active Disturbance Rejection Controller
│   └── ...
├── Reinforce_learning/
│   ├── RLg/
│   │   └── TSA_LADRC.py          # [New] RL Agent for parameter tuning (Paper A)
│   ├── buffers/
│   │   └── PyramidBuffer.py      # [New] Hierarchical Experience Replay (Paper B)
│   └── ...
└── Trainer/
    └── curriculum/
        └── DynamicCurriculum.py  # [New] Success-rate based curriculum switcher
```

---

## 2. Phase 1: Control Layer Implementation (Paper A)

### 2.1 Component: `LADRC_Controller`
**File**: `lc/Gym_env/LADRC_Controller.py`
**Responsibility**: Execute the core control law. Pure math, no neural networks.

**Vibe Specs (Prompts for Implementation)**:
1.  **Class Structure**: `class LADRC:`
2.  **Inputs**: `update(ref_value, measured_value, dt)`
3.  **Internal State**: 
    -   `z`: State vector of LESO $[z_1, z_2, z_3]$.
    -   `u_last`: Previous control output (critical for LESO update).
4.  **Tunable Params**: `omega_c` (Bandwidth), `b0` (System gain). **Crucial**: These must be settable at runtime via a `set_params()` method.
5.  **Math**: Implement the discrete-time LESO equations (Euler integration is sufficient for 100Hz).

### 2.2 Component: `TSA_LADRC_Agent`
**File**: `lc/Reinforce_learning/RLg/TSA_LADRC.py`
**Responsibility**: The "Brain" of the controller. Adapts `omega_c` based on error states.

**Vibe Specs**:
1.  **Inheritance**: Must inherit from `BaseAlgo` (to fit existing Trainer).
2.  **State Processing (TSA Core)**:
    -   **State Stacking**: Implement a `deque(maxlen=k)` to store the last `k` error states. The network input should be `flat(stack)`.
    -   **Input Normalization**: Error derivatives can be large; ensure proper scaling before feeding to NN.
3.  **Action Logic (Action Holding)**:
    -   Implement a counter `self.hold_steps`.
    -   Only query the Actor network when `counter == 0`. Otherwise, return `self.last_action`.
4.  **Reward Function**: 
    -   Needs a custom reward tailored for **smoothness**.
    -   `R = -|error| - 0.1 * |action_diff|^2` (Penalize bang-bang control).

---

## 3. Phase 2: Planning Layer Implementation (Paper B)

### 3.1 Component: `DynamicCurriculum`
**File**: `lc/Trainer/curriculum/DynamicCurriculum.py`
**Responsibility**: Decide *when* to upgrade the task difficulty.

**Vibe Specs**:
1.  **State Machine**: Define states `[Phase1_Hover, Phase2_Avoid, Phase3_Pursuit]`.
2.  **Metric Tracking**: Maintain a rolling window of `success_rate` (e.g., last 100 episodes).
3.  **Trigger Condition**: 
    -   `if win_rate > 0.85 AND reward_std < threshold: advance_stage()`
4.  **Env Interface**: Must have a method `get_env_config()` that returns the obstacle density and target speed for the current stage.

### 3.2 Component: `PyramidBuffer` (PHM)
**File**: `lc/Reinforce_learning/buffers/PyramidBuffer.py`
**Responsibility**: The memory bank. Prevents forgetting.

**Vibe Specs**:
1.  **Data Structure**: Three distinct internal lists/deques:
    -   `L0_Raw`: Standard FIFO buffer.
    -   `L1_Surprise`: Stores transitions where `TD_Error > threshold`.
    -   `L2_Success`: Stores transitions where `info['is_success'] == True`.
2.  **Add Logic**: When `add(transition)` is called, route data to L0. *Also* copy to L2 if it's a success trajectory. L1 is populated during the update step (after calculating TD errors).
3.  **Sample Logic**:
    -   Input: `batch_size`, `current_stage_progress` (lambda).
    -   Logic: Draw `batch_size * lambda` from L0, and the rest mixed from L1/L2.
    -   **Annealing**: As training progresses, reduce sampling from L0 and increase L2.

---

## 4. Verification & Testing Strategy

**Prompt for writing tests**:
> "Create a unit test for `TSA_LADRC_Agent`. Mock the environment state. Verify that `select_action` returns the *same* action for `k` consecutive calls (Action Holding check)."

**Prompt for Simulation**:
> "Write a script `sim_ladrc.py`. Run a PID baseline vs. our LADRC on a step response task with random wind noise. Plot the `error` and `control_output` side-by-side."

---

*This blueprint defines the "What" and "Why". The coding agent should focus on the "How" (syntax, imports, tensor shapes).*

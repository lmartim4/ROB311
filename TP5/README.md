# TP5 - ROB311
[This TP](ROB311%20-%20RL.pdf) gives us a basic idea of how Reinforcement Learning principles work.

## Learning Case Scenario
To illustrate it, we'll be answering some questions based on this diagram of state transition and actions.

![State Action Diagram](StateAction_diagram.png)

| Transition Probabilities | State Rewards |
| --- | --- |
| ![](TransitionProbabilities.png) | ![](Reward.png) |

## Synthetic Description

**States:** S0, S1, S2, S3

**Transitions:**
- S0 ==a1==> S1 (deterministic)
- S0 ==a2==> S2 (deterministic)
- S1 ==a0==> S1 (prob: 1-x) or S3 (prob: x)
- S2 ==a0==> S3 (prob: y) or S0 (prob: 1-y)
- S3 ==a0==> S0 (deterministic)

**Rewards:**
| State | Reward |
| --- | --- |
| S0 | 0 |
| S1 | 0 |
| S2 | 1 |
| S3 | 10 |

## Questions

### Question 1: Possible Policies

### Question 2: Bellman Optimality Equations

### Question 3: Optimal Policy for π*(s0) = a2

### Question 4: Optimal Policy for π*(s0) = a1

### Question 5: Value Iteration Implementation

# Household Documentation Guide

This is the entrypoint for the household solar-battery track.

If you are new to the repository, read [../README.md](../README.md), [../architecture.md](../architecture.md), and [../development.md](../development.md) first.

## Which Document Should I Read?

| If you want to... | Read this | What it covers |
| --- | --- | --- |
| Run the main household workflow | [workflow.md](workflow.md) | Data preparation, notebooks, DT training, and output locations |
| Build diverse synthetic households | [workflow.md](workflow.md) | Whole-day recomposition, validation gates, corpus CLI, and manifest semantics |
| Understand the household environment | [environment.md](environment.md) | PV + load dynamics, action space, reward, and the household control loop |
| Understand battery degradation modeling | [degradation.md](degradation.md) | Muenzel-style degradation, rainflow counting, and implementation details |
| Understand the wider repo structure | [../architecture.md](../architecture.md) | How the household track fits into the full codebase |

## Recommended Reading Order

1. [workflow.md](workflow.md)
2. [environment.md](environment.md)
3. [degradation.md](degradation.md)
4. [../architecture.md](../architecture.md) if you are extending the surrounding tooling

# Multi-Criteria Decision Making Methods in Crisis Management

A bachelor's thesis project (FESB Split) that explores how multi-criteria decision making (MCDM) methods can support fast, high-stakes decisions in crisis situations, tested inside the StarCraft II Learning Environment (SC2LE).

## Overview

When disaster strikes (an earthquake, flood, fire or any other event threatening large numbers of people), evacuation decisions have to be made within minutes. The information available is usually incomplete, contradictory or outdated. The real difficulty is rarely a lack of data. It is that the criteria that matter most, such as safety, speed and accessibility, rarely align: the shortest route is rarely the safest one.

This project argues that this decision problem shares a common structure regardless of the type of crisis. It follows that a single, general decision-support system can be built for it using MCDM methods.

## Methods

The project examines three of the most widely documented MCDM methods in the crisis-management literature:

| Method | Approach |
|--------|----------|
| **AHP** (Analytic Hierarchy Process) | Derives criteria weights from pairwise comparisons |
| **TOPSIS** | Ranks alternatives by closeness to an ideal solution |
| **VIKOR** | Finds a compromise solution balancing overall benefit and worst-case regret |

For each method, the thesis covers its mathematical foundations, ranking procedure, strengths and limitations, illustrated with evacuation-route selection.

## Simulation Environment

To go beyond a static, one-off ranking, the methods are tested in **SC2LE**, a dynamic and time-critical environment. An agent whose decisions are driven by MCDM methods makes a sequence of strategic choices under tight resource constraints. This setting is structurally analogous to a real crisis, though not identical to one.

## Repository Contents

| Path | What it is |
|------|------------|
| `final_machine.py` | The final agent: a Terran player in PySC2 whose strategic priority at each step is chosen by an MCDM method (SAW, TOPSIS or VIKOR), with hard RTS constraints enforced before the MCDM decision |
| `results_<method>_<seed>.csv` | Per-decision logs for SAW, TOPSIS and VIKOR, each run with seeds 101, 202 and 303 |
| `old/`, `old 2/` | Earlier development versions of the agent |
| `*.pdf` | The full thesis |

## Running the Agent

Requirements: Python 3.11, StarCraft II with the `Simple64` map, and the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
python final_machine.py --mcdm topsis --episodes 1 --random_seed 101 --decision_log results_topsis_101.csv
```

Useful flags:

| Flag | Meaning | Default |
|------|---------|---------|
| `--mcdm` | `saw`, `topsis`, `vikor`, or `none` (ablation) | `saw` |
| `--normalization` | `classic` (SAW/VIKOR linear, TOPSIS vector) or `shared` (linear for all) | `classic` |
| `--vikor_v` | VIKOR weight of group utility | `0.5` |
| `--random_seed` | Use the same seed across methods for paired comparisons | `-1` |
| `--decision_log` | CSV path for per-decision matrices, scores and rankings | none |
| `--difficulty` | Built-in Zerg bot difficulty | `easy` |
| `--visualize` | Show the PySC2 visualizer | off |

Run `python final_machine.py --helpfull` for every option.

## Results Files

Each row of a `results_*.csv` file is one decision: the episode and game loop, the method and normalization, each candidate action with its criteria values (economic effect, military effect, urgency, mineral cost, risk, latency), its score and rank, whether it was chosen, and the game state at that moment (workers, army, barracks, minerals, supply, enemy units, threats).

## Key Findings

- The choice of MCDM method is **not a neutral implementation detail**.
- Given identical inputs, AHP, TOPSIS and VIKOR produce measurably different priorities and different agent behaviour.

## Limitations and Ethics

The thesis closes with a critical look at:

- the limits of using a game environment as a stand-in for real crises
- the security risks of automated decision-making in real emergencies

It argues for transparent systems and for keeping a **human in the loop** wherever a flawed recommendation could cost a human life.

## Keywords

`multi-criteria decision making` `crisis management` `AHP` `VIKOR` `TOPSIS` `emergency evacuation` `StarCraft II Learning Environment`

## Thesis

The full thesis (in Croatian, with an English abstract) is available in repo.

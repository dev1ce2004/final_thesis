# Multi-Criteria Decision Making Methods in Crisis Management

A master's thesis project that explores how multi-criteria decision making (MCDM) methods can support fast, high-stakes decisions in crisis situations, tested inside the StarCraft II Learning Environment (SC2LE).

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

The full thesis (in Croatian, with an English abstract) is available in [`thesis.pdf`](thesis.pdf).

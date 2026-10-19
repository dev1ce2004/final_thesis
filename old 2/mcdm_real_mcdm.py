
from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np
from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units


# ============================================================
# SC2 MCDM AGENT
#
# One real decision pipeline:
#
#   SC2 state
#      -> legal candidate actions
#      -> AHP weights
#      -> TOPSIS / VIKOR / AHP
#      -> execute ONE action
#      -> observe result
#      -> re-evaluate
#
# Terran vs Zerg Easy, Simple64, RAW actions.
#
# This is intentionally small and debuggable.  It does not
# pretend that "EKONOMIJA" itself is an executable action.
# MCDM ranks REAL executable alternatives.
# ============================================================


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "method",
    "TOPSIS",
    "Decision method: TOPSIS, VIKOR or AHP.",
)
flags.DEFINE_integer(
    "game_steps",
    25000,
    "Maximum game steps.",
)

STEP_MUL = 8

SCV = int(units.Terran.SCV)
CC = int(units.Terran.CommandCenter)
ORBITAL = int(units.Terran.OrbitalCommand)
BARRACKS = int(units.Terran.Barracks)
BARRACKS_FLYING = int(units.Terran.BarracksFlying)
DEPOT = int(units.Terran.SupplyDepot)
DEPOT_LOWERED = int(units.Terran.SupplyDepotLowered)
MARINE = int(units.Terran.Marine)


def dist(a, b):
    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


class MCDM:
    """
    AHP supplies the criterion weights.
    TOPSIS and VIKOR rank the SAME set of real SC2 actions.

    Criteria:
      military     - immediate military value
      survival     - safety / risk
      economy      - economic value
      opportunity  - exploitable enemy weakness
      urgency      - how badly the current state needs it
    """

    CRITERIA = [
        "military",
        "survival",
        "economy",
        "opportunity",
        "urgency",
    ]

    # Fixed AHP-derived weights. They sum to 1.
    # The important point is that AHP weights the criteria;
    # it does not directly choose an SC2 action.
    WEIGHTS = np.array([
        0.35,  # military
        0.25,  # survival
        0.15,  # economy
        0.15,  # opportunity
        0.10,  # urgency
    ], dtype=float)

    def __init__(self, method="TOPSIS"):
        method = method.upper()
        if method not in {"TOPSIS", "VIKOR", "AHP"}:
            raise ValueError(
                "method must be TOPSIS, VIKOR or AHP"
            )
        self.method = method

    def _normalize(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        denom = np.sqrt(
            np.sum(matrix * matrix, axis=0)
        )
        denom[denom < 1e-12] = 1.0
        return matrix / denom

    def _topsis(self, matrix):
        r = self._normalize(matrix)
        v = r * self.WEIGHTS

        ideal = np.max(v, axis=0)
        anti = np.min(v, axis=0)

        d_plus = np.sqrt(
            np.sum((v - ideal) ** 2, axis=1)
        )
        d_minus = np.sqrt(
            np.sum((v - anti) ** 2, axis=1)
        )

        return d_minus / (
            d_plus + d_minus + 1e-12
        )

    def _vikor(self, matrix):
        """
        All criteria are benefits here: higher is better.
        VIKOR returns utility-like values where higher is better.
        """
        x = np.asarray(matrix, dtype=float)

        best = np.max(x, axis=0)
        worst = np.min(x, axis=0)

        gap = np.maximum(
            best - worst,
            1e-9
        )

        normalized_regret = (
            (best - x) / gap
        )

        s = np.sum(
            normalized_regret * self.WEIGHTS,
            axis=1
        )

        r = np.max(
            normalized_regret * self.WEIGHTS,
            axis=1
        )

        s_best = np.min(s)
        s_worst = np.max(s)
        r_best = np.min(r)
        r_worst = np.max(r)

        qs = 0.5 * (
            (s - s_best) /
            max(1e-9, s_worst - s_best)
        ) + 0.5 * (
            (r - r_best) /
            max(1e-9, r_worst - r_best)
        )

        # Lower Q is better, convert to higher-is-better utility.
        return 1.0 - qs

    def rank(self, candidates):
        """
        candidates = [(name, [c1,c2,c3,c4,c5]), ...]

        Returns:
          selected_name,
          ranking dictionary,
          matrix
        """
        names = [
            x[0] for x in candidates
        ]

        matrix = np.asarray(
            [x[1] for x in candidates],
            dtype=float
        )

        if len(names) == 1:
            return (
                names[0],
                {names[0]: 1.0},
                matrix,
            )

        if self.method == "TOPSIS":
            values = self._topsis(matrix)

        elif self.method == "VIKOR":
            values = self._vikor(matrix)

        else:
            # AHP-weighted sum baseline.
            values = matrix @ self.WEIGHTS

        scores = {
            n: float(v)
            for n, v in zip(names, values)
        }

        selected = names[
            int(np.argmax(values))
        ]

        return selected, scores, matrix


class Agent:

    def __init__(self, method):
        self.mcdm = MCDM(method)

        self.last_decision = -9999
        self.last_scv = -9999
        self.last_marine = -9999
        self.last_attack = -9999

        self.orders = Counter()

        # A placement is not trusted just because it looks empty.
        # A point becomes blacklisted if a build is ordered there and
        # the structure does not appear afterwards.
        self.failed_points = {
            "barracks": set(),
            "depot": set(),
        }

        self.pending_build = None

        self.last_known_enemy = None

        self.attack_point = (20, 20)

        # Only a small number of production buildings are needed.
        self.max_barracks = 3

    # --------------------------------------------------------
    # Observation
    # --------------------------------------------------------

    def raw(self, obs):
        return list(
            getattr(obs, "raw_units", [])
        )

    def own(self, obs):
        return [
            u for u in self.raw(obs)
            if int(u.alliance) == 1
        ]

    def enemies(self, obs):
        return [
            u for u in self.raw(obs)
            if int(u.alliance) == 4
        ]

    def by_type(self, obs, type_id, own=True):
        source = (
            self.own(obs)
            if own
            else self.enemies(obs)
        )
        return [
            u for u in source
            if int(u.unit_type) == int(type_id)
        ]

    def structures(self, obs, ids):
        ids = {int(x) for x in ids}
        return [
            u for u in self.own(obs)
            if int(u.unit_type) in ids
        ]

    def complete(self, xs):
        return [
            u for u in xs
            if float(
                getattr(
                    u,
                    "build_progress",
                    1.0,
                )
            ) >= 0.999
        ]

    def pos(self, u):
        return (
            float(u.x),
            float(u.y),
        )

    def queue(self, u):
        try:
            orders = getattr(
                u,
                "orders",
                None,
            )
            return len(orders) if orders else 0
        except Exception:
            return 0

    def state(self, obs):
        p = obs.player

        minerals = int(p.minerals)
        food_used = int(p.food_used)
        food_cap = int(p.food_cap)

        scvs = self.complete(
            self.by_type(obs, SCV)
        )

        marines = self.complete(
            self.by_type(obs, MARINE)
        )

        barracks_all = self.structures(
            obs,
            [BARRACKS, BARRACKS_FLYING],
        )

        barracks_done = self.complete(
            self.by_type(obs, BARRACKS)
        )

        barracks_building = [
            u for u in self.by_type(
                obs,
                BARRACKS,
            )
            if float(
                getattr(
                    u,
                    "build_progress",
                    1.0,
                )
            ) < 0.999
        ]

        depots = self.structures(
            obs,
            [DEPOT, DEPOT_LOWERED],
        )

        depot_building = [
            u for u in depots
            if float(
                getattr(
                    u,
                    "build_progress",
                    1.0,
                )
            ) < 0.999
        ]

        cc = self.complete(
            self.structures(
                obs,
                [CC, ORBITAL],
            )
        )

        enemy = self.enemies(obs)

        # Visible Zerg combat units.
        enemy_combat = [
            u for u in enemy
            if int(u.unit_type) not in {
                104,  # Drone
                106,  # Overlord
                86,   # Hatchery
                100,  # Lair
                101,  # Hive
            }
        ]

        if enemy_combat:
            # Remember the last useful enemy location.
            avg_x = sum(
                float(u.x)
                for u in enemy_combat
            ) / len(enemy_combat)

            avg_y = sum(
                float(u.y)
                for u in enemy_combat
            ) / len(enemy_combat)

            self.last_known_enemy = (
                avg_x,
                avg_y,
            )

        army = len(marines)
        enemy_count = len(enemy_combat)

        # Approximate army strength, not just unit count.
        our_power = (
            army * 1.0
        )

        enemy_power = 0.0
        for u in enemy_combat:
            uid = int(u.unit_type)

            if uid == 105:       # zergling
                enemy_power += 0.7
            elif uid == 107:     # hydralisk
                enemy_power += 1.5
            elif uid == 110:     # roach
                enemy_power += 2.0
            elif uid == 126:     # queen
                enemy_power += 2.5
            else:
                enemy_power += 1.0

        if enemy_power <= 0:
            safety = 1.0
        else:
            safety = clamp(
                our_power /
                (our_power + enemy_power)
            )

        return {
            "minerals": minerals,
            "food_used": food_used,
            "food_cap": food_cap,
            "supply_left": food_cap - food_used,

            "workers": len(scvs),
            "scvs": scvs,

            "marines": marines,
            "army": army,

            "enemy_units": enemy_combat,
            "enemy": enemy_count,
            "enemy_power": enemy_power,
            "our_power": our_power,
            "safety": safety,

            "cc": cc,

            "barracks": len(barracks_done),
            "barracks_all": barracks_all,
            "barracks_building":
                barracks_building,

            "depots": depots,
            "depot_building":
                depot_building,
        }

    # --------------------------------------------------------
    # RAW actions
    # --------------------------------------------------------

    def valid(self, obs, raw_name):
        try:
            fn = getattr(
                actions.RAW_FUNCTIONS,
                raw_name,
            )
            return (
                int(fn.id)
                in {
                    int(x)
                    for x in obs.available_actions
                }
            )
        except Exception:
            return False

    def call(self, raw_name, *args):
        try:
            fn = getattr(
                actions.RAW_FUNCTIONS,
                raw_name,
            )
            return fn(*args)
        except Exception as e:
            print(
                f"[RAW ACTION ERROR] "
                f"{raw_name}: {e}"
            )
            return actions.RAW_FUNCTIONS.no_op()

    def noop(self):
        return actions.RAW_FUNCTIONS.no_op()

    # --------------------------------------------------------
    # Placement
    # --------------------------------------------------------

    def occupied(self, obs, pt, radius=4.5):
        building_ids = {
            CC,
            ORBITAL,
            BARRACKS,
            BARRACKS_FLYING,
            DEPOT,
            DEPOT_LOWERED,
        }

        for u in self.own(obs):
            if int(u.unit_type) not in building_ids:
                continue

            if dist(
                pt,
                self.pos(u),
            ) < radius:
                return True

        return False

    def candidates(self, obs, kind):
        ccs = self.complete(
            self.structures(
                obs,
                [CC, ORBITAL],
            )
        )

        if not ccs:
            return []

        cx, cy = self.pos(ccs[0])

        if kind == "barracks":
            offsets = [
                (12, 0),
                (12, 5),
                (12, -5),
                (8, 8),
                (8, -8),
                (15, 3),
                (15, -3),
                (5, 11),
                (5, -11),
            ]
        else:
            offsets = [
                (7, 7),
                (7, -7),
                (11, 5),
                (11, -5),
                (5, 10),
                (5, -10),
                (14, 0),
            ]

        points = []

        for ox, oy in offsets:
            x = int(round(cx + ox))
            y = int(round(cy + oy))

            if 3 <= x <= 60 and 3 <= y <= 60:
                points.append((x, y))

        return points

    def next_point(self, obs, kind):
        for pt in self.candidates(
            obs,
            kind,
        ):
            if pt in self.failed_points[kind]:
                continue

            if self.occupied(
                obs,
                pt,
            ):
                continue

            return pt

        return None

    # --------------------------------------------------------
    # Candidate generation
    #
    # THIS is where MCDM is actually used.
    # --------------------------------------------------------

    def candidates_for_state(
        self,
        obs,
        s,
        loop,
    ):
        result = []

        army = s["army"]
        enemy = s["enemy"]
        minerals = s["minerals"]
        supply = s["supply_left"]
        workers = s["workers"]
        barracks = s["barracks"]

        # ----------------------------------------------------
        # TRAIN MARINE
        # ----------------------------------------------------
        if (
            s["barracks_all"]
            and minerals >= 50
            and supply >= 1
        ):
            ready = [
                b for b in s["barracks_all"]
                if int(b.unit_type) == BARRACKS
                and float(
                    getattr(
                        b,
                        "build_progress",
                        1.0,
                    )
                ) >= 0.999
                and self.queue(b) == 0
            ]

            if ready and self.valid(
                obs,
                "Train_Marine_quick",
            ):
                result.append(
                    (
                        "TRAIN_MARINE",
                        self.score_action(
                            "TRAIN_MARINE",
                            s,
                        ),
                    )
                )

        # ----------------------------------------------------
        # TRAIN SCV
        # ----------------------------------------------------
        if (
            s["cc"]
            and minerals >= 50
            and supply >= 1
            and workers < 16
        ):
            ready_cc = [
                cc for cc in s["cc"]
                if self.queue(cc) == 0
            ]

            if ready_cc and self.valid(
                obs,
                "Train_SCV_quick",
            ):
                result.append(
                    (
                        "TRAIN_SCV",
                        self.score_action(
                            "TRAIN_SCV",
                            s,
                        ),
                    )
                )

        # ----------------------------------------------------
        # DEPOT
        # ----------------------------------------------------
        if (
            minerals >= 100
            and supply <= 3
            and s["scvs"]
            and not s["depot_building"]
        ):
            pt = self.next_point(
                obs,
                "depot",
            )

            if (
                pt is not None
                and self.valid(
                    obs,
                    "Build_SupplyDepot_pt",
                )
            ):
                result.append(
                    (
                        f"BUILD_DEPOT@{pt[0]},{pt[1]}",
                        self.score_action(
                            "BUILD_DEPOT",
                            s,
                        ),
                    )
                )

        # ----------------------------------------------------
        # BARRACKS
        # ----------------------------------------------------
        if (
            minerals >= 150
            and s["scvs"]
            and (
                barracks
                +
                len(
                    s["barracks_building"]
                )
                < self.max_barracks
            )
        ):
            pt = self.next_point(
                obs,
                "barracks",
            )

            if (
                pt is not None
                and self.valid(
                    obs,
                    "Build_Barracks_pt",
                )
            ):
                result.append(
                    (
                        f"BUILD_BARRACKS@{pt[0]},{pt[1]}",
                        self.score_action(
                            "BUILD_BARRACKS",
                            s,
                        ),
                    )
                )

        # ----------------------------------------------------
        # ATTACK
        # ----------------------------------------------------
        if army >= 8:
            result.append(
                (
                    "ATTACK",
                    self.score_action(
                        "ATTACK",
                        s,
                    ),
                )
            )

        # ----------------------------------------------------
        # RETREAT / DEFEND
        # ----------------------------------------------------
        if army > 0 and enemy > 0:
            result.append(
                (
                    "DEFEND",
                    self.score_action(
                        "DEFEND",
                        s,
                    ),
                )
            )

        if army > 0:
            result.append(
                (
                    "RALLY",
                    self.score_action(
                        "RALLY",
                        s,
                    ),
                )
            )

        # ----------------------------------------------------
        # SCOUT
        # ----------------------------------------------------
        if army >= 3:
            result.append(
                (
                    "SCOUT",
                    self.score_action(
                        "SCOUT",
                        s,
                    ),
                )
            )

        return result

    def score_action(self, action, s):
        """
        Return five benefit criteria in [0,1].
        """

        army = s["army"]
        enemy = s["enemy"]
        minerals = s["minerals"]
        supply = s["supply_left"]
        workers = s["workers"]
        barracks = s["barracks"]
        safety = s["safety"]

        ratio = (
            army /
            max(
                1.0,
                s["enemy_power"] + 1.0,
            )
        )

        military = 0.0
        survival = safety
        economy = 0.0
        opportunity = 0.0
        urgency = 0.0

        if action == "TRAIN_MARINE":
            military = 0.75
            survival = clamp(
                safety + 0.15
            )
            economy = 0.40
            opportunity = clamp(
                enemy / 8.0
            )
            urgency = (
                0.9
                if enemy > army
                else 0.55
            )

        elif action == "TRAIN_SCV":
            military = 0.05
            survival = 0.85
            economy = 0.95
            opportunity = 0.10
            urgency = (
                0.8
                if workers < 12
                else 0.2
            )

        elif action == "BUILD_DEPOT":
            military = 0.05
            survival = 0.90
            economy = 0.65
            opportunity = 0.15
            urgency = (
                1.0
                if supply <= 1
                else 0.7
            )

        elif action == "BUILD_BARRACKS":
            military = 0.60
            survival = 0.85
            economy = 0.65
            opportunity = 0.40
            urgency = (
                0.9
                if barracks < 2
                else 0.45
            )

        elif action == "ATTACK":
            military = clamp(
                ratio
            )
            survival = clamp(
                0.25 + 0.65 * ratio
            )
            economy = 0.55
            opportunity = clamp(
                0.35
                +
                0.65 *
                (enemy / 8.0)
            )
            urgency = (
                0.95
                if army >= 12
                else 0.70
            )

        elif action == "DEFEND":
            military = 0.45
            survival = clamp(
                0.85 + 0.10 * (1.0 - safety)
            )
            economy = 0.35
            opportunity = 0.35
            urgency = clamp(
                enemy / 6.0
            )

        elif action == "RALLY":
            military = 0.65
            survival = 0.75
            economy = 0.20
            opportunity = 0.60
            urgency = 0.55

        elif action == "SCOUT":
            military = 0.15
            survival = 0.80
            economy = 0.10
            opportunity = 0.85
            urgency = (
                0.8
                if enemy == 0
                else 0.35
            )

        return [
            clamp(military),
            clamp(survival),
            clamp(economy),
            clamp(opportunity),
            clamp(urgency),
        ]

    # --------------------------------------------------------
    # Execute selected action
    # --------------------------------------------------------

    def target_enemy(self, s):
        enemies = s["enemy_units"]

        if not enemies:
            return None

        priority = {
            105: 0,  # zergling
            107: 1,  # hydralisk
            110: 2,  # roach
            126: 3,  # queen
            104: 4,  # drone
            86: 5,   # hatchery
        }

        return min(
            enemies,
            key=lambda u: (
                priority.get(
                    int(u.unit_type),
                    10,
                ),
                float(
                    getattr(
                        u,
                        "health",
                        999999,
                    )
                ),
            ),
        )

    def execute(
        self,
        obs,
        s,
        selected,
        loop,
    ):
        # ----------------------------------------------------
        # Marine
        # ----------------------------------------------------
        if selected == "TRAIN_MARINE":
            if loop - self.last_marine < 80:
                return None

            ready = [
                b for b in s["barracks_all"]
                if int(b.unit_type) == BARRACKS
                and float(
                    getattr(
                        b,
                        "build_progress",
                        1.0,
                    )
                ) >= 0.999
                and self.queue(b) == 0
            ]

            if not ready:
                return None

            b = ready[0]

            print(
                f"SC2 ACTION: TRAIN_MARINE "
                f"BARRACKS={int(b.tag)}"
            )

            self.last_marine = loop
            self.orders["marine"] += 1

            return self.call(
                "Train_Marine_quick",
                "now",
                [int(b.tag)],
            )

        # ----------------------------------------------------
        # SCV
        # ----------------------------------------------------
        if selected == "TRAIN_SCV":
            if loop - self.last_scv < 80:
                return None

            ready = [
                cc for cc in s["cc"]
                if self.queue(cc) == 0
            ]

            if not ready:
                return None

            cc = ready[0]

            print(
                f"SC2 ACTION: TRAIN_SCV "
                f"CC={int(cc.tag)}"
            )

            self.last_scv = loop
            self.orders["scv"] += 1

            return self.call(
                "Train_SCV_quick",
                "now",
                [int(cc.tag)],
            )

        # ----------------------------------------------------
        # Depot
        # ----------------------------------------------------
        if selected.startswith(
            "BUILD_DEPOT@"
        ):
            xy = selected.split("@", 1)[1]
            x, y = [
                int(v)
                for v in xy.split(",")
            ]

            if not s["scvs"]:
                return None

            worker = min(
                s["scvs"],
                key=lambda u:
                    dist(
                        self.pos(u),
                        (x, y),
                    ),
            )

            print(
                f"SC2 ACTION: BUILD_DEPOT "
                f"SCV={int(worker.tag)} "
                f"POS=({x},{y})"
            )

            self.orders["depot"] += 1

            self.pending_build = {
                "kind": "depot",
                "point": (x, y),
            }

            return self.call(
                "Build_SupplyDepot_pt",
                "now",
                [int(worker.tag)],
                [x, y],
            )

        # ----------------------------------------------------
        # Barracks
        # ----------------------------------------------------
        if selected.startswith(
            "BUILD_BARRACKS@"
        ):
            xy = selected.split("@", 1)[1]
            x, y = [
                int(v)
                for v in xy.split(",")
            ]

            if not s["scvs"]:
                return None

            worker = min(
                s["scvs"],
                key=lambda u:
                    dist(
                        self.pos(u),
                        (x, y),
                    ),
            )

            print(
                f"SC2 ACTION: BUILD_BARRACKS "
                f"SCV={int(worker.tag)} "
                f"POS=({x},{y})"
            )

            self.orders["barracks"] += 1

            self.pending_build = {
                "kind": "barracks",
                "point": (x, y),
            }

            return self.call(
                "Build_Barracks_pt",
                "now",
                [int(worker.tag)],
                [x, y],
            )

        # ----------------------------------------------------
        # Attack
        # ----------------------------------------------------
        if selected == "ATTACK":
            marines = s["marines"]

            if len(marines) < 8:
                return None

            if loop - self.last_attack < 160:
                return None

            tags = [
                int(m.tag)
                for m in marines
            ]

            target = self.target_enemy(s)

            if target is not None:
                print(
                    f"SC2 ACTION: ATTACK_UNIT "
                    f"MARINES={len(tags)} "
                    f"TARGET={int(target.tag)}"
                )

                self.last_attack = loop
                self.orders["attack"] += 1

                return self.call(
                    "Attack_unit",
                    "now",
                    tags,
                    int(target.tag),
                )

            point = (
                self.last_known_enemy
                if self.last_known_enemy is not None
                else self.attack_point
            )

            print(
                f"SC2 ACTION: ATTACK_POINT "
                f"MARINES={len(tags)} "
                f"POS={point}"
            )

            self.last_attack = loop
            self.orders["attack"] += 1

            return self.call(
                "Attack_pt",
                "now",
                tags,
                list(point),
            )

        # ----------------------------------------------------
        # Rally
        # ----------------------------------------------------
        if selected == "RALLY":
            marines = s["marines"]

            if not marines:
                return None

            if loop - self.last_attack < 60:
                return None

            tags = [
                int(m.tag)
                for m in marines
            ]

            point = (
                self.last_known_enemy
                if self.last_known_enemy is not None
                else (30, 28)
            )

            print(
                f"SC2 ACTION: RALLY "
                f"MARINES={len(tags)} "
                f"POS={point}"
            )

            self.orders["move"] += 1

            return self.call(
                "Move_pt",
                "now",
                tags,
                list(point),
            )

        # ----------------------------------------------------
        # Defend
        # ----------------------------------------------------
        if selected == "DEFEND":
            marines = s["marines"]

            if not marines:
                return None

            tags = [
                int(m.tag)
                for m in marines
            ]

            point = (
                self.pos(s["cc"][0])
                if s["cc"]
                else (19, 23)
            )

            print(
                f"SC2 ACTION: DEFEND "
                f"MARINES={len(tags)} "
                f"POS={point}"
            )

            self.orders["defend"] += 1

            return self.call(
                "Move_pt",
                "now",
                tags,
                list(point),
            )

        # ----------------------------------------------------
        # Scout
        # ----------------------------------------------------
        if selected == "SCOUT":
            if not s["scvs"]:
                return None

            worker = s["scvs"][0]

            point = (
                35,
                35,
            )

            print(
                f"SC2 ACTION: SCOUT "
                f"SCV={int(worker.tag)} "
                f"POS={point}"
            )

            self.orders["scout"] += 1

            return self.call(
                "Move_pt",
                "now",
                [int(worker.tag)],
                list(point),
            )

        return None

    # --------------------------------------------------------
    # Decision logging
    # --------------------------------------------------------

    def print_decision(
        self,
        loop,
        candidates,
        selected,
        scores,
    ):
        print()
        print("=" * 68)
        print(
            f"MCDM | METHOD={self.mcdm.method} "
            f"GAME_LOOP={loop}"
        )
        print("=" * 68)

        print(
            "AHP WEIGHTS: "
            "MIL=0.35 SURV=0.25 ECO=0.15 "
            "OPP=0.15 URG=0.10"
        )

        for name, criteria in candidates:
            print(
                f"{name:28s} "
                f"SCORE={scores.get(name, 0.0):.4f} "
                f"| "
                f"M={criteria[0]:.2f} "
                f"S={criteria[1]:.2f} "
                f"E={criteria[2]:.2f} "
                f"O={criteria[3]:.2f} "
                f"U={criteria[4]:.2f}"
            )

        print("-" * 68)
        print(
            f"SELECTED -> {selected}"
        )
        print("=" * 68)

    # --------------------------------------------------------
    # Main decision cycle
    # --------------------------------------------------------

    def step(self, timestep):
        if timestep.last():
            return self.noop()

        obs = timestep.observation

        raw_loop = getattr(
            obs,
            "game_loop",
            0,
        )

        arr = np.asarray(
            raw_loop
        )

        if arr.ndim:
            loop = int(
                arr.reshape(-1)[0]
            )
        else:
            loop = int(raw_loop)

        s = self.state(obs)

        # If a previous building did not materialize after enough time,
        # blacklist its point. This is feedback, not blind retry.
        if self.pending_build is not None:
            kind = self.pending_build["kind"]
            point = self.pending_build["point"]

            if kind == "barracks":
                exists = any(
                    dist(
                        self.pos(u),
                        point,
                    ) < 5.0
                    for u in self.by_type(
                        obs,
                        BARRACKS,
                    )
                )
            else:
                exists = any(
                    dist(
                        self.pos(u),
                        point,
                    ) < 4.5
                    for u in self.structures(
                        obs,
                        [DEPOT, DEPOT_LOWERED],
                    )
                )

            if exists:
                self.pending_build = None

        # Re-plan every 16 game loops.
        if (
            loop - self.last_decision
            >= 16
        ):
            self.last_decision = loop

            candidates = (
                self.candidates_for_state(
                    obs,
                    s,
                    loop,
                )
            )

            if candidates:
                selected, scores, _ = (
                    self.mcdm.rank(
                        candidates
                    )
                )

                self.print_decision(
                    loop,
                    candidates,
                    selected,
                    scores,
                )

                action = self.execute(
                    obs,
                    s,
                    selected,
                    loop,
                )

                if action is not None:
                    return action

        # If the selected action was temporarily unavailable, do not stall.
        # Re-run the legal candidate generator on the next cycle.
        return self.noop()


def main(_):
    method = FLAGS.method.upper()

    print("=" * 68)
    print(
        "SC2 MCDM REAL-ACTION AGENT"
    )
    print(
        f"METHOD = {method}"
    )
    print(
        "AHP -> criteria weights | "
        "TOPSIS/VIKOR/AHP -> real action ranking"
    )
    print(
        "Terran vs Zerg Easy | Simple64 | RAW"
    )
    print("=" * 68)

    interface = (
        features.AgentInterfaceFormat(
            use_raw_units=True,
            raw_resolution=64,
            action_space=actions.ActionSpace.RAW,
        )
    )

    players = [
        sc2_env.Agent(
            sc2_env.Race.terran,
            name="MCDM_Terran",
        ),
        sc2_env.Bot(
            sc2_env.Race.zerg,
            sc2_env.Difficulty.easy,
        ),
    ]

    env = sc2_env.SC2Env(
        map_name="Simple64",
        players=players,
        agent_interface_format=interface,
        step_mul=STEP_MUL,
        realtime=False,
        visualize=False,
        game_steps_per_episode=FLAGS.game_steps,
        score_index=-1,
        disable_fog=False,
        ensure_available_actions=False,
    )

    agent = Agent(method)

    try:
        timesteps = env.reset()

        while True:
            action = agent.step(
                timesteps[0]
            )

            timesteps = env.step(
                [action]
            )

            if timesteps[0].last():
                break

    finally:
        print()
        print("=" * 68)
        print("SIMULATION FINISHED")
        print("=" * 68)
        print(
            f"METHOD      : {method}"
        )
        print(
            f"DEPOTS      : {agent.orders['depot']}"
        )
        print(
            f"BARRACKS    : {agent.orders['barracks']}"
        )
        print(
            f"SCVs        : {agent.orders['scv']}"
        )
        print(
            f"MARINES     : {agent.orders['marine']}"
        )
        print(
            f"ATTACKS     : {agent.orders['attack']}"
        )
        print(
            f"DEFENDS     : {agent.orders['defend']}"
        )
        print(
            f"RALLIES     : {agent.orders['move']}"
        )
        print(
            f"SCOUTS      : {agent.orders['scout']}"
        )
        print("=" * 68)

        env.close()


if __name__ == "__main__":
    app.run(main)

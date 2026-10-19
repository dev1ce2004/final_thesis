
from __future__ import annotations
import math
import time
from collections import Counter

import numpy as np
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units


# ============================================================
# MCDM + RTS Terran bot for PySC2 4.0
# - RAW actions + RAW unit tags
# - Terran vs Zerg Easy on Simple64
# - hard RTS safety priorities over strategic MCDM
# ============================================================

STEP_MUL = 8

# Costs
SCV_COST = 50
MARINE_COST = 50
DEPOT_COST = 100
BARRACKS_COST = 150
BUNKER_COST = 100

# Unit ids
SCV = int(units.Terran.SCV)
CC = int(units.Terran.CommandCenter)
ORBITAL = int(units.Terran.OrbitalCommand)
BARRACKS = int(units.Terran.Barracks)
BARRACKS_FLYING = int(units.Terran.BarracksFlying)
DEPOT = int(units.Terran.SupplyDepot)
DEPOT_LOWERED = int(units.Terran.SupplyDepotLowered)
MARINE = int(units.Terran.Marine)
BUNKER = int(units.Terran.Bunker)

# Candidate placements for the standard Simple64 Terran start.
BARRACKS_POINTS = [
    (31, 27), (31, 37), (31, 32),
    (34, 25), (34, 39), (44, 39),
    (46, 39), (44, 25), (47, 35),
]
DEPOT_POINTS = [
    (30, 45), (34, 45), (30, 40), (35, 42),
    (28, 38), (43, 44), (45, 42),
    (31, 24), (35, 23), (43, 24),
]
BUNKER_POINTS = [(34, 34), (34, 39), (44, 38)]


def pxy(u):
    return float(u.x), float(u.y)


def dist(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


class MCDM:
    """Small, deterministic AHP/TOPSIS/VIKOR layer.

    It is deliberately not allowed to override survival/build constraints.
    """

    ACTIONS = [
        "EKONOMIJA", "PROIZVODNJA", "NAPAD", "OBRANA",
        "IZVIDANJE", "POVLACENJE", "ODRZAVANJE"
    ]

    def decide(self, s):
        enemy = s["enemy"]
        army = s["army"]
        workers = s["workers"]
        minerals = s["minerals"]
        safety = s["safety"]
        supply_left = s["supply_left"]
        barracks = s["barracks"]

        # Features are benefit-like scores in [0, 1].
        enemy_pressure = min(1.0, enemy / max(1.0, army * 1.5 + 2))
        army_power = min(1.0, army / 20.0)
        economy = min(1.0, workers / 20.0) * 0.7 + min(1.0, minerals / 1000.0) * 0.3
        production = min(1.0, barracks / 3.0) * 0.6 + (1.0 if supply_left > 0 else 0.0) * 0.4
        attack = min(1.0, army / max(1.0, enemy + 1.0)) * 0.8 + (1.0 if army >= 10 else 0.0) * 0.2
        defense = enemy_pressure * 0.75 + (1.0 - safety) * 0.25
        scout = min(1.0, enemy / 5.0) * 0.5 + (0.3 if army >= 4 else 0.0)
        retreat = min(1.0, enemy / max(1.0, army + 1.0)) * 0.8 + (1.0 - safety) * 0.2
        maintenance = (1.0 - safety) * 0.6 + (1.0 if army < 6 else 0.0) * 0.4

        scores = {
            "EKONOMIJA": economy,
            "PROIZVODNJA": production,
            "NAPAD": attack,
            "OBRANA": defense,
            "IZVIDANJE": scout,
            "POVLACENJE": retreat,
            "ODRZAVANJE": maintenance,
        }

        # TOPSIS-like closeness: score itself.
        topsis = scores.copy()

        # VIKOR-like utility: lower regret is better, represented as utility.
        best = max(scores.values()) if scores else 1.0
        worst = min(scores.values()) if scores else 0.0
        span = max(1e-9, best - worst)
        vikor = {k: 1.0 - (best - v) / span for k, v in scores.items()}

        top = max(topsis, key=topsis.get)
        vik = max(vikor, key=vikor.get)

        # Consensus unless the two methods disagree; use the stronger score.
        if top == vik:
            action = top
            consensus = True
        else:
            action = top if topsis[top] >= vikor[vk := vik] else vik
            consensus = False

        # The MCDM layer is strategic only. Hard RTS rules are handled outside.
        return action, top, vik, consensus, topsis, vikor


class WinAgent:
    def __init__(self):
        self.mcdm = MCDM()

        self.last_loop = -1
        self.last_build_loop = -9999
        self.last_supply_loop = -9999
        self.last_train_scv_loop = -9999
        self.last_train_marine_loop = -9999
        self.last_attack_loop = -9999
        self.last_move_loop = -9999
        self.last_decision_loop = -9999

        self.barracks_candidate = 0
        self.depot_candidate = 0
        self.bunker_candidate = 0

        self.barracks_order_loop = -9999
        self.depot_order_loop = -9999
        self.bunker_order_loop = -9999

        # A Terran build command is a multi-step operation from the bot's
        # point of view. Never issue another build until the previous one
        # either appears in raw_units or has genuinely timed out.
        self.pending_build = {
            "kind": None,
            "point": None,
            "worker_tag": None,
            "issued_loop": -9999,
        }

        self.attack_target_tag = None
        self.attack_point = (20.0, 20.0)
        self.retreat_point = (39.0, 32.0)

        self.orders = Counter()
        self.mcdm_decisions = 0

        # Points on Simple64 can be rejected by SC2 even when they look free
        # from raw-unit occupancy alone. Keep rejected points out of the
        # candidate rotation so the bot does not repeatedly click the same
        # invalid location.
        self.failed_build_points = {
            "depot": set(),
            "barracks": set(),
            "bunker": set(),
        }

    # ---------- observation helpers ----------

    def raw(self, obs):
        return list(getattr(obs, "raw_units", []))

    def own(self, obs):
        return [u for u in self.raw(obs) if int(u.alliance) == 1]

    def enemy(self, obs):
        return [u for u in self.raw(obs) if int(u.alliance) == 4]

    def by_type(self, obs, type_id, own=True):
        src = self.own(obs) if own else self.enemy(obs)
        return [u for u in src if int(u.unit_type) == int(type_id)]

    def structures(self, obs, type_ids):
        ids = {int(x) for x in type_ids}
        return [u for u in self.own(obs)
                if int(u.unit_type) in ids]

    def complete(self, us):
        return [u for u in us if float(getattr(u, "build_progress", 1.0)) >= 0.999]

    def position(self, u):
        return (float(u.x), float(u.y))

    def state(self, obs):
        p = obs.player
        minerals = int(p.minerals)
        vespene = int(p.vespene)
        food_used = int(p.food_used)
        food_cap = int(p.food_cap)

        scvs = self.complete(self.by_type(obs, SCV))
        marines = self.complete(self.by_type(obs, MARINE))
        barracks_all = self.structures(obs, [BARRACKS, BARRACKS_FLYING])
        barracks_done = self.complete(self.by_type(obs, BARRACKS))
        barracks_building = [u for u in self.by_type(obs, BARRACKS)
                             if float(getattr(u, "build_progress", 1.0)) < 0.999]
        depots = self.structures(obs, [DEPOT, DEPOT_LOWERED])
        depot_done = self.complete(depots)
        depot_building = [u for u in depots
                          if float(getattr(u, "build_progress", 1.0)) < 0.999]
        bunkers = self.structures(obs, [BUNKER])
        cc = self.complete(self.structures(obs, [CC, ORBITAL]))
        enemies = self.enemy(obs)

        # Estimate safety from visible enemy count and army count.
        enemy_combat = [u for u in enemies if int(u.unit_type) not in
                        {104, 106, 86, 100, 101}]  # drones/overlord/hatchery/lair/hive
        army = len(marines)
        pressure = len(enemy_combat)
        safety = max(0.0, min(1.0, 1.0 - pressure / max(4.0, army * 1.5 + 4.0)))

        return {
            "minerals": minerals,
            "vespene": vespene,
            "food_used": food_used,
            "food_cap": food_cap,
            "supply_left": food_cap - food_used,
            "workers": len(scvs),
            "army": army,
            "enemy": len(enemy_combat),
            "enemy_units": enemy_combat,
            "marines": marines,
            "scvs": scvs,
            "cc": cc,
            "barracks": len(barracks_done),
            "barracks_all": barracks_all,
            "barracks_building": barracks_building,
            "depots": depots,
            "depot_done": depot_done,
            "depot_building": depot_building,
            "bunkers": bunkers,
            "safety": safety,
        }

    # ---------- raw action helpers ----------

    def call(self, name, *args):
        try:
            fn = getattr(actions.RAW_FUNCTIONS, name)
            return fn(*args)
        except Exception:
            return actions.RAW_FUNCTIONS.no_op()

    def valid(self, obs, name):
        try:
            fn = getattr(actions.RAW_FUNCTIONS, name)
            return int(fn.id) in set(int(x) for x in obs.available_actions)
        except Exception:
            return True

    def no_op(self):
        return actions.RAW_FUNCTIONS.no_op()

    def production_queue(self, unit):
        """Number of current/queued orders on a production structure."""
        try:
            q = getattr(unit, "orders", None)
            return len(q) if q is not None else 0
        except Exception:
            return 0

    def _feature_minimap(self, obs):
        try:
            return obs.feature_minimap
        except Exception:
            return None

    def _buildable_map(self, obs):
        mm = self._feature_minimap(obs)
        if mm is None:
            return None
        try:
            return mm[features.MINIMAP_FEATURES.buildable.index]
        except Exception:
            try:
                return mm[features.MINIMAP_FEATURES.buildable]
            except Exception:
                return None

    def _footprint_buildable(self, buildable, x, y, radius):
        if buildable is None:
            return True
        h, w = buildable.shape[-2], buildable.shape[-1]
        x0, x1 = x-radius, x+radius
        y0, y1 = y-radius, y+radius
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            return False
        patch = buildable[y0:y1+1, x0:x1+1]
        return bool(np.all(patch > 0))

    def build_candidates(self, obs, kind):
        """
        Find real buildable WORLD coordinates from the 64x64 minimap.
        This removes the old hard-coded placement assumption.
        """
        ccs = self.complete(self.structures(obs, [CC, ORBITAL]))
        if not ccs:
            return []

        cc = ccs[0]
        cx, cy = self.position(cc)
        buildable = self._buildable_map(obs)

        if kind == "barracks":
            footprint = 1       # 3x3 footprint around center
            min_r, max_r = 5, 16
        elif kind == "depot":
            footprint = 1
            min_r, max_r = 4, 14
        else:
            footprint = 1
            min_r, max_r = 4, 12

        # Prefer positions around the actual CC, ordered by distance.
        candidates = []
        for y in range(3, 61):
            for x in range(3, 61):
                r = math.hypot(x - cx, y - cy)
                if r < min_r or r > max_r:
                    continue
                if not self._footprint_buildable(buildable, x, y, footprint):
                    continue
                pt = (int(x), int(y))
                if self.point_is_occupied(obs, pt, 3.0):
                    continue
                if not self.mineral_safe(obs, pt):
                    continue
                candidates.append(pt)

        candidates.sort(key=lambda p: (
            abs(math.hypot(p[0]-cx, p[1]-cy) - 8.0),
            p[1], p[0]
        ))

        # If minimap buildability isn't present for some reason, retain the
        # known Simple64 fallback rather than failing completely.
        if not candidates:
            fallback = {
                "depot": DEPOT_POINTS,
                "barracks": BARRACKS_POINTS,
                "bunker": BUNKER_POINTS,
            }[kind]
            candidates = [
                pt for pt in fallback
                if pt not in self.failed_build_points[kind]
                and not self.point_is_occupied(obs, pt, 3.0)
                and self.mineral_safe(obs, pt)
            ]

        return candidates

    def next_build_point(self, obs, kind, min_dist):
        candidates = self.build_candidates(obs, kind)
        failed = self.failed_build_points[kind]

        for pt in candidates:
            if pt in failed:
                continue
            if self.point_is_occupied(obs, pt, min_dist):
                continue
            if not self.mineral_safe(obs, pt):
                continue
            return pt

        return None

    # ---------- placement ----------

    def point_is_occupied(self, obs, point, radius=4.5):
        own = self.own(obs)
        for u in own:
            # Only buildings matter for coarse exclusion.
            if int(u.unit_type) in {
                CC, ORBITAL, BARRACKS, BARRACKS_FLYING,
                DEPOT, DEPOT_LOWERED, BUNKER
            }:
                if dist(point, self.position(u)) < radius:
                    return True
        return False

    def mineral_safe(self, obs, point):
        # Keep structures away from visible mineral/geyser fields.
        neutral = [u for u in self.raw(obs) if int(u.alliance) == 3]
        for u in neutral:
            if int(u.unit_type) in (341, 483, 146, 147, 342, 343, 344):
                if dist(point, self.position(u)) < 4.0:
                    return False
        return True

    def next_point(self, obs, candidates, start_index, min_dist=4.5):
        n = len(candidates)
        for off in range(n):
            idx = (start_index + off) % n
            pt = candidates[idx]
            if not self.point_is_occupied(obs, pt, min_dist) and self.mineral_safe(obs, pt):
                return idx, pt
        return None, None

    # ---------- build / train ----------

    def issue_depot(self, obs, s, loop):
        if s["minerals"] < DEPOT_COST:
            return None
        if s["depot_building"]:
            return None

        # Build before the cap is exhausted.
        if s["supply_left"] > 2:
            return None

        if loop - self.last_supply_loop < 80:
            return None

        workers = s["scvs"]
        if not workers:
            return None

        # Do not interfere with an already pending barracks build.
        if self.pending_build["kind"] == "barracks":
            return None

        pt = self.next_build_point(obs, "depot", 3.0)
        if pt is None:
            return None

        idle_workers = [
            u for u in workers
            if self.production_queue(u) == 0
        ]
        worker = min(
            idle_workers or workers,
            key=lambda u: dist(self.position(u), pt)
        )

        if not self.valid(obs, "Build_SupplyDepot_pt"):
            return None

        self.pending_build = {
            "kind": "depot",
            "point": tuple(pt),
            "worker_tag": int(worker.tag),
            "issued_loop": loop,
        }
        self.last_supply_loop = loop
        self.depot_order_loop = loop

        print(
            f"SC2 RAW ACTION: BUILD_SUPPLY_DEPOT "
            f"SCV={int(worker.tag)} POS={pt}"
        )
        self.orders["depot"] += 1

        return self.call(
            "Build_SupplyDepot_pt",
            "now",
            [int(worker.tag)],
            list(pt),
        )

    def issue_barracks(self, obs, s, loop):
        # Desired production capacity grows with the army.
        desired = 1
        if s["army"] >= 10:
            desired = 2
        if s["army"] >= 22:
            desired = 3

        existing = s["barracks"] + len(s["barracks_building"])

        if existing >= desired:
            self.pending_build["kind"] = None
            return None

        if s["minerals"] < BARRACKS_COST or not s["scvs"]:
            return None

        # ---- A real build is already in progress ----
        pending = self.pending_build
        if pending["kind"] == "barracks":
            # Success: the structure is now visible.
            if s["barracks"] + len(s["barracks_building"]) > 0:
                self.pending_build = {
                    "kind": None, "point": None,
                    "worker_tag": None, "issued_loop": -9999
                }
                return None

            # If the worker disappeared, allow a new attempt after timeout.
            worker_alive = any(
                int(u.tag) == int(pending["worker_tag"])
                for u in s["scvs"]
            )
            worker = next(
                (u for u in s["scvs"]
                 if int(u.tag) == int(pending["worker_tag"])),
                None
            )
            worker_busy = worker is not None and self.production_queue(worker) > 0
            elapsed = loop - pending["issued_loop"]

            # A legal build can take a while before the raw structure entry
            # is visible. Never issue a second build during this window.
            if elapsed < 320 and (worker_alive or worker_busy):
                return None

            # Only NOW do we call the point failed.
            if pending["point"] is not None:
                self.failed_build_points["barracks"].add(
                    tuple(pending["point"])
                )

            self.pending_build = {
                "kind": None, "point": None,
                "worker_tag": None, "issued_loop": -9999
            }
            return None

        if loop - self.last_build_loop < 40:
            return None

        pt = self.next_build_point(obs, "barracks", 3.5)
        if pt is None:
            # Do not spam. Wait for a new observation / resource state.
            return None

        # Prefer a genuinely idle SCV.  A worker with an existing order is
        # much more likely to ignore a competing build command.
        idle_workers = [
            u for u in s["scvs"]
            if self.production_queue(u) == 0
        ]
        workers = idle_workers or s["scvs"]
        worker = min(
            workers,
            key=lambda u: dist(self.position(u), pt)
        )

        if not self.valid(obs, "Build_Barracks_pt"):
            return None

        self.pending_build = {
            "kind": "barracks",
            "point": tuple(pt),
            "worker_tag": int(worker.tag),
            "issued_loop": loop,
        }
        self.last_build_loop = loop
        self.barracks_order_loop = loop

        print(
            f"SC2 RAW ACTION: BUILD_BARRACKS "
            f"SCV={int(worker.tag)} POS={pt}"
        )
        self.orders["barracks"] += 1

        return self.call(
            "Build_Barracks_pt",
            "now",
            [int(worker.tag)],
            list(pt),
        )

    def issue_bunker(self, obs, s, loop):
        # One bunker is enough for early defense; never spam bunkers.
        if s["bunkers"] or s["enemy"] < 6:
            return None
        if self.pending_build["kind"] is not None:
            return None
        if s["minerals"] < BUNKER_COST or not s["scvs"]:
            return None
        if loop - self.bunker_order_loop < 200:
            return None

        pt = self.next_build_point(obs, "bunker", 5.0)
        if pt is None:
            return None
        worker = min(s["scvs"], key=lambda u: dist(self.position(u), pt))
        if not self.valid(obs, "Build_Bunker_pt"):
            return None

        self.failed_build_points["bunker"].add(pt)
        self.bunker_order_loop = loop
        print(f"SC2 RAW ACTION: BUILD_BUNKER SCV={int(worker.tag)} POS={pt}")
        self.orders["bunker"] += 1
        return self.call("Build_Bunker_pt", "now", [int(worker.tag)], list(pt))

    def issue_scv(self, obs, s, loop):
        if not s["cc"]:
            return None
        if s["workers"] >= 20:
            return None
        if s["minerals"] < SCV_COST:
            return None
        if s["supply_left"] < 1:
            return None
        if loop - self.last_train_scv_loop < 60:
            return None

        ready_ccs = [
            u for u in s["cc"]
            if float(getattr(u, "build_progress", 1.0)) >= 0.999
            and self.production_queue(u) < 5
        ]
        if not ready_ccs:
            return None

        cc = min(
            ready_ccs,
            key=lambda u: (
                self.production_queue(u),
                -float(getattr(u, "build_progress", 1.0)),
            ),
        )
        if not self.valid(obs, "Train_SCV_quick"):
            return None

        self.last_train_scv_loop = loop
        print(f"SC2 RAW ACTION: TRAIN_SCV CC={int(cc.tag)}")
        self.orders["scv"] += 1
        return self.call("Train_SCV_quick", "now", [int(cc.tag)])

    def issue_marines(self, obs, s, loop):
        if not s["barracks_all"]:
            return None
        if s["supply_left"] < 1 or s["minerals"] < MARINE_COST:
            return None
        if loop - self.last_train_marine_loop < 20:
            return None

        ready = [
            b for b in s["barracks_all"]
            if int(b.unit_type) == BARRACKS
            and float(getattr(b, "build_progress", 1.0)) >= 0.999
            and self.production_queue(b) < 5
        ]
        if not ready:
            return None

        # Always use the barracks with the shortest queue. Never send a
        # sixth order to a full production queue.
        b = min(
            ready,
            key=lambda x: (
                self.production_queue(x),
                -float(getattr(x, "health", 999999)),
            ),
        )
        if not self.valid(obs, "Train_Marine_quick"):
            return None

        self.last_train_marine_loop = loop
        print(f"SC2 RAW ACTION: TRAIN_MARINE BARRACKS={int(b.tag)}")
        self.orders["marine"] += 1
        return self.call("Train_Marine_quick", "now", [int(b.tag)])

    # ---------- combat ----------

    def choose_enemy_target(self, s):
        enemies = s["enemy_units"]
        if not enemies:
            return None
        # Prioritize combat units, then workers, then structures.
        priority = {
            105: 1,   # zergling
            107: 2,   # hydralisk
            110: 3,   # roach
            126: 4,   # queen
            89: 5,    # spawning pool
            86: 6,    # hatchery
            104: 7,   # drone
        }
        return min(enemies, key=lambda u: (
            priority.get(int(u.unit_type), 10),
            float(getattr(u, "health", 99999))
        ))

    def enemy_point(self, s):
        e = self.choose_enemy_target(s)
        if e is not None:
            return self.position(e)

        # No enemy visible yet: go toward the opposite side of the map
        # relative to our actual Command Center.
        if s["cc"]:
            cx, cy = self.position(s["cc"][0])
            return (
                max(4.0, min(60.0, 64.0 - cx)),
                max(4.0, min(60.0, 64.0 - cy)),
            )
        return (32.0, 32.0)

    def issue_combat(self, obs, s, action, loop):
        marines = s["marines"]
        if not marines:
            return None

        tags = [int(m.tag) for m in marines]
        target = self.choose_enemy_target(s)

        # Hard safety layer.
        if s["enemy"] >= max(6, int(len(marines) * 1.5)) and s["safety"] < 0.45:
            if loop - self.last_move_loop >= 35:
                self.last_move_loop = loop
                self.orders["retreat"] += 1
                print(f"SC2 RAW ACTION: RETREAT MARINES={len(tags)} POS={self.retreat_point}")
                return self.call("Move_pt", "now", tags, list(self.retreat_point))

        # Attack only with a meaningful force.
        if len(marines) >= 8 and (action in {"NAPAD", "IZVIDANJE", "PROIZVODNJA", "OBRANA", "EKONOMIJA"}):
            if loop - self.last_attack_loop >= 100:
                self.last_attack_loop = loop
                if target is not None:
                    print(f"SC2 RAW ACTION: ATTACK_UNIT MARINES={len(tags)} TARGET={int(target.tag)}")
                    self.orders["attack"] += 1
                    return self.call("Attack_unit", "now", tags, int(target.tag))
                pt = self.enemy_point(s)
                print(f"SC2 RAW ACTION: ATTACK_POINT MARINES={len(tags)} POS={pt}")
                self.orders["attack"] += 1
                return self.call("Attack_pt", "now", tags, list(pt))

        # Rally smaller groups toward the enemy / center, but don't suicide.
        if len(marines) >= 4 and loop - self.last_move_loop >= 80:
            self.last_move_loop = loop
            pt = self.enemy_point(s) if s["enemy"] else (30.0, 28.0)
            print(f"SC2 RAW ACTION: MOVE MARINES={len(tags)} POS={pt}")
            self.orders["move"] += 1
            return self.call("Move_pt", "now", tags, list(pt))

        return None

    # ---------- MCDM ----------

    def print_mcdm(self, action, top, vik, consensus, topsis, vikor):
        print("\n" + "=" * 52)
        print("MCDM DECISION")
        print("=" * 52)
        labels = {
            "EKONOMIJA": "EKONOMIJA",
            "PROIZVODNJA": "PROIZVODNJA",
            "NAPAD": "NAPAD",
            "OBRANA": "OBRANA",
            "IZVIDANJE": "IZVIĐANJE",
            "POVLACENJE": "POVLAČENJE",
            "ODRZAVANJE": "ODRŽAVANJE",
        }
        for k in MCDM.ACTIONS:
            print(f"{labels[k]:14s} TOPSIS={topsis[k]:.3f} VIKOR={vikor[k]:.3f}")
        print("-" * 52)
        print(f"TOPSIS -> {labels[top]}")
        print(f"VIKOR  -> {labels[vik]}")
        print(f"{'CONSENSUS' if consensus else 'NO CONSENSUS'} -> {labels[action]}")
        print(f"ACTION: {labels[action]}")

    def step(self, timestep):
        obs = timestep.observation
        loop = int(np.asarray(getattr(obs, "game_loop", 0)).reshape(-1)[0]) if np.asarray(getattr(obs, "game_loop", 0)).ndim else int(getattr(obs, "game_loop", 0))
        self.last_loop = loop
        s = self.state(obs)

        if loop < 16 and s["cc"]:
            cc = s["cc"][0]
            print(
                f"START POSITION: CC=({float(cc.x):.1f},{float(cc.y):.1f}) "
                f"SCVS={len(s['scvs'])}"
            )

        # If the game is nearly lost, prioritize defense and survival.
        if s["food_cap"] > 0 and s["food_used"] > s["food_cap"]:
            print("EMERGENCY: OVER SUPPLY -> REBUILD DEPOT")
        if s["enemy"] >= 10 and s["army"] < 8:
            strategic = "OBRANA"
        else:
            strategic, top, vik, consensus, topsis, vikor = self.mcdm.decide(s)
            if loop - self.last_decision_loop >= 112:
                self.last_decision_loop = loop
                self.mcdm_decisions += 1
                self.print_mcdm(strategic, top, vik, consensus, topsis, vikor)

        # 1) First objective: get ONE Barracks online. This is a hard
        # execution prerequisite for the marine win condition.
        if s["barracks"] == 0 and not s["barracks_building"]:
            act = self.issue_barracks(obs, s, loop)
            if act is not None:
                return act

        # 2) Supply.
        act = self.issue_depot(obs, s, loop)
        if act is not None:
            return act

        # 3) Keep workers producing until 20.
        act = self.issue_scv(obs, s, loop)
        if act is not None:
            return act

        # 3) Recover/build production. If under attack, still rebuild one
        # barracks, but do not spam placements.
        act = self.issue_barracks(obs, s, loop)
        if act is not None:
            return act

        # 4) Defensive bunker if pressure is real.
        act = self.issue_bunker(obs, s, loop)
        if act is not None:
            return act

        # 5) Marines are the main win condition.
        act = self.issue_marines(obs, s, loop)
        if act is not None:
            return act

        # 6) Combat control.
        act = self.issue_combat(obs, s, strategic, loop)
        if act is not None:
            return act

        return self.no_op()


def main(_):
    print("=" * 60)
    print("SC2 MCDM WIN AGENT")
    print("AHP + TOPSIS + VIKOR | RAW ACTIONS | Terran vs Zerg Easy")
    print("=" * 60)

    interface = features.AgentInterfaceFormat(
        feature_dimensions=features.Dimensions(screen=64, minimap=64),
        use_raw_units=True,
        raw_resolution=64,
        action_space=actions.ActionSpace.RAW,
    )

    players = [
        sc2_env.Agent(sc2_env.Race.terran, name="MCDM_Terran"),
        sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.easy),
    ]

    env = sc2_env.SC2Env(
        map_name="Simple64",
        players=players,
        agent_interface_format=interface,
        step_mul=STEP_MUL,
        realtime=False,
        visualize=False,
        game_steps_per_episode=25000,
        score_index=-1,
        disable_fog=False,
        ensure_available_actions=False,
    )

    agent = WinAgent()

    try:
        timesteps = env.reset()
        while True:
            action = agent.step(timesteps[0])
            timesteps = env.step([action])

            if timesteps[0].last():
                break
    finally:
        print("\n" + "=" * 60)
        print("SIMULATION FINISHED")
        print("=" * 60)
        print(f"MCDM decisions : {agent.mcdm_decisions}")
        print(f"Orders         : {sum(agent.orders.values())}")
        print(f"Build orders   : {agent.orders['depot'] + agent.orders['barracks'] + agent.orders['bunker']}")
        print(f"SCV orders     : {agent.orders['scv']}")
        print(f"Marine orders  : {agent.orders['marine']}")
        print(f"Combat orders  : {agent.orders['attack'] + agent.orders['move'] + agent.orders['retreat']}")
        env.close()


if __name__ == "__main__":
    app.run(main)

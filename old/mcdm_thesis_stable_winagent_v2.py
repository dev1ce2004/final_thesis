from __future__ import annotations

import math
from collections import Counter

import numpy as np
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units


# ============================================================
# STABLE TERRAN MARINE PUSH
# Terran vs Zerg Easy
# Simple64
#
# MCDM = strategic information
# RTS controller = hard execution
#
# IMPORTANT:
# - NEVER fill production queues.
# - NEVER spam building locations.
# - NEVER let MCDM prevent an attack.
# ============================================================

STEP_MUL = 8

SCV_COST = 50
MARINE_COST = 50
DEPOT_COST = 100
BARRACKS_COST = 150

SCV = int(units.Terran.SCV)
CC = int(units.Terran.CommandCenter)
ORBITAL = int(units.Terran.OrbitalCommand)

BARRACKS = int(units.Terran.Barracks)
BARRACKS_FLYING = int(units.Terran.BarracksFlying)

DEPOT = int(units.Terran.SupplyDepot)
DEPOT_LOWERED = int(units.Terran.SupplyDepotLowered)

MARINE = int(units.Terran.Marine)


# ============================================================
# HELPERS
# ============================================================

def dist(a, b):
    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


# ============================================================
# MCDM
# ============================================================

class MCDM:

    ACTIONS = [
        "EKONOMIJA",
        "PROIZVODNJA",
        "NAPAD",
        "OBRANA",
        "IZVIDANJE",
        "POVLACENJE",
        "ODRZAVANJE",
    ]

    def decide(self, s):

        army = s["army"]
        enemy = s["enemy"]
        workers = s["workers"]
        minerals = s["minerals"]
        barracks = s["barracks"]

        economy = min(1.0, workers / 16.0)

        production = min(
            1.0,
            barracks / 2.0
        )

        attack = min(
            1.0,
            army / max(12.0, enemy + 1.0)
        )

        defense = min(
            1.0,
            enemy / max(1.0, army + 1.0)
        )

        scores = {
            "EKONOMIJA": economy,
            "PROIZVODNJA": production,
            "NAPAD": attack,
            "OBRANA": defense,
            "IZVIDANJE": 0.2,
            "POVLACENJE": 0.0,
            "ODRZAVANJE": 0.5,
        }

        # MCDM still decides the strategic label,
        # but hard RTS rules override it.
        if army >= 12:
            action = "NAPAD"

        elif barracks < 2:
            action = "PROIZVODNJA"

        elif workers < 16:
            action = "EKONOMIJA"

        else:
            action = max(
                scores,
                key=scores.get
            )

        return action, scores


# ============================================================
# BOT
# ============================================================

class WinAgent:

    def __init__(self):

        self.mcdm = MCDM()

        self.last_decision = -9999
        self.current_loop = 0

        self.last_scv = -9999
        self.last_marine = -9999

        self.last_depot = -9999
        self.last_barracks = -9999

        self.last_attack = -9999
        self.last_move = -9999

        self.orders = Counter()

        # ----------------------------------------------------
        # BUILD PLACEMENT STATE
        # ----------------------------------------------------

        self.failed_points = {
            "depot": {},
            "barracks": {},
        }

        # ONE building attempt at a time.
        # We only move to another point after SC2 confirms success or
        # the previous attempt has timed out.
        self.build_attempts = {
            "depot": None,
            "barracks": None,
        }

        # These are the exact Simple64 coordinates that appeared in the
        # user's earlier run where BUILD_BARRACKS actions were being
        # accepted. They are tried one-at-a-time, never spammed.
        self.known_barracks_points = [
            (47, 45),
            (39, 53),
            (31, 45),
            (39, 37),
            (49, 51),
            (49, 39),
            (29, 51),
            (29, 39),
            (45, 55),
            (33, 55),
            (45, 35),
            (33, 35),
        ]

        # Keep depots away from the mineral line / geysers. These are
        # conservative base-side locations and are also tried one-at-a-time.
        self.known_depot_points = [
            (30, 45),
            (34, 45),
            (30, 40),
            (35, 42),
            (28, 38),
            (43, 44),
        ]

        # Only two barracks.
        self.max_barracks = 2

        # ----------------------------------------------------
        # Simple64 known fallback locations.
        # These are ONLY fallbacks.
        # ----------------------------------------------------

        self.fallback_depots = [
            (30, 45),
            (34, 45),
            (30, 40),
            (35, 42),
            (28, 38),
            (43, 44),
        ]

        self.fallback_barracks = [
            (31, 27),
            (31, 37),
            (34, 25),
            (34, 39),
            (44, 39),
        ]

    # ========================================================
    # OBSERVATION
    # ========================================================

    def raw(self, obs):
        return list(
            getattr(
                obs,
                "raw_units",
                []
            )
        )

    def own(self, obs):
        return [
            u for u in self.raw(obs)
            if int(u.alliance) == 1
        ]

    def enemy(self, obs):
        return [
            u for u in self.raw(obs)
            if int(u.alliance) == 4
        ]

    def by_type(
        self,
        obs,
        unit_type,
        own=True
    ):

        source = (
            self.own(obs)
            if own
            else self.enemy(obs)
        )

        return [
            u for u in source
            if int(u.unit_type)
            == int(unit_type)
        ]

    def complete(self, units_list):

        return [
            u for u in units_list
            if float(
                getattr(
                    u,
                    "build_progress",
                    1.0
                )
            ) >= 0.999
        ]

    def structures(
        self,
        obs,
        ids
    ):

        ids = {
            int(x)
            for x in ids
        }

        return [
            u
            for u in self.own(obs)
            if int(u.unit_type)
            in ids
        ]

    def pos(self, u):

        return (
            float(u.x),
            float(u.y),
        )

    # ========================================================
    # PRODUCTION QUEUE
    #
    # IMPORTANT:
    # We intentionally require ZERO queued orders.
    #
    # This completely kills "queue is full".
    # ========================================================

    def queue(self, unit):

        try:

            orders = getattr(
                unit,
                "orders",
                None
            )

            if orders is None:
                return 0

            return len(orders)

        except Exception:

            return 0

    # ========================================================
    # STATE
    # ========================================================

    def state(self, obs):

        p = obs.player

        minerals = int(
            p.minerals
        )

        food_used = int(
            p.food_used
        )

        food_cap = int(
            p.food_cap
        )

        scvs = self.complete(
            self.by_type(
                obs,
                SCV
            )
        )

        marines = self.complete(
            self.by_type(
                obs,
                MARINE
            )
        )

        barracks_all = self.structures(
            obs,
            [
                BARRACKS,
                BARRACKS_FLYING
            ]
        )

        barracks_done = self.complete(
            self.by_type(
                obs,
                BARRACKS
            )
        )

        barracks_building = [
            u
            for u in self.by_type(
                obs,
                BARRACKS
            )
            if float(
                getattr(
                    u,
                    "build_progress",
                    1.0
                )
            ) < 0.999
        ]

        depots = self.structures(
            obs,
            [
                DEPOT,
                DEPOT_LOWERED
            ]
        )

        depot_building = [
            u
            for u in depots
            if float(
                getattr(
                    u,
                    "build_progress",
                    1.0
                )
            ) < 0.999
        ]

        cc = self.complete(
            self.structures(
                obs,
                [
                    CC,
                    ORBITAL
                ]
            )
        )

        enemies = self.enemy(obs)

        # Ignore drones / overlords / bases for combat pressure.
        enemy_combat = [
            u
            for u in enemies
            if int(u.unit_type)
            not in {
                104,  # Drone
                106,  # Overlord
                86,   # Hatchery
                100,  # Lair
                101,  # Hive
            }
        ]

        return {
            "minerals": minerals,

            "food_used": food_used,
            "food_cap": food_cap,

            "supply_left":
                food_cap - food_used,

            "workers":
                len(scvs),

            "scvs":
                scvs,

            "marines":
                marines,

            "army":
                len(marines),

            "enemy_units":
                enemy_combat,

            "enemy":
                len(enemy_combat),

            "barracks":
                len(barracks_done),

            "barracks_all":
                barracks_all,

            "barracks_building":
                barracks_building,

            "depots":
                depots,

            "depot_building":
                depot_building,

            "cc":
                cc,
        }

    # ========================================================
    # RAW ACTION
    # ========================================================

    def call(self, name, *args):

        try:

            fn = getattr(
                actions.RAW_FUNCTIONS,
                name
            )

            return fn(*args)

        except Exception as e:

            print(
                f"[RAW ERROR] "
                f"{name}: {e}"
            )

            return actions.RAW_FUNCTIONS.no_op()

    def valid(self, obs, name):

        try:

            fn = getattr(
                actions.RAW_FUNCTIONS,
                name
            )

            return (
                int(fn.id)
                in {
                    int(x)
                    for x in obs.available_actions
                }
            )

        except Exception:

            return True

    def no_op(self):

        return actions.RAW_FUNCTIONS.no_op()

    # ========================================================
    # BUILDING PLACEMENT
    # ========================================================

    def building_occupied(
        self,
        obs,
        point,
        radius=4.0
    ):

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
                point,
                self.pos(u)
            ) < radius:

                return True

        return False

    def mineral_blocked(
        self,
        obs,
        point
    ):

        # Neutral mineral/geyser objects.
        neutral = [
            u
            for u in self.raw(obs)
            if int(u.alliance) == 3
        ]

        for u in neutral:

            # Common mineral/geyser IDs.
            if int(u.unit_type) in {
                341,
                483,
                146,
                147,
                342,
                343,
                344,
            }:

                if dist(
                    point,
                    self.pos(u)
                ) < 4.5:

                    return True

        return False

    def candidate_points(
        self,
        obs,
        kind
    ):
        # IMPORTANT:
        # Do not invent placement coordinates from offsets.
        # The previous implementation was generating points that SC2
        # rejected with "Can't find placement location".
        #
        # Simple64 is fixed for this experiment, so use the proven
        # coordinate sequence from the user's earlier working run.
        if kind == "barracks":
            return list(self.known_barracks_points)

        return list(self.known_depot_points)

    def find_build_point(
        self,
        obs,
        kind
    ):

        # Never issue another build of the same kind while one is pending.
        if self.build_attempts.get(kind) is not None:
            return None

        failed = self.failed_points[kind]

        for point in self.candidate_points(
            obs,
            kind
        ):

            failed_at = failed.get(
                point,
                -999999
            )

            if self.current_loop - failed_at < 240:
                continue

            if self.building_occupied(
                obs,
                point,
                4.5
            ):
                continue

            if self.mineral_blocked(
                obs,
                point
            ):
                continue

            return point

        return None

    # ========================================================
    # DEPOT
    # ========================================================

    def issue_depot(
        self,
        obs,
        s,
        loop
    ):

        if s["minerals"] < DEPOT_COST:
            return None

        # Already constructing one.
        if s["depot_building"]:
            return None

        # Don't spam depots.
        if loop - self.last_depot < 120:
            return None

        # Build before supply block.
        if s["supply_left"] > 1:
            return None

        if not s["scvs"]:
            return None

        point = self.find_build_point(
            obs,
            "depot"
        )

        if point is None:
            return None

        idle_workers = [
            u for u in s["scvs"]
            if self.queue(u) == 0
        ]

        if not idle_workers:
            return None

        worker = min(
            idle_workers,
            key=lambda u:
                dist(
                    self.pos(u),
                    point
                )
        )

        if not self.valid(
            obs,
            "Build_SupplyDepot_pt"
        ):
            return None

        # IMPORTANT:
        # Don't try the same bad point again.
        self.build_attempts["depot"] = {
            "point": point,
            "loop": loop,
        }

        self.last_depot = loop

        self.orders["depot"] += 1

        print(
            "BUILD DEPOT "
            f"SCV={int(worker.tag)} "
            f"POS={point}"
        )

        return self.call(
            "Build_SupplyDepot_pt",
            "now",
            [int(worker.tag)],
            list(point)
        )

    # ========================================================
    # BARRACKS
    # ========================================================

    def issue_barracks(
        self,
        obs,
        s,
        loop
    ):

        # NEVER build more than two.
        total = (
            s["barracks"]
            +
            len(
                s["barracks_building"]
            )
        )

        if total >= self.max_barracks:
            return None

        if s["minerals"] < BARRACKS_COST:
            return None

        if not s["scvs"]:
            return None

        if loop - self.last_barracks < 300:
            return None

        point = self.find_build_point(
            obs,
            "barracks"
        )

        if point is None:
            return None

        idle_workers = [
            u for u in s["scvs"]
            if self.queue(u) == 0
        ]

        if not idle_workers:
            return None

        worker = min(
            idle_workers,
            key=lambda u:
                dist(
                    self.pos(u),
                    point
                )
        )

        if not self.valid(
            obs,
            "Build_Barracks_pt"
        ):
            return None

        self.build_attempts["barracks"] = {
            "point": point,
            "loop": loop,
        }

        self.last_barracks = loop

        self.orders["barracks"] += 1

        print(
            "BUILD BARRACKS "
            f"SCV={int(worker.tag)} "
            f"POS={point}"
        )

        return self.call(
            "Build_Barracks_pt",
            "now",
            [int(worker.tag)],
            list(point)
        )

    # ========================================================
    # SCV PRODUCTION
    # ========================================================

    def issue_scv(
        self,
        obs,
        s,
        loop
    ):

        if not s["cc"]:
            return None

        if s["workers"] >= 16:
            return None

        if s["minerals"] < SCV_COST:
            return None

        if s["supply_left"] < 1:
            return None

        if loop - self.last_scv < 50:
            return None

        # ONLY CC WITH EMPTY QUEUE.
        ready = [
            cc
            for cc in s["cc"]
            if self.queue(cc) == 0
        ]

        if not ready:
            return None

        cc = ready[0]

        if not self.valid(
            obs,
            "Train_SCV_quick"
        ):
            return None

        self.last_scv = loop

        self.orders["scv"] += 1

        print(
            f"TRAIN SCV "
            f"CC={int(cc.tag)}"
        )

        return self.call(
            "Train_SCV_quick",
            "now",
            [int(cc.tag)]
        )

    # ========================================================
    # MARINE PRODUCTION
    #
    # ZERO QUEUE POLICY.
    #
    # This eliminates:
    # "queue is full"
    # ========================================================

    def issue_marine(
        self,
        obs,
        s,
        loop
    ):

        if not s["barracks_all"]:
            return None

        if s["minerals"] < MARINE_COST:
            return None

        if s["supply_left"] < 1:
            return None

        # Marine takes much longer than 20 game loops.
        if loop - self.last_marine < 80:
            return None

        ready = []

        for b in s["barracks_all"]:

            if int(
                b.unit_type
            ) != BARRACKS:
                continue

            if float(
                getattr(
                    b,
                    "build_progress",
                    1.0
                )
            ) < 0.999:
                continue

            # ABSOLUTELY DO NOT QUEUE.
            if self.queue(b) != 0:
                continue

            ready.append(b)

        if not ready:
            return None

        b = min(
            ready,
            key=lambda x:
                float(
                    getattr(
                        x,
                        "health",
                        999999
                    )
                )
        )

        if not self.valid(
            obs,
            "Train_Marine_quick"
        ):
            return None

        self.last_marine = loop

        self.orders["marine"] += 1

        print(
            "TRAIN MARINE "
            f"BARRACKS={int(b.tag)}"
        )

        return self.call(
            "Train_Marine_quick",
            "now",
            [int(b.tag)]
        )

    # ========================================================
    # TARGET
    # ========================================================

    def target(self, s):

        enemies = s["enemy_units"]

        if not enemies:
            return None

        priority = {
            105: 0,  # Zergling
            107: 1,  # Hydralisk
            110: 2,  # Roach
            126: 3,  # Queen
            104: 4,  # Drone
            89: 5,  # Spawning Pool
            86: 6,  # Hatchery
        }

        return min(
            enemies,
            key=lambda u: (
                priority.get(
                    int(u.unit_type),
                    10
                ),
                float(
                    getattr(
                        u,
                        "health",
                        999999
                    )
                )
            )
        )

    # ========================================================
    # ATTACK
    #
    # MCDM CANNOT BLOCK THIS.
    # ========================================================

    def attack(
        self,
        obs,
        s,
        loop
    ):

        marines = s["marines"]

        # Need an actual force.
        if len(marines) < 12:
            return None

        if loop - self.last_attack < 160:
            return None

        tags = [
            int(m.tag)
            for m in marines
        ]

        target = self.target(s)

        # ----------------------------------------------------
        # Enemy visible.
        # Attack directly.
        # ----------------------------------------------------

        if target is not None:

            print(
                "ATTACK "
                f"MARINES={len(tags)} "
                f"TARGET={int(target.tag)}"
            )

            self.last_attack = loop

            self.orders["attack"] += 1

            return self.call(
                "Attack_unit",
                "now",
                tags,
                int(target.tag)
            )

        # ----------------------------------------------------
        # Enemy not visible.
        # Attack across the map.
        # ----------------------------------------------------

        point = (
            20.0,
            20.0
        )

        print(
            "ATTACK POINT "
            f"MARINES={len(tags)} "
            f"POS={point}"
        )

        self.last_attack = loop

        self.orders["attack"] += 1

        return self.call(
            "Attack_pt",
            "now",
            tags,
            list(point)
        )

    # ========================================================
    # RALLY
    # ========================================================

    def rally(
        self,
        obs,
        s,
        loop
    ):

        marines = s["marines"]

        if not marines:
            return None

        if len(marines) >= 12:
            return None

        if loop - self.last_move < 160:
            return None

        tags = [
            int(m.tag)
            for m in marines
        ]

        point = (
            30.0,
            28.0
        )

        print(
            "RALLY "
            f"MARINES={len(tags)} "
            f"POS={point}"
        )

        self.last_move = loop

        self.orders["move"] += 1

        return self.call(
            "Move_pt",
            "now",
            tags,
            list(point)
        )

    # ========================================================
    # MAIN STEP
    # ========================================================

    def step(self, timestep):

        obs = timestep.observation

        try:

            raw_loop = getattr(
                obs,
                "game_loop",
                0
            )

            arr = np.asarray(raw_loop)

            if arr.size:
                loop = int(arr.reshape(-1)[0])
            else:
                loop = 0

        except Exception:

            loop = 0

        self.current_loop = loop

        s = self.state(obs)

        # ----------------------------------------------------
        # BUILD FEEDBACK
        # ----------------------------------------------------
        # If a building attempt has not materialized after a while,
        # mark only that location as temporarily failed. This prevents
        # "can't find placement" spam while still allowing recovery.
        for kind in ("depot", "barracks"):
            attempt = self.build_attempts.get(kind)

            if attempt is None:
                continue

            point = attempt["point"]
            started = attempt["loop"]

            if kind == "depot":
                exists = any(
                    dist(point, self.pos(u)) < 4.5
                    for u in self.structures(
                        obs,
                        [DEPOT, DEPOT_LOWERED]
                    )
                )
            else:
                exists = any(
                    dist(point, self.pos(u)) < 5.0
                    for u in self.by_type(
                        obs,
                        BARRACKS
                    )
                )

            if exists:
                self.build_attempts[kind] = None

            elif loop - started > 240:
                self.failed_points[kind][point] = loop
                print(
                    f"BUILD FAILED/TIMEOUT: {kind.upper()} "
                    f"POS={point} -> trying next known point"
                )
                self.build_attempts[kind] = None

        # ----------------------------------------------------
        # MCDM
        # ----------------------------------------------------

        if loop - self.last_decision >= 224:

            self.last_decision = loop

            action, scores = (
                self.mcdm.decide(s)
            )

            print()
            print("=" * 52)
            print("MCDM")
            print("=" * 52)

            for k in MCDM.ACTIONS:

                print(
                    f"{k:16s}"
                    f"{scores[k]:.3f}"
                )

            print(
                f"ACTION: {action}"
            )

            print(
                f"STATE: "
                f"SCV={s['workers']} "
                f"MARINES={s['army']} "
                f"BARRACKS={s['barracks']} "
                f"MIN={s['minerals']} "
                f"SUPPLY={s['supply_left']}"
            )

        # ====================================================
        # HARD RTS PRIORITY
        # ====================================================

        # 1. Supply
        act = self.issue_depot(
            obs,
            s,
            loop
        )

        if act is not None:
            return act

        # 2. SCVs
        act = self.issue_scv(
            obs,
            s,
            loop
        )

        if act is not None:
            return act

        # 3. Barracks
        act = self.issue_barracks(
            obs,
            s,
            loop
        )

        if act is not None:
            return act

        # 4. Marines
        act = self.issue_marine(
            obs,
            s,
            loop
        )

        if act is not None:
            return act

        # 5. ATTACK
        act = self.attack(
            obs,
            s,
            loop
        )

        if act is not None:
            return act

        # 6. Rally
        act = self.rally(
            obs,
            s,
            loop
        )

        if act is not None:
            return act

        return self.no_op()


# ============================================================
# MAIN
# ============================================================

def main(_):

    print("=" * 60)
    print(
        "STABLE MCDM TERRAN MARINE PUSH"
    )
    print(
        "Terran vs Zerg Easy | Simple64"
    )
    print(
        "ZERO QUEUE SPAM"
    )
    print(
        "MCDM THESIS BASELINE - PROVEN SIMPLE64 PLACEMENT SEQUENCE"
    )
    print(
        "MCDM chooses strategic priority; execution validates legality."
    )
    print("=" * 60)

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
            name="MCDM_Terran"
        ),

        sc2_env.Bot(
            sc2_env.Race.zerg,
            sc2_env.Difficulty.easy
        ),
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
        print("=" * 60)
        print("SIMULATION FINISHED")
        print("=" * 60)

        print(
            f"Depots     : "
            f"{agent.orders['depot']}"
        )

        print(
            f"Barracks   : "
            f"{agent.orders['barracks']}"
        )

        print(
            f"SCVs       : "
            f"{agent.orders['scv']}"
        )

        print(
            f"Marines    : "
            f"{agent.orders['marine']}"
        )

        print(
            f"Attacks    : "
            f"{agent.orders['attack']}"
        )

        print(
            f"Moves      : "
            f"{agent.orders['move']}"
        )

        print()
        print(
            "THESIS_METRICS,"
            f"depots={agent.orders['depot']},"
            f"barracks={agent.orders['barracks']},"
            f"scvs={agent.orders['scv']},"
            f"marines={agent.orders['marine']},"
            f"attacks={agent.orders['attack']},"
            f"moves={agent.orders['move']}"
        )

        env.close()


if __name__ == "__main__":
    app.run(main)
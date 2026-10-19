import math
import numpy as np

from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.agents import base_agent


class MCDMAgent(base_agent.BaseAgent):
    """
    Deterministic Terran RTS agent for Simple64.

    Architecture:
      MCDM = strategic decision layer.
      RTS executor = deterministic execution layer.

    The executor uses PySC2 RAW units/actions for unit-tag commands:
      SCV tag -> Build_Barracks_pt -> Barracks tag -> Train_Marine_quick
      Marine tags -> Attack_unit / Move_pt

    Feature layers are kept only as optional spatial information; no
    feature-screen unit selection is used for the core game control.
    """

    SCV = int(units.Terran.SCV)
    COMMAND_CENTER = int(units.Terran.CommandCenter)
    SUPPLY_DEPOT = int(units.Terran.SupplyDepot)
    BARRACKS = int(units.Terran.Barracks)
    MARINE = int(units.Terran.Marine)

    ECONOMY = 0
    PRODUCTION = 1
    ATTACK = 2
    DEFENSE = 3
    SCOUT = 4
    RETREAT = 5
    MAINTENANCE = 6

    NAMES = [
        "EKONOMIJA",
        "PROIZVODNJA",
        "NAPAD",
        "OBRANA",
        "IZVIĐANJE",
        "POVLAČENJE",
        "ODRŽAVANJE",
    ]

    def __init__(self):
        super().__init__()

        # ---------------- MCDM ----------------
        self.weights = self.ahp_weights()
        self.current_decision = self.PRODUCTION
        self.last_decision_loop = -999999
        self.decision_interval = 320  # SC2 game loops

        # ---------------- RTS execution ----------------
        self.last_scv_train_loop = -999999
        self.last_marine_train_loop = -999999
        self.last_build_loop = -999999
        self.last_combat_loop = -999999

        self.build_attempt = 0
        self.build_wait_until = 0
        self.build_point = None

        self.last_print_loop = -999999
        self.print_interval = 160

        self.total_orders = 0
        self.build_orders = 0
        self.scv_orders = 0
        self.marine_orders = 0
        self.combat_orders = 0
        self.mcdm_decisions = 0

        print("=" * 52)
        print("SC2 MCDM RAW RTS AGENT")
        print("AHP + TOPSIS + VIKOR")
        print("Terran vs Zerg Easy | Simple64")
        print("=" * 52)
        print("RAW BUILD:", actions.RAW_FUNCTIONS.Build_Barracks_pt)
        print("RAW SCV  :", actions.RAW_FUNCTIONS.Train_SCV_quick)
        print("RAW MARINE:", actions.RAW_FUNCTIONS.Train_Marine_quick)
        print()

    # ============================================================
    # MCDM
    # ============================================================

    def ahp_weights(self):
        matrix = np.array([
            [1.0, 5.0, 3.0],
            [1.0 / 5.0, 1.0, 1.0 / 2.0],
            [1.0 / 3.0, 2.0, 1.0],
        ])

        normalized = matrix / matrix.sum(axis=0)
        return normalized.mean(axis=1)

    def topsis(self, matrix):
        matrix = np.asarray(matrix, dtype=float)

        denom = np.sqrt((matrix ** 2).sum(axis=0))
        denom[denom < 1e-12] = 1.0

        weighted = matrix / denom * self.weights

        ideal = weighted.max(axis=0)
        anti = weighted.min(axis=0)

        d_plus = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
        d_minus = np.sqrt(((weighted - anti) ** 2).sum(axis=1))

        return d_minus / (d_plus + d_minus + 1e-12)

    def vikor(self, matrix, v=0.5):
        matrix = np.asarray(matrix, dtype=float)

        best = matrix.max(axis=0)
        worst = matrix.min(axis=0)

        span = best - worst
        span[span < 1e-12] = 1.0

        regret = (best - matrix) / span
        weighted = regret * self.weights

        S = weighted.sum(axis=1)
        R = weighted.max(axis=1)

        s_min, s_max = S.min(), S.max()
        r_min, r_max = R.min(), R.max()

        s_norm = (
            np.zeros_like(S)
            if s_max - s_min < 1e-12
            else (S - s_min) / (s_max - s_min)
        )

        r_norm = (
            np.zeros_like(R)
            if r_max - r_min < 1e-12
            else (R - r_min) / (r_max - r_min)
        )

        return v * s_norm + (1.0 - v) * r_norm

    # ============================================================
    # RAW OBSERVATION HELPERS
    # ============================================================

    def raw_units(self, obs):
        return getattr(obs, "raw_units", [])

    def own_units(self, obs, unit_type=None):
        result = []

        for u in self.raw_units(obs):
            if int(u.alliance) != 1:
                continue

            if unit_type is not None and int(u.unit_type) != int(unit_type):
                continue

            result.append(u)

        return result

    def enemy_units(self, obs):
        return [
            u for u in self.raw_units(obs)
            if int(u.alliance) == 4
        ]

    def unit_tag(self, u):
        return int(u.tag)

    def unit_xy(self, u):
        return float(u.x), float(u.y)

    def distance(self, a, b):
        ax, ay = self.unit_xy(a)
        bx, by = self.unit_xy(b)
        return math.hypot(ax - bx, ay - by)

    def complete(self, u):
        return float(getattr(u, "build_progress", 1.0)) >= 0.999

    def first_complete(self, units_list):
        for u in units_list:
            if self.complete(u):
                return u
        return None

    def game_loop(self, obs):
        return int(getattr(obs, "game_loop", 0))

    # ============================================================
    # STATE
    # ============================================================

    def get_state(self, obs):
        p = obs.player

        scvs = self.own_units(obs, self.SCV)
        ccs = self.own_units(obs, self.COMMAND_CENTER)
        depots = self.own_units(obs, self.SUPPLY_DEPOT)
        barracks = self.own_units(obs, self.BARRACKS)
        marines = self.own_units(obs, self.MARINE)
        enemies = self.enemy_units(obs)

        complete_barracks = [b for b in barracks if self.complete(b)]

        minerals = float(p.minerals)
        gas = float(p.vespene)
        used = float(p.food_used)
        cap = float(p.food_cap)
        free = max(0.0, cap - used)

        enemy_combat = 0
        for e in enemies:
            # Count visible enemy units that are not resources/neutral objects.
            if int(e.unit_type) not in (
                int(units.Neutral.MineralField),
                int(units.Neutral.RichMineralField),
                int(units.Neutral.VespeneGeyser),
            ):
                enemy_combat += 1

        marines_count = len(marines)

        if enemy_combat == 0:
            danger = 0.0
            safety = 10.0
        else:
            danger = min(10.0, enemy_combat * 0.75)

            if marines_count == 0:
                safety = max(0.0, 3.0 - danger * 0.35)
            else:
                ratio = marines_count / max(1.0, enemy_combat)
                safety = max(
                    0.0,
                    min(10.0, 3.0 + ratio * 2.0 - danger * 0.20),
                )

        economy_score = min(
            10.0,
            3.0 + minerals / 600.0 + len(scvs) * 0.15,
        )

        mobility = min(10.0, marines_count * 1.5 + free * 0.3)

        return {
            "minerals": minerals,
            "gas": gas,
            "used": used,
            "cap": cap,
            "free": free,
            "scvs": len(scvs),
            "ccs": len(ccs),
            "depots": len(depots),
            "barracks": len(complete_barracks),
            "barracks_total": len(barracks),
            "marines": marines_count,
            "enemies": enemy_combat,
            "danger": danger,
            "safety": safety,
            "economy": economy_score,
            "mobility": mobility,
        }

    # ============================================================
    # DECISION MATRIX
    # ============================================================

    def decision_matrix(self, s):
        safety = s["safety"]
        danger = s["danger"]
        minerals = s["minerals"]
        marines = s["marines"]
        free = s["free"]
        mobility = s["mobility"]

        economy = [
            safety,
            min(10.0, 4.0 + minerals / 500.0),
            min(10.0, 6.0 + free * 0.30),
        ]

        production = [
            min(10.0, safety + (3.0 if marines < 8 else 1.0)),
            min(10.0, 5.0 + minerals / 600.0),
            min(10.0, 6.0 + free * 0.35),
        ]

        attack = [
            min(10.0, safety + marines * 0.40),
            min(10.0, 2.0 + marines * 0.85),
            min(10.0, mobility + 1.0),
        ]

        defense = [
            min(10.0, safety + danger),
            min(10.0, 5.0 + danger),
            min(10.0, mobility + 2.0),
        ]

        scout = [
            min(10.0, safety + 1.0),
            min(10.0, 4.0 + mobility * 0.55),
            min(10.0, 5.0 + mobility * 0.50),
        ]

        retreat = [
            max(0.1, 10.0 - danger),
            min(10.0, 4.0 + danger),
            min(10.0, mobility + danger),
        ]

        maintenance = [
            safety,
            min(10.0, 5.0 + minerals / 1000.0),
            min(10.0, mobility + 1.0),
        ]

        return np.array([
            economy,
            production,
            attack,
            defense,
            scout,
            retreat,
            maintenance,
        ])

    def decide(self, obs, s):
        matrix = self.decision_matrix(s)

        topsis = self.topsis(matrix)
        vikor = self.vikor(matrix)

        # Impossible combat actions are suppressed.
        if s["marines"] == 0:
            topsis[self.ATTACK] = -1.0
            topsis[self.SCOUT] = -0.5
            topsis[self.RETREAT] = -1.0

            vikor[self.ATTACK] = 999.0
            vikor[self.SCOUT] = 999.0
            vikor[self.RETREAT] = 999.0

        # If the enemy is near and we have no army, prioritize production.
        if s["enemies"] > 0 and s["marines"] < 4:
            topsis[self.PRODUCTION] += 0.35

        topsis_choice = int(np.argmax(topsis))
        vikor_choice = int(np.argmin(vikor))

        # Deterministic TOPSIS tie-break.
        choice = topsis_choice

        self.current_decision = choice
        self.mcdm_decisions += 1

        print()
        print("=" * 52)
        print("MCDM DECISION")
        print("=" * 52)

        for i, name in enumerate(self.NAMES):
            print(
                f"{name:15} "
                f"TOPSIS={topsis[i]:.3f} "
                f"VIKOR={vikor[i]:.3f}"
            )

        print("-" * 52)
        print("TOPSIS ->", self.NAMES[topsis_choice])
        print("VIKOR  ->", self.NAMES[vikor_choice])

        if topsis_choice == vikor_choice:
            print("CONSENSUS ->", self.NAMES[choice])
        else:
            print("NO CONSENSUS -> TOPSIS")

        print("ACTION:", self.NAMES[choice])

        return choice

    # ============================================================
    # RAW ACTION HELPERS
    # ============================================================

    def raw_noop(self):
        return actions.RAW_FUNCTIONS.no_op()

    def raw_available(self, obs, fn):
        # Raw abilities are not represented by the feature-layer
        # available_actions list. We therefore use game-state
        # prerequisites plus conservative cooldowns.
        return True

    def issue(self, label):
        self.total_orders += 1
        print("SC2 RAW ACTION:", label)

    # ============================================================
    # BASE / BUILDING LOCATIONS
    # ============================================================

    def get_cc(self, obs):
        ccs = [
            c for c in self.own_units(obs, self.COMMAND_CENTER)
            if self.complete(c)
        ]

        if not ccs:
            return None

        return ccs[0]

    def get_build_scv(self, obs, cc):
        scvs = [
            s for s in self.own_units(obs, self.SCV)
            if self.complete(s)
        ]

        if not scvs:
            return None

        if cc is None:
            return scvs[0]

        return min(scvs, key=lambda s: self.distance(s, cc))

    def build_candidates(self, cc):
        """
        Simple64 world coordinates are 64x64.

        We try multiple deterministic points around the starting CC.
        If one point is blocked, the next attempt uses another point.
        """
        x, y = self.unit_xy(cc)

        offsets = [
            (7.0, 0.0),
            (7.0, 5.0),
            (7.0, -5.0),
            (0.0, 7.0),
            (0.0, -7.0),
            (-7.0, 5.0),
            (-7.0, -5.0),
            (5.0, 7.0),
            (-5.0, 7.0),
        ]

        points = []

        for dx, dy in offsets:
            px = min(60.0, max(4.0, x + dx))
            py = min(60.0, max(4.0, y + dy))
            points.append((px, py))

        return points

    # ============================================================
    # REAL GAME EXECUTION
    # ============================================================

    def train_scv(self, obs, s):
        loop = self.game_loop(obs)

        if s["ccs"] == 0:
            return None

        if s["free"] <= 0:
            return None

        if s["minerals"] < 50:
            return None

        if loop - self.last_scv_train_loop < 360:
            return None

        cc = self.get_cc(obs)

        if cc is None:
            return None

        tag = self.unit_tag(cc)

        self.last_scv_train_loop = loop
        self.scv_orders += 1

        self.issue(f"TRAIN_SCV CC={tag}")

        return actions.RAW_FUNCTIONS.Train_SCV_quick(
            "now",
            [tag],
        )

    def build_barracks(self, obs, s):
        loop = self.game_loop(obs)

        if s["barracks"] > 0 or s["barracks_total"] > 0:
            return None

        if s["minerals"] < 150:
            return None

        if loop < self.build_wait_until:
            return None

        if loop - self.last_build_loop < 80:
            return None

        cc = self.get_cc(obs)
        if cc is None:
            return None

        scv = self.get_build_scv(obs, cc)
        if scv is None:
            return None

        candidates = self.build_candidates(cc)

        if self.build_attempt >= len(candidates):
            self.build_attempt = 0

        point = candidates[self.build_attempt]
        self.build_point = point

        scv_tag = self.unit_tag(scv)

        self.last_build_loop = loop
        self.build_wait_until = loop + 120
        self.build_attempt += 1
        self.build_orders += 1

        self.issue(
            f"BUILD_BARRACKS SCV={scv_tag} "
            f"POS=({point[0]:.1f},{point[1]:.1f})"
        )

        return actions.RAW_FUNCTIONS.Build_Barracks_pt(
            "now",
            [scv_tag],
            point,
        )

    def train_marine(self, obs, s):
        loop = self.game_loop(obs)

        if s["barracks"] == 0:
            return None

        if s["free"] <= 0:
            return None

        if s["minerals"] < 50:
            return None

        if loop - self.last_marine_train_loop < 400:
            return None

        barracks = [
            b for b in self.own_units(obs, self.BARRACKS)
            if self.complete(b)
        ]

        if not barracks:
            return None

        # Pick a Barracks that is not currently under construction.
        b = barracks[0]
        tag = self.unit_tag(b)

        self.last_marine_train_loop = loop
        self.marine_orders += 1

        self.issue(f"TRAIN_MARINE BARRACKS={tag}")

        return actions.RAW_FUNCTIONS.Train_Marine_quick(
            "now",
            [tag],
        )

    # ============================================================
    # COMBAT
    # ============================================================

    def combat_tags(self, obs):
        return [
            self.unit_tag(m)
            for m in self.own_units(obs, self.MARINE)
            if self.complete(m)
        ]

    def nearest_enemy(self, obs, marines):
        enemies = self.enemy_units(obs)

        if not enemies:
            return None

        if not marines:
            return enemies[0]

        mx = sum(self.unit_xy(m)[0] for m in marines) / len(marines)
        my = sum(self.unit_xy(m)[1] for m in marines) / len(marines)

        return min(
            enemies,
            key=lambda e: math.hypot(
                self.unit_xy(e)[0] - mx,
                self.unit_xy(e)[1] - my,
            ),
        )

    def attack(self, obs, s):
        loop = self.game_loop(obs)

        if s["marines"] < 4:
            return None

        if loop - self.last_combat_loop < 240:
            return None

        marines = [
            m for m in self.own_units(obs, self.MARINE)
            if self.complete(m)
        ]

        enemy = self.nearest_enemy(obs, marines)

        if enemy is None:
            return None

        tags = [self.unit_tag(m) for m in marines]
        target = self.unit_tag(enemy)

        self.last_combat_loop = loop
        self.combat_orders += 1

        self.issue(
            f"ATTACK_UNIT MARINES={len(tags)} TARGET={target}"
        )

        return actions.RAW_FUNCTIONS.Attack_unit(
            "now",
            tags,
            target,
        )

    def defense(self, obs, s):
        loop = self.game_loop(obs)

        if s["marines"] == 0:
            return None

        if loop - self.last_combat_loop < 240:
            return None

        cc = self.get_cc(obs)

        if cc is None:
            return None

        x, y = self.unit_xy(cc)

        # Defensive point just outside the base.
        point = (
            min(60.0, max(4.0, x + 4.0)),
            min(60.0, max(4.0, y + 4.0)),
        )

        tags = self.combat_tags(obs)

        if not tags:
            return None

        self.last_combat_loop = loop
        self.combat_orders += 1

        self.issue(
            f"DEFEND_MOVE MARINES={len(tags)} "
            f"POS=({point[0]:.1f},{point[1]:.1f})"
        )

        return actions.RAW_FUNCTIONS.Move_pt(
            "now",
            tags,
            point,
        )

    def retreat(self, obs, s):
        loop = self.game_loop(obs)

        if s["marines"] == 0:
            return None

        if loop - self.last_combat_loop < 240:
            return None

        cc = self.get_cc(obs)

        if cc is None:
            return None

        tags = self.combat_tags(obs)
        if not tags:
            return None

        x, y = self.unit_xy(cc)

        point = (
            min(60.0, max(4.0, x)),
            min(60.0, max(4.0, y)),
        )

        self.last_combat_loop = loop
        self.combat_orders += 1

        self.issue(
            f"RETREAT MARINES={len(tags)} "
            f"POS=({point[0]:.1f},{point[1]:.1f})"
        )

        return actions.RAW_FUNCTIONS.Move_pt(
            "now",
            tags,
            point,
        )

    # ============================================================
    # EXECUTOR
    # ============================================================

    def execute(self, obs, s):
        """
        Priority order is deliberate:

        1. Never get supply blocked.
        2. Get the first Barracks.
        3. Keep SCVs coming.
        4. Keep Marines coming.
        5. Execute MCDM combat decision.
        """

        # First priority: Barracks prerequisite.
        if s["barracks"] == 0 and s["barracks_total"] == 0:
            action = self.build_barracks(obs, s)

            if action is not None:
                return action

            # While waiting for a valid build order, continue economy.
            action = self.train_scv(obs, s)

            if action is not None:
                return action

            return self.raw_noop()

        # Supply management.
        if s["free"] <= 1:
            action = self.train_scv(obs, s)

            if action is not None:
                return action

            return self.raw_noop()

        # Economic upkeep.
        if s["scvs"] < 16:
            action = self.train_scv(obs, s)

            if action is not None:
                return action

        # Production upkeep.
        if s["marines"] < 12:
            action = self.train_marine(obs, s)

            if action is not None:
                return action

        # MCDM strategy.
        if self.current_decision == self.ATTACK:
            action = self.attack(obs, s)
            if action is not None:
                return action

        elif self.current_decision == self.DEFENSE:
            action = self.defense(obs, s)
            if action is not None:
                return action

        elif self.current_decision == self.RETREAT:
            action = self.retreat(obs, s)
            if action is not None:
                return action

        elif self.current_decision == self.SCOUT:
            # A small deterministic scouting move using one Marine.
            action = self.defense(obs, s)
            if action is not None:
                return action

        return self.raw_noop()

    # ============================================================
    # MAIN STEP
    # ============================================================

    def step(self, timestep):
        super().step(timestep)

        if timestep.last():
            return self.raw_noop()

        obs = timestep.observation
        loop = self.game_loop(obs)
        s = self.get_state(obs)

        # Recalculate MCDM periodically.
        if (
            self.last_decision_loop < 0
            or loop - self.last_decision_loop >= self.decision_interval
        ):
            self.current_decision = self.decide(obs, s)
            self.last_decision_loop = loop

        # State display at a slower rate so the console remains readable.
        if loop - self.last_print_loop >= self.print_interval:
            self.last_print_loop = loop

            print()
            print("-" * 52)
            print("STANJE SC2")
            print("-" * 52)
            print(f"Minerali          : {s['minerals']:.0f}")
            print(f"Vespene           : {s['gas']:.0f}")
            print(f"Supply            : {s['used']:.0f}/{s['cap']:.0f}")
            print(f"SCV               : {s['scvs']}")
            print(f"Barracks          : {s['barracks']}/{s['barracks_total']}")
            print(f"Marinci           : {s['marines']}")
            print(f"Neprijatelji      : {s['enemies']}")
            print(f"Opasnost          : {s['danger']:.2f}")
            print(f"Sigurnost         : {s['safety']:.2f}")
            print(f"MCDM cilj         : {self.NAMES[self.current_decision]}")
            print("-" * 52)

        return self.execute(obs, s)


def main(argv):
    del argv

    interface = sc2_env.parse_agent_interface_format(
        feature_screen=84,
        feature_minimap=64,
        use_feature_units=False,
        use_raw_units=True,
        raw_resolution=64,
        action_space="raw",
    )

    agent = MCDMAgent()

    print()
    print("RAW ACTION SPACE ENABLED")
    print("RAW UNIT TAG CONTROL ENABLED")
    print("=" * 52)

    with sc2_env.SC2Env(
        map_name="Simple64",
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(
                sc2_env.Race.zerg,
                sc2_env.Difficulty.easy,
            ),
        ],
        agent_interface_format=interface,
        step_mul=8,
        realtime=False,
    ) as env:

        timesteps = env.reset()

        while not timesteps[0].last():
            action = agent.step(timesteps[0])
            timesteps = env.step([action])

    print()
    print("=" * 52)
    print("SIMULATION FINISHED")
    print("=" * 52)
    print("MCDM decisions :", agent.mcdm_decisions)
    print("Total orders   :", agent.total_orders)
    print("Build orders   :", agent.build_orders)
    print("SCV orders     :", agent.scv_orders)
    print("Marine orders  :", agent.marine_orders)
    print("Combat orders  :", agent.combat_orders)
    print("=" * 52)


if __name__ == "__main__":
    app.run(main)

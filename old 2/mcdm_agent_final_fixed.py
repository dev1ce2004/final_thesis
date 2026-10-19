import math
import numpy as np

from absl import app
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from pysc2.agents import base_agent


class MCDMAgent(base_agent.BaseAgent):
    SCV = 45
    MARINE = 48
    BARRACKS = 21
    SUPPLY_DEPOT = 19
    COMMAND_CENTER = 18

    ECONOMY = 0
    PRODUCTION = 1
    ATTACK = 2
    DEFENSE = 3
    SCOUT = 4
    RETREAT = 5
    MAINTENANCE = 6

    ACTION_NAMES = [
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
        self.criteria_weights = self.ahp_weights()
        self.internal_step = 0
        self.decision_interval = 16
        self.last_decision = None
        self.barracks_candidate_index = 0
        self.depot_candidate_index = 0
        self.mcdm_decisions = 0
        self.actions_sent = 0

        print("\n========================================")
        print("SC2 MCDM AGENT")
        print("AHP + TOPSIS + VIKOR")
        print("PySC2 4.0 RAW ACTIONS")
        print("========================================")
        print("\n========================================")
        print("AHP WEIGHTS")
        print("========================================")
        print(f"Sigurnost          : {self.criteria_weights[0]:.3f}")
        print(f"Vrijeme            : {self.criteria_weights[1]:.3f}")
        print(f"Prohodnost         : {self.criteria_weights[2]:.3f}")
        print("========================================\n")

    # ---------------- AHP / TOPSIS / VIKOR ----------------

    def ahp_weights(self):
        matrix = np.array([
            [1.0, 5.0, 3.0],
            [1.0 / 5.0, 1.0, 1.0 / 2.0],
            [1.0 / 3.0, 2.0, 1.0],
        ])
        return (matrix / matrix.sum(axis=0)).mean(axis=1)

    def topsis(self, matrix, weights):
        matrix = np.asarray(matrix, dtype=float)
        denom = np.sqrt((matrix ** 2).sum(axis=0))
        denom[denom < 1e-12] = 1.0
        weighted = (matrix / denom) * weights
        ideal = weighted.max(axis=0)
        anti = weighted.min(axis=0)
        d1 = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
        d2 = np.sqrt(((weighted - anti) ** 2).sum(axis=1))
        return d2 / (d1 + d2 + 1e-12)

    def vikor(self, matrix, weights, v=0.5):
        matrix = np.asarray(matrix, dtype=float)
        best = matrix.max(axis=0)
        worst = matrix.min(axis=0)
        span = best - worst
        span[span < 1e-12] = 1.0
        regret = (best - matrix) / span
        weighted = regret * weights
        s = weighted.sum(axis=1)
        r = weighted.max(axis=1)
        s0, s1 = s.min(), s.max()
        r0, r1 = r.min(), r.max()
        st = np.zeros_like(s) if s1 - s0 < 1e-12 else (s - s0) / (s1 - s0)
        rt = np.zeros_like(r) if r1 - r0 < 1e-12 else (r - r0) / (r1 - r0)
        return v * st + (1.0 - v) * rt

    # ---------------- RAW UNIT STATE ----------------

    def own_units(self, obs):
        return [u for u in obs.raw_units if u.alliance == 1]

    def enemy_units(self, obs):
        return [u for u in obs.raw_units if u.alliance == 4]

    def units_of_type(self, obs, unit_type):
        return [u for u in self.own_units(obs) if u.unit_type == unit_type]

    def workers(self, obs):
        workers = self.units_of_type(obs, self.SCV)
        idle = [u for u in workers if getattr(u, "order_length", 0) == 0]
        return idle if idle else workers

    def barracks(self, obs):
        return self.units_of_type(obs, self.BARRACKS)

    def ready_barracks(self, obs):
        return [
            u for u in self.barracks(obs)
            if getattr(u, "build_progress", 1.0) >= 0.99
        ]

    def command_centers(self, obs):
        return self.units_of_type(obs, self.COMMAND_CENTER)

    def marines(self, obs):
        return self.units_of_type(obs, self.MARINE)

    def supply_depots(self, obs):
        return self.units_of_type(obs, self.SUPPLY_DEPOT)

    def game_state(self, obs):
        p = obs.player
        enemies = self.enemy_units(obs)
        own = self.own_units(obs)
        workers = self.workers(obs)
        marines = self.marines(obs)

        enemy_count = len(enemies)
        marine_count = len(marines)

        danger = min(
            10.0,
            enemy_count * 0.55
            + max(0, enemy_count - marine_count) * 0.30,
        )

        if enemy_count == 0:
            safety = 10.0
        else:
            ratio = marine_count / max(1, enemy_count)
            safety = max(
                0.0,
                min(10.0, 2.0 + ratio * 3.0 - danger * 0.25),
            )

        minerals = float(p.minerals)
        vespene = float(p.vespene)
        food_used = float(p.food_used)
        food_cap = float(p.food_cap)
        free_supply = max(0.0, food_cap - food_used)

        economy = min(10.0, minerals / 400.0)
        gas = min(10.0, vespene / 200.0)

        time_score = max(
            0.0,
            min(
                10.0,
                0.60 * economy
                + 0.20 * gas
                + 2.0 * (1.0 - min(1.0, danger / 10.0)),
            ),
        )

        mobility = max(
            0.0,
            min(10.0, min(10.0, marine_count * 1.5) + min(3.0, free_supply * 0.3)),
        )

        return {
            "minerals": minerals,
            "vespene": vespene,
            "food_used": food_used,
            "food_cap": food_cap,
            "free_supply": free_supply,
            "own_units": len(own),
            "workers": len(workers),
            "marines": marine_count,
            "military": marine_count,
            "barracks": len(self.barracks(obs)),
            "depots": len(self.supply_depots(obs)),
            "enemies": enemy_count,
            "danger": danger,
            "safety": safety,
            "time": time_score,
            "mobility": mobility,
        }

    # ---------------- MCDM ----------------

    def alternatives(self, s):
        danger = s["danger"]
        safety = s["safety"]
        mobility = s["mobility"]
        minerals = s["minerals"]
        marines = s["marines"]
        free_supply = s["free_supply"]

        economy = [
            safety,
            min(10.0, 4.0 + minerals / 600.0),
            min(10.0, 6.0 + free_supply * 0.4),
        ]

        production = [
            min(10.0, safety + (3.0 if marines < 4 else 1.0)),
            min(10.0, 5.0 + minerals / 500.0),
            min(10.0, 7.0 + free_supply * 0.3),
        ]

        attack = (
            [0.0, 0.0, 0.0]
            if marines == 0
            else [
                min(10.0, safety + marines * 0.4),
                min(10.0, 3.0 + marines * 0.8),
                min(10.0, mobility + marines * 0.2),
            ]
        )

        defense = [
            min(10.0, safety + danger * 0.8),
            min(10.0, 5.0 + danger),
            min(10.0, mobility + 2.0),
        ]

        scouting = [
            min(10.0, safety + 1.0),
            min(10.0, 5.0 + mobility * 0.5),
            min(10.0, 5.0 + mobility * 0.5),
        ]

        retreat = (
            [0.0, 0.0, 0.0]
            if marines == 0
            else [
                max(0.1, 10.0 - danger),
                min(10.0, 4.0 + danger),
                min(10.0, mobility + danger),
            ]
        )

        maintenance = [
            safety,
            min(10.0, 5.0 + minerals / 1000.0),
            min(10.0, mobility + 1.5),
        ]

        return np.array([
            economy,
            production,
            attack,
            defense,
            scouting,
            retreat,
            maintenance,
        ], dtype=float)

    def choose_action(self, obs):
        state = self.game_state(obs)
        matrix = self.alternatives(state)

        topsis = self.topsis(matrix, self.criteria_weights)
        vikor = self.vikor(matrix, self.criteria_weights)

        if state["marines"] == 0:
            topsis[self.ATTACK] = -1.0
            topsis[self.RETREAT] = -1.0
            vikor[self.ATTACK] = 999.0
            vikor[self.RETREAT] = 999.0

        t_choice = int(np.argmax(topsis))
        v_choice = int(np.argmin(vikor))
        choice = t_choice

        self.mcdm_decisions += 1

        print("\n========================================")
        print("STANJE SC2")
        print("========================================")
        print(f"Minerali          : {state['minerals']:.0f}")
        print(f"Vespene           : {state['vespene']:.0f}")
        print(f"Supply            : {state['food_used']:.0f}/{state['food_cap']:.0f}")
        print(f"Vlastite jedinice : {state['own_units']}")
        print(f"Vojne jedinice    : {state['military']}")
        print(f"Marinci           : {state['marines']}")
        print(f"Barracks          : {state['barracks']}")
        print(f"Supply Depot      : {state['depots']}")
        print(f"Radnici           : {state['workers']}")
        print(f"Neprijatelji      : {state['enemies']}")
        print(f"Opasnost          : {state['danger']:.2f}")
        print(f"Sigurnost         : {state['safety']:.2f}")
        print(f"Vrijeme            : {state['time']:.2f}")
        print(f"Prohodnost        : {state['mobility']:.2f}")

        print("\n========================================")
        print("MCDM DECISION")
        print("========================================")
        for i, name in enumerate(self.ACTION_NAMES):
            print(
                f"{name:15} "
                f"TOPSIS={topsis[i]:.3f} "
                f"VIKOR={vikor[i]:.3f}"
            )

        print("----------------------------------------")
        print(f"TOPSIS -> {self.ACTION_NAMES[t_choice]}")
        print(f"VIKOR  -> {self.ACTION_NAMES[v_choice]}")

        if t_choice == v_choice:
            print(f"CONSENSUS -> {self.ACTION_NAMES[choice]}")
        else:
            print(f"NO CONSENSUS -> TOPSIS ({self.ACTION_NAMES[choice]})")

        print(f"ACTION: {self.ACTION_NAMES[choice]}")
        self.last_decision = choice
        return choice

    # ---------------- BUILD PLACEMENT ----------------

    def _camera_center_world(self, obs):
        """
        The feature screen is a camera view, while RAW actions use world
        coordinates.  We explicitly center the camera on our Command Center
        and then convert the chosen feature-screen pixel back to world space.
        """
        ccs = self.command_centers(obs)
        if ccs:
            cc = ccs[0]
            return float(cc.x), float(cc.y)
        return 32.0, 32.0

    def _feature_screen_array(self, obs, name):
        try:
            return np.asarray(getattr(obs.feature_screen, name))
        except (AttributeError, TypeError):
            return None

    def _buildable_screen_candidates(self, obs):
        """
        Find several safe buildable patches in the current camera view.

        This is deliberately conservative. We don't guess arbitrary world
        coordinates; SC2's own buildable feature layer decides where terrain
        is suitable. Existing visible units/resources are rejected too.
        """
        buildable = self._feature_screen_array(obs, "buildable")
        player_relative = self._feature_screen_array(obs, "player_relative")

        if buildable is None:
            return []

        h, w = buildable.shape[:2]
        occupied = None
        if player_relative is not None:
            occupied = player_relative != 0

        cx, cy = w // 2, h // 2
        radius = 6
        candidates = []

        for y in range(radius + 2, h - radius - 2):
            for x in range(radius + 2, w - radius - 2):
                patch = buildable[
                    y-radius:y+radius+1,
                    x-radius:x+radius+1
                ]

                if patch.size == 0:
                    continue

                # Barracks footprint is several world units, so require a
                # reasonably large contiguous buildable area.
                if float(np.mean(patch > 0)) < 0.97:
                    continue

                if occupied is not None:
                    opatch = occupied[
                        y-radius:y+radius+1,
                        x-radius:x+radius+1
                    ]
                    if np.any(opatch):
                        continue

                distance = math.hypot(x - cx, y - cy)

                # Do not overlap the Command Center.
                if distance < 11:
                    continue

                candidates.append((distance, x, y))

        candidates.sort(key=lambda v: v[0])

        # Deduplicate points that are too close to one another.
        result = []
        for _, x, y in candidates:
            if all(math.hypot(x - px, y - py) >= 10
                   for px, py in result):
                result.append((x, y))
            if len(result) >= 20:
                break

        return result

    def _screen_to_world(self, obs, screen_point):
        """
        Convert feature-screen coordinates to world coordinates.

        PySC2 defaults the feature camera width to 24 world units.  We use the
        same value explicitly in main() below, so the conversion is stable.
        """
        sx, sy = screen_point
        cx, cy = self._camera_center_world(obs)

        screen_w = 84.0
        screen_h = 84.0
        camera_width = 24.0
        camera_height = camera_width * (screen_h / screen_w)

        wx = cx + (float(sx) - screen_w / 2.0) * camera_width / screen_w
        wy = cy + (float(sy) - screen_h / 2.0) * camera_height / screen_h

        wx = max(1.0, min(63.0, wx))
        wy = max(1.0, min(63.0, wy))
        return (int(round(wx)), int(round(wy)))

    def _find_build_worker(self, obs, point):
        workers = self.workers(obs)
        if not workers:
            return None

        return min(
            workers,
            key=lambda w: math.hypot(
                float(w.x) - point[0],
                float(w.y) - point[1],
            ),
        )

    def _move_camera_to_base(self, obs):
        cc = self.command_centers(obs)
        if not cc:
            return actions.RAW_FUNCTIONS.no_op()

        c = cc[0]
        target = (float(c.x), float(c.y))

        print(f"SC2 RAW ACTION: MOVE_CAMERA -> {target}")
        self.actions_sent += 1
        self.build_state = "camera"

        return actions.RAW_FUNCTIONS.raw_move_camera(target)

    def _start_barracks_build(self, obs):
        if self.build_cooldown > 0:
            return actions.RAW_FUNCTIONS.no_op()

        if obs.player.minerals < 150:
            return actions.RAW_FUNCTIONS.no_op()

        # First center the camera. The next observation then contains the
        # exact buildable feature layer around our base.
        if self.build_state != "camera":
            return self._move_camera_to_base(obs)

        candidates = self._buildable_screen_candidates(obs)

        if not candidates:
            print("SC2: NO SAFE BARRACKS LOCATION FOUND")
            self.build_state = "idle"
            self.build_cooldown = 8
            return actions.RAW_FUNCTIONS.no_op()

        # Try candidates in order. If one placement is rejected, the next
        # attempt uses a different point instead of hammering the same point.
        index = self.barracks_candidate_index % len(candidates)
        screen_point = candidates[index]
        self.barracks_candidate_index += 1

        world_point = self._screen_to_world(obs, screen_point)
        worker = self._find_build_worker(obs, world_point)

        if worker is None:
            print("SC2: NO SCV AVAILABLE FOR BARRACKS")
            self.build_state = "idle"
            self.build_cooldown = 8
            return actions.RAW_FUNCTIONS.no_op()

        self.build_target_screen = screen_point
        self.build_target_world = world_point
        self.build_state = "waiting"

        print(
            "SC2 RAW ACTION: BUILD_BARRACKS "
            f"SCV={worker.tag} "
            f"SCREEN={screen_point} "
            f"WORLD={world_point}"
        )

        self.actions_sent += 1
        self.build_cooldown = 32

        return actions.RAW_FUNCTIONS.Build_Barracks_pt(
            "now",
            [int(worker.tag)],
            world_point,
        )

    def build_barracks(self, obs):
        if self.barracks(obs):
            self.build_state = "idle"
            self.build_cooldown = 0
            return actions.RAW_FUNCTIONS.no_op()

        if self.build_state == "waiting":
            # A Barracks can take a while to appear. Never issue another
            # build command during this period.
            if self.build_cooldown > 0:
                return actions.RAW_FUNCTIONS.no_op()

            # If the previous point failed, move the camera back to the base
            # and try another safe point once.
            self.build_state = "idle"
            return self._start_barracks_build(obs)

        return self._start_barracks_build(obs)

    def construction_points(self, obs, kind):
        """Compatibility helper: returns world-coordinate candidates."""
        points = []
        for screen in self._buildable_screen_candidates(obs):
            points.append(self._screen_to_world(obs, screen))
        return points

    def choose_worker_for(self, obs, point):
        return self._find_build_worker(obs, point)

    def build_depot(self, obs):
        if self.build_state != "idle" or self.build_cooldown > 0:
            return actions.RAW_FUNCTIONS.no_op()

        if obs.player.minerals < 100:
            return actions.RAW_FUNCTIONS.no_op()

        if self.build_state != "camera":
            return self._move_camera_to_base(obs)

        candidates = self._buildable_screen_candidates(obs)
        if not candidates:
            self.build_cooldown = 8
            return actions.RAW_FUNCTIONS.no_op()

        index = self.depot_candidate_index % len(candidates)
        screen_point = candidates[index]
        self.depot_candidate_index += 1

        world_point = self._screen_to_world(obs, screen_point)
        worker = self._find_build_worker(obs, world_point)

        if worker is None:
            self.build_cooldown = 8
            return actions.RAW_FUNCTIONS.no_op()

        print(
            "SC2 RAW ACTION: BUILD_SUPPLY_DEPOT "
            f"SCV={worker.tag} WORLD={world_point}"
        )

        self.actions_sent += 1
        self.build_state = "waiting"
        self.build_cooldown = 24

        return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
            "now",
            [int(worker.tag)],
            world_point,
        )

    # ---------------- PRODUCTION ----------------

    def train_scv(self, obs):
        if obs.player.minerals < 50:
            return actions.RAW_FUNCTIONS.no_op()

        if obs.player.food_cap - obs.player.food_used < 1:
            return actions.RAW_FUNCTIONS.no_op()

        for cc in self.command_centers(obs):
            if getattr(cc, "build_progress", 1.0) >= 0.99:
                print(f"SC2 RAW ACTION: TRAIN_SCV CC={cc.tag}")
                self.actions_sent += 1
                return actions.RAW_FUNCTIONS.Train_SCV_quick(
                    "now", [int(cc.tag)]
                )

        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):
        if obs.player.minerals < 50:
            return actions.RAW_FUNCTIONS.no_op()

        if obs.player.food_cap - obs.player.food_used < 1:
            return actions.RAW_FUNCTIONS.no_op()

        ready = self.ready_barracks(obs)
        tags = [
            int(b.tag)
            for b in ready
            if getattr(b, "order_length", 0) == 0
        ]

        if not tags:
            return actions.RAW_FUNCTIONS.no_op()

        print(f"SC2 RAW ACTION: TRAIN_MARINE BARRACKS={tags}")
        self.actions_sent += 1
        return actions.RAW_FUNCTIONS.Train_Marine_quick(
            "now", tags
        )

    # ---------------- COMBAT ----------------

    def combat_target(self, obs):
        enemies = self.enemy_units(obs)
        if not enemies:
            return None

        marines = self.marines(obs)
        if not marines:
            return enemies[0]

        cx = sum(float(m.x) for m in marines) / len(marines)
        cy = sum(float(m.y) for m in marines) / len(marines)

        return min(
            enemies,
            key=lambda e: math.hypot(
                float(e.x) - cx,
                float(e.y) - cy,
            ),
        )

    def attack(self, obs):
        marines = self.marines(obs)
        target = self.combat_target(obs)

        if not marines or target is None:
            return actions.RAW_FUNCTIONS.no_op()

        tags = [int(m.tag) for m in marines]
        print(f"SC2 RAW ACTION: ATTACK target={target.tag}")
        self.actions_sent += 1

        return actions.RAW_FUNCTIONS.Attack_unit(
            "now",
            tags,
            int(target.tag),
        )

    def defense(self, obs):
        # Defend by attacking the nearest visible enemy.
        return self.attack(obs)

    def retreat(self, obs):
        marines = self.marines(obs)
        ccs = self.command_centers(obs)

        if not marines or not ccs:
            return actions.RAW_FUNCTIONS.no_op()

        cc = ccs[0]
        target = (
            max(2, min(61, int(round(cc.x)))),
            max(2, min(61, int(round(cc.y)))),
        )
        tags = [int(m.tag) for m in marines]

        print(f"SC2 RAW ACTION: RETREAT -> {target}")
        self.actions_sent += 1

        return actions.RAW_FUNCTIONS.Move_pt(
            "now", tags, target
        )

    def scout(self, obs):
        marines = self.marines(obs)
        if not marines:
            return actions.RAW_FUNCTIONS.no_op()

        ccs = self.command_centers(obs)
        if ccs:
            cc = ccs[0]
            target = (
                max(2, min(61, int(round(64 - cc.x)))),
                max(2, min(61, int(round(64 - cc.y)))),
            )
        else:
            target = (48, 48)

        marine = marines[0]
        print(
            f"SC2 RAW ACTION: SCOUT "
            f"Marine={marine.tag} -> {target}"
        )
        self.actions_sent += 1

        return actions.RAW_FUNCTIONS.Move_pt(
            "now", [int(marine.tag)], target
        )

    # ---------------- EXECUTION ----------------

    def execute(self, choice, obs):
        state = self.game_state(obs)

        # Hard prerequisite: no Barracks means production cannot
        # happen yet. Build it using RAW actions.
        if not self.barracks(obs):
            if state["minerals"] >= 150:
                return self.build_barracks(obs)
            return actions.RAW_FUNCTIONS.no_op()

        # Do not issue another build while a Barracks is unfinished.
        if any(
            getattr(b, "build_progress", 1.0) < 0.99
            for b in self.barracks(obs)
        ):
            return actions.RAW_FUNCTIONS.no_op()

        if state["free_supply"] <= 1 and state["minerals"] >= 100:
            return self.build_depot(obs)

        if choice == self.ECONOMY:
            if state["workers"] < 12:
                return self.train_scv(obs)
            return actions.RAW_FUNCTIONS.no_op()

        if choice == self.PRODUCTION:
            return self.train_marine(obs)

        if choice == self.ATTACK:
            return self.attack(obs)

        if choice == self.DEFENSE:
            return self.defense(obs)

        if choice == self.SCOUT:
            return self.scout(obs)

        if choice == self.RETREAT:
            return self.retreat(obs)

        return actions.RAW_FUNCTIONS.no_op()

    # ---------------- STEP ----------------

    def step(self, timestep):
        super().step(timestep)

        if timestep.last():
            return actions.RAW_FUNCTIONS.no_op()

        self.internal_step += 1
        obs = timestep.observation
        state = self.game_state(obs)

        # 1. Barracks is a real prerequisite. Do not wait for a
        # screen selection or Build_Barracks_screen availability.
        if not self.barracks(obs):
            if state["minerals"] >= 150:
                return self.build_barracks(obs)

            if self.internal_step % 16 == 0 and state["workers"] < 12:
                return self.train_scv(obs)

            return actions.RAW_FUNCTIONS.no_op()

        # 2. Supply protection.
        if state["free_supply"] <= 1 and state["minerals"] >= 100:
            return self.build_depot(obs)

        # 3. MCDM.
        if (
            self.internal_step == 1
            or self.internal_step % self.decision_interval == 0
        ):
            choice = self.choose_action(obs)
            return self.execute(choice, obs)

        # 4. Until the army is established, keep producing.
        if (
            self.internal_step % 8 == 0
            and state["free_supply"] > 1
            and state["minerals"] >= 50
            and state["marines"] < 4
        ):
            return self.train_marine(obs)

        return actions.RAW_FUNCTIONS.no_op()


def main(argv):
    del argv

    agent = MCDMAgent()

    interface = features.AgentInterfaceFormat(
        feature_dimensions=features.Dimensions(
            screen=84,
            minimap=64,
        ),
        use_feature_units=True,
        use_raw_units=True,
        use_raw_actions=True,
        raw_resolution=(64, 64),
        max_raw_actions=512,
        max_selected_units=30,
        show_placeholders=True,
    )

    print("\n========================================")
    print("STARTING SC2")
    print("MAP: Simple64")
    print("TERRAN vs ZERG EASY")
    print("RAW WORLD COORDINATES: 64x64")
    print("========================================\n")

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
        timestep = env.reset()

        while not timestep[0].last():
            action = agent.step(timestep[0])
            timestep = env.step([action])

    print("\n========================================")
    print("SIMULATION FINISHED")
    print("========================================")
    print(f"MCDM decisions : {agent.mcdm_decisions}")
    print(f"SC2 actions    : {agent.actions_sent}")
    print("========================================")


if __name__ == "__main__":
    app.run(main)

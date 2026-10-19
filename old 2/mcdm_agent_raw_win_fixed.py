import numpy as np
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import features, actions
from pysc2.agents import base_agent

# ------------------------------------------------------------
# MCDM + DIRECT RAW-UNIT SC2 EXECUTION
# ------------------------------------------------------------
# This version deliberately does NOT use select_point/screen actions.
# SC2 receives unit tags directly, so the agent cannot lose its SCV
# merely because the camera/screen coordinate changed.
#
# PySC2 raw actions are designed for direct unit-tag control and raw
# world coordinates.  See DeepMind PySC2 action definitions.
# ------------------------------------------------------------

class MCDMAgent(base_agent.BaseAgent):
    SCV = 45
    MARINE = 48
    BARRACKS = 21
    COMMAND_CENTER = 18
    SUPPLY_DEPOT = 19

    def __init__(self):
        super().__init__()

        # MCDM
        self.weights = self.ahp_weights()
        self.internal_step = 0
        self.decision_interval = 32
        self.last_choice = 1

        # Production / execution memory
        self.last_train_scv = -999999
        self.last_train_marine = -999999
        self.last_build = -999999
        self.last_attack = -999999

        self.scv_cooldown = 16
        self.marine_cooldown = 18
        self.build_cooldown = 48
        self.attack_cooldown = 32

        self.next_build_index = 0
        self.supply_target = 14

        # Construction / combat goals
        self.last_enemy_tag = None

        print("\n======================================================")
        print(" SC2 MCDM RAW AGENT - STABLE EXECUTION")
        print(" AHP + TOPSIS + VIKOR")
        print(" DIRECT UNIT-TAG ACTIONS")
        print(" NO SCREEN SELECTION")
        print(" TERRAN vs ZERG - EASY")
        print("======================================================\n")

    # ---------------- MCDM ----------------

    def ahp_weights(self):
        matrix = np.array([
            [1, 5, 3],
            [1/5, 1, 1/2],
            [1/3, 2, 1]
        ], dtype=float)
        return (matrix / matrix.sum(axis=0)).mean(axis=1)

    def topsis(self, matrix, weights):
        matrix = np.asarray(matrix, dtype=float)
        denom = np.sqrt((matrix ** 2).sum(axis=0))
        denom[denom < 1e-12] = 1.0
        weighted = matrix / denom * weights
        ideal = weighted.max(axis=0)
        anti = weighted.min(axis=0)
        dp = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
        dn = np.sqrt(((weighted - anti) ** 2).sum(axis=1))
        return dn / (dp + dn + 1e-12)

    def vikor(self, matrix, weights, v=0.5):
        matrix = np.asarray(matrix, dtype=float)
        best = matrix.max(axis=0)
        worst = matrix.min(axis=0)
        rng = best - worst
        rng[rng < 1e-12] = 1.0
        w = ((best - matrix) / rng) * weights
        S = w.sum(axis=1)
        R = w.max(axis=1)

        smin, smax = S.min(), S.max()
        rmin, rmax = R.min(), R.max()

        Sn = np.zeros_like(S) if smax-smin < 1e-12 else (S-smin)/(smax-smin)
        Rn = np.zeros_like(R) if rmax-rmin < 1e-12 else (R-rmin)/(rmax-rmin)
        return v*Sn + (1-v)*Rn

    # ---------------- RAW OBSERVATION ----------------

    def units(self, obs):
        return list(getattr(obs, "raw_units", []))

    def own(self, obs):
        return [u for u in self.units(obs) if int(u.alliance) == 1]

    def enemy(self, obs):
        return [u for u in self.units(obs) if int(u.alliance) == 4]

    def of_type(self, obs, unit_type, own=True):
        pool = self.own(obs) if own else self.enemy(obs)
        return [u for u in pool if int(u.unit_type) == unit_type]

    def ready(self, unit):
        return float(getattr(unit, "build_progress", 1.0)) >= 0.999

    def workers(self, obs):
        return [u for u in self.of_type(obs, self.SCV) if self.ready(u)]

    def command_centers(self, obs):
        return [u for u in self.of_type(obs, self.COMMAND_CENTER) if self.ready(u)]

    def barracks(self, obs):
        return [u for u in self.of_type(obs, self.BARRACKS) if self.ready(u)]

    def barracks_all(self, obs):
        return self.of_type(obs, self.BARRACKS)

    def depots(self, obs):
        return [u for u in self.of_type(obs, self.SUPPLY_DEPOT) if self.ready(u)]

    def marines(self, obs):
        return [u for u in self.of_type(obs, self.MARINE) if self.ready(u)]

    def minerals(self, obs):
        return float(obs.player.minerals)

    def vespene(self, obs):
        return float(obs.player.vespene)

    def food_used(self, obs):
        return float(obs.player.food_used)

    def food_cap(self, obs):
        return float(obs.player.food_cap)

    def state(self, obs):
        ws = self.workers(obs)
        bs = self.barracks(obs)
        ms = self.marines(obs)
        enemies = self.enemy(obs)

        m = self.minerals(obs)
        g = self.vespene(obs)
        used = self.food_used(obs)
        cap = self.food_cap(obs)

        enemy_count = len(enemies)
        military = len(ms)

        danger = min(
            10.0,
            enemy_count * 0.5 +
            max(0, enemy_count - military) * 0.3
        )

        if enemy_count:
            safety = 2.0 + military / enemy_count * 3.0
        else:
            safety = 10.0

        safety = max(0.0, min(10.0, safety - min(4.0, danger*0.25)))
        economy = min(10.0, m / 500.0)
        time_score = min(
            10.0,
            economy * 0.6 +
            (1.0 - min(1.0, danger/10.0)) * 2.0 +
            min(10.0, g/200.0) * 0.2
        )
        mobility = min(10.0, military*0.7 + (1.0-used/max(cap,1.0))*3.0)

        return {
            "minerals": m,
            "vespene": g,
            "food_used": used,
            "food_cap": cap,
            "supply_left": cap-used,
            "workers": len(ws),
            "barracks": len(bs),
            "barracks_all": len(self.barracks_all(obs)),
            "marines": military,
            "enemy": enemy_count,
            "danger": danger,
            "safety": safety,
            "time": time_score,
            "mobility": mobility,
        }

    def choose_mcdm(self, obs):
        s = self.state(obs)

        # Columns: safety, time, mobility.
        alternatives = np.array([
            # economy
            [s["safety"], min(10, 4+s["minerals"]/700), max(0, 8-s["marines"]*.25)],
            # production
            [min(10, s["safety"] + (3 if s["marines"] == 0 else 1)),
             min(10, 6+s["minerals"]/700),
             min(10, 7+(2 if s["marines"] == 0 else .5))],
            # attack
            [min(10, s["safety"]+s["marines"]*.4),
             min(10, 4+s["marines"]*.7),
             min(10, s["mobility"]+s["marines"]*.3)],
            # defense
            [min(10, s["safety"]+s["danger"]*.6),
             min(10, 5+s["danger"]),
             min(10, s["mobility"]+2)],
            # scouting
            [max(1,s["safety"]), min(10,5+s["mobility"]), min(10,5+s["mobility"])],
            # retreat
            [max(.1,10-s["danger"]), min(10,5+s["danger"]), max(.1,s["mobility"]+s["danger"])],
            # maintenance
            [s["safety"], min(10,5+s["minerals"]/1000), min(10,s["mobility"]+2)],
        ], dtype=float)

        t = self.topsis(alternatives, self.weights)
        v = self.vikor(alternatives, self.weights)

        # Never attack/retreat with no army.
        if s["marines"] == 0:
            t[2] = -1
            t[5] = -1
            v[2] = 999
            v[5] = 999

        names = [
            "EKONOMIJA", "PROIZVODNJA", "NAPAD",
            "OBRANA", "IZVIDANJE", "POVLACENJE", "ODRZAVANJE"
        ]

        ti = int(np.argmax(t))
        vi = int(np.argmin(v))

        # Operational overrides. MCDM chooses the strategic direction,
        # while RTS constraints decide the executable action.
        choice = ti

        if s["barracks"] == 0:
            choice = 1
        elif s["supply_left"] <= 1:
            choice = 1
        elif s["enemy"] > 0 and s["marines"] < s["enemy"]:
            choice = 3
        elif s["marines"] >= 8 and s["enemy"] > 0:
            choice = 2

        print("\n======================================================")
        print(" MCDM DECISION")
        print("======================================================")
        print(f"TOPSIS -> {names[ti]}")
        print(f"VIKOR  -> {names[vi]}")
        print(f"ACTION -> {names[choice]}")
        print("======================================================")

        return choice

    # ---------------- RAW ACTION HELPERS ----------------

    def raw_available(self, obs, func):
        # Raw ability functions in PySC2 4.x are executable through
        # RAW_FUNCTIONS. Resource/prerequisite checks are handled here.
        return func is not None

    def no_op(self):
        return actions.RAW_FUNCTIONS.no_op()

    def issue(self, func, queued, unit_tags, point=None, target_tag=None):
        if target_tag is not None:
            return func(queued, unit_tags, target_tag)
        if point is not None:
            return func(queued, unit_tags, point)
        return func(queued, unit_tags)

    def choose_worker(self, obs):
        workers = self.workers(obs)
        if not workers:
            return None

        # Prefer a worker not currently constructing.
        idle = []
        for w in workers:
            orders = getattr(w, "orders", [])
            if not orders:
                idle.append(w)

        return idle[0] if idle else workers[0]

    def choose_cc(self, obs):
        ccs = self.command_centers(obs)
        return ccs[0] if ccs else None

    def choose_build_point(self, obs):
        cc = self.choose_cc(obs)
        if cc is None:
            return (32, 32)

        x = int(cc.x)
        y = int(cc.y)

        # Raw coordinates are world coordinates, not screen pixels.
        offsets = [
            (8, 0), (0, 8), (-8, 0), (0, -8),
            (10, 6), (10, -6), (-10, 6), (-10, -6),
            (6, 10), (-6, 10), (6, -10), (-6, -10),
        ]

        p = offsets[self.next_build_index % len(offsets)]
        self.next_build_index += 1

        return (
            max(3, min(61, x+p[0])),
            max(3, min(61, y+p[1]))
        )

    # ---------------- PRODUCTION ----------------

    def train_scv(self, obs):
        s = self.state(obs)
        cc = self.choose_cc(obs)

        if cc is None:
            return self.no_op(), "NO_CC"

        if s["workers"] >= 16:
            return self.no_op(), "WORKER_TARGET_REACHED"

        if s["minerals"] < 50:
            return self.no_op(), "WAIT_50_MINERALS"

        if self.internal_step-self.last_train_scv < self.scv_cooldown:
            return self.no_op(), "SCV_COOLDOWN"

        self.last_train_scv = self.internal_step

        print(f"SC2 RAW ACTION: TRAIN_SCV CC={int(cc.tag)}")
        return (
            actions.RAW_FUNCTIONS.Train_SCV_quick(
                "now", [int(cc.tag)]
            ),
            "TRAIN_SCV"
        )

    def build_supply(self, obs):
        s = self.state(obs)
        w = self.choose_worker(obs)

        if s["supply_left"] > 2:
            return self.no_op(), "SUPPLY_OK"

        if s["minerals"] < 100:
            return self.no_op(), "WAIT_100_MINERALS"

        if w is None:
            return self.no_op(), "NO_SCV"

        if self.internal_step-self.last_build < 16:
            return self.no_op(), "BUILD_COOLDOWN"

        p = self.choose_build_point(obs)
        self.last_build = self.internal_step

        print(
            f"SC2 RAW ACTION: BUILD_SUPPLY_DEPOT "
            f"SCV={int(w.tag)} POS={p}"
        )

        return (
            actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", [int(w.tag)], p
            ),
            "BUILD_SUPPLY_DEPOT"
        )

    def build_barracks(self, obs):
        s = self.state(obs)

        # Critical: distinguish a Barracks under construction from
        # having no Barracks at all.
        if s["barracks"] > 0 or s["barracks_all"] > 0:
            return self.no_op(), "BARRACKS_EXISTS"

        if s["minerals"] < 150:
            return self.no_op(), "WAIT_150_MINERALS"

        if self.internal_step-self.last_build < self.build_cooldown:
            return self.no_op(), "BUILD_COOLDOWN"

        w = self.choose_worker(obs)
        if w is None:
            return self.no_op(), "NO_SCV"

        p = self.choose_build_point(obs)
        self.last_build = self.internal_step

        print(
            f"SC2 RAW ACTION: BUILD_BARRACKS "
            f"SCV={int(w.tag)} POS={p}"
        )

        return (
            actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", [int(w.tag)], p
            ),
            "BUILD_BARRACKS"
        )

    def train_marines(self, obs):
        s = self.state(obs)

        if s["barracks"] <= 0:
            return self.no_op(), "NO_READY_BARRACKS"

        if s["minerals"] < 50:
            return self.no_op(), "WAIT_50_MINERALS"

        if s["supply_left"] < 1:
            return self.no_op(), "WAIT_SUPPLY"

        if self.internal_step-self.last_train_marine < self.marine_cooldown:
            return self.no_op(), "MARINE_COOLDOWN"

        # Find every ready Barracks. If several are available, train from
        # all of them at once; raw tags make this safe.
        ready = [b for b in self.barracks(obs) if self.ready(b)]
        tags = [int(b.tag) for b in ready]

        if not tags:
            return self.no_op(), "NO_READY_BARRACKS"

        self.last_train_marine = self.internal_step

        print(
            f"SC2 RAW ACTION: TRAIN_MARINE "
            f"BARRACKS={tags}"
        )

        return (
            actions.RAW_FUNCTIONS.Train_Marine_quick(
                "now", tags
            ),
            "TRAIN_MARINE"
        )

    # ---------------- COMBAT ----------------

    def attack(self, obs):
        s = self.state(obs)
        marines = self.marines(obs)
        enemies = self.enemy(obs)

        if len(marines) < 6:
            return self.no_op(), "ATTACK_WAIT_ARMY"

        if not enemies:
            return self.no_op(), "ATTACK_NO_ENEMY"

        if self.internal_step-self.last_attack < self.attack_cooldown:
            return self.no_op(), "ATTACK_COOLDOWN"

        target = min(
            enemies,
            key=lambda e: (
                (float(e.x)-float(marines[0].x))**2 +
                (float(e.y)-float(marines[0].y))**2
            )
        )

        tags = [int(m.tag) for m in marines]
        self.last_attack = self.internal_step
        self.last_enemy_tag = int(target.tag)

        print(
            f"SC2 RAW ACTION: ATTACK "
            f"MARINES={len(tags)} TARGET={int(target.tag)}"
        )

        return (
            actions.RAW_FUNCTIONS.Attack_unit(
                "now", tags, int(target.tag)
            ),
            "ATTACK"
        )

    def defend(self, obs):
        marines = self.marines(obs)
        enemies = self.enemy(obs)

        if not marines or not enemies:
            return self.no_op(), "DEFENSE_WAIT"

        target = min(
            enemies,
            key=lambda e: (
                (float(e.x)-float(marines[0].x))**2 +
                (float(e.y)-float(marines[0].y))**2
            )
        )

        tags = [int(m.tag) for m in marines]

        if self.internal_step-self.last_attack < self.attack_cooldown:
            return self.no_op(), "DEFENSE_COOLDOWN"

        self.last_attack = self.internal_step

        print(
            f"SC2 RAW ACTION: DEFEND "
            f"MARINES={len(tags)} TARGET={int(target.tag)}"
        )

        return (
            actions.RAW_FUNCTIONS.Attack_unit(
                "now", tags, int(target.tag)
            ),
            "DEFEND"
        )

    # ---------------- STEP ----------------

    def step(self, timestep):
        super().step(timestep)
        self.internal_step += 1

        if timestep.last():
            return self.no_op()

        obs = timestep.observation
        s = self.state(obs)

        # --------------------------------------------------------
        # HARD RTS PRIORITIES
        # These execute before MCDM because the game itself imposes
        # prerequisites: supply, Barracks, then production, then fight.
        # --------------------------------------------------------

        # 1. Never get supply blocked.
        if s["supply_left"] <= 1:
            action, desc = self.build_supply(obs)
            if desc not in ("SUPPLY_OK", "WAIT_100_MINERALS", "NO_SCV",
                            "BUILD_COOLDOWN"):
                return action

        # 2. Build the first Barracks as soon as it is affordable.
        if s["barracks"] == 0 and s["barracks_all"] == 0:
            if s["minerals"] >= 150:
                action, desc = self.build_barracks(obs)
                if desc not in ("WAIT_150_MINERALS", "NO_SCV",
                                "BUILD_COOLDOWN"):
                    return action
            return self.no_op()

        # 3. Keep SCVs coming, but do not consume the whole economy.
        if self.internal_step % 16 == 0 and s["workers"] < 16:
            action, desc = self.train_scv(obs)
            if desc == "TRAIN_SCV":
                return action

        # 4. Once Barracks exists, spend minerals on Marines.
        if s["barracks"] > 0:
            action, desc = self.train_marines(obs)
            if desc == "TRAIN_MARINE":
                return action

        # 5. If enough Marines are present and an enemy is visible,
        #    attack instead of idling.
        if s["marines"] >= 6 and s["enemy"] > 0:
            action, desc = self.attack(obs)
            if desc == "ATTACK":
                return action

        # 6. MCDM strategic decision at fixed intervals.
        if self.internal_step == 1 or self.internal_step % self.decision_interval == 0:
            self.last_choice = self.choose_mcdm(obs)

            if self.last_choice == 2:
                action, desc = self.attack(obs)
            elif self.last_choice == 3:
                action, desc = self.defend(obs)
            elif self.last_choice == 0:
                action, desc = self.train_scv(obs)
            else:
                # Production strategy.
                if s["barracks"] == 0:
                    action, desc = self.build_barracks(obs)
                else:
                    action, desc = self.train_marines(obs)

            print(f"SC2 ACTION: {desc}")
            return action

        return self.no_op()


def main(argv):
    del argv

    agent = MCDMAgent()

    # IMPORTANT:
    # Raw action space + raw units. No screen selection.
    interface = features.AgentInterfaceFormat(
        use_raw_units=True,
        raw_resolution=64,
        action_space=actions.ActionSpace.RAW,
    )

    with sc2_env.SC2Env(
        map_name="Simple64",
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.easy),
        ],
        agent_interface_format=interface,
        step_mul=8,
        realtime=False,
        visualize=False,
    ) as env:

        timesteps = env.reset()

        while not timesteps[0].last():
            action = agent.step(timesteps[0])
            timesteps = env.step([action])

        result = timesteps[0]

        print("\n======================================================")
        print(" SIMULATION FINISHED")
        print("======================================================")
        print("If Outcome is [1], Terran won.")
        print("If Outcome is [-1], Terran lost.")
        print("======================================================\n")


if __name__ == "__main__":
    app.run(main)

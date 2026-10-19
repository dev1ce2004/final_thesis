import numpy as np
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import features, actions
from pysc2.agents import base_agent

class MCDMAgent(base_agent.BaseAgent):
    SCV = 45
    MARINE = 48
    BARRACKS = 21
    COMMAND_CENTER = 18
    SUPPLY_DEPOT = 19

    IDLE = None
    BUILD_BARRACKS = "BUILD_BARRACKS"
    TRAIN_MARINE = "TRAIN_MARINE"
    ATTACK = "ATTACK"
    DEFENSE = "DEFENSE"
    RETREAT = "RETREAT"

    def __init__(self):
        super().__init__()
        self.criteria_weights = self.ahp_weights()
        self.internal_step = 0
        self.decision_interval = 32
        self.last_mcdm_action = 0
        self.action_state = self.IDLE
        self.action_target = None
        self.action_unit_tag = None

        self.last_train_scv_step = -9999
        self.last_train_marine_step = -9999
        self.last_build_step = -9999
        self.last_combat_step = -9999

        self.train_scv_cooldown = 18
        self.train_marine_cooldown = 18
        self.build_cooldown = 48
        self.combat_cooldown = 12

        self.failed_build_positions = []

        print("\n========================================")
        print("SC2 MCDM AGENT - FIXED")
        print("AHP + TOPSIS + VIKOR")
        print("FEATURE ACTIONS + ACTION VALIDATOR")
        print("Terran vs Zerg Easy")
        print("========================================")
        print("\n========================================")
        print("AHP WEIGHTS")
        print("========================================")
        print(f"Sigurnost          : {self.criteria_weights[0]:.3f}")
        print(f"Vrijeme evakuacije : {self.criteria_weights[1]:.3f}")
        print(f"Prohodnost         : {self.criteria_weights[2]:.3f}")
        print("========================================\n")

    def ahp_weights(self):
        matrix = np.array([[1,5,3],[1/5,1,1/2],[1/3,2,1]], dtype=float)
        return (matrix / matrix.sum(axis=0)).mean(axis=1)

    def topsis(self, matrix, weights):
        matrix = np.asarray(matrix, dtype=float)
        denominator = np.sqrt((matrix ** 2).sum(axis=0))
        denominator[denominator < 1e-12] = 1.0
        weighted = (matrix / denominator) * weights
        ideal = weighted.max(axis=0)
        antiideal = weighted.min(axis=0)
        dp = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
        dn = np.sqrt(((weighted - antiideal) ** 2).sum(axis=1))
        return dn / (dp + dn + 1e-12)

    def vikor(self, matrix, weights, v=0.5):
        matrix = np.asarray(matrix, dtype=float)
        best = matrix.max(axis=0)
        worst = matrix.min(axis=0)
        denominator = best - worst
        denominator[denominator < 1e-12] = 1.0
        weighted = ((best - matrix) / denominator) * weights
        S = weighted.sum(axis=1)
        R = weighted.max(axis=1)
        sb, sw, rb, rw = S.min(), S.max(), R.min(), R.max()
        st = np.zeros_like(S) if sw-sb < 1e-12 else (S-sb)/(sw-sb)
        rt = np.zeros_like(R) if rw-rb < 1e-12 else (R-rb)/(rw-rb)
        return v*st + (1-v)*rt

    def is_available(self, observation, function):
        return function.id in observation.available_actions

    def get_units_state(self, observation):
        units = observation.feature_units
        own, enemy, workers, military, marines = [], [], [], [], []
        barracks, ccs, depots = [], [], []
        for unit in units:
            if unit.alliance == 1:
                own.append(unit)
                if unit.unit_type == self.SCV: workers.append(unit)
                elif unit.unit_type == self.BARRACKS: barracks.append(unit)
                elif unit.unit_type == self.COMMAND_CENTER: ccs.append(unit)
                elif unit.unit_type == self.SUPPLY_DEPOT: depots.append(unit)
                elif unit.unit_type == self.MARINE:
                    marines.append(unit); military.append(unit)
                else: military.append(unit)
            elif unit.alliance == 4:
                enemy.append(unit)
        return {"own_units":own,"enemy_units":enemy,"workers":workers,
                "military":military,"marines":marines,"barracks":barracks,
                "command_centers":ccs,"supply_depots":depots}

    def get_game_state(self, observation):
        p = observation.player
        u = self.get_units_state(observation)
        minerals, vespene = float(p.minerals), float(p.vespene)
        food_used, food_cap = float(p.food_used), float(p.food_cap)
        own_count, enemy_count = len(u["own_units"]), len(u["enemy_units"])
        worker_count, military_count = len(u["workers"]), len(u["military"])
        marine_count, barracks_count, cc_count = len(u["marines"]), len(u["barracks"]), len(u["command_centers"])
        supply_ratio = food_used/food_cap if food_cap > 0 else 0.0
        danger = min(10.0, enemy_count*0.5 + max(0.0, enemy_count-military_count)*0.3)
        if enemy_count:
            safety = 2.0 + (military_count/enemy_count)*3.0
        else:
            safety = 10.0
        safety = max(0.0, min(10.0, safety - min(4.0, danger*0.25)))
        economy_score = min(10.0, minerals/500.0)
        gas_score = min(10.0, vespene/200.0)
        time_score = max(0.0, min(10.0, economy_score*0.6 + gas_score*0.2 + (1.0-min(1.0,danger/10.0))*2.0))
        mobility = max(0.0, min(10.0, min(10.0,float(military_count))*0.7 + (1.0-supply_ratio)*3.0))
        return {"minerals":minerals,"vespene":vespene,"food_used":food_used,"food_cap":food_cap,
                "supply_ratio":supply_ratio,"own_units":own_count,"enemy_units":enemy_count,
                "military_units":military_count,"workers":worker_count,"marines":marine_count,
                "barracks":barracks_count,"command_centers":cc_count,"danger":danger,
                "safety":safety,"time":time_score,"mobility":mobility}

    def generate_alternatives(self, observation):
        s = self.get_game_state(observation)
        m, mil, danger, safety, mobility = s["minerals"], s["military_units"], s["danger"], s["safety"], s["mobility"]
        economy = [safety, min(10,4+m/700), max(0,8-mil*0.25)]
        production = [min(10,safety+(3 if mil==0 else 1)), min(10,6+m/700), min(10,7+(2 if mil==0 else .5))]
        attack = [.01,.01,.01] if mil==0 else [min(10,safety+mil*.4),min(10,4+mil*.7),min(10,mobility+mil*.3)]
        defense = [min(10,safety+danger*.6),min(10,5+danger),min(10,mobility+2)]
        scouting = [max(1,safety),min(10,5+mobility),min(10,5+mobility)]
        retreat = [.01,.01,.01] if mil==0 else [max(.1,10-danger),min(10,5+danger),max(.1,mobility+danger)]
        maintenance = [safety,min(10,5+m/1000),min(10,mobility+2)]
        return np.array([economy,production,attack,defense,scouting,retreat,maintenance], dtype=float)

    def choose_action(self, observation):
        s = self.get_game_state(observation)
        matrix = self.generate_alternatives(observation)
        ts = self.topsis(matrix,self.criteria_weights)
        vs = self.vikor(matrix,self.criteria_weights)
        names = ["EKONOMIJA","PROIZVODNJA","NAPAD","OBRANA","IZVIĐANJE","POVLAČENJE","ODRŽAVANJE"]
        if s["military_units"] == 0:
            ts[2], ts[5], vs[2], vs[5] = -1, -1, 999, 999
        tc, vc = int(np.argmax(ts)), int(np.argmin(vs))
        final = tc
        supply_left = s["food_cap"] - s["food_used"]
        if supply_left <= 1 or (s["barracks"] == 0 and s["workers"] > 0 and s["minerals"] >= 150) or s["military_units"] == 0:
            final = 1
        elif s["enemy_units"] > 0 and s["danger"] >= 5 and s["military_units"] < s["enemy_units"]:
            final = 3
        print("\n========================================\nSTANJE SC2\n========================================")
        print(f"Minerali          : {s['minerals']:.0f}\nVespene           : {s['vespene']:.0f}")
        print(f"Supply            : {s['food_used']:.0f}/{s['food_cap']:.0f}")
        print(f"Vlastite jedinice : {s['own_units']}\nVojne jedinice    : {s['military_units']}\nMarinci           : {s['marines']}")
        print(f"Barracks          : {s['barracks']}\nRadnici           : {s['workers']}\nNeprijatelji      : {s['enemy_units']}")
        print(f"Opasnost          : {s['danger']:.2f}\nSigurnost         : {s['safety']:.2f}\nVrijeme           : {s['time']:.2f}\nProhodnost        : {s['mobility']:.2f}")
        print("========================================\n\n========================================\nMCDM DECISION\n========================================")
        for i,n in enumerate(names): print(f"{n:15} TOPSIS={ts[i]:.3f} VIKOR={vs[i]:.3f}")
        print("----------------------------------------")
        print("TOPSIS ->",names[tc]); print("VIKOR  ->",names[vc])
        print("CONSENSUS ->" if tc==vc else "NO CONSENSUS ->",names[final])
        print("ACTION:",names[final])
        return final

    def get_worker(self,o):
        x=self.get_units_state(o)["workers"]; return x[0] if x else None
    def get_barracks(self,o):
        x=self.get_units_state(o)["barracks"]; return x[0] if x else None
    def get_marine(self,o):
        x=self.get_units_state(o)["marines"]; return x[0] if x else None
    def get_command_center(self,o):
        x=self.get_units_state(o)["command_centers"]; return x[0] if x else None

    def clear_action_state(self):
        self.action_state=None; self.action_target=None; self.action_unit_tag=None

    def target_still_exists(self, observation):
        if self.action_unit_tag is None: return False
        return any(u.alliance==1 and int(u.tag)==int(self.action_unit_tag) for u in observation.feature_units)

    def build_positions(self,o):
        cc=self.get_command_center(o)
        bx,by=(int(cc.x),int(cc.y)) if cc else (42,42)
        offsets=[(9,6),(10,-6),(-9,6),(-10,-6),(7,10),(-7,10),(7,-10),(-7,-10),(13,5),(-13,5),(13,-5),(-13,-5)]
        return [(max(5,min(78,bx+dx)),max(5,min(78,by+dy))) for dx,dy in offsets]

    def position_failed(self,p):
        return any(abs(p[0]-q[0])<=2 and abs(p[1]-q[1])<=2 for q in self.failed_build_positions)

    def choose_build_position(self,o):
        for p in self.build_positions(o):
            if not self.position_failed(p): return p
        self.failed_build_positions=[]; return self.build_positions(o)[0]

    def start_build_barracks(self,o):
        s=self.get_game_state(o)
        if s["barracks"]>0: return actions.FUNCTIONS.no_op(),"BARRACKS_READY"
        if s["minerals"]<150: return actions.FUNCTIONS.no_op(),"WAIT_150_MINERALS"
        if self.internal_step-self.last_build_step<self.build_cooldown: return actions.FUNCTIONS.no_op(),"BUILD_COOLDOWN"
        w=self.get_worker(o)
        if w is None: return actions.FUNCTIONS.no_op(),"NO_SCV"
        if not self.is_available(o,actions.FUNCTIONS.select_point): return actions.FUNCTIONS.no_op(),"SELECT_SCV_NOT_AVAILABLE"
        self.action_state=self.BUILD_BARRACKS; self.action_target=w; self.action_unit_tag=w.tag
        print(f"SC2 ACTION: SELECT_SCV_FOR_BARRACKS SCV={w.tag}")
        return actions.FUNCTIONS.select_point("select",(int(w.x),int(w.y))),"SELECT_SCV_FOR_BARRACKS"

    def continue_build_barracks(self,o):
        s=self.get_game_state(o)
        if s["barracks"]>0: self.clear_action_state(); return actions.FUNCTIONS.no_op(),"BARRACKS_READY"
        if not self.target_still_exists(o): self.clear_action_state(); return actions.FUNCTIONS.no_op(),"BUILD_SCV_LOST"
        if s["minerals"]<150: return actions.FUNCTIONS.no_op(),"WAIT_150_MINERALS"
        f=actions.FUNCTIONS.Build_Barracks_screen
        if not self.is_available(o,f): return actions.FUNCTIONS.no_op(),"WAIT_BUILD_BARRACKS"
        p=self.choose_build_position(o)
        print(f"SC2 ACTION: BUILD_BARRACKS POS={p}")
        self.last_build_step=self.internal_step; self.clear_action_state()
        return f("now",p),"BUILD_BARRACKS"

    def validate_train_marine(self,o):
        s=self.get_game_state(o)
        if s["barracks"]<=0:return False,"NO_BARRACKS"
        if s["minerals"]<50:return False,"WAIT_50_MINERALS"
        if s["food_cap"]-s["food_used"]<1:return False,"WAIT_SUPPLY"
        if self.internal_step-self.last_train_marine_step<self.train_marine_cooldown:return False,"TRAIN_MARINE_COOLDOWN"
        if not self.is_available(o,actions.FUNCTIONS.Train_Marine_quick):return False,"TRAIN_MARINE_NOT_AVAILABLE"
        return True,"OK"

    def start_train_marine(self,o):
        ok,r=self.validate_train_marine(o)
        if not ok:return actions.FUNCTIONS.no_op(),r
        b=self.get_barracks(o)
        if b is None:return actions.FUNCTIONS.no_op(),"NO_BARRACKS"
        if not self.is_available(o,actions.FUNCTIONS.select_point):return actions.FUNCTIONS.no_op(),"SELECT_BARRACKS_NOT_AVAILABLE"
        self.action_state=self.TRAIN_MARINE; self.action_target=b; self.action_unit_tag=b.tag
        return actions.FUNCTIONS.select_point("select",(int(b.x),int(b.y))),"SELECT_BARRACKS"

    def continue_train_marine(self,o):
        ok,r=self.validate_train_marine(o)
        if not ok:
            if r=="NO_BARRACKS":self.clear_action_state()
            return actions.FUNCTIONS.no_op(),r
        if not self.target_still_exists(o):self.clear_action_state();return actions.FUNCTIONS.no_op(),"BARRACKS_TARGET_LOST"
        print(f"SC2 ACTION: TRAIN_MARINE BARRACKS={self.action_unit_tag}")
        self.last_train_marine_step=self.internal_step; self.clear_action_state()
        return actions.FUNCTIONS.Train_Marine_quick("now"),"TRAIN_MARINE"

    def start_attack(self,o):
        s=self.get_game_state(o)
        if s["military_units"]<4:return actions.FUNCTIONS.no_op(),"ATTACK_WAIT_ARMY"
        enemies=self.get_units_state(o)["enemy_units"]
        if not enemies:return actions.FUNCTIONS.no_op(),"ATTACK_NO_ENEMY"
        if self.internal_step-self.last_combat_step<self.combat_cooldown:return actions.FUNCTIONS.no_op(),"COMBAT_COOLDOWN"
        m=self.get_marine(o)
        if m is None:return actions.FUNCTIONS.no_op(),"ATTACK_NO_MARINE"
        if not self.is_available(o,actions.FUNCTIONS.select_point):return actions.FUNCTIONS.no_op(),"ATTACK_SELECT_NOT_AVAILABLE"
        self.action_state=self.ATTACK; self.action_unit_tag=m.tag
        return actions.FUNCTIONS.select_point("select",(int(m.x),int(m.y))),"SELECT_MARINE_FOR_ATTACK"

    def continue_attack(self,o):
        e=self.get_units_state(o)["enemy_units"]
        if not e:self.clear_action_state();return actions.FUNCTIONS.no_op(),"ATTACK_NO_ENEMY"
        if not self.is_available(o,actions.FUNCTIONS.Attack_screen):return actions.FUNCTIONS.no_op(),"WAIT_ATTACK"
        t=e[0]; self.last_combat_step=self.internal_step; self.clear_action_state()
        return actions.FUNCTIONS.Attack_screen("now",(int(t.x),int(t.y))),"ATTACK"

    def start_defense(self,o):
        s=self.get_game_state(o); e=self.get_units_state(o)["enemy_units"]
        if not e:return actions.FUNCTIONS.no_op(),"DEFENSE_NO_ENEMY"
        if s["military_units"]<=0:return self.start_train_or_build(o)
        if self.internal_step-self.last_combat_step<self.combat_cooldown:return actions.FUNCTIONS.no_op(),"COMBAT_COOLDOWN"
        m=self.get_marine(o)
        if m is None:return actions.FUNCTIONS.no_op(),"DEFENSE_NO_MARINE"
        if not self.is_available(o,actions.FUNCTIONS.select_point):return actions.FUNCTIONS.no_op(),"DEFENSE_SELECT_NOT_AVAILABLE"
        self.action_state=self.DEFENSE;self.action_unit_tag=m.tag
        return actions.FUNCTIONS.select_point("select",(int(m.x),int(m.y))),"SELECT_MARINE_FOR_DEFENSE"

    def continue_defense(self,o):
        e=self.get_units_state(o)["enemy_units"]
        if not e:self.clear_action_state();return actions.FUNCTIONS.no_op(),"DEFENSE_NO_ENEMY"
        if not self.is_available(o,actions.FUNCTIONS.Attack_screen):return actions.FUNCTIONS.no_op(),"WAIT_DEFENSE"
        t=e[0];self.last_combat_step=self.internal_step;self.clear_action_state()
        return actions.FUNCTIONS.Attack_screen("now",(int(t.x),int(t.y))),"DEFEND"

    def start_retreat(self,o):
        s=self.get_game_state(o);m=self.get_marine(o);cc=self.get_command_center(o)
        if s["military_units"]<=0 or m is None or cc is None:return actions.FUNCTIONS.no_op(),"RETREAT_NO_TARGET"
        if not self.is_available(o,actions.FUNCTIONS.select_point):return actions.FUNCTIONS.no_op(),"RETREAT_SELECT_NOT_AVAILABLE"
        self.action_state=self.RETREAT;self.action_unit_tag=m.tag
        return actions.FUNCTIONS.select_point("select",(int(m.x),int(m.y))),"SELECT_MARINE_FOR_RETREAT"

    def continue_retreat(self,o):
        cc=self.get_command_center(o)
        if cc is None:self.clear_action_state();return actions.FUNCTIONS.no_op(),"RETREAT_NO_CC"
        if not self.is_available(o,actions.FUNCTIONS.Move_screen):return actions.FUNCTIONS.no_op(),"WAIT_RETREAT"
        self.last_combat_step=self.internal_step;self.clear_action_state()
        return actions.FUNCTIONS.Move_screen("now",(int(cc.x),int(cc.y))),"RETREAT"

    def train_scv(self,o):
        s=self.get_game_state(o)
        if s["workers"]>=16:return actions.FUNCTIONS.no_op(),"WORKER_TARGET_REACHED"
        if s["minerals"]<50:return actions.FUNCTIONS.no_op(),"WAIT_50_MINERALS"
        if self.internal_step-self.last_train_scv_step<self.train_scv_cooldown:return actions.FUNCTIONS.no_op(),"SCV_COOLDOWN"
        cc=self.get_command_center(o)
        if cc is None:return actions.FUNCTIONS.no_op(),"NO_CC"
        if not self.is_available(o,actions.FUNCTIONS.select_point):return actions.FUNCTIONS.no_op(),"SELECT_CC_NOT_AVAILABLE"
        self.action_state="TRAIN_SCV";self.action_unit_tag=cc.tag
        return actions.FUNCTIONS.select_point("select",(int(cc.x),int(cc.y))),"SELECT_CC_FOR_SCV"

    def continue_train_scv(self,o):
        s=self.get_game_state(o)
        if s["workers"]>=16:self.clear_action_state();return actions.FUNCTIONS.no_op(),"WORKER_TARGET_REACHED"
        if s["minerals"]<50:return actions.FUNCTIONS.no_op(),"WAIT_50_MINERALS"
        if not self.is_available(o,actions.FUNCTIONS.Train_SCV_quick):return actions.FUNCTIONS.no_op(),"WAIT_TRAIN_SCV"
        self.last_train_scv_step=self.internal_step;self.clear_action_state()
        return actions.FUNCTIONS.Train_SCV_quick("now"),"TRAIN_SCV"

    def start_build_supply_depot(self,o):
        s=self.get_game_state(o)
        if s["minerals"]<100:return actions.FUNCTIONS.no_op(),"WAIT_100_MINERALS"
        w=self.get_worker(o)
        if w is None:return actions.FUNCTIONS.no_op(),"NO_SCV"
        if not self.is_available(o,actions.FUNCTIONS.select_point):return actions.FUNCTIONS.no_op(),"SELECT_SCV_NOT_AVAILABLE"
        self.action_state="BUILD_SUPPLY";self.action_unit_tag=w.tag
        return actions.FUNCTIONS.select_point("select",(int(w.x),int(w.y))),"SELECT_SCV_FOR_SUPPLY"

    def continue_build_supply_depot(self,o):
        s=self.get_game_state(o)
        if s["minerals"]<100:return actions.FUNCTIONS.no_op(),"WAIT_100_MINERALS"
        f=actions.FUNCTIONS.Build_SupplyDepot_screen
        if not self.is_available(o,f):return actions.FUNCTIONS.no_op(),"WAIT_BUILD_SUPPLY"
        cc=self.get_command_center(o)
        if cc:
            p=(max(5,min(78,int(cc.x)+6)),max(5,min(78,int(cc.y)+8)))
        else:p=(40,40)
        self.clear_action_state()
        return f("now",p),"BUILD_SUPPLY_DEPOT"

    def start_train_or_build(self,o):
        s=self.get_game_state(o);left=s["food_cap"]-s["food_used"]
        if left<=1:return self.start_build_supply_depot(o)
        if s["barracks"]==0:
            if s["minerals"]<150:return actions.FUNCTIONS.no_op(),"WAIT_150_MINERALS"
            return self.start_build_barracks(o)
        return self.start_train_marine(o)

    def continue_current_action(self,o):
        if self.action_state==self.BUILD_BARRACKS:return self.continue_build_barracks(o)
        if self.action_state==self.TRAIN_MARINE:return self.continue_train_marine(o)
        if self.action_state=="TRAIN_SCV":return self.continue_train_scv(o)
        if self.action_state=="BUILD_SUPPLY":return self.continue_build_supply_depot(o)
        if self.action_state==self.ATTACK:return self.continue_attack(o)
        if self.action_state==self.DEFENSE:return self.continue_defense(o)
        if self.action_state==self.RETREAT:return self.continue_retreat(o)
        self.clear_action_state();return actions.FUNCTIONS.no_op(),"NO_OP"

    def execute_mcdm_action(self,choice,o):
        if choice==0:return self.train_scv(o)
        if choice==1:return self.start_train_or_build(o)
        if choice==2:return self.start_attack(o)
        if choice==3:return self.start_defense(o)
        if choice==4:return self.start_train_or_build(o)
        if choice==5:return self.start_retreat(o)
        if choice==6:return self.start_train_or_build(o)
        return actions.FUNCTIONS.no_op(),"NO_OP"

    def step(self,obs):
        super().step(obs);self.internal_step+=1;o=obs.observation
        if obs.last():return actions.FUNCTIONS.no_op()
        if self.action_state is not None:
            action,desc=self.continue_current_action(o)
            if not desc.startswith(("WAIT_","NO_","WORKER_","SCV_COOLDOWN","TRAIN_MARINE_COOLDOWN","BUILD_COOLDOWN","COMBAT_COOLDOWN","SELECT_")):
                print("SC2 ACTION:",desc)
            return action

        s=self.get_game_state(o);left=s["food_cap"]-s["food_used"]
        if left<=1:
            a,d=self.start_build_supply_depot(o)
            if d not in ("WAIT_100_MINERALS","NO_SCV","SELECT_SCV_NOT_AVAILABLE"):return a
        if s["barracks"]==0 and s["minerals"]>=150 and s["workers"]>0:
            a,d=self.start_build_barracks(o)
            if d not in ("WAIT_150_MINERALS","BUILD_COOLDOWN","NO_SCV","SELECT_SCV_NOT_AVAILABLE"):return a
        if s["barracks"]>0 and s["minerals"]>=50 and left>1:
            a,d=self.start_train_marine(o)
            if d=="SELECT_BARRACKS":return a

        if self.internal_step==1 or self.internal_step%self.decision_interval==0:
            self.last_mcdm_action=self.choose_action(o)
            a,d=self.execute_mcdm_action(self.last_mcdm_action,o)
            print("SC2 ACTION:",d)
            return a
        return actions.FUNCTIONS.no_op()

def main(argv):
    agent=MCDMAgent()
    interface=features.AgentInterfaceFormat(
        feature_dimensions=features.Dimensions(screen=84,minimap=64),
        use_feature_units=True,
    )
    with sc2_env.SC2Env(
        map_name="Simple64",
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.zerg,sc2_env.Difficulty.easy),
        ],
        agent_interface_format=interface,
        step_mul=8,
        realtime=False,
        visualize=False,
    ) as env:
        timesteps=env.reset()
        while not timesteps[0].last():
            action=agent.step(timesteps[0])
            timesteps=env.step([action])
    print("\n========================================")
    print("SIMULATION FINISHED")
    print("========================================")

if __name__=="__main__":
    app.run(main)

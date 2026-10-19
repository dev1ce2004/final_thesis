from absl import app
from pysc2 import run_configs
from pysc2 import maps
from pysc2.env import sc2_env


def main(argv):
    run_config = run_configs.get()

    print("Pokrecem StarCraft II...")

    with sc2_env.SC2Env(
        map_name="Simple64",
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.easy),
        ],
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_screen=64,
            feature_minimap=64,
        ),
        step_mul=8,
        realtime=False,
    ) as env:

        print("SC2 SESIJA USPJESNO POKRENUTA!")

        timestep = env.reset()[0]

        print("Observation dobiven!")
        print("Reward:", timestep.reward)
        print("Game loop:", timestep.observation.game_loop)

    print("SC2 TEST ZAVRSEN.")


if __name__ == "__main__":
    app.run(main)
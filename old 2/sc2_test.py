from absl import app
from pysc2 import run_configs


def main(argv):
    print("Pokrecem SC2...")

    run_config = run_configs.get()

    print("PySC2 konfiguracija pronadena!")
    print("Tip:", type(run_config))

    print("TEST USPJESAN")


if __name__ == "__main__":
    app.run(main)
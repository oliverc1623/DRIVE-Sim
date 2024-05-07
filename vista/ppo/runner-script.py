import subprocess

def run_script(script_name, args=None):
    if args is None:
        args = []
    subprocess.run(["python", script_name] + args, check=True)

def main():
    # Running the first script with arguments
    try:
        trial = 5
        print(f"Running PPO Trial {trial}")
        run_script("train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

    try:
        trial = 6
        print(f"Running PPO Trial {trial}")
        run_script("train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

    try:
        trial = 7
        print(f"Running PPO Trial {trial}")
        run_script("train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

    try:
        trial = 8
        print(f"Running PPO Trial {trial}")
        run_script("train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

if __name__ == "__main__":
    main()

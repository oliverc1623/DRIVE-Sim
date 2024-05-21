import subprocess

def run_script(script_name, args=None):
    if args is None:
        args = []
    subprocess.run(["python", script_name] + args, check=True)

def main():
    # Running the first script with arguments
    try:
        trial = 100
        print(f"Running PPO Trial {trial}")
        run_script("train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

    try:
        trial = 101
        print(f"Running PPO Trial {trial}")
        run_script("train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

     try:
        trial = 102
        print(f"Running PPO Trial {trial}")
        run_script("train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

    try:
        trial = 103
        print(f"Running PPO Trial {trial}")
        run_script("train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

     try:
        trial = 104
        print(f"Running PPO Trial {trial}")
        run_script("train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

    try:
        trial = 105
        print(f"Running PPO Trial {trial}")
        run_script("train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

    ### LSTM ###
    try:
        trial = 100
        print(f"Running PPO Trial {trial}")
        run_script("lstm_train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

    ### LSTM ###
    try:
        trial = 101
        print(f"Running PPO Trial {trial}")
        run_script("lstm_train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

    ### LSTM ###
    try:
        trial = 102
        print(f"Running PPO Trial {trial}")
        run_script("lstm_train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

    ### LSTM ###
    try:
        trial = 103
        print(f"Running PPO Trial {trial}")
        run_script("lstm_train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

    ### LSTM ###
    try:
        trial = 104
        print(f"Running PPO Trial {trial}")
        run_script("lstm_train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

    ### LSTM ###
    try:
        trial = 105
        print(f"Running PPO Trial {trial}")
        run_script("lstm_train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return

    ### LSTM ###
    try:
        trial = 106
        print(f"Running PPO Trial {trial}")
        run_script("lstm_train_ppo.py", [f"{trial}"])
        print(f"ppo trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"ppo trial {trial}  failed.")
        return
if __name__ == "__main__":
    main()

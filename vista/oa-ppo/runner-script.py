import subprocess

def run_script(script_name, args=None):
    if args is None:
        args = []
    subprocess.run(["python", script_name] + args, check=True)

def main():
    # Running the first script with arguments
    try:
        run_script("train_ppo.py", ["1"])
        print("ppo trial 1 finished successfully.")
    except subprocess.CalledProcessError:
        print("ppo trial 1  failed.")
        return

    try:
        run_script("train_ppo.py", ["2"])
        print("ppo trial 2 finished successfully.")
    except subprocess.CalledProcessError:
        print("ppo trial 2  failed.")
        return

    try:
        run_script("train_ppo.py", ["3"])
        print("ppo trial 3 finished successfully.")
    except subprocess.CalledProcessError:
        print("ppo trial 3  failed.")
        return

    try:
        run_script("train_ppo.py", ["4"])
        print("ppo trial 4 finished successfully.")
    except subprocess.CalledProcessError:
        print("ppo trial 4  failed.")
        return

if __name__ == "__main__":
    main()

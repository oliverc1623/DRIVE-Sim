import subprocess

def run_script(script_name, args=None):
    if args is None:
        args = []
    subprocess.run(["python", script_name] + args, check=True)

def main():
    # Running the first script with arguments
    try:
        run_script("train_ddpg.py", ["2"])
        print("ddpg trial 2 finished successfully.")
    except subprocess.CalledProcessError:
        print("ddpg trial 2  failed.")
        return

if __name__ == "__main__":
    main()

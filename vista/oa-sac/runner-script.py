import subprocess

def run_script(script_name, args=None):
    if args is None:
        args = []
    subprocess.run(["python", script_name] + args, check=True)

def main():
    # Running the first script with arguments
    try:
        run_script("train_sac.py", ["1"])
        print("sac trial 1 finished successfully.")
    except subprocess.CalledProcessError:
        print("sac trial 1  failed.")
        return

    # Running the second script without arguments
    try:
        run_script("train_sac.py", ["2"])
        print("sac trial 2 finished successfully.")
    except subprocess.CalledProcessError:
        print("sac trial 2  failed.")
        return

if __name__ == "__main__":
    main()

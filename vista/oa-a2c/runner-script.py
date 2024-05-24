import subprocess

def run_script(script_name, args=None):
    if args is None:
        args = []
    subprocess.run(["python", script_name] + args, check=True)

def main():
    # Running the first script with arguments
    try:
        run_script("train_a2c.py", ["1"])
        print("a2c trial 1 finished successfully.")
    except subprocess.CalledProcessError:
        print("a2c trial 1  failed.")
        return

    try:
        run_script("train_a2c.py", ["2"])
        print("a2c trial 2 finished successfully.")
    except subprocess.CalledProcessError:
        print("a2c trial 2  failed.")
        return

    try:
        run_script("train_a2c.py", ["3"])
        print("a2c trial 3 finished successfully.")
    except subprocess.CalledProcessError:
        print("a2c trial 3  failed.")
        return

if __name__ == "__main__":
    main()

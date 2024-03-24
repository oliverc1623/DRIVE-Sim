import subprocess

def run_script(script_name, args=None):
    if args is None:
        args = []
    subprocess.run(["python", script_name] + args, check=True)

def main():
    Running the first script with arguments
    try:
        run_script("train_td3.py", ["1"])
        print("td3 trial 1 finished successfully.")
    except subprocess.CalledProcessError:
        print("td3 trial 1  failed.")
        return

    try:
        run_script("train_td3.py", ["2"])
        print("td3 trial 2 finished successfully.")
    except subprocess.CalledProcessError:
        print("td3 trial 2  failed.")
        return

    try:
        run_script("train_td3.py", ["3"])
        print("td3 trial 3 finished successfully.")
    except subprocess.CalledProcessError:
        print("td3 trial 3  failed.")
        return

    try:
        run_script("train_td3.py", ["4"])
        print("td3 trial 4 finished successfully.")
    except subprocess.CalledProcessError:
        print("td3 trial 4  failed.")
        return

if __name__ == "__main__":
    main()

import subprocess

def run_script(script_name, args=None):
    if args is None:
        args = []
    subprocess.run(["python", script_name] + args, check=True)

def main():
    # Running the first script with arguments
    try:
        trial = 1
        print(f"Running td3 Trial {trial}")
        run_script("train_td3.py", [f"{trial}"])
        print(f"td3 trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"td3 trial {trial}  failed.")
        return

    try:
        trial = 2
        print(f"Running td3 Trial {trial}")
        run_script("train_td3.py", [f"{trial}"])
        print(f"td3 trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"td3 trial {trial}  failed.")
        return

    try:
        trial = 3
        print(f"Running td3 Trial {trial}")
        run_script("train_td3.py", [f"{trial}"])
        print(f"td3 trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"td3 trial {trial}  failed.")
        return

    try:
        trial = 4
        print(f"Running td3 Trial {trial}")
        run_script("train_td3.py", [f"{trial}"])
        print(f"td3 trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"td3 trial {trial}  failed.")
        return

    try:
        trial = 5
        print(f"Running td3 Trial {trial}")
        run_script("train_td3.py", [f"{trial}"])
        print(f"td3 trial {trial} finished successfully.")
    except subprocess.CalledProcessError:
        print(f"td3 trial {trial}  failed.")
        return

if __name__ == "__main__":
    main()

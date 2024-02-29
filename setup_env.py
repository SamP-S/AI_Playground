import os
import sys

def get_next_version(dir):
    folders = [folder for folder in os.listdir(dir) if os.path.isdir(os.path.join(dir, folder))]
    
    if len(folders) == 0:
        print("WARNING: no previous version found, assuming v1")
        return os.path.join(dir, "v1")
    
    highest_v = 0
    for folder in folders:
        try:
            v = int(folder[1:])  
            if v > highest_v:
                highest_v = v
        except ValueError:
            pass
    return os.path.join(dir, "v" + str(highest_v + 1))


def setup_output_dir(path):
    os.makedirs(path, exist_ok=True)
    v_path = get_next_version(path)
    os.mkdir(v_path)
    print(f"INFO: created dir @ {v_path}")
    return v_path

if __name__ == "__main__":
    if len(sys.argv) == 1:
        setup_output_dir("models/test")
    else:
        setup_output_dir(sys.argv[1])

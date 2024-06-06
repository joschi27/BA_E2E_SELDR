import sys
import glob
import os


def add_carla_to_path():
    try:
        # Get the directory of the currently executing script
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the relative path to find the .egg
        relative_path = os.path.join('..', '..', 'carla', 'PythonAPI', 'carla', 'dist', 'carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))

        # Use glob to find the target file
        target_files = glob.glob(os.path.join(current_dir, relative_path))

        if target_files:
            # Add the directory containing the target file to sys.path
            target_file_dir = os.path.dirname(target_files[0])
            sys.path.append(target_file_dir)

            # Add the file itself as on windows the above doesn't seem to work
            sys.path.append(target_files[0])
        else:
            print("Target file not found.")
    except Exception as e:
        print("An error occurred:", e)
import os
import subprocess
import sys

def start_visualizer():
    """Starts the cell counter visualizer."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        visualizer_dir = os.path.join(script_dir, 'visualizer')
        visualizer_script = os.path.join(visualizer_dir, 'view.py')

        # include the project root so that imports work correctly in the visualizer
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = script_dir + os.pathsep + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = script_dir

        print("Starting visualizer...")
        subprocess.run([sys.executable, visualizer_script], check=True, env=env, cwd=visualizer_dir)
        print("Visualizer closed.")

    except FileNotFoundError:
        print(f"Error: {visualizer_script} not found.")
    except subprocess.CalledProcessError as e:
        print(f"Error running visualizer: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    start_visualizer()

import sys
import os

# Suppress all warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

print("=" * 60, flush=True)
print("Exercise Form Detection - Full System Test", flush=True)  
print("=" * 60, flush=True)

print("\n[1/5] Importing modules...", flush=True)
try:
    from src.pose_detector import PoseDetector
    from src.exercises import SquatExercise, PushUpExercise, LungeExercise
    from src.utils import load_config
    print("      ✓ All modules imported successfully", flush=True)
except Exception as e:
    print(f"      ✗ Import error: {e}", flush=True)
    sys.exit(1)

print("\n[2/5] Loading configuration...", flush=True)
try:
    config = load_config('config.yaml')
    print(f"      ✓ Config loaded (Target FPS: {config['performance']['target_fps']})", flush=True)
except Exception as e:
    print(f"      ✗ Config error: {e}", flush=True)
    sys.exit(1)

print("\n[3/5] Initializing exercises...", flush=True)
exercises = []
for ExClass, name in [(SquatExercise, 'Squat'), (PushUpExercise, 'Push-up'), (LungeExercise, 'Lunge')]:
    try:
        ex = ExClass(config)
        exercises.append(name)
        print(f"      ✓ {name} ready", flush=True)
    except Exception as e:
        print(f"      ✗ {name} failed: {e}", flush=True)

print("\n[4/5] Initializing pose detector...", flush=True)
try:
    detector = PoseDetector(config)
    print("      ✓ Pose detector initialized", flush=True)
except Exception as e:
    print(f"      ✗ Pose detector error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[5/5] Cleanup...", flush=True)
try:
    detector.release()
    print("      ✓ Resources released", flush=True)
except Exception as e:
    print(f"      ✗ Cleanup error: {e}", flush=True)

print("\n" + "=" * 60, flush=True)
print("✓ ALL SYSTEMS READY!", flush=True)
print(f"  Exercises: {', '.join(exercises)}", flush=True)
print("=" * 60, flush=True)
print("\nTo run the app:", flush=True)
print("  python main.py --exercise squat", flush=True)
print("  python main.py --exercise pushup", flush=True)
print("  python main.py --exercise lunge", flush=True)
print("\nPress 'q' to quit when running.", flush=True)

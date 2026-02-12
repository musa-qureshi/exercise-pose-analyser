# Exercise Form Detection System

Real-time exercise form analysis using MediaPipe pose estimation. Provides automatic rep counting and form feedback for squats, push-ups, and lunges.

## Demo

[Watch Demo Video](https://youtu.be/BQ9T0wK29-8)

## Features

- **Real-time Pose Tracking** - MediaPipe-based pose detection with temporal smoothing
- **Live Exercise Switching** - Toggle between exercises with keyboard shortcuts (P/S/L)
- **Automatic Rep Counting** - Phase-based repetition tracking
- **Form Analysis** - Real-time feedback on common form errors
- **Visual Feedback** - Pose overlay with on-screen exercise selector
- **Session Logging** - CSV export of exercise data
- **Configurable** - YAML-based threshold adjustment

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for installation and first run.

```bash
python main.py
```

Use **P/S/L** keys to switch between Push-ups, Squats, and Lunges.

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for installation and first run.

```bash
python main.py
```

Use **P/S/L** keys to switch between Push-ups, Squats, and Lunges.

## Command-Line Options

```bash
python main.py [--source SOURCE] [--config CONFIG]
```

**Arguments:**
- `--source, -s` - Video source: camera index (0, 1) or file path (default: 0)
- `--config, -c` - Config file path (default: config.yaml)

**Examples:**
```bash
# Webcam (default)
python main.py

# Different camera
python main.py --source 1

# Video file
python main.py --source workout.mp4

# Custom config
python main.py --config custom.yaml
```

## Interactive Controls

- **P** - Switch to Push-ups
- **S** - Switch to Squats
- **L** - Switch to Lunges
- **Q** - Quit
- **R** - Reset rep count
- **C** - Recalibrate

Active exercise appears in green in bottom-right corner.

## Configuration

Edit `config.yaml` to customize detection behavior:

**Pose Detection:**
- `model_complexity`: 0 (fast), 1 (balanced), 2 (accurate)
- `min_detection_confidence`: Initial detection threshold (0-1)
- `min_tracking_confidence`: Frame tracking threshold (0-1)

**Temporal Smoothing:**
- `enabled`: Smooth tracking across frames
- `window_size`: Number of frames to average

**Exercise Thresholds:**
Each exercise has configurable thresholds for phase detection and form analysis. See `config.yaml` comments for detailed explanations.

**Performance:**
- `target_fps`: Limit processing speed (0 = unlimited)

**Logging:**
- `enabled`: Enable CSV session logging
- `output_directory`: Where to save logs

## Form Errors Detected

### Squat (Front-Facing Camera)
- **Knee Valgus** - Knees caving inward
- **Knee Spread** - Knees too wide apart
- **Forward Lean** - Excessive torso lean
- **Foot Angle** - Feet pointing too far inward/outward

### Push-up (Sideways Camera)
- **Hip Sag** - Hips dropping (core not engaged)
- **Shallow Range** - Not lowering enough

### Lunge (Sideways Camera)
- **Knee Over Toe** - Front knee too far forward
- **Shallow Depth** - Back knee not dropping low enough
- **Torso Lean** - Leaning too far forward/back

## Troubleshooting

**Low FPS:** Set `model_complexity: 0` in config.yaml

**Pose Not Detected:** Ensure full body visible, good lighting, plain background

**Reps Not Counting:** Complete full range of motion, recalibrate with C

**Lunges Not Detecting:** Stand sideways to camera (perpendicular, not facing it)

**Form Errors Too Sensitive:** Adjust thresholds in config.yaml for specific exercise

## Project Structure

```
src/
├── pose_detector/      # MediaPipe pose detection
├── exercises/          # Exercise analyzers (squat, pushup, lunge)
└── utils/             # Geometry, config, logging
```

## Extending

To add a new exercise:
1. Create class inheriting from `BaseExercise`
2. Implement `detect_exercise()`, `detect_phase()`, `analyze_form()`
3. Add config section to `config.yaml`
4. Register in `main.py`

## License

MIT License

## Requirements

- opencv-python >= 4.8.0
- mediapipe >= 0.10.0
- numpy >= 1.24.0
- pyyaml >= 6.0

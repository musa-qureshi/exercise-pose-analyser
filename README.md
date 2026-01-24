# Exercise Form Detection System

Production-grade real-time exercise form detection and feedback system using single-camera pose estimation with temporal modeling. Provides intelligent form analysis, rep counting, and actionable feedback for squat, push-up, and lunge exercises.

## Features

✨ **Real-time Pose Estimation** using MediaPipe with temporal smoothing for stable tracking  
🎯 **Exercise Classification** - Automatically detects squat, push-up, and lunge exercises  
📊 **Phase Segmentation** - Tracks exercise phases (up, down, hold)  
🔢 **Accurate Rep Counting** - Automatic repetition counting based on phase transitions  
⚠️ **Form Error Detection** - Identifies and provides feedback on:
- **Squat**: Knee valgus, shallow depth, excessive forward lean
- **Push-up**: Sagging hips, incomplete ROM
- **Lunge**: Knee over toe, insufficient depth, torso lean

🎨 **Visual Feedback** - Real-time pose overlay and text feedback  
👤 **User Calibration** - Automatic calibration to user's body proportions  
📝 **CSV Logging** - Session data logging for analysis  
⚙️ **Config-driven** - Easily adjustable thresholds via YAML  
🏗️ **Modular Design** - Clean, extensible architecture  
⚡ **High Performance** - Optimized for ≥20 FPS on standard hardware

## Installation

### Prerequisites
- **Python 3.9+** (including Python 3.13)
- Webcam or video file for input
- (Optional) CUDA-capable GPU for better performance

### Setup

1. Clone the repository:
```bash
git clone https://github.com/musa-qureshi/pose-based-exercise-analyser.git
cd pose-based-exercise-analyser
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** On first run, the app will automatically download the MediaPipe pose model (~30MB).

## Usage

### Basic Usage

Run with default settings (squat exercise, webcam):
```bash
python main.py
```

### Specify Exercise Type

Analyze push-ups:
```bash
python main.py --exercise pushup
```

Analyze lunges:
```bash
python main.py --exercise lunge
```

### Use Video File

Analyze from video file instead of webcam:
```bash
python main.py --exercise squat --source path/to/video.mp4
```

### Custom Configuration

Use custom configuration file:
```bash
python main.py --config my_config.yaml
```

### Command-Line Options

```
usage: main.py [-h] [--exercise {squat,pushup,push-up,lunge}] 
               [--source SOURCE] [--config CONFIG]

Real-time Exercise Form Detection System

optional arguments:
  -h, --help            show this help message and exit
  --exercise {squat,pushup,push-up,lunge}, -e
                        Exercise type to detect (default: squat)
  --source SOURCE, -s SOURCE
                        Video source: camera index (0, 1, ...) or 
                        video file path (default: 0)
  --config CONFIG, -c CONFIG
                        Path to configuration file (default: config.yaml)
```

### Interactive Controls

While running:
- **'q'** - Quit the application
- **'r'** - Reset rep count
- **'c'** - Recalibrate

## Configuration

Edit `config.yaml` to customize detection thresholds and behavior:

### Key Configuration Sections

**Pose Detection:**
- `model_complexity`: 0-2 (higher = more accurate but slower)
- `min_detection_confidence`: Detection confidence threshold (0-1)
- `min_tracking_confidence`: Tracking confidence threshold (0-1)

**Temporal Smoothing:**
- `enabled`: Enable/disable smoothing
- `window_size`: Number of frames to average (higher = smoother but more lag)

**Exercise Thresholds:**
Each exercise has specific thresholds for form analysis:
- **Squat**: Depth thresholds, knee valgus angle, forward lean angle
- **Push-up**: Hip sag threshold, ROM angles
- **Lunge**: Knee alignment, depth threshold, torso lean angle

**Performance:**
- `target_fps`: Target frames per second
- `max_fps`: Maximum FPS cap

**Logging:**
- `enabled`: Enable/disable CSV logging
- `output_directory`: Directory for log files
- `session_prefix`: Prefix for log filenames

## Architecture

### Project Structure

```
pose-based-exercise-analyser/
├── main.py                    # Main application entry point
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
│
├── src/
│   ├── pose_detector/       # Pose detection module
│   │   └── pose_detector.py # MediaPipe pose detection with smoothing
│   │
│   ├── exercises/           # Exercise analysis modules
│   │   ├── base_exercise.py # Abstract base class for exercises
│   │   ├── squat.py        # Squat analysis
│   │   ├── pushup.py       # Push-up analysis
│   │   └── lunge.py        # Lunge analysis
│   │
│   └── utils/              # Utility modules
│       ├── geometry.py     # Angle and distance calculations
│       ├── config_loader.py # Configuration loading
│       └── logger.py       # CSV session logging
│
└── logs/                   # Session logs (created at runtime)
```

### Key Components

**PoseDetector** (`src/pose_detector/`)
- MediaPipe-based pose detection
- Temporal smoothing for stability
- Landmark extraction and visualization

**Exercise Analyzers** (`src/exercises/`)
- Base class defining exercise interface
- Exercise-specific implementations (squat, push-up, lunge)
- Phase detection and rep counting
- Form error detection with severity levels

**Geometry Utilities** (`src/utils/geometry.py`)
- Angle calculations
- Distance measurements
- Body side detection
- Landmark normalization

**Session Logger** (`src/utils/logger.py`)
- CSV logging of frame-by-frame data
- Session summary statistics
- Performance metrics

## Form Error Detection

### Squat
1. **Knee Valgus** - Detects knees caving inward
2. **Shallow Depth** - Ensures hips go below parallel
3. **Forward Lean** - Checks for excessive torso lean

### Push-up
1. **Hip Sag** - Detects broken plank position
2. **Incomplete ROM** - Ensures elbows reach 90°

### Lunge
1. **Knee Over Toe** - Ensures front knee stays over ankle
2. **Insufficient Depth** - Checks back knee approaches ground
3. **Torso Lean** - Maintains upright torso position

## Performance Optimization

The system is optimized for ≥20 FPS on standard hardware:

- Efficient MediaPipe Pose model (complexity level 1 by default)
- Temporal smoothing with small window (5 frames)
- Minimal overhead processing
- Configurable FPS targeting

**Tips for better performance:**
- Lower `model_complexity` to 0 for faster processing
- Reduce video resolution if needed
- Ensure good lighting for better pose detection
- Use a solid background for easier detection

## Extending the System

### Adding a New Exercise

1. Create new exercise class inheriting from `BaseExercise`
2. Implement required methods:
   - `detect_exercise()` - Detect if exercise is being performed
   - `detect_phase()` - Determine current phase
   - `analyze_form()` - Detect form errors
3. Add configuration section to `config.yaml`
4. Register in `main.py` exercise map

Example:
```python
from src.exercises.base_exercise import BaseExercise, ExercisePhase, FormError

class MyExercise(BaseExercise):
    def __init__(self, config):
        super().__init__(config)
        self.exercise_config = config['my_exercise']
        self.name = "My Exercise"
    
    def detect_exercise(self, landmarks):
        # Implementation
        pass
    
    def detect_phase(self, landmarks):
        # Implementation
        pass
    
    def analyze_form(self, landmarks):
        # Implementation
        pass
```

## Troubleshooting

**Low FPS:**
- Reduce `model_complexity` in config.yaml
- Close other applications
- Ensure good lighting

**Pose Not Detected:**
- Ensure full body is visible in frame
- Check lighting conditions
- Move closer or adjust camera angle
- Ensure plain background

**Inaccurate Rep Counting:**
- Adjust phase detection thresholds in config.yaml
- Ensure complete range of motion
- Recalibrate using 'c' key

**Form Errors Not Detected:**
- Adjust sensitivity thresholds in config.yaml
- Ensure proper camera angle (side view for most exercises)
- Check that relevant body parts are visible

## Requirements

- opencv-python >= 4.8.0
- mediapipe >= 0.10.0
- numpy >= 1.24.0
- pyyaml >= 6.0

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- MediaPipe for pose estimation
- OpenCV for video processing
- The open-source community

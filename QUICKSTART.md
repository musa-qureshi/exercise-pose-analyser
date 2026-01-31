# Quick Start Guide

Get the Exercise Form Detection System running in under 2 minutes.

## Installation

```bash
git clone https://github.com/musa-qureshi/exercise-pose-analyser.git
cd exercise-pose-analyser
pip install -r requirements.txt
```

## Run It

```bash
python main.py
```

That's it! The webcam will open with lunge detection enabled.

## Controls

**Switch Exercise:**
- **P** - Push-ups
- **S** - Squats  
- **L** - Lunges

**Other:**
- **Q** - Quit
- **R** - Reset rep count
- **C** - Recalibrate

Current exercise shows in green in the bottom-right corner.

## Camera Setup

**Position:**
- **Squats** - Face camera directly (front view)
- **Push-ups** - Stand sideways to camera (side view)
- **Lunges** - Stand sideways to camera (side view)
  - *Note: Try facing right side of screen first - MediaPipe's pose detection tends to track better in this orientation*

**Distance:** Full body visible with some margin above head and below feet

**Lighting:** Good, even lighting (avoid backlighting from windows)

## Video File

```bash
python main.py --source path/to/video.mp4
```

## Logs

Session data saves automatically to `logs/` folder as CSV files.

## Troubleshooting

**"No pose detected"** - Move back, ensure full body visible, check lighting

**Low FPS** - Edit config.yaml, set `model_complexity: 0`

**Reps not counting** - Complete full range of motion, recalibrate with C

**Lunge not detecting** - Stand sideways to camera (right side toward screen works best)

## Next Steps

- Edit `config.yaml` to adjust sensitivity thresholds
- Check `README.md` for detailed documentation
- See `examples/` for programmatic usage

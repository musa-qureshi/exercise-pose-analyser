# Quick Start Guide

Get started with the Exercise Form Detection System in just a few minutes!

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/musa-qureshi/pose-based-exercise-analyser.git
   cd pose-based-exercise-analyser
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   That's it! The system is ready to use.

## First Run

### Test with Webcam

Run the analyzer with your webcam doing squats:

```bash
python main.py --exercise squat
```

**What you'll see:**
- Your video feed with pose skeleton overlay
- Rep counter in the top-left
- Current phase (Up/Down)
- Form feedback (Good or specific errors)
- FPS counter in top-right

**Controls while running:**
- Press **'q'** to quit
- Press **'r'** to reset rep count
- Press **'c'** to recalibrate

### Try Different Exercises

**Push-ups:**
```bash
python main.py --exercise pushup
```

**Lunges:**
```bash
python main.py --exercise lunge
```

## Understanding the Feedback

### Squat Feedback
- ✅ **"Form: Good!"** - Perfect squat form
- ⚠️ **"Squat deeper"** - Not going low enough
- ⚠️ **"Keep knees aligned"** - Knees caving inward (valgus)
- ⚠️ **"Keep chest up"** - Leaning too far forward

### Push-up Feedback
- ✅ **"Form: Good!"** - Excellent push-up form
- ⚠️ **"Engage core"** - Hips sagging
- ⚠️ **"Go lower"** - Not reaching full depth
- ⚠️ **"Keep elbows closer"** - Elbows flaring out

### Lunge Feedback
- ✅ **"Form: Good!"** - Great lunge form
- ⚠️ **"Keep knee over ankle"** - Front knee too far forward
- ⚠️ **"Go lower"** - Not deep enough
- ⚠️ **"Keep torso upright"** - Leaning too much

## Tips for Best Results

### Camera Setup
1. **Position**: Side view works best for most exercises
   - Squat: Side view (90° from front)
   - Push-up: Side view
   - Lunge: Side view

2. **Distance**: 
   - Stand 6-10 feet from camera
   - Entire body should be visible
   - Leave some space above head and below feet

3. **Lighting**:
   - Good, even lighting
   - Avoid backlighting (windows behind you)
   - Face well-lit area

4. **Background**:
   - Plain, solid background works best
   - Avoid busy or cluttered backgrounds

### Getting Accurate Counts
- **Full range of motion**: Complete each rep fully
- **Controlled movement**: Avoid jerky or too-fast movements
- **Clear phases**: Pause briefly at top and bottom of movement
- **Proper form**: System is more accurate with good form

## Session Logs

Exercise sessions are automatically logged to `logs/` directory.

**Each log contains:**
- Frame-by-frame data
- Timestamps
- Exercise detection status
- Phase information
- Rep counts
- Form errors with timestamps
- Session summary with average FPS

**View your logs:**
```bash
ls logs/
cat logs/exercise_session_YYYYMMDD_HHMMSS.csv
```

## Next Steps

### Customize Thresholds
The default thresholds work well for most people, but you can adjust them:

```bash
# Edit config.yaml with your preferred text editor
nano config.yaml

# Run with custom settings
python main.py --exercise squat
```

### Use with Video Files
Analyze pre-recorded videos:

```bash
python main.py --exercise squat --source path/to/your/video.mp4
```

### Programmatic Usage
Check out `examples/programmatic_usage.py` to integrate the analyzer into your own code.

## Troubleshooting

**"No pose detected"**
- Ensure full body is visible in frame
- Check lighting - make sure you're well lit
- Move camera back if too close

**Low FPS (< 20)**
- Lower model_complexity in config.yaml (try 0)
- Close other applications
- Ensure sufficient CPU/GPU

**Inaccurate rep counting**
- Ensure complete range of motion
- Check form errors - system may not count partial reps
- Recalibrate by pressing 'c'
- Adjust phase thresholds in config.yaml

**Form errors not showing**
- Check camera angle (side view recommended)
- Ensure proper exercise form
- Adjust sensitivity in config.yaml

## Need Help?

- Read the full README.md for detailed documentation
- Check examples/ directory for usage examples
- Review config.yaml for all available settings

Happy training! 💪

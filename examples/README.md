# Exercise Form Detection System - Examples

This directory contains example scripts showing various ways to use the exercise form detection system.

## Examples

### 1. Custom Configuration (`custom_config.py`)
Shows how to create and use a custom configuration file with adjusted thresholds.

```bash
python examples/custom_config.py
python main.py --config config_custom.yaml --exercise squat
```

### 2. Programmatic Usage (`programmatic_usage.py`)
Demonstrates how to use the ExerciseAnalyzer class programmatically in your own code.

Features:
- Analyze video files and save output
- Process webcam feed with custom callbacks
- Access exercise state and form errors

```bash
python examples/programmatic_usage.py
```

## Creating Your Own Integration

To integrate the exercise analyzer into your application:

```python
from main import ExerciseAnalyzer

# Initialize
analyzer = ExerciseAnalyzer(
    config_path='config.yaml',
    exercise_type='squat'
)

# Process frames
frame, exercise_state = analyzer.process_frame(input_frame)

# Access results
if exercise_state:
    print(f"Reps: {exercise_state['rep_count']}")
    print(f"Phase: {exercise_state['phase']}")
    print(f"Errors: {exercise_state['form_errors']}")

# Cleanup
analyzer.cleanup()
```

## Additional Ideas

Here are some additional use cases you can implement:

1. **REST API Server**: Wrap the analyzer in a Flask/FastAPI server for web integration
2. **Batch Processing**: Process multiple videos in a batch
3. **Real-time Feedback**: Integrate with text-to-speech for audio feedback
4. **Performance Tracking**: Store results in a database for long-term tracking
5. **Multi-user Support**: Track multiple users with separate profiles

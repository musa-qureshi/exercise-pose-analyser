"""
Example: Programmatic usage of the exercise analyzer

This example shows how to use the ExerciseAnalyzer class
programmatically without using the CLI.
"""

import cv2
import numpy as np
from main import ExerciseAnalyzer

def analyze_video_file(video_path: str, exercise_type: str = 'squat'):
    """
    Analyze a video file and save output with visualizations.
    
    Args:
        video_path: Path to input video file
        exercise_type: Type of exercise to analyze
    """
    # Initialize analyzer
    analyzer = ExerciseAnalyzer(
        config_path='config.yaml',
        exercise_type=exercise_type
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    # video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #output video writer
    output_path = f"output_{exercise_type}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, exercise_state = analyzer.process_frame(frame)
            
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print every second (assuming 30fps)
                print(f"Processed {frame_count} frames, "
                      f"Reps: {analyzer.exercise.rep_count}, "
                      f"FPS: {analyzer.fps:.1f}")
    
    finally:
        cap.release()
        out.release()
        analyzer.cleanup()
    
    print(f"\nAnalysis complete!")
    print(f"Total frames: {frame_count}")
    print(f"Final rep count: {analyzer.exercise.rep_count}")
    print(f"Output saved to: {output_path}")


def analyze_webcam_with_callback(exercise_type: str = 'squat'):
    def on_form_error(error):
        """Callback function when form error is detected."""
        print(f"⚠️  Form Error: {error.error_type} - {error.message}")
    
    analyzer = ExerciseAnalyzer(
        config_path='config.yaml',
        exercise_type=exercise_type
    )
    
    cap = cv2.VideoCapture(0)
    
    print("Starting webcam analysis with error callbacks...")
    print("Press 'q' to quit")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, exercise_state = analyzer.process_frame(frame)
            
            if exercise_state and exercise_state.get('form_errors'):
                for error in exercise_state['form_errors']:
                    on_form_error(error)
            
            cv2.imshow('Exercise Analysis', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        analyzer.cleanup()


if __name__ == '__main__':
    import sys
    
    print("Exercise Analyzer Examples")
    print("1. Analyze video file")
    print("2. Analyze webcam with callbacks")
    
    choice = input("\nSelect example (1 or 2): ")
    
    if choice == '1':
        video_path = input("Enter video file path: ")
        exercise = input("Enter exercise type (squat/pushup/lunge) [squat]: ") or 'squat'
        analyze_video_file(video_path, exercise)
    
    elif choice == '2':
        exercise = input("Enter exercise type (squat/pushup/lunge) [squat]: ") or 'squat'
        analyze_webcam_with_callback(exercise)
    
    else:
        print("Invalid choice")

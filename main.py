"""
Exercise Form Detection System
Main application for real-time exercise form detection and feedback.
"""

# Suppress warnings before any other imports
import os
import sys

# Force unbuffered output so messages appear immediately
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
warnings.filterwarnings('ignore')

# Force output flushing
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import cv2
import time
import argparse
import numpy as np
from typing import Optional, Dict

from src.pose_detector import PoseDetector
from src.exercises import SquatExercise, PushUpExercise, LungeExercise
from src.utils import load_config
from src.utils.logger import SessionLogger


class ExerciseAnalyzer:
    """
    Main application class for exercise form detection and analysis.
    """
    
    def __init__(self, config_path: str = 'config.yaml', exercise_type: str = 'squat'):
        """
        Initialize the exercise analyzer.
        
        Args:
            config_path: Path to configuration file
            exercise_type: Type of exercise ('squat', 'pushup', 'lunge')
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize pose detector
        self.pose_detector = PoseDetector(self.config)
        
        # Initialize exercise analyzer
        self.exercise_type = exercise_type.lower()
        self.exercise = self._create_exercise(self.exercise_type)
        
        # Initialize logger
        self.logger = SessionLogger(self.config)
        
        # Performance tracking
        self.fps = 0
        self.frame_times = []
        
        # Calibration
        self.calibration_frames = 0
        self.calibration_required = self.config['calibration']['required_frames']
        self.is_calibrated = False
        
        print(f"Exercise Analyzer initialized for: {self.exercise.name}")
        print(f"Target FPS: {self.config['performance']['target_fps']}")
    
    def _create_exercise(self, exercise_type: str):
        """Create exercise analyzer based on type."""
        exercise_map = {
            'squat': SquatExercise,
            'pushup': PushUpExercise,
            'push-up': PushUpExercise,
            'lunge': LungeExercise
        }
        
        if exercise_type not in exercise_map:
            raise ValueError(f"Unknown exercise type: {exercise_type}. "
                           f"Choose from: {list(exercise_map.keys())}")
        
        return exercise_map[exercise_type](self.config)
    
    def calibrate_frame(self, landmarks: np.ndarray) -> bool:
        """
        Process calibration frame.
        
        Args:
            landmarks: Pose landmarks
            
        Returns:
            True if calibration complete
        """
        if self.is_calibrated:
            return True
        
        self.calibration_frames += 1
        
        # Calibrate on first valid frame
        if self.calibration_frames == 1:
            self.exercise.calibrate(landmarks)
            self.is_calibrated = True
            print("Calibration complete!")
            return True
        
        return False
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (processed frame, exercise state)
        """
        start_time = time.time()
        
        # Detect pose
        landmarks, detected = self.pose_detector.detect_pose(frame)
        
        exercise_state = None
        
        if detected and landmarks is not None:
            # Calibration phase
            if not self.is_calibrated:
                self.calibrate_frame(landmarks)
                self._draw_calibration_message(frame)
            else:
                # Analyze exercise
                exercise_state = self.exercise.update(landmarks)
                
                # Draw pose
                frame = self.pose_detector.draw_pose(frame, landmarks)
                
                # Draw feedback
                self._draw_feedback(frame, exercise_state)
                
                # Log data
                self.logger.log_frame(self.exercise.name, exercise_state)
        else:
            self._draw_no_pose_message(frame)
        
        # Calculate FPS
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        # Draw FPS
        self._draw_fps(frame)
        
        return frame, exercise_state
    
    def _draw_feedback(self, frame: np.ndarray, exercise_state: Dict):
        """Draw exercise feedback on frame."""
        if not exercise_state or not exercise_state['exercise_detected']:
            cv2.putText(frame, "Exercise not detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return
        
        feedback_lines = self.exercise.get_feedback_text()
        
        viz_config = self.config['visualization']
        font_scale = viz_config['feedback_font_scale']
        thickness = viz_config['feedback_font_thickness']
        
        y_offset = 30
        for i, line in enumerate(feedback_lines):
            # Color based on content
            if 'Issue' in line or 'Error' in line:
                color = (0, 0, 255)  # Red for errors
            elif 'Good' in line:
                color = (0, 255, 0)  # Green for good form
            else:
                color = (255, 255, 255)  # White for info
            
            cv2.putText(frame, line, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    def _draw_fps(self, frame: np.ndarray):
        """Draw FPS counter."""
        fps_text = f"FPS: {self.fps:.1f}"
        color = (0, 255, 0) if self.fps >= 20 else (0, 165, 255)
        cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _draw_calibration_message(self, frame: np.ndarray):
        """Draw calibration in progress message."""
        msg = "Calibrating... Stand in starting position"
        cv2.putText(frame, msg, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    def _draw_no_pose_message(self, frame: np.ndarray):
        """Draw no pose detected message."""
        msg = "No pose detected - ensure full body is visible"
        cv2.putText(frame, msg, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def run_video(self, video_source: int = 0):
        """
        Run analyzer on video source (webcam or video file).
        
        Args:
            video_source: Camera index or video file path
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_source}")
            return
        
        print("\n=== Exercise Form Detection Started ===")
        print(f"Exercise: {self.exercise.name}")
        print("Press 'q' to quit, 'r' to reset rep count, 'c' to recalibrate")
        print("=" * 45 + "\n")
        
        # Create window and bring to front
        window_name = 'Exercise Form Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame, exercise_state = self.process_frame(frame)
                
                # Display
                cv2.imshow(window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.exercise.reset()
                    print("Rep count reset")
                elif key == ord('c'):
                    self.is_calibrated = False
                    self.calibration_frames = 0
                    print("Recalibrating...")
                
                # FPS limiting (optional)
                target_fps = self.config['performance']['target_fps']
                if target_fps > 0:
                    time.sleep(max(0, 1.0/target_fps - sum(self.frame_times[-1:])))
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.logger.close()
        self.pose_detector.release()
        print("\nExercise session ended.")
        print(f"Final rep count: {self.exercise.rep_count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Real-time Exercise Form Detection System'
    )
    parser.add_argument(
        '--exercise', '-e',
        type=str,
        default='squat',
        choices=['squat', 'pushup', 'push-up', 'lunge'],
        help='Exercise type to detect (default: squat)'
    )
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='0',
        help='Video source: camera index (0, 1, ...) or video file path (default: 0)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Parse video source
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source
    
    # Create and run analyzer
    try:
        analyzer = ExerciseAnalyzer(
            config_path=args.config,
            exercise_type=args.exercise
        )
        analyzer.run_video(video_source)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

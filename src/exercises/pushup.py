"""
Push-up Exercise Analyzer
Detects and analyzes push-up form with rep counting and error detection.
"""

import numpy as np
from typing import List, Optional
from .base_exercise import BaseExercise, ExercisePhase, FormError
from ..utils.geometry import (
    calculate_angle, calculate_vertical_angle, point_to_line_distance,
    PoseLandmark, get_side_of_body, calculate_distance
)


class PushUpExercise(BaseExercise):
    """
    Push-up exercise detection and analysis.
    
    Detects:
    - Sagging hips (core not engaged)
    - Incomplete range of motion
    - Elbow flare (elbows too far from body)
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.exercise_config = config['pushup']
        self.name = "Push-up"
        
    def detect_exercise(self, landmarks: np.ndarray) -> bool:
        """
        Detect if person is in push-up position (plank-like).
        """
        # Get key landmarks
        left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[PoseLandmark.LEFT_HIP]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP]
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        # Check visibility
        min_vis = min(
            left_shoulder[2], right_shoulder[2],
            left_hip[2], right_hip[2],
            left_ankle[2], right_ankle[2]
        )
        if min_vis < 0.5:
            return False
        
        # Check if body is roughly horizontal (plank position)
        # Shoulders and hips should be at similar heights
        shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_avg_y = (left_hip[1] + right_hip[1]) / 2
        ankle_avg_y = (left_ankle[1] + right_ankle[1]) / 2
        
        # In push-up position, all should be at similar height (horizontal body)
        height_diff_shoulder_hip = abs(shoulder_avg_y - hip_avg_y)
        height_diff_hip_ankle = abs(hip_avg_y - ankle_avg_y)
        
        # Body should be relatively horizontal
        return height_diff_shoulder_hip < 0.3 and height_diff_hip_ankle < 0.3
    
    def detect_phase(self, landmarks: np.ndarray) -> ExercisePhase:
        """
        Detect push-up phase based on elbow angle.
        """
        side = get_side_of_body(landmarks)
        
        if side == 'left':
            shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks[PoseLandmark.LEFT_ELBOW]
            wrist = landmarks[PoseLandmark.LEFT_WRIST]
        else:
            shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
            elbow = landmarks[PoseLandmark.RIGHT_ELBOW]
            wrist = landmarks[PoseLandmark.RIGHT_WRIST]
        
        # Calculate elbow angle
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        
        # Determine phase based on elbow angle
        top_threshold = self.exercise_config['top_elbow_angle']
        bottom_threshold = self.exercise_config['bottom_elbow_angle']
        
        if elbow_angle > top_threshold - 10:
            return ExercisePhase.UP
        elif elbow_angle < bottom_threshold + 15:
            return ExercisePhase.DOWN
        else:
            # Transitioning
            if self.current_phase == ExercisePhase.UP:
                return ExercisePhase.DOWN
            else:
                return ExercisePhase.UP
    
    def analyze_form(self, landmarks: np.ndarray) -> List[FormError]:
        """
        Analyze push-up form for common errors.
        """
        errors = []
        side = get_side_of_body(landmarks)
        
        # Select landmarks based on visible side
        if side == 'left':
            shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks[PoseLandmark.LEFT_ELBOW]
            wrist = landmarks[PoseLandmark.LEFT_WRIST]
            hip = landmarks[PoseLandmark.LEFT_HIP]
            ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        else:
            shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
            elbow = landmarks[PoseLandmark.RIGHT_ELBOW]
            wrist = landmarks[PoseLandmark.RIGHT_WRIST]
            hip = landmarks[PoseLandmark.RIGHT_HIP]
            ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        # 1. Check for sagging hips
        hip_sag_error = self._check_hip_sag(shoulder, hip, ankle)
        if hip_sag_error:
            errors.append(hip_sag_error)
        
        # 2. Check range of motion
        rom_error = self._check_range_of_motion(shoulder, elbow, wrist)
        if rom_error:
            errors.append(rom_error)
        
        # 3. Check elbow flare
        flare_error = self._check_elbow_flare(shoulder, elbow, hip)
        if flare_error:
            errors.append(flare_error)
        
        return errors
    
    def _check_hip_sag(self, shoulder: np.ndarray, hip: np.ndarray, 
                       ankle: np.ndarray) -> Optional[FormError]:
        """Check for sagging hips (broken plank position)."""
        # Calculate how far hip deviates from straight line shoulder-ankle
        hip_deviation = point_to_line_distance(hip, shoulder, ankle)
        
        # Also check the angle - hips should form roughly straight line
        body_angle = calculate_angle(shoulder, hip, ankle)
        
        # Straight body should have angle close to 180
        threshold = self.exercise_config['hip_sag_threshold']
        
        if body_angle < (180 - threshold):
            return FormError(
                error_type='hip_sag',
                severity='high',
                message='Engage core - hips are sagging',
                value=body_angle
            )
        
        return None
    
    def _check_range_of_motion(self, shoulder: np.ndarray, elbow: np.ndarray,
                               wrist: np.ndarray) -> Optional[FormError]:
        """Check if going low enough in push-up."""
        # Only check at bottom of movement
        if self.current_phase != ExercisePhase.DOWN:
            return None
        
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        
        shallow_threshold = self.exercise_config['shallow_threshold']
        
        if elbow_angle > shallow_threshold:
            return FormError(
                error_type='shallow_pushup',
                severity='medium',
                message='Go lower - elbows should reach 90 degrees',
                value=elbow_angle
            )
        
        return None
    
    def _check_elbow_flare(self, shoulder: np.ndarray, elbow: np.ndarray,
                          hip: np.ndarray) -> Optional[FormError]:
        """Check if elbows are flaring out too much."""
        # Only check at bottom of movement
        if self.current_phase != ExercisePhase.DOWN:
            return None
        
        # Calculate angle between elbow and body line (shoulder-hip)
        # Elbow should stay relatively close to body
        body_line_angle = calculate_vertical_angle(shoulder, hip)
        elbow_line_angle = calculate_vertical_angle(shoulder, elbow)
        
        # Angle difference indicates flare
        flare_angle = abs(body_line_angle - elbow_line_angle)
        
        threshold = self.exercise_config['elbow_flare_threshold']
        
        if flare_angle > threshold:
            return FormError(
                error_type='elbow_flare',
                severity='low',
                message='Keep elbows closer to body',
                value=flare_angle
            )
        
        return None
    
    def calibrate(self, landmarks: np.ndarray):
        """Calibrate based on user's plank position."""
        side = get_side_of_body(landmarks)
        
        if side == 'left':
            shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
            ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        else:
            shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
            ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        # Store body length for reference
        self.calibration_data['body_length'] = calculate_distance(shoulder, ankle)
        self.is_calibrated = True

"""
Squat Exercise Analyzer
Detects and analyzes squat form with rep counting and error detection.
"""

import numpy as np
from typing import List, Optional
from .base_exercise import BaseExercise, ExercisePhase, FormError
from ..utils.geometry import (
    calculate_angle, calculate_distance, calculate_vertical_angle,
    calculate_midpoint, PoseLandmark, get_side_of_body
)


class SquatExercise(BaseExercise):
    """
    Squat exercise detection and analysis.
    
    Detects:
    - Knee valgus (knees caving in)
    - Shallow depth (not going low enough)
    - Forward lean (torso leaning too far forward)
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.exercise_config = config['squat']
        self.name = "Squat"
        
    def detect_exercise(self, landmarks: np.ndarray) -> bool:
        """
        Detect if person is in squat position.
        Check if person is standing/squatting (feet roughly under hips).
        """
        # Get hip and ankle positions
        left_hip = landmarks[PoseLandmark.LEFT_HIP]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP]
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        # Check visibility
        if min(left_hip[2], right_hip[2], left_ankle[2], right_ankle[2]) < 0.5:
            return False
        
        # Calculate horizontal distance between feet and hips
        hip_center = calculate_midpoint(left_hip, right_hip)
        ankle_center = calculate_midpoint(left_ankle, right_ankle)
        
        horizontal_dist = abs(hip_center[0] - ankle_center[0])
        
        # Feet should be roughly under hips (within reasonable range)
        # This is a simple heuristic - person is standing/squatting if feet are under torso
        return horizontal_dist < 0.3  # Normalized coordinate threshold
    
    def detect_phase(self, landmarks: np.ndarray) -> ExercisePhase:
        """
        Detect squat phase based on hip angle.
        """
        side = get_side_of_body(landmarks)
        
        if side == 'left':
            shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
            hip = landmarks[PoseLandmark.LEFT_HIP]
            knee = landmarks[PoseLandmark.LEFT_KNEE]
        else:
            shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
            hip = landmarks[PoseLandmark.RIGHT_HIP]
            knee = landmarks[PoseLandmark.RIGHT_KNEE]
        
        # Calculate hip angle (torso-hip-thigh angle)
        hip_angle = calculate_angle(shoulder, hip, knee)
        
        # Determine phase based on hip angle
        standing_threshold = self.exercise_config['standing_hip_angle']
        bottom_threshold = self.exercise_config['bottom_hip_angle']
        
        if hip_angle > standing_threshold:
            return ExercisePhase.UP
        elif hip_angle < bottom_threshold + 20:  # Allow some margin
            return ExercisePhase.DOWN
        else:
            # Transitioning
            if self.current_phase == ExercisePhase.UP:
                return ExercisePhase.DOWN
            else:
                return ExercisePhase.UP
    
    def analyze_form(self, landmarks: np.ndarray) -> List[FormError]:
        """
        Analyze squat form for common errors.
        """
        errors = []
        side = get_side_of_body(landmarks)
        
        # Select landmarks based on visible side
        if side == 'left':
            shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
            hip = landmarks[PoseLandmark.LEFT_HIP]
            knee = landmarks[PoseLandmark.LEFT_KNEE]
            ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        else:
            shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
            hip = landmarks[PoseLandmark.RIGHT_HIP]
            knee = landmarks[PoseLandmark.RIGHT_KNEE]
            ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        # 1. Check squat depth
        depth_error = self._check_depth(hip, ankle)
        if depth_error:
            errors.append(depth_error)
        
        # 2. Check knee alignment (valgus)
        knee_error = self._check_knee_valgus(landmarks, side)
        if knee_error:
            errors.append(knee_error)
        
        # 3. Check forward lean
        lean_error = self._check_forward_lean(shoulder, hip, ankle)
        if lean_error:
            errors.append(lean_error)
        
        return errors
    
    def _check_depth(self, hip: np.ndarray, ankle: np.ndarray) -> Optional[FormError]:
        """Check if squat depth is adequate."""
        # Calculate vertical distance ratio
        vertical_dist = abs(hip[1] - ankle[1])
        
        # Only check depth when in down phase
        if self.current_phase != ExercisePhase.DOWN:
            return None
        
        # Compare to threshold
        shallow_threshold = self.exercise_config['depth_shallow_threshold']
        
        if vertical_dist > shallow_threshold:
            return FormError(
                error_type='shallow_depth',
                severity='medium',
                message='Squat deeper - hips should go below knees',
                value=vertical_dist
            )
        
        return None
    
    def _check_knee_valgus(self, landmarks: np.ndarray, side: str) -> Optional[FormError]:
        """Check for knee valgus (knees caving in)."""
        # Get both knees and ankles to check alignment
        left_knee = landmarks[PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        # Only check when in down phase
        if self.current_phase != ExercisePhase.DOWN:
            return None
        
        # Check if knees are closer together than ankles (simplified valgus check)
        knee_distance = calculate_distance(left_knee, right_knee)
        ankle_distance = calculate_distance(left_ankle, right_ankle)
        
        # Knee valgus occurs when knees are significantly closer than ankles
        if knee_distance < ankle_distance * 0.8:
            return FormError(
                error_type='knee_valgus',
                severity='high',
                message='Keep knees aligned with toes - knees are caving in',
                value=knee_distance / ankle_distance
            )
        
        return None
    
    def _check_forward_lean(self, shoulder: np.ndarray, hip: np.ndarray, 
                           ankle: np.ndarray) -> Optional[FormError]:
        """Check for excessive forward lean."""
        # Calculate torso angle from vertical
        torso_angle = calculate_vertical_angle(hip, shoulder)
        
        # Only check when in down phase
        if self.current_phase != ExercisePhase.DOWN:
            return None
        
        threshold = self.exercise_config['forward_lean_threshold']
        
        if torso_angle > threshold:
            return FormError(
                error_type='forward_lean',
                severity='medium',
                message='Keep chest up - torso leaning too far forward',
                value=torso_angle
            )
        
        return None
    
    def calibrate(self, landmarks: np.ndarray):
        """Calibrate based on user's standing position."""
        side = get_side_of_body(landmarks)
        
        if side == 'left':
            hip = landmarks[PoseLandmark.LEFT_HIP]
            ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        else:
            hip = landmarks[PoseLandmark.RIGHT_HIP]
            ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        # Store user's leg length for depth calculations
        self.calibration_data['leg_length'] = calculate_distance(hip, ankle)
        self.is_calibrated = True

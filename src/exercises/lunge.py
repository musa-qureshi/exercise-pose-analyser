"""
Lunge Exercise Analyzer
Detects and analyzes lunge form with rep counting and error detection.
"""

import numpy as np
from typing import List, Optional
from .base_exercise import BaseExercise, ExercisePhase, FormError
from ..utils.geometry import (
    calculate_angle, calculate_distance, calculate_vertical_angle,
    PoseLandmark, get_side_of_body
)


class LungeExercise(BaseExercise):
    """
    Lunge exercise detection and analysis.
    
    Detects:
    - Knee over toe (front knee too far forward)
    - Insufficient depth
    - Poor balance (excessive torso lean)
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.exercise_config = config['lunge']
        self.name = "Lunge"
        
        # Track which leg is forward
        self.forward_leg = None
        
    def detect_exercise(self, landmarks: np.ndarray) -> bool:
        """
        Detect if person is in lunge position.
        One foot should be forward, one back, with vertical torso.
        """
        # Get ankle positions
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        left_hip = landmarks[PoseLandmark.LEFT_HIP]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP]
        
        # Check visibility
        if min(left_ankle[2], right_ankle[2], left_hip[2], right_hip[2]) < 0.5:
            return False
        
        # Check if one foot is significantly forward of the other
        foot_separation = abs(left_ankle[0] - right_ankle[0])
        
        # In lunge, feet should be separated horizontally
        if foot_separation > 0.15:  # Normalized threshold
            # Determine which leg is forward
            if left_ankle[0] < right_ankle[0]:
                self.forward_leg = 'left'
            else:
                self.forward_leg = 'right'
            return True
        
        return False
    
    def detect_phase(self, landmarks: np.ndarray) -> ExercisePhase:
        """
        Detect lunge phase based on front knee angle.
        """
        if self.forward_leg == 'left':
            hip = landmarks[PoseLandmark.LEFT_HIP]
            knee = landmarks[PoseLandmark.LEFT_KNEE]
            ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        else:
            hip = landmarks[PoseLandmark.RIGHT_HIP]
            knee = landmarks[PoseLandmark.RIGHT_KNEE]
            ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        # Calculate front knee angle
        knee_angle = calculate_angle(hip, knee, ankle)
        
        # Determine phase based on knee angle
        standing_threshold = self.exercise_config['standing_front_knee_angle']
        lunge_threshold = self.exercise_config['lunge_front_knee_angle']
        
        if knee_angle > standing_threshold - 10:
            return ExercisePhase.UP
        elif knee_angle < lunge_threshold + 20:
            return ExercisePhase.DOWN
        else:
            # Transitioning
            if self.current_phase == ExercisePhase.UP:
                return ExercisePhase.DOWN
            else:
                return ExercisePhase.UP
    
    def analyze_form(self, landmarks: np.ndarray) -> List[FormError]:
        """
        Analyze lunge form for common errors.
        """
        errors = []
        
        if self.forward_leg is None:
            return errors
        
        # Get landmarks based on forward leg
        if self.forward_leg == 'left':
            front_hip = landmarks[PoseLandmark.LEFT_HIP]
            front_knee = landmarks[PoseLandmark.LEFT_KNEE]
            front_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
            back_hip = landmarks[PoseLandmark.RIGHT_HIP]
            back_knee = landmarks[PoseLandmark.RIGHT_KNEE]
            back_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
            shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
        else:
            front_hip = landmarks[PoseLandmark.RIGHT_HIP]
            front_knee = landmarks[PoseLandmark.RIGHT_KNEE]
            front_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
            back_hip = landmarks[PoseLandmark.LEFT_HIP]
            back_knee = landmarks[PoseLandmark.LEFT_KNEE]
            back_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
            shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
        
        # 1. Check knee alignment (knee over toe)
        knee_error = self._check_knee_alignment(front_knee, front_ankle)
        if knee_error:
            errors.append(knee_error)
        
        # 2. Check depth
        depth_error = self._check_depth(back_knee, back_ankle)
        if depth_error:
            errors.append(depth_error)
        
        # 3. Check torso position (balance)
        balance_error = self._check_torso_lean(shoulder, front_hip, front_ankle)
        if balance_error:
            errors.append(balance_error)
        
        return errors
    
    def _check_knee_alignment(self, knee: np.ndarray, 
                             ankle: np.ndarray) -> Optional[FormError]:
        """Check if front knee is going too far past the toes."""
        # Only check at bottom of lunge
        if self.current_phase != ExercisePhase.DOWN:
            return None
        
        # Calculate horizontal distance between knee and ankle
        knee_ankle_dist = abs(knee[0] - ankle[0])
        
        threshold = self.exercise_config['knee_over_toe_threshold']
        
        if knee_ankle_dist > threshold:
            return FormError(
                error_type='knee_over_toe',
                severity='high',
                message='Front knee too far forward - keep knee over ankle',
                value=knee_ankle_dist
            )
        
        return None
    
    def _check_depth(self, back_knee: np.ndarray, 
                     back_ankle: np.ndarray) -> Optional[FormError]:
        """Check if lunge is deep enough."""
        # Only check at bottom of lunge
        if self.current_phase != ExercisePhase.DOWN:
            return None
        
        # Calculate vertical distance from back knee to ground (ankle level)
        knee_height = abs(back_knee[1] - back_ankle[1])
        
        threshold = self.exercise_config['depth_threshold']
        
        if knee_height > threshold:
            return FormError(
                error_type='shallow_lunge',
                severity='medium',
                message='Go lower - back knee should nearly touch ground',
                value=knee_height
            )
        
        return None
    
    def _check_torso_lean(self, shoulder: np.ndarray, hip: np.ndarray,
                         ankle: np.ndarray) -> Optional[FormError]:
        """Check for excessive torso lean (balance issue)."""
        # Calculate torso angle from vertical
        torso_angle = calculate_vertical_angle(hip, shoulder)
        
        threshold = self.exercise_config['torso_lean_threshold']
        
        if torso_angle > threshold:
            return FormError(
                error_type='torso_lean',
                severity='medium',
                message='Keep torso upright - you are leaning too much',
                value=torso_angle
            )
        
        return None
    
    def calibrate(self, landmarks: np.ndarray):
        """Calibrate based on user's standing position."""
        # Store leg length for depth calculations
        left_hip = landmarks[PoseLandmark.LEFT_HIP]
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        
        self.calibration_data['leg_length'] = calculate_distance(left_hip, left_ankle)
        self.is_calibrated = True
    
    def reset(self):
        """Reset lunge state including forward leg tracking."""
        super().reset()
        self.forward_leg = None

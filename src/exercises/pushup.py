"""
Push-up Exercise Analyzer
"""

import numpy as np
from typing import List, Optional
from .base_exercise import BaseExercise, ExercisePhase, FormError
from ..utils.geometry import (
    calculate_angle, point_to_line_distance,
    PoseLandmark, get_side_of_body, calculate_distance
)


class PushUpExercise(BaseExercise):
    """
    Push-up exercise detection and analysis.
    
    Detects:
    - Sagging hips (core not engaged)
    - Incomplete range of motion
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.exercise_config = config['pushup']
        self.name = "Push-up"
        
        self._went_to_bottom = False 
        
    def detect_exercise(self, landmarks: np.ndarray) -> bool:
        left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[PoseLandmark.LEFT_HIP]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP]
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        left_wrist = landmarks[PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[PoseLandmark.RIGHT_WRIST]
        left_knee = landmarks[PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[PoseLandmark.RIGHT_KNEE]
        
      
        shoulder_vis = max(left_shoulder[2], right_shoulder[2])
        hip_vis = max(left_hip[2], right_hip[2])
        ankle_vis = max(left_ankle[2], right_ankle[2])
        
        if min(shoulder_vis, hip_vis, ankle_vis) < 0.5:
            return False
        
        side = get_side_of_body(landmarks)
        
        if side == 'left':
            shoulder = left_shoulder
            hip = left_hip
            ankle = left_ankle
            wrist = left_wrist
            knee = left_knee
        else:
            shoulder = right_shoulder
            hip = right_hip
            ankle = right_ankle
            wrist = right_wrist
            knee = right_knee
        
        # Check if body is roughly horizontal (plank position)
        height_diff_shoulder_hip = abs(shoulder[1] - hip[1])
        height_diff_hip_ankle = abs(hip[1] - ankle[1])
        
        if not (height_diff_shoulder_hip < 0.35 and height_diff_hip_ankle < 0.35):
            return False
        
        # Check that person is actually on the ground (not crouching in air)
        # Only check if wrist is visible enough
        if wrist[2] > 0.4:
            if wrist[1] < hip[1] - 0.15: #Wrists shouldn't be way above hips
                return False
        
        if knee[2] > 0.4:
            if knee[1] < hip[1] - 0.25:  # Knees shouldn't be way above hips
                return False
        
        return True
    
    def detect_phase(self, landmarks: np.ndarray) -> ExercisePhase:
        side = get_side_of_body(landmarks)
        
        if side == 'left':
            shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks[PoseLandmark.LEFT_ELBOW]
            wrist = landmarks[PoseLandmark.LEFT_WRIST]
        else:
            shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
            elbow = landmarks[PoseLandmark.RIGHT_ELBOW]
            wrist = landmarks[PoseLandmark.RIGHT_WRIST]
        
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        
        top_threshold = self.exercise_config['top_elbow_angle']
        bottom_threshold = self.exercise_config['bottom_elbow_angle']
        
        hysteresis = 15
        down_leniency = 25 
        
        if self.current_phase == ExercisePhase.UP or self.current_phase == ExercisePhase.UNKNOWN:
            if elbow_angle < bottom_threshold + hysteresis + down_leniency:
                self._went_to_bottom = True 
                return ExercisePhase.DOWN
            else:
                return ExercisePhase.UP
        
        elif self.current_phase == ExercisePhase.DOWN:
            if elbow_angle > top_threshold - hysteresis:
                return ExercisePhase.UP
            else:
                return ExercisePhase.DOWN
        
        return self.current_phase
    
    def analyze_form(self, landmarks: np.ndarray) -> List[FormError]:
        errors = []
        side = get_side_of_body(landmarks)
        
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
        
        hip_sag_error = self._check_hip_sag(shoulder, hip, ankle)
        if hip_sag_error:
            errors.append(hip_sag_error)
        
        rom_error = self._check_range_of_motion(shoulder, elbow, wrist)
        if rom_error:
            errors.append(rom_error)
        
        return errors
    
    def _check_hip_sag(self, shoulder: np.ndarray, hip: np.ndarray, 
                       ankle: np.ndarray) -> Optional[FormError]:
        """Check for sagging hips (broken plank position)."""
        # Calculate how far hip deviates from straight line shoulder-ankle
        hip_deviation = point_to_line_distance(hip, shoulder, ankle)
        
        body_angle = calculate_angle(shoulder, hip, ankle)
        
        threshold = self.exercise_config['hip_sag_threshold']
        
        if body_angle < (180 - threshold):
            return FormError(
                error_type='hip_sag',
                severity='high',
                message='Hips are sagging - Core not engaged',
                value=body_angle
            )
        
        return None
    
    def _check_range_of_motion(self, shoulder: np.ndarray, elbow: np.ndarray,
                               wrist: np.ndarray) -> Optional[FormError]:
        if self.current_phase != ExercisePhase.DOWN:
            return None
        
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        
        shallow_threshold = self.exercise_config['shallow_threshold']
        
        if elbow_angle > shallow_threshold:
            return FormError(
                error_type='shallow_pushup',
                severity='medium',
                message='Go lower in your push-up',
                value=elbow_angle
            )
        
        return None
    
    def calibrate(self, landmarks: np.ndarray):
        side = get_side_of_body(landmarks)
        
        if side == 'left':
            shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
            ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        else:
            shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
            ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        self.calibration_data['body_length'] = calculate_distance(shoulder, ankle)
        self.is_calibrated = True
    
    def _is_rep_completed(self) -> bool:
        if (self.previous_phase == ExercisePhase.DOWN and 
            self.current_phase == ExercisePhase.UP and
            self._went_to_bottom):
            self._went_to_bottom = False
            return True
        return False
    
    def reset(self):
        super().reset()
        self._went_to_bottom = False

"""
Squat Exercise Analyzer
"""

import numpy as np
from typing import List, Optional
from .base_exercise import BaseExercise, ExercisePhase, FormError
from ..utils.geometry import (
    calculate_angle, calculate_distance, calculate_vertical_angle,
    calculate_midpoint, PoseLandmark, get_side_of_body, calculate_foot_angle
)

class SquatExercise(BaseExercise):
    """
    Squat exercise detection and analysis.
    
    Detects:
    - Knee valgus (knees caving in)
    - Forward lean (torso leaning too far forward)
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.exercise_config = config['squat']
        self.name = "Squat"
        
        self._phase_locked = False  
        self._went_to_bottom = False 
        self._last_hip_angle = None 
        
        self._is_descending = False  
        self._lowest_hip_angle = 180  # Track how low they went this rep
        self._rep_had_form_errors = []  # Track form errors during rep attempt
        
    def detect_exercise(self, landmarks: np.ndarray) -> bool:
        left_hip = landmarks[PoseLandmark.LEFT_HIP]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP]
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        # visibility
        if min(left_hip[2], right_hip[2], left_ankle[2], right_ankle[2]) < 0.5:
            return False
        
        hip_center = calculate_midpoint(left_hip, right_hip)
        ankle_center = calculate_midpoint(left_ankle, right_ankle)
        
        horizontal_dist = abs(hip_center[0] - ankle_center[0])
        
        return horizontal_dist < 0.3 
    
    def detect_phase(self, landmarks: np.ndarray) -> ExercisePhase:
        side = get_side_of_body(landmarks)
        
        if side == 'left':
            shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
            hip = landmarks[PoseLandmark.LEFT_HIP]
            knee = landmarks[PoseLandmark.LEFT_KNEE]
        else:
            shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
            hip = landmarks[PoseLandmark.RIGHT_HIP]
            knee = landmarks[PoseLandmark.RIGHT_KNEE]
        
        hip_angle = calculate_angle(shoulder, hip, knee)
        
        self._last_hip_angle = hip_angle
        
        #if user is descending (for feedback)
        standing_threshold = self.exercise_config['standing_hip_angle']  # 160
        bottom_threshold = self.exercise_config['bottom_hip_angle']  # 110
        
        # descent
        if hip_angle < standing_threshold - 10:  # Below ~150 degrees
            self._is_descending = True
            if hip_angle < self._lowest_hip_angle:
                self._lowest_hip_angle = hip_angle
        
        # Hysteresis buffer - prevents flickering at boundaries
        hysteresis = 15  # degrees
        down_leniency = 20  # Extra leniency for down phase (allows shallow squats to count)
        
        if self.current_phase == ExercisePhase.UP or self.current_phase == ExercisePhase.UNKNOWN:
            if hip_angle < bottom_threshold + hysteresis + down_leniency:
                self._went_to_bottom = True
                return ExercisePhase.DOWN
            else:
                return ExercisePhase.UP
        
        elif self.current_phase == ExercisePhase.DOWN:
            if hip_angle > standing_threshold - hysteresis:
                return ExercisePhase.UP
            else:
                return ExercisePhase.DOWN
        
        return self.current_phase  # Default: maintain current phase
    
    def analyze_form(self, landmarks: np.ndarray) -> List[FormError]:
        errors = []
        side = get_side_of_body(landmarks)
        
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
        
        knee_error = self._check_knee_alignment(landmarks, side)
        if knee_error:
            errors.append(knee_error)
            self._rep_had_form_errors.append(knee_error)
        
        lean_error = self._check_forward_lean(shoulder, hip, ankle)
        if lean_error:
            errors.append(lean_error)
        
        foot_errors = self._check_foot_angle(landmarks)
        errors.extend(foot_errors)
        
        return errors
    
    def _check_knee_valgus(self, landmarks: np.ndarray, side: str) -> Optional[FormError]:
        return self._check_knee_alignment(landmarks, side)
    
    def _check_knee_alignment(self, landmarks: np.ndarray, side: str) -> Optional[FormError]:
        left_knee = landmarks[PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        left_hip = landmarks[PoseLandmark.LEFT_HIP]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP]
        
        if self.current_phase != ExercisePhase.DOWN and not self._is_descending:
            return None
        
        knee_distance = calculate_distance(left_knee, right_knee)
        ankle_distance = calculate_distance(left_ankle, right_ankle)
        hip_distance = calculate_distance(left_hip, right_hip)
        
        knee_ankle_ratio = knee_distance / ankle_distance if ankle_distance > 0 else 1.0
        
        knee_hip_ratio = knee_distance / hip_distance if hip_distance > 0 else 1.0
        
        valgus_threshold = self.exercise_config.get('knee_valgus_threshold', 0.8)
        spread_threshold = self.exercise_config.get('knee_spread_threshold', 1.8)
        
        if knee_ankle_ratio < valgus_threshold:
            return FormError(
                error_type='knee_valgus',
                severity='high',
                message='Knees caving in! Push knees outward over toes',
                value=knee_ankle_ratio
            )
        
        # Check for knee SPREAD knees much wider than hips
        if knee_hip_ratio > spread_threshold:
            return FormError(
                error_type='knee_spread',
                severity='medium',
                message='Knees too wide! Keep knees aligned over toes',
                value=knee_hip_ratio
            )
        
        return None
    
    def _check_forward_lean(self, shoulder: np.ndarray, hip: np.ndarray, 
                           ankle: np.ndarray) -> Optional[FormError]:
        torso_angle = calculate_vertical_angle(hip, shoulder)
        
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
    
    def _check_foot_angle(self, landmarks: np.ndarray) -> List[FormError]:
        errors = []
        
        left_heel = landmarks[PoseLandmark.LEFT_HEEL]
        right_heel = landmarks[PoseLandmark.RIGHT_HEEL]
        left_toe = landmarks[PoseLandmark.LEFT_FOOT_INDEX]
        right_toe = landmarks[PoseLandmark.RIGHT_FOOT_INDEX]
        
        min_visibility = 0.4
        left_visible = min(left_heel[2], left_toe[2]) >= min_visibility
        right_visible = min(right_heel[2], right_toe[2]) >= min_visibility
        
        if not (left_visible or right_visible):
            return errors
        
        # outward_threshold "too outward" (duck feet)
        # inward_threshold "too inward" (pigeon toed)
        outward_threshold = self.exercise_config.get('foot_outward_threshold', 60)  # degrees
        inward_threshold = self.exercise_config.get('foot_inward_threshold', 20)    # degrees
        
        worst_outward = 0
        worst_inward = 0
        
        if left_visible:
            left_angle = calculate_foot_angle(left_heel, left_toe, is_left_foot=True)
            if left_angle > outward_threshold:
                worst_outward = max(worst_outward, left_angle)
            elif left_angle < -inward_threshold:
                worst_inward = max(worst_inward, abs(left_angle))
        
        if right_visible:
            right_angle = calculate_foot_angle(right_heel, right_toe, is_left_foot=False)
            if right_angle > outward_threshold:
                worst_outward = max(worst_outward, right_angle)
            elif right_angle < -inward_threshold:
                worst_inward = max(worst_inward, abs(right_angle))
        
        if worst_outward > 0:
            # 50-75°: medium, 75°+: high (approaching perpendicular)
            severity = 'high' if worst_outward > 75 else 'medium'
            errors.append(FormError(
                error_type='feet_too_outward',
                severity=severity,
                message='Feet pointing too outward - keep feet more parallel',
                value=worst_outward
            ))
        
        if worst_inward > 0:
            severity = 'high' if worst_inward > 25 else 'medium'
            errors.append(FormError(
                error_type='feet_too_inward',
                severity=severity,
                message='Feet pointing too inward - keep feet more parallel',
                value=worst_inward
            ))
        
        return errors
    
    def calibrate(self, landmarks: np.ndarray):
        side = get_side_of_body(landmarks)
        
        if side == 'left':
            hip = landmarks[PoseLandmark.LEFT_HIP]
            ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        else:
            hip = landmarks[PoseLandmark.RIGHT_HIP]
            ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        self.calibration_data['leg_length'] = calculate_distance(hip, ankle)
        self.is_calibrated = True

    def _is_rep_completed(self) -> bool:
        """
        Check if a rep was completed.
        A rep is only complete when:
        1. User went DOWN (reached bottom position)
        2. User came back UP (returned to standing)
        
        This is the fix for the counting during continuous motion or staying in one position.
        """
        if (self.previous_phase == ExercisePhase.DOWN and 
            self.current_phase == ExercisePhase.UP and
            self._went_to_bottom):
            #Reset tracking for next rep
            self._went_to_bottom = False
            self._is_descending = False
            self._lowest_hip_angle = 180
            self._rep_had_form_errors = []
            return True
        return False
    
    def reset(self):
        super().reset()
        self._phase_locked = False
        self._went_to_bottom = False
        self._last_hip_angle = None
        self._is_descending = False
        self._lowest_hip_angle = 180
        self._rep_had_form_errors = []
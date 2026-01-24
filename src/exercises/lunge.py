"""
Lunge Exercise Analyzer
Detects and analyzes lunge form with rep counting and error detection.
"""

import numpy as np
from typing import List, Optional, Tuple
from .base_exercise import BaseExercise, ExercisePhase, FormError
from ..utils.geometry import (
    calculate_angle, calculate_distance, calculate_vertical_angle,
    calculate_midpoint, PoseLandmark, get_side_of_body
)

class LungeExercise(BaseExercise):
    """
    Lunge exercise detection and analysis.
    
    Detects:
    - Front knee going too far over toes
    - Back knee not dropping low enough
    - Torso leaning forward/backward excessively
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.exercise_config = config['lunge']
        self.name = "Lunge"
        
        self._went_to_bottom = False
        
        self._front_leg_side: Optional[str] = None
        self._last_front_knee_angle = None
        self._last_hip_height = None
        
        self._detection_frames = 0
        self._detection_threshold = 3
        
    def detect_exercise(self, landmarks: np.ndarray) -> bool:
        
        left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[PoseLandmark.LEFT_HIP]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        min_vis = min(
            left_shoulder[2], right_shoulder[2],
            left_hip[2], right_hip[2],
            left_knee[2], right_knee[2],
            left_ankle[2], right_ankle[2]
        )
        if min_vis < 0.4:
            self._detection_frames = 0
            return False
        
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        hip_width = abs(left_hip[0] - right_hip[0])
        
        sideways_threshold = self.exercise_config.get('sideways_threshold', 0.12)
        
        # If shoulders/hips are too wide, person is not sideways to the camera
        if shoulder_width > sideways_threshold or hip_width > sideways_threshold:
            self._detection_frames = 0
            return False
        
        foot_x_separation = abs(left_ankle[0] - right_ankle[0])
        foot_y_separation = abs(left_ankle[1] - right_ankle[1])
        
        min_foot_separation = self.exercise_config.get('min_foot_separation', 0.10)
        
        if foot_x_separation < min_foot_separation:
            self._detection_frames = 0
            return False
        
        if foot_x_separation <= foot_y_separation:
            self._detection_frames = 0
            return False
        
        left_knee_angle = calculate_angle(
            landmarks[PoseLandmark.LEFT_HIP],
            landmarks[PoseLandmark.LEFT_KNEE],
            landmarks[PoseLandmark.LEFT_ANKLE]
        )
        right_knee_angle = calculate_angle(
            landmarks[PoseLandmark.RIGHT_HIP],
            landmarks[PoseLandmark.RIGHT_KNEE],
            landmarks[PoseLandmark.RIGHT_ANKLE]
        )
        
        if left_knee_angle < right_knee_angle:
            self._front_leg_side = 'left'
        else:
            self._front_leg_side = 'right'
        
        if left_knee_angle > 165 and right_knee_angle > 165 and foot_x_separation < 0.20:
            self._detection_frames = 0
            return False
        
        self._detection_frames += 1
        if self._detection_frames < self._detection_threshold:
            return False
        
        return True
    
    def detect_phase(self, landmarks: np.ndarray) -> ExercisePhase:
        
        if self._front_leg_side is None:
            return ExercisePhase.UNKNOWN
        
        if self._front_leg_side == 'left':
            hip = landmarks[PoseLandmark.LEFT_HIP]
            knee = landmarks[PoseLandmark.LEFT_KNEE]
            ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        else:
            hip = landmarks[PoseLandmark.RIGHT_HIP]
            knee = landmarks[PoseLandmark.RIGHT_KNEE]
            ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        front_knee_angle = calculate_angle(hip, knee, ankle)
        self._last_front_knee_angle = front_knee_angle
        
        self._last_hip_height = hip[1]
        
        standing_threshold = self.exercise_config.get('standing_front_knee_angle', 140)
        lunge_threshold = self.exercise_config.get('lunge_front_knee_angle', 115)
        
        down_trigger = lunge_threshold + 40  # ~155°
        up_trigger = standing_threshold + 20  # ~160°
        
        if self.current_phase == ExercisePhase.UP or self.current_phase == ExercisePhase.UNKNOWN:
            if front_knee_angle < down_trigger:
                self._went_to_bottom = True
                return ExercisePhase.DOWN
            else:
                return ExercisePhase.UP
        
        elif self.current_phase == ExercisePhase.DOWN:
            if front_knee_angle > up_trigger:
                return ExercisePhase.UP
            else:
                return ExercisePhase.DOWN
        
        return self.current_phase
    
    def analyze_form(self, landmarks: np.ndarray) -> List[FormError]:
        
        errors = []
        
        if self._front_leg_side is None:
            return errors
        
        if self._front_leg_side == 'left':
            front_hip = landmarks[PoseLandmark.LEFT_HIP]
            front_knee = landmarks[PoseLandmark.LEFT_KNEE]
            front_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
            front_toe = landmarks[PoseLandmark.LEFT_FOOT_INDEX]
            back_hip = landmarks[PoseLandmark.RIGHT_HIP]
            back_knee = landmarks[PoseLandmark.RIGHT_KNEE]
            back_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        else:
            front_hip = landmarks[PoseLandmark.RIGHT_HIP]
            front_knee = landmarks[PoseLandmark.RIGHT_KNEE]
            front_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
            front_toe = landmarks[PoseLandmark.RIGHT_FOOT_INDEX]
            back_hip = landmarks[PoseLandmark.LEFT_HIP]
            back_knee = landmarks[PoseLandmark.LEFT_KNEE]
            back_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        
        side = get_side_of_body(landmarks)
        if side == 'left':
            shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
            hip_for_torso = landmarks[PoseLandmark.LEFT_HIP]
        else:
            shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
            hip_for_torso = landmarks[PoseLandmark.RIGHT_HIP]
        
        knee_error = self._check_knee_over_toe(front_knee, front_ankle, front_toe)
        if knee_error:
            errors.append(knee_error)
        
        left_knee = landmarks[PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        depth_error = self._check_back_knee_depth(left_knee, right_knee, left_ankle, right_ankle)
        if depth_error:
            errors.append(depth_error)
        
        lean_error = self._check_torso_lean(shoulder, hip_for_torso)
        if lean_error:
            errors.append(lean_error)
        
        return errors
    
    def _check_knee_over_toe(self, front_knee: np.ndarray, 
                              front_ankle: np.ndarray,
                              front_toe: np.ndarray) -> Optional[FormError]:
        
        if self.current_phase != ExercisePhase.DOWN:
            return None
        
        forward_direction = front_toe[0] - front_ankle[0]
        
        knee_offset = front_knee[0] - front_ankle[0]
        
        if abs(forward_direction) < 0.01:
            # Foot is nearly perpendicular to camera - can't determine forward direction
            return None
        
        if forward_direction > 0:
            knee_forward = knee_offset
        else:
            knee_forward = -knee_offset
        
        threshold = self.exercise_config.get('knee_over_toe_threshold', 0.08)
        
        if knee_forward > threshold:
            return FormError(
                error_type='knee_over_toe',
                severity='medium',
                message='Front knee too far forward - keep knee over ankle',
                value=knee_forward
            )
        
        return None
    
    def _check_back_knee_depth(self, left_knee: np.ndarray, 
                                right_knee: np.ndarray,
                                left_ankle: np.ndarray,
                                right_ankle: np.ndarray) -> Optional[FormError]:
        
        if self.current_phase != ExercisePhase.DOWN:
            return None
        
        # Ground reference = the lowest point (highest Y) among both ankles
        ground_level = max(left_ankle[1], right_ankle[1])
        
        left_knee_height = ground_level - left_knee[1]
        right_knee_height = ground_level - right_knee[1]
        
        lower_knee_height = min(left_knee_height, right_knee_height)
        
        depth_threshold = self.exercise_config.get('depth_threshold', 0.15)
        
        if lower_knee_height > depth_threshold:
            return FormError(
                error_type='shallow_lunge',
                severity='medium',
                message='Drop back knee lower toward ground',
                value=lower_knee_height
            )
        
        return None
    
    def _check_torso_lean(self, shoulder: np.ndarray, 
                          hip: np.ndarray) -> Optional[FormError]:
        
        if self.current_phase != ExercisePhase.DOWN:
            return None
        
        torso_angle = calculate_vertical_angle(hip, shoulder)
        
        threshold = self.exercise_config.get('torso_lean_threshold', 35)
        
        if torso_angle > threshold:
            return FormError(
                error_type='torso_lean',
                severity='medium',
                message='Keep torso more upright - leaning too far',
                value=torso_angle
            )
        
        return None
    
    def calibrate(self, landmarks: np.ndarray):
        
        side = get_side_of_body(landmarks)
        
        if side == 'left':
            hip = landmarks[PoseLandmark.LEFT_HIP]
            knee = landmarks[PoseLandmark.LEFT_KNEE]
            ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        else:
            hip = landmarks[PoseLandmark.RIGHT_HIP]
            knee = landmarks[PoseLandmark.RIGHT_KNEE]
            ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        thigh_length = calculate_distance(hip, knee)
        shin_length = calculate_distance(knee, ankle)
        
        self.calibration_data['thigh_length'] = thigh_length
        self.calibration_data['shin_length'] = shin_length
        self.calibration_data['leg_length'] = thigh_length + shin_length
        
        self.is_calibrated = True
    
    def _is_rep_completed(self) -> bool:
        
        if (self.previous_phase == ExercisePhase.DOWN and 
            self.current_phase == ExercisePhase.UP and
            self._went_to_bottom):
            # Reset tracking for next rep
            self._went_to_bottom = False
            return True
        return False
    
    def reset(self):
        
        super().reset()
        self._went_to_bottom = False
        self._front_leg_side = None
        self._last_front_knee_angle = None
        self._last_hip_height = None
        self._detection_frames = 0
    
    def get_debug_info(self) -> dict:
        
        return {
            'front_leg': self._front_leg_side,
            'last_knee_angle': self._last_front_knee_angle,
            'last_hip_height': self._last_hip_height,
            'went_to_bottom': self._went_to_bottom,
            'current_phase': self.current_phase.value,
            'detection_frames': self._detection_frames
        }
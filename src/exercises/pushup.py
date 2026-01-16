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
        
        # Phase detection state for hysteresis
        self._went_to_bottom = False  # Must reach bottom before counting rep
        
    def detect_exercise(self, landmarks: np.ndarray) -> bool:
        """
        Detect if person is in push-up position (plank-like).
        Ensures person is actually on the ground, not crouching in air.
        """
        # Get key landmarks
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
        if not (height_diff_shoulder_hip < 0.3 and height_diff_hip_ankle < 0.3):
            return False
        
        # NEW: Check that person is actually on the ground
        # Wrists should be low (on ground) - Y coordinate should be high (normalized coords)
        wrist_avg_y = (left_wrist[1] + right_wrist[1]) / 2
        
        # Knees and ankles should also be relatively low (on ground or elevated behind)
        knee_avg_y = (left_knee[1] + right_knee[1]) / 2
        
        # In pushup position, wrists are on ground and should be BELOW (higher Y) the hips
        # This prevents detecting crouching in air as pushup
        if wrist_avg_y < hip_avg_y - 0.1:  # Wrists should be below hips
            return False
        
        # Also verify knees are in reasonable position (either on ground or elevated, not floating)
        # Knees should be at similar level or below hips
        if knee_avg_y < hip_avg_y - 0.2:  # Knees shouldn't be way above hips
            return False
        
        return True
    
    def detect_phase(self, landmarks: np.ndarray) -> ExercisePhase:
        """
        Detect push-up phase based on elbow angle with hysteresis to prevent flickering.
        Uses clear thresholds: must go clearly DOWN before counting, then clearly UP to complete rep.
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
        
        # Hysteresis buffer - prevents flickering at boundaries
        hysteresis = 15  # degrees
        down_leniency = 25  # Extra leniency for down phase (don't need to go as low)
        
        # Determine phase with hysteresis
        if self.current_phase == ExercisePhase.UP or self.current_phase == ExercisePhase.UNKNOWN:
            # Currently UP - only switch to DOWN if clearly below threshold
            # Use extra leniency so you don't need to go all the way down
            if elbow_angle < bottom_threshold + hysteresis + down_leniency:
                self._went_to_bottom = True  # Mark that we reached bottom
                return ExercisePhase.DOWN
            else:
                return ExercisePhase.UP
        
        elif self.current_phase == ExercisePhase.DOWN:
            # Currently DOWN - only switch to UP if clearly above threshold
            if elbow_angle > top_threshold - hysteresis:
                return ExercisePhase.UP
            else:
                return ExercisePhase.DOWN
        
        return self.current_phase  # Default: maintain current phase
    
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
                message='Hips are sagging - Core not engaged',
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
                message='Go lower in your push-up',
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
    
    def _is_rep_completed(self) -> bool:
        """
        Check if a rep was completed.
        A rep is only complete when:
        1. User went DOWN (reached bottom position)
        2. User came back UP (returned to starting position)
        
        This prevents counting during continuous motion or staying in one position.
        """
        # Rep completes when transitioning from DOWN to UP AND we actually went to bottom
        if (self.previous_phase == ExercisePhase.DOWN and 
            self.current_phase == ExercisePhase.UP and
            self._went_to_bottom):
            # Reset tracking for next rep
            self._went_to_bottom = False
            return True
        return False
    
    def reset(self):
        """Reset exercise state including pushup-specific tracking."""
        super().reset()
        self._went_to_bottom = False

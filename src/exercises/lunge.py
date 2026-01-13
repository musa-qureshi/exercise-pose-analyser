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
    
    IMPORTANT: User must stand SIDEWAYS to camera for proper detection.
    
    Terminology:
    - Front leg: The leg that's forward in the frame (bears weight, knee bends ~90°)
    - Back leg: The leg stretched behind, knee goes toward ground
    
    Detects:
    - Knee over toe (front knee too far forward)
    - Insufficient depth
    - Poor balance (excessive torso lean)
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.exercise_config = config['lunge']
        self.name = "Lunge"
        
        # Track which leg is forward (based on X position when sideways)
        self.forward_leg = None
        
    def _is_sideways_to_camera(self, landmarks: np.ndarray) -> bool:
        """
        Check if user is standing sideways to camera (not facing it).
        
        When facing camera: shoulders appear wide (large X separation)
        When sideways: shoulders appear narrow (small X separation)
        
        Returns:
            True if user is sideways to camera
        """
        left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[PoseLandmark.LEFT_HIP]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP]
        
        # Calculate horizontal width of shoulders and hips
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        hip_width = abs(left_hip[0] - right_hip[0])
        
        # When sideways, both widths will be small (< 0.12 typically)
        # When facing camera, widths will be larger (> 0.2 typically)
        sideways_threshold = self.exercise_config.get('sideways_threshold', 0.12)
        
        # Both shoulder and hip width should be narrow for sideways stance
        return shoulder_width < sideways_threshold and hip_width < sideways_threshold
    
    def _is_lunge_stance(self, landmarks: np.ndarray) -> bool:
        """
        Check if feet are in lunge position (one foot forward, one back).
        
        Key insight:
        - Lunge (sideways): feet have large X separation, small Y separation
        - Squat (facing): feet have small X separation, may have some Y separation
        
        We require:
        1. Significant horizontal (X) foot separation (front-to-back)
        2. X separation must be GREATER than Y separation
        
        Returns:
            True if feet are in lunge stance
        """
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        
        # Calculate foot separation in both axes
        x_separation = abs(left_ankle[0] - right_ankle[0])  # Front-to-back when sideways
        y_separation = abs(left_ankle[1] - right_ankle[1])  # Vertical difference
        
        min_x_separation = self.exercise_config.get('min_foot_separation', 0.10)
        
        # Must have enough front-to-back separation
        if x_separation < min_x_separation:
            return False
        
        # X separation should be greater than Y separation
        # In a lunge, feet are front-to-back (large X diff, small Y diff)
        # In a squat facing camera with offset feet, it's the opposite
        if x_separation <= y_separation:
            return False
        
        return True
        
    def detect_exercise(self, landmarks: np.ndarray) -> bool:
        """
        Detect if person is in lunge position.
        
        Requirements:
        1. User must be SIDEWAYS to camera (shoulders appear narrow)
        2. Feet must be in lunge stance (significant X separation, X > Y separation)
        
        CAMERA POSITIONING: Stand SIDEWAYS to the camera (left or right side facing camera).
        """
        # Get key landmarks
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        left_hip = landmarks[PoseLandmark.LEFT_HIP]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP]
        
        # Check basic visibility
        if min(left_ankle[2], right_ankle[2], left_hip[2], right_hip[2]) < 0.3:
            return False
        
        # CHECK 1: Must be sideways to camera (shoulders/hips appear narrow)
        if not self._is_sideways_to_camera(landmarks):
            return False
        
        # CHECK 2: Feet must be in lunge stance (front-to-back, not side-by-side)
        if not self._is_lunge_stance(landmarks):
            return False
        
        # Determine front leg by X position (reliable when sideways)
        # The leg with smaller X is more to the "left" of frame = front when facing that way
        if left_ankle[0] < right_ankle[0]:
            self.forward_leg = 'left'
        else:
            self.forward_leg = 'right'
        
        return True
    
    def detect_phase(self, landmarks: np.ndarray) -> ExercisePhase:
        """
        Detect lunge phase based on front knee angle.
        
        Phase transitions:
        - UP: Standing or leg relatively straight (knee angle > 140°)
        - DOWN: In lunge position (knee angle < 120°)
        
        A rep counts when: DOWN -> UP transition occurs
        """
        if self.forward_leg is None:
            return ExercisePhase.UNKNOWN
            
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
        
        # Get thresholds from config (with more lenient defaults)
        standing_threshold = self.exercise_config.get('standing_front_knee_angle', 140)
        lunge_threshold = self.exercise_config.get('lunge_front_knee_angle', 120)
        
        # Hysteresis to prevent flickering
        hysteresis = 10
        
        if self.current_phase == ExercisePhase.DOWN:
            # In DOWN phase - need to go back UP (knee straightens)
            if knee_angle > standing_threshold - hysteresis:
                return ExercisePhase.UP
            return ExercisePhase.DOWN
        else:
            # In UP phase (or unknown) - need to go DOWN (knee bends)
            if knee_angle < lunge_threshold + hysteresis:
                return ExercisePhase.DOWN
            return ExercisePhase.UP
    
    def analyze_form(self, landmarks: np.ndarray) -> List[FormError]:
        """
        Analyze lunge form for common errors.
        Only checks form for landmarks that are sufficiently visible.
        """
        errors = []
        
        if self.forward_leg is None:
            return errors
        
        # Get landmarks based on forward leg
        # Front leg = the one that's forward, knee bends ~90°, bears weight
        # Back leg = the one stretched behind, knee goes down toward ground
        if self.forward_leg == 'left':
            front_hip = landmarks[PoseLandmark.LEFT_HIP]
            front_knee = landmarks[PoseLandmark.LEFT_KNEE]
            front_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
            front_heel = landmarks[PoseLandmark.LEFT_HEEL]
            back_hip = landmarks[PoseLandmark.RIGHT_HIP]
            back_knee = landmarks[PoseLandmark.RIGHT_KNEE]
            back_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
            shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
        else:
            front_hip = landmarks[PoseLandmark.RIGHT_HIP]
            front_knee = landmarks[PoseLandmark.RIGHT_KNEE]
            front_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
            front_heel = landmarks[PoseLandmark.RIGHT_HEEL]
            back_hip = landmarks[PoseLandmark.LEFT_HIP]
            back_knee = landmarks[PoseLandmark.LEFT_KNEE]
            back_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
            shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
        
        # Visibility threshold - only check form if landmarks are reliable
        min_visibility = 0.5
        
        # 1. Check knee alignment (knee over toe) - only if front leg is visible
        front_leg_visible = min(front_knee[2], front_ankle[2]) >= min_visibility
        if front_leg_visible:
            knee_error = self._check_knee_alignment(front_knee, front_ankle)
            if knee_error:
                errors.append(knee_error)
        
        # 2. Check depth (back knee close to ground) - only if back leg is visible
        back_leg_visible = min(back_knee[2], back_ankle[2]) >= min_visibility
        if back_leg_visible:
            depth_error = self._check_depth(back_knee, back_ankle)
            if depth_error:
                errors.append(depth_error)
        
        # 3. Check torso position (balance) - only if shoulder/hip visible
        torso_visible = min(shoulder[2], front_hip[2]) >= min_visibility
        if torso_visible:
            balance_error = self._check_torso_lean(shoulder, front_hip, front_ankle)
            if balance_error:
                errors.append(balance_error)
        
        # 4. Check back knee position relative to front heel (back leg not too extended)
        # Only if both back knee and front heel are visible
        both_visible = min(back_knee[2], front_heel[2]) >= min_visibility
        if both_visible:
            extension_error = self._check_back_leg_extension(back_knee, front_heel)
            if extension_error:
                errors.append(extension_error)
        
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
        """
        Check if lunge is deep enough - back knee should go close to the ground.
        
        We measure the vertical (Y) distance from back knee to back ankle.
        Smaller distance = knee closer to ground = deeper lunge.
        
        Config: depth_threshold (normalized Y distance)
        ─────────────────────────────────────────────────
        EASIER (higher threshold): 0.5 - allows knee to stay high
        MODERATE: 0.3-0.4 - knee should drop noticeably
        HARDER (lower threshold): 0.15-0.25 - knee must nearly touch ground
        ─────────────────────────────────────────────────
        """
        # Only check at bottom of lunge
        if self.current_phase != ExercisePhase.DOWN:
            return None
        
        # Calculate vertical distance from back knee to ground (ankle level)
        # In normalized coords, Y increases downward, so we want knee Y close to ankle Y
        knee_height = abs(back_knee[1] - back_ankle[1])
        
        threshold = self.exercise_config.get('depth_threshold', 0.25)
        
        if knee_height > threshold:
            return FormError(
                error_type='shallow_lunge',
                severity='medium',
                message='Go lower - back knee should nearly touch ground',
                value=knee_height
            )
        
        return None
    
    def _check_back_leg_extension(self, back_knee: np.ndarray, 
                                   front_heel: np.ndarray) -> Optional[FormError]:
        """
        Check that back leg isn't too straight/extended.
        
        In a proper lunge, the back knee should drop down relatively close to 
        (but behind) the front foot. If the back knee is too far away horizontally,
        it means the back leg is too straight.
        
        We measure horizontal (X) distance from back knee to front heel.
        Smaller distance = knee closer = proper lunge form.
        
        Config: back_knee_max_distance (normalized X distance)
        ─────────────────────────────────────────────────
        EASIER (higher threshold): 0.35 - allows extended back leg
        MODERATE: 0.25-0.30 - back knee should be reasonably close
        HARDER (lower threshold): 0.15-0.20 - back knee must be close to front heel
        ─────────────────────────────────────────────────
        """
        # Only check at bottom of lunge
        if self.current_phase != ExercisePhase.DOWN:
            return None
        
        # Calculate horizontal distance from back knee to front heel
        # The back knee should drop down somewhat close to (but behind) the front foot
        horizontal_dist = abs(back_knee[0] - front_heel[0])
        
        threshold = self.exercise_config.get('back_knee_max_distance', 0.30)
        
        if horizontal_dist > threshold:
            return FormError(
                error_type='back_leg_too_straight',
                severity='medium',
                message='Back leg too extended - bring back knee closer under body',
                value=horizontal_dist
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

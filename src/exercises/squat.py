"""
Squat Exercise Analyzer
Detects and analyzes squat form with rep counting and error detection.
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
    - Shallow depth (not going low enough)
    - Forward lean (torso leaning too far forward)
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.exercise_config = config['squat']
        self.name = "Squat"
        
        # Phase detection state for hysteresis
        self._phase_locked = False  # Prevents rapid phase switching
        self._went_to_bottom = False  # Must reach bottom before counting rep
        self._last_hip_angle = None  # Track angle for direction detection
        
        # Tracking for user feedback
        self._is_descending = False  # User started going down
        self._lowest_hip_angle = 180  # Track how low they went this rep
        self._rep_had_form_errors = []  # Track form errors during rep attempt
        
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
        Detect squat phase based on hip angle with hysteresis to prevent flickering.
        Uses clear thresholds: must go clearly DOWN before counting, then clearly UP to complete rep.
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
        
        # Store for tracking
        self._last_hip_angle = hip_angle
        
        # Track if user is descending (for feedback)
        standing_threshold = self.exercise_config['standing_hip_angle']  # 160
        bottom_threshold = self.exercise_config['bottom_hip_angle']  # 110
        
        # Detect descent - user started going down
        if hip_angle < standing_threshold - 10:  # Below ~150 degrees
            self._is_descending = True
            # Track lowest point reached
            if hip_angle < self._lowest_hip_angle:
                self._lowest_hip_angle = hip_angle
        
        # Hysteresis buffer - prevents flickering at boundaries
        hysteresis = 15  # degrees
        
        # Determine phase with hysteresis
        if self.current_phase == ExercisePhase.UP or self.current_phase == ExercisePhase.UNKNOWN:
            # Currently UP - only switch to DOWN if clearly below threshold
            if hip_angle < bottom_threshold + hysteresis:
                self._went_to_bottom = True  # Mark that we reached bottom
                return ExercisePhase.DOWN
            else:
                return ExercisePhase.UP
        
        elif self.current_phase == ExercisePhase.DOWN:
            # Currently DOWN - only switch to UP if clearly above threshold
            if hip_angle > standing_threshold - hysteresis:
                return ExercisePhase.UP
            else:
                return ExercisePhase.DOWN
        
        return self.current_phase  # Default: maintain current phase
    
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
        
        # 1. Check squat depth (only give feedback when descending/down)
        depth_error = self._check_depth(hip, ankle)
        if depth_error:
            errors.append(depth_error)
        
        # 2. Check knee alignment - both valgus (knees in) AND spread (knees out)
        knee_error = self._check_knee_alignment(landmarks, side)
        if knee_error:
            errors.append(knee_error)
            self._rep_had_form_errors.append(knee_error)
        
        # 3. Check forward lean
        lean_error = self._check_forward_lean(shoulder, hip, ankle)
        if lean_error:
            errors.append(lean_error)
        
        # 4. Check foot angle (feet should be roughly parallel, not too inward/outward)
        foot_errors = self._check_foot_angle(landmarks)
        errors.extend(foot_errors)
        
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
        """Legacy method - redirects to new alignment check."""
        return self._check_knee_alignment(landmarks, side)
    
    def _check_knee_alignment(self, landmarks: np.ndarray, side: str) -> Optional[FormError]:
        """Check for knee alignment issues - both valgus (knees in) AND spread (knees too far out)."""
        # Get both knees and ankles to check alignment
        left_knee = landmarks[PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
        left_hip = landmarks[PoseLandmark.LEFT_HIP]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP]
        
        # Only check when in down phase or descending
        if self.current_phase != ExercisePhase.DOWN and not self._is_descending:
            return None
        
        # Calculate distances
        knee_distance = calculate_distance(left_knee, right_knee)
        ankle_distance = calculate_distance(left_ankle, right_ankle)
        hip_distance = calculate_distance(left_hip, right_hip)
        
        # Ratio of knee spread to ankle spread
        knee_ankle_ratio = knee_distance / ankle_distance if ankle_distance > 0 else 1.0
        
        # Ratio of knee spread to hip width (for detecting excessive spread)
        knee_hip_ratio = knee_distance / hip_distance if hip_distance > 0 else 1.0
        
        # Get thresholds from config (with defaults)
        valgus_threshold = self.exercise_config.get('knee_valgus_threshold', 0.8)
        spread_threshold = self.exercise_config.get('knee_spread_threshold', 1.8)
        
        # Check for knee VALGUS (knees caving IN) - knees closer than ankles
        if knee_ankle_ratio < valgus_threshold:
            return FormError(
                error_type='knee_valgus',
                severity='high',
                message='Knees caving in! Push knees outward over toes',
                value=knee_ankle_ratio
            )
        
        # Check for knee SPREAD (knees too far OUT) - knees much wider than hips
        # Knees should roughly track over toes, not splay outward excessively
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
    
    def _check_foot_angle(self, landmarks: np.ndarray) -> List[FormError]:
        """
        Check foot angle - feet should be roughly parallel (pointing forward).
        
        This detects:
        - Feet pointing too far outward (duck feet / external rotation)
        - Feet pointing too far inward (pigeon toed / internal rotation)
        
        The math:
        - We measure the angle from heel to toe relative to "forward" (Y-axis)
        - Positive angle = outward, Negative angle = inward
        - Natural stance allows some variation (up to ~30° outward is common)
        - We only warn for extreme angles (like 45°+ outward or any significant inward)
        
        Returns:
            List of FormError objects for any foot angle issues detected
        """
        errors = []
        
        # Get foot landmarks
        left_heel = landmarks[PoseLandmark.LEFT_HEEL]
        right_heel = landmarks[PoseLandmark.RIGHT_HEEL]
        left_toe = landmarks[PoseLandmark.LEFT_FOOT_INDEX]
        right_toe = landmarks[PoseLandmark.RIGHT_FOOT_INDEX]
        
        # Check visibility - need decent visibility for accurate measurement
        min_visibility = 0.4
        left_visible = min(left_heel[2], left_toe[2]) >= min_visibility
        right_visible = min(right_heel[2], right_toe[2]) >= min_visibility
        
        if not (left_visible or right_visible):
            return errors  # Can't check if feet aren't visible
        
        # Get thresholds from config (with sensible defaults)
        # outward_threshold: angle beyond which feet are "too outward" (duck feet)
        # inward_threshold: angle beyond which feet are "too inward" (pigeon toed)
        outward_threshold = self.exercise_config.get('foot_outward_threshold', 60)  # degrees
        inward_threshold = self.exercise_config.get('foot_inward_threshold', 20)    # degrees
        
        # Track worst angles for reporting
        worst_outward = 0
        worst_inward = 0
        
        # Check left foot
        if left_visible:
            left_angle = calculate_foot_angle(left_heel, left_toe, is_left_foot=True)
            if left_angle > outward_threshold:
                worst_outward = max(worst_outward, left_angle)
            elif left_angle < -inward_threshold:
                worst_inward = max(worst_inward, abs(left_angle))
        
        # Check right foot
        if right_visible:
            right_angle = calculate_foot_angle(right_heel, right_toe, is_left_foot=False)
            if right_angle > outward_threshold:
                worst_outward = max(worst_outward, right_angle)
            elif right_angle < -inward_threshold:
                worst_inward = max(worst_inward, abs(right_angle))
        
        # Generate errors based on detected issues
        if worst_outward > 0:
            # Severity based on how extreme the angle is
            # 50-75°: medium, 75°+: high (approaching perpendicular)
            severity = 'high' if worst_outward > 75 else 'medium'
            errors.append(FormError(
                error_type='feet_too_outward',
                severity=severity,
                message='Feet pointing too outward - keep feet more parallel',
                value=worst_outward
            ))
        
        if worst_inward > 0:
            # Inward feet are less common but also problematic
            severity = 'high' if worst_inward > 25 else 'medium'
            errors.append(FormError(
                error_type='feet_too_inward',
                severity=severity,
                message='Feet pointing too inward - keep feet more parallel',
                value=worst_inward
            ))
        
        return errors
    
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

    def _is_rep_completed(self) -> bool:
        """
        Check if a rep was completed.
        A rep is only complete when:
        1. User went DOWN (reached bottom position)
        2. User came back UP (returned to standing)
        
        This prevents counting during continuous motion or staying in one position.
        """
        # Rep completes when transitioning from DOWN to UP AND we actually went to bottom
        if (self.previous_phase == ExercisePhase.DOWN and 
            self.current_phase == ExercisePhase.UP and
            self._went_to_bottom):
            # Reset tracking for next rep
            self._went_to_bottom = False
            self._is_descending = False
            self._lowest_hip_angle = 180
            self._rep_had_form_errors = []
            return True
        return False
    
    def reset(self):
        """Reset exercise state including squat-specific tracking."""
        super().reset()
        self._phase_locked = False
        self._went_to_bottom = False
        self._last_hip_angle = None
        self._is_descending = False
        self._lowest_hip_angle = 180
        self._rep_had_form_errors = []
"""
Base Exercise Class
Abstract base class for all exercise implementations.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Optional
import numpy as np


class ExercisePhase(Enum):
    """Common exercise phases."""
    UNKNOWN = "unknown"
    READY = "ready"
    DOWN = "down"
    UP = "up"
    HOLD = "hold"


class FormError:    
    def __init__(self, error_type: str, severity: str, message: str, value: Optional[float] = None):
        """
        Initialize form error.
        Args:
            error_type: Type of error (e.g., 'knee_valgus', 'shallow_depth')
            severity: 'low', 'medium', or 'high'
            message: Human-readable error message
            value: Optional numeric value associated with the error
        """
        self.error_type = error_type
        self.severity = severity
        self.message = message
        self.value = value
    
    def __repr__(self):
        return f"FormError({self.error_type}, {self.severity}): {self.message}"


class BaseExercise(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.exercise_config = None  # Set by subclass
        
        # State tracking
        self.current_phase = ExercisePhase.UNKNOWN
        self.previous_phase = ExercisePhase.UNKNOWN
        self.rep_count = 0
        self.form_errors = []
        
        # Calibration
        self.is_calibrated = False
        self.calibration_data = {}
        
    @abstractmethod
    def detect_exercise(self, landmarks: np.ndarray) -> bool:
        """
        Detect if the person is in position to perform this exercise.
        
        Args:
            landmarks: Pose landmarks array
            
        Returns:
            True if exercise is detected
        """
        pass
    
    @abstractmethod
    def detect_phase(self, landmarks: np.ndarray) -> ExercisePhase:
        """
        Detect the current phase of the exercise.
        
        Args:
            landmarks: Pose landmarks array
            
        Returns:
            Current exercise phase
        """
        pass
    
    @abstractmethod
    def analyze_form(self, landmarks: np.ndarray) -> List[FormError]:
        """
        Analyze exercise form and detect errors.
        
        Args:
            landmarks: Pose landmarks array
            
        Returns:
            List of form errors detected
        """
        pass
    
    def update(self, landmarks: np.ndarray) -> Dict:
        """
        Update exercise state with new landmarks.
        
        Args:
            landmarks: Pose landmarks array
            
        Returns:
            Dictionary containing exercise state information
        """
        exercise_detected = self.detect_exercise(landmarks)
        
        if not exercise_detected:
            return {
                'exercise_detected': False,
                'phase': ExercisePhase.UNKNOWN.value,
                'rep_count': self.rep_count,
                'form_errors': []
            }
        
        self.previous_phase = self.current_phase
        self.current_phase = self.detect_phase(landmarks)  #Update phase
        
        if self._is_rep_completed():
            self.rep_count += 1
        
        # Analyze form
        self.form_errors = self.analyze_form(landmarks)
        
        return {
            'exercise_detected': True,
            'phase': self.current_phase.value,
            'rep_count': self.rep_count,
            'form_errors': self.form_errors
        }
    
    def _is_rep_completed(self) -> bool:
        """
        Check if a rep was completed based on phase transition. Returns
        True if rep completed
        """
        # Rep completes when returning to ready/up position from down position
        return (
            self.previous_phase == ExercisePhase.DOWN and 
            self.current_phase == ExercisePhase.UP
        )
    
    def calibrate(self, landmarks: np.ndarray):
        # Subclasses can override for specific calibration
        self.is_calibrated = True
    
    def reset(self):
        self.current_phase = ExercisePhase.UNKNOWN
        self.previous_phase = ExercisePhase.UNKNOWN
        self.rep_count = 0
        self.form_errors = []
    
    def get_feedback_text(self) -> List[str]:
        feedback = []
        
        # Add rep count
        feedback.append(f"Reps: {self.rep_count}")
        
        # Add phase
        feedback.append(f"Phase: {self.current_phase.value.title()}")
        
        # Add form errors
        if self.form_errors:
            feedback.append("Form Issues:")
            for error in self.form_errors:
                feedback.append(f"  - {error.message}")
        else:
            feedback.append("Form: Good!")
        
        return feedback

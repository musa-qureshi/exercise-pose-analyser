"""
Pose Detector Module
Uses MediaPipe for pose estimation with temporal smoothing.
Compatible with both old (mp.solutions) and new (mp.tasks) APIs.
"""

import cv2
import numpy as np
import os
from collections import deque
from typing import Optional, Tuple, List
from dataclasses import dataclass
import mediapipe as mp


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class PoseLandmarks:
    landmark: List[Landmark]


# Check which API is available
_USE_NEW_API = not hasattr(mp, 'solutions')


class PoseDetector:
    """
    Handles pose detection using MediaPipe with temporal smoothing for stability.
    """
    
    def __init__(self, config: dict):
        self.config = config
        pose_config = config['pose_detection']
        
        if _USE_NEW_API:
            # New API (mediapipe >= 0.10.30 for Python 3.13)
            self._init_new_api(pose_config)
        else:
            # Old API (mediapipe <= 0.10.13)
            self._init_old_api(pose_config)
        
        # Temporal smoothing
        self.smoothing_enabled = config['smoothing']['enabled']
        self.window_size = config['smoothing']['window_size']
        self.landmark_history = deque(maxlen=self.window_size)
    
    def _init_new_api(self, pose_config: dict):
        """Initialize using the new MediaPipe Tasks API."""
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker, 
            PoseLandmarkerOptions,
            RunningMode
        )
        
        self.use_new_api = True
        self.mp_pose = None
        self.mp_drawing = None
        
        # Find model file
        model_path = self._find_model_file()
        
        # Set up options for new API
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=pose_config['min_detection_confidence'],
            min_pose_presence_confidence=pose_config['min_tracking_confidence'],
            min_tracking_confidence=pose_config['min_tracking_confidence'],
            output_segmentation_masks=pose_config.get('enable_segmentation', False)
        )
        
        self.pose = PoseLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0
        
        print("Using MediaPipe Tasks API (Python 3.13 compatible)")
    
    def _init_old_api(self, pose_config: dict):
        """Initialize using the old MediaPipe Solutions API."""
        self.use_new_api = False
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            model_complexity=pose_config['model_complexity'],
            min_detection_confidence=pose_config['min_detection_confidence'],
            min_tracking_confidence=pose_config['min_tracking_confidence'],
            enable_segmentation=pose_config['enable_segmentation']
        )
        
        print("Using MediaPipe Solutions API (legacy)")
    
    def _find_model_file(self) -> str:
        # Check common locations
        possible_paths = [
            'models/pose_landmarker.task',
            'pose_landmarker.task',
            os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'pose_landmarker.task'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If not found, try to download it
        return self._download_model()
    
    def _download_model(self) -> str:
        """Download the pose landmarker model if not present."""
        model_dir = 'models'
        model_path = os.path.join(model_dir, 'pose_landmarker.task')
        
        if os.path.exists(model_path):
            return model_path
        
        print("Downloading pose landmarker model...")
        os.makedirs(model_dir, exist_ok=True)
        
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
        
        urllib.request.urlretrieve(url, model_path)
        print(f"Model downloaded to {model_path}")
        
        return model_path
        
    def detect_pose(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """
        Detect pose landmarks in the frame.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            Tuple of (landmarks array, detection success flag)
            landmarks: Nx3 array of normalized [x, y, visibility] coordinates
        """
        if self.use_new_api:
            return self._detect_new_api(frame)
        else:
            return self._detect_old_api(frame)
    
    def _detect_new_api(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Process the frame
        self.frame_timestamp_ms += 33  # Approx 30 fps
        results = self.pose.detect_for_video(mp_image, self.frame_timestamp_ms)
        
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            # Extract landmarks as numpy array
            landmarks = np.array([
                [lm.x, lm.y, lm.visibility if hasattr(lm, 'visibility') else 1.0] 
                for lm in results.pose_landmarks[0]
            ])
            
            # Apply temporal smoothing if enabled
            if self.smoothing_enabled:
                landmarks = self._smooth_landmarks(landmarks)
            
            return landmarks, True
        
        return None, False
    
    def _detect_old_api(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Extract landmarks as numpy array
            landmarks = np.array([
                [lm.x, lm.y, lm.visibility] 
                for lm in results.pose_landmarks.landmark
            ])
            
            # Apply temporal smoothing if enabled
            if self.smoothing_enabled:
                landmarks = self._smooth_landmarks(landmarks)
            
            return landmarks, True
        
        return None, False
    
    def _smooth_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        self.landmark_history.append(landmarks)
        
        if len(self.landmark_history) < 2:
            return landmarks
        
        # Average over the history window
        smoothed = np.mean(self.landmark_history, axis=0)
        return smoothed
    
    def draw_pose(self, frame: np.ndarray, landmarks: np.ndarray, 
                  connections: bool = True) -> np.ndarray:
        if landmarks is None:
            return frame
        
        h, w = frame.shape[:2]
        viz_config = self.config['visualization']
        
        if not viz_config['show_landmarks']:
            return frame
        
        # Define pose connections (MediaPipe format)
        POSE_CONNECTIONS = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),
            # Torso
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Left arm
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            # Right arm
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            # Left leg
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            # Right leg
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
        ]
        
        if connections: #draw connections
            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_vis = landmarks[start_idx][2]
                    end_vis = landmarks[end_idx][2]
                    
                    if start_vis > 0.5 and end_vis > 0.5:
                        start_point = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                        end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
                        cv2.line(frame, start_point, end_point, (255, 255, 255), viz_config['line_thickness'])
        
        #draw landmarks
        for i, landmark in enumerate(landmarks):
            x, y, vis = landmark
            if vis > 0.5:  # Only visible landmarks
                cv2.circle(frame, (int(x * w), int(y * h)), 
                         viz_config['landmark_radius'], (0, 255, 0), -1)
        
        return frame
    
    def get_landmark_coords(self, landmarks: np.ndarray, 
                           landmark_idx: int, 
                           frame_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Get pixel coordinates for a specific landmark.
        
        Args:
            landmarks: Pose landmarks array
            landmark_idx: Index of the landmark
            frame_shape: (height, width) of the frame
            
        Returns:
            (x, y) pixel coordinates
        """
        h, w = frame_shape
        x, y, _ = landmarks[landmark_idx]
        return int(x * w), int(y * h)
    
    def reset_smoothing(self):
        self.landmark_history.clear()
    
    def release(self):
        self.pose.close()

"""
Geometry Utilities
Helper functions for calculating angles, distances, and ratios from pose landmarks.
"""

import numpy as np
import mediapipe as mp
from typing import Tuple


# MediaPipe Pose Landmark indices for easy reference
class PoseLandmark:
    """MediaPipe Pose landmark indices."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate angle at point b formed by points a-b-c.
    
    Args:
        a, b, c: 2D or 3D coordinate arrays
        
    Returns:
        Angle in degrees (0-180)
    """
    a = np.array(a[:2])  # Use only x, y
    b = np.array(b[:2])
    c = np.array(c[:2])
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle


def calculate_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        a, b: 2D or 3D coordinate arrays
        
    Returns:
        Distance in normalized coordinates
    """
    a = np.array(a[:2])
    b = np.array(b[:2])
    return np.linalg.norm(a - b)


def calculate_vertical_angle(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate angle from vertical (y-axis) for line segment a-b.
    
    Args:
        a: Start point (2D or 3D)
        b: End point (2D or 3D)
        
    Returns:
        Angle from vertical in degrees (0 = vertical, 90 = horizontal)
    """
    a = np.array(a[:2])
    b = np.array(b[:2])
    
    # Vector from a to b
    vector = b - a
    
    # Vertical reference vector (pointing down)
    vertical = np.array([0, 1])
    
    # Calculate angle
    cosine_angle = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(np.abs(cosine_angle)))
    
    return angle


def calculate_midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate midpoint between two points.
    
    Args:
        a, b: Coordinate arrays
        
    Returns:
        Midpoint coordinates
    """
    return (np.array(a) + np.array(b)) / 2


def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, 
                          line_end: np.ndarray) -> float:
    """
    Calculate perpendicular distance from point to line segment.
    
    Args:
        point: Point coordinates
        line_start: Line segment start
        line_end: Line segment end
        
    Returns:
        Distance in normalized coordinates
    """
    point = np.array(point[:2])
    line_start = np.array(line_start[:2])
    line_end = np.array(line_end[:2])
    
    # Line vector
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-6:
        return np.linalg.norm(point - line_start)
    
    # Normalized line vector
    line_unitvec = line_vec / line_len
    
    # Vector from line start to point
    point_vec = point - line_start
    
    # Project point onto line
    projection = np.dot(point_vec, line_unitvec)
    projection = np.clip(projection, 0, line_len)
    
    # Closest point on line
    closest = line_start + projection * line_unitvec
    
    # Distance from point to closest point
    return np.linalg.norm(point - closest)


def get_side_of_body(landmarks: np.ndarray) -> str:
    """
    Determine which side of the body is more visible to the camera.
    
    Args:
        landmarks: Pose landmarks array
        
    Returns:
        'left' or 'right' indicating more visible side
    """
    # Compare visibility of left and right landmarks
    left_vis = (
        landmarks[PoseLandmark.LEFT_SHOULDER][2] +
        landmarks[PoseLandmark.LEFT_HIP][2] +
        landmarks[PoseLandmark.LEFT_KNEE][2]
    ) / 3
    
    right_vis = (
        landmarks[PoseLandmark.RIGHT_SHOULDER][2] +
        landmarks[PoseLandmark.RIGHT_HIP][2] +
        landmarks[PoseLandmark.RIGHT_KNEE][2]
    ) / 3
    
    return 'left' if left_vis > right_vis else 'right'


def normalize_landmark(landmark: np.ndarray, reference_point: np.ndarray, 
                       scale: float) -> np.ndarray:
    """
    Normalize landmark coordinates relative to a reference point and scale.
    
    Args:
        landmark: Landmark coordinates
        reference_point: Reference point (e.g., hip midpoint)
        scale: Scale factor (e.g., torso length)
        
    Returns:
        Normalized coordinates
    """
    return (np.array(landmark) - np.array(reference_point)) / (scale + 1e-6)


def calculate_foot_angle(heel: np.ndarray, toe: np.ndarray, is_left_foot: bool) -> float:
    """
    Calculate foot angle relative to the forward (vertical) direction.
    
    The foot direction is determined by the vector from heel to toe.
    We measure how much this vector deviates from pointing straight forward.
    
    In normalized coordinates:
    - X increases to the right
    - Y increases downward (forward direction when facing camera)
    
    Args:
        heel: Heel landmark coordinates (x, y, visibility)
        toe: Foot index (toe) landmark coordinates (x, y, visibility)
        is_left_foot: True if this is the left foot, False for right foot
        
    Returns:
        Angle in degrees:
        - Positive = foot pointing outward (toes away from body midline)
        - Negative = foot pointing inward (toes toward body midline)
        - 0 = foot pointing straight forward
        
    Math explanation:
        The forward direction is (0, 1) - straight down in image coordinates.
        We compute the signed angle between the foot vector and forward.
        For left foot: positive X deviation = outward
        For right foot: negative X deviation = outward
    """
    heel = np.array(heel[:2])
    toe = np.array(toe[:2])
    
    # Foot direction vector (heel to toe)
    foot_vec = toe - heel
    
    # Forward direction (straight ahead, Y-positive in image coords)
    forward = np.array([0, 1])
    
    # Calculate angle using atan2 for signed angle
    # atan2(cross, dot) gives signed angle
    cross = foot_vec[0] * forward[1] - foot_vec[1] * forward[0]  # 2D cross product (scalar)
    dot = np.dot(foot_vec, forward)
    
    angle_rad = np.arctan2(cross, dot)
    angle_deg = np.degrees(angle_rad)
    
    # Adjust sign based on which foot:
    # For left foot: positive angle (toe going right/outward) = outward
    # For right foot: positive angle (toe going right) = inward, so we flip
    if not is_left_foot:
        angle_deg = -angle_deg
    
    return angle_deg

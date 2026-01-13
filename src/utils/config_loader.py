"""
Configuration Loader
Handles loading and validation of configuration from YAML file.
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = [
        'pose_detection', 'smoothing', 'performance',
        'squat', 'pushup', 'lunge', 'visualization', 'logging'
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    return config


def get_exercise_config(config: Dict[str, Any], exercise_name: str) -> Dict[str, Any]:
    """
    Get configuration for specific exercise.
    
    Args:
        config: Full configuration dictionary
        exercise_name: Name of exercise ('squat', 'pushup', 'lunge')
        
    Returns:
        Exercise-specific configuration
        
    Raises:
        ValueError: If exercise not found in config
    """
    if exercise_name not in config:
        raise ValueError(f"Exercise '{exercise_name}' not found in configuration")
    
    return config[exercise_name]

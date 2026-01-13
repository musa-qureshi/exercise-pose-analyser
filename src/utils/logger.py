"""
Session Logger
Handles CSV logging of exercise session data.
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional


class SessionLogger:
    """
    Logs exercise session data to CSV file.
    """
    
    def __init__(self, config: dict):
        """
        Initialize session logger.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logging_config = config['logging']
        
        self.log_file = None
        self.csv_writer = None
        self.session_start_time = None
        self.frame_count = 0
        
        if self.logging_config['enabled']:
            self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Create log directory and file."""
        try:
            # Create logs directory if it doesn't exist
            log_dir = self.logging_config['output_directory']
            os.makedirs(log_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prefix = self.logging_config['session_prefix']
            filename = f"{prefix}_{timestamp}.csv"
            filepath = os.path.join(log_dir, filename)
            
            # Open file and create CSV writer
            self.log_file = open(filepath, 'w', newline='')
            self.csv_writer = csv.writer(self.log_file)
            
            # Write header
            header = [
                'frame',
                'timestamp',
                'exercise',
                'exercise_detected',
                'phase',
                'rep_count',
                'form_errors',
                'error_count'
            ]
            self.csv_writer.writerow(header)
            
            self.session_start_time = datetime.now()
            print(f"Logging session data to: {filepath}")
        except Exception as e:
            print(f"Warning: Could not initialize log file: {e}")
            self.log_file = None
            self.csv_writer = None
    
    def log_frame(self, exercise_name: str, exercise_state: Dict):
        """
        Log data for a single frame.
        
        Args:
            exercise_name: Name of the exercise being performed
            exercise_state: Dictionary containing exercise state from update()
        """
        if not self.logging_config['enabled'] or self.csv_writer is None:
            return
        
        self.frame_count += 1
        
        # Calculate timestamp relative to session start
        elapsed = (datetime.now() - self.session_start_time).total_seconds()
        
        # Format form errors
        form_errors = exercise_state.get('form_errors', [])
        error_messages = [f"{err.error_type}:{err.severity}" for err in form_errors]
        error_str = ';'.join(error_messages) if error_messages else 'none'
        
        # Write row
        row = [
            self.frame_count,
            f"{elapsed:.2f}",
            exercise_name,
            exercise_state.get('exercise_detected', False),
            exercise_state.get('phase', 'unknown'),
            exercise_state.get('rep_count', 0),
            error_str,
            len(form_errors)
        ]
        
        self.csv_writer.writerow(row)
    
    def close(self):
        """Close log file and write summary."""
        if self.log_file:
            # Write summary at end
            if self.session_start_time:
                duration = (datetime.now() - self.session_start_time).total_seconds()
                avg_fps = self.frame_count / duration if duration > 0 else 0
                
                self.csv_writer.writerow([])
                self.csv_writer.writerow(['=== Session Summary ==='])
                self.csv_writer.writerow(['Total Frames', self.frame_count])
                self.csv_writer.writerow(['Duration (s)', f"{duration:.2f}"])
                self.csv_writer.writerow(['Avg FPS', f"{avg_fps:.1f}"])
            
            self.log_file.close()
            print(f"Session log closed. Total frames: {self.frame_count}")

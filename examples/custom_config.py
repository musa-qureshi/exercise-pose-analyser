"""
Example: Custom exercise configuration

This example shows how to create a custom configuration file
with adjusted thresholds for your specific needs.
"""

import yaml
import shutil

# Copy default config
shutil.copy('config.yaml', 'config_custom.yaml')

# Load and modify
with open('config_custom.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Make squats easier (less strict)
config['squat']['depth_shallow_threshold'] = 0.65  # Allow shallower squats
config['squat']['knee_valgus_threshold'] = 20      # More lenient on knee valgus

# Make push-ups stricter
config['pushup']['bottom_elbow_angle'] = 85        # Must go deeper
config['pushup']['hip_sag_threshold'] = 10         # Less tolerance for hip sag

# Adjust visualization
config['visualization']['feedback_font_scale'] = 0.9  # Bigger text

# Save custom config
with open('config_custom.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("Custom configuration saved to: config_custom.yaml")
print("\nTo use it, run:")
print("python main.py --config config_custom.yaml --exercise squat")

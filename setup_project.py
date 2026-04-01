"""
Setup script to ensure all project files are in place.
"""

import os
import shutil

def create_directory_structure():
    """Create the required directory structure."""
    directories = [
        'config',
        'src',
        'src/ga',
        'data/raw',
        'data/processed',
        'outputs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created {directory}/")

def create_config_files():
    """Create configuration files."""
    
    # config/constants.py
    constants_content = '''"""
Physical and operational constants for the scheduling system.
"""

# Physical constraints
OBS_DURATION_SEC = 90  # Fixed observation duration
SLEW_GAP_SEC = 120     # Telescope slew and settling time

# Night window boundaries (UTC) - Hanle station
# IST: 18:00 to 06:00 = UTC: 12:30 to 00:30 (next day)
NIGHT_START_UTC = "12:30"  # 18:00 IST = 12:30 UTC
NIGHT_END_UTC = "00:30"    # 06:00 IST = 00:30 UTC (next day)

# Time format constants
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
TIME_FORMAT_HM = "%H:%M"

# Genetic Algorithm parameters
GA_POPULATION_SIZE = 50
GA_GENERATIONS = 100
GA_TOURNAMENT_SIZE = 3
GA_MUTATION_RATE = 0.1
GA_EARLY_STOP_PATIENCE = 20

# Multi-day scheduling
MAX_NIGHTS = 5

# File paths
RAW_DATA_PATH = "data/raw/geo_visibility.csv"
PROCESSED_DATA_PATH = "data/processed/night_filtered.csv"

print(f"Night window configured: {NIGHT_START_UTC} to {NIGHT_END_UTC} UTC")
'''
    
    with open('config/constants.py', 'w') as f:
        f.write(constants_content)
    print("✓ Created config/constants.py")
    
    # config/__init__.py
    init_content = '''"""
Configuration package for GEO satellite scheduling system.
Exposes configuration values from constants.
"""

from .constants import *

__all__ = ['OBS_DURATION_SEC', 'SLEW_GAP_SEC', 'NIGHT_START_UTC', 
           'NIGHT_END_UTC', 'TIME_FORMAT', 'TIME_FORMAT_HM',
           'GA_POPULATION_SIZE', 'GA_GENERATIONS', 'GA_TOURNAMENT_SIZE',
           'GA_MUTATION_RATE', 'GA_EARLY_STOP_PATIENCE', 'MAX_NIGHTS',
           'RAW_DATA_PATH', 'PROCESSED_DATA_PATH']
'''
    
    with open('config/__init__.py', 'w') as f:
        f.write(init_content)
    print("✓ Created config/__init__.py")
    
    # config/night_window.py
    night_window_content = '''"""
Night window configuration for Hanle station.
"""

NIGHT_WINDOW = {
    'start_utc': "12:30",
    'end_utc': "00:30",
    'crossing_midnight': True,
    'timezone': "UTC"
}
'''
    
    with open('config/night_window.py', 'w') as f:
        f.write(night_window_content)
    print("✓ Created config/night_window.py")

def create_src_files():
    """Create source code files."""
    
    # src/__init__.py
    init_content = '''"""
GEO Satellite Debris Observation Scheduling System
Main package for scheduling algorithms and utilities.
"""

from . import preprocessing
from . import constraints
from . import greedy_scheduler
from . import ga
from . import utils

__all__ = ['preprocessing', 'constraints', 'greedy_scheduler', 'ga', 'utils']
'''
    
    with open('src/__init__.py', 'w') as f:
        f.write(init_content)
    print("✓ Created src/__init__.py")
    
    # Check if data file exists
    if not os.path.exists('data/raw/geo_visibility.csv'):
        print("\n⚠️  WARNING: data/raw/geo_visibility.csv not found!")
        print("Please copy your CSV file to data/raw/geo_visibility.csv")
        return False
    
    return True

def check_dependencies():
    """Check and install required dependencies."""
    print("\nChecking dependencies...")
    
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'pytz']
    
    import subprocess
    import sys
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")

def main():
    """Main setup function."""
    print("="*60)
    print("GEO Satellite Scheduler - Project Setup")
    print("="*60)
    
    # Create directories
    create_directory_structure()
    
    # Create config files
    create_config_files()
    
    # Create src files
    data_exists = create_src_files()
    
    if not data_exists:
        return
    
    # Check dependencies
    check_dependencies()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Ensure your geo_visibility.csv is in data/raw/")
    print("2. Run: python main.py --date 2025-10-07")
    print("3. Check outputs/ directory for results")
    
    # Check if main.py exists
    if not os.path.exists('main.py'):
        print("\n⚠️  WARNING: main.py not found in current directory!")
        print("Please ensure main.py is in the project root.")
    else:
        print("\n✓ main.py found")

if __name__ == "__main__":
    main()
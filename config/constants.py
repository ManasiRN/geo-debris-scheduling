"""
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

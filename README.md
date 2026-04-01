<<<<<<< HEAD
# GEO Satellite Debris Observation Scheduling System

A constrained time-slot scheduling system for ground-based telescope observation of GEO satellite debris from Hanle station.

## Overview

This system schedules observations of ~600 GEO satellite debris objects across multiple nights, respecting hard physical constraints:

- Single telescope at Hanle station
- Fixed observation duration: 90 seconds
- Mandatory slew/settling time: 120 seconds
- Night window: 12:30–00:30 UTC (18:00–06:00 IST)
- Maximum 5 nights of observation

## Features

- **Multi-day scheduling**: Distributes observations across ≤5 nights
- **Dual scheduling algorithms**: Greedy baseline + Genetic Algorithm optimizer
- **Constraint validation**: Ensures all physical limits are respected
- **Coverage analysis**: Identifies schedulable vs. unschedulable debris
- **Visualization**: Gantt charts and GA convergence plots

## System Architecture
=======
# geo-debris-scheduling
>>>>>>> 51602a478ed6874351420cff3d89d505a3b22e7c

"""Fixture-only physics-guided teacher/student baselines."""
from .teacher import CableTeacherConfig, Stimulus, simulate_teacher, make_stimulus_panel
from .students import MLPDynamics, GraphDynamics, PhysicsResidualDynamics
__all__ = ["CableTeacherConfig", "Stimulus", "simulate_teacher", "make_stimulus_panel", "MLPDynamics", "GraphDynamics", "PhysicsResidualDynamics"]

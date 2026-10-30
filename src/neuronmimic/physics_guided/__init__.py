"""Physics-guided differentiable dynamics benchmark components."""

from .teacher import CableTeacherConfig, Stimulus, simulate_teacher, make_stimulus_panel
from .students import MLPDynamics, GraphDynamics, PhysicsResidualDynamics, PhysicsOnlyDynamics
from .metrics import evaluate_student

__all__ = [
    "CableTeacherConfig", "Stimulus", "simulate_teacher", "make_stimulus_panel",
    "MLPDynamics", "GraphDynamics", "PhysicsResidualDynamics", "evaluate_student", "PhysicsOnlyDynamics",
]

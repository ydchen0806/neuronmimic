"""Thin connectome adapters with explicit provenance boundaries."""
from .male_cns_adapter import MaleCNSAdapterConfig, audit_male_cns, normalize_tables, write_adapter_outputs
__all__ = ["MaleCNSAdapterConfig", "audit_male_cns", "normalize_tables", "write_adapter_outputs"]

# MaleCNS adapter

The optional `neuronmimic.connectome.male_cns_adapter` module normalizes a
MaleCNS v1.0 annotation table and directed aggregate edge table. It preserves
64-bit body IDs as strings and writes source/output hashes. It does not convert
coordinate units, infer synaptic polarity, or map contacts to cable branches.

The parent BioFly project contains the audited local source inventory and the
actual command. This standalone release contains the schema-level adapter so a
user can test the contract without downloading restricted raw data.

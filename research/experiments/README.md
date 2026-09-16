# V5 Experiment Metadata

Each V5 research run records an immutable JSON document in this directory through
`quant.research.ExperimentRegistry`. Metadata documents are intentionally ignored
because they contain machine-local run timestamps and generated result metrics.

Every record includes the experiment ID, UTC timestamp, Git commit, data coverage,
universe, factor set, model, parameters, result metrics, and notes. Do not edit a
record after it has been created; create a new experiment ID instead.

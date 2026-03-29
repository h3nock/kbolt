# ADR 0002: Role-Specific Inference Config

## Status
Superseded by ADR 0003

This ADR captured one correct direction and several wrong ones.

What survived:
- inference is role-shaped, not one global model knob
- reranking and expansion need independent contracts
- unset roles should stay unset rather than being synthesized heuristically

What did not survive:
- in-process local runtime ownership inside role config
- local model-file/runtime knobs as part of the user-facing inference schema
- treating local inference as something kbolt downloads and manages itself

The target architecture now standardizes on:
- one local backend family: `llama.cpp server`
- multiple local deployment profiles of that same kind
- many remote backends
- per-role provider choice through provider profiles and role bindings

See [ADR 0003](./0003-provider-profiles-and-shared-local-inference.md).

## Historical Takeaway
ADR 0002 was a useful intermediate step because it proved that inference should be modeled by
role. It is preserved only as historical context. The concrete config shape and local-runtime
assumptions in this ADR were removed and must not be revived.

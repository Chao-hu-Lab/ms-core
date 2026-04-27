# ms-core

Shared core library for mass spectrometry data processing, calibration, and
pipeline contracts.

`ms-core` is intended to be consumed by downstream projects such as the MS
preprocessing toolkit and DNP through pinned commits or tags. Consumer projects
own their own GUI, packaging, launch flow, and adapter tests; shared processing
behavior belongs in this repository first.

## Testing

See `docs/TESTING.md` for the durable testing strategy, marker policy, local
commands, and consumer coordination rules.

Common local checks from the `ms-core` root:

```powershell
python -m pytest -m contract -v --tb=short
python -m pytest -m algorithm -v --tb=short
python -m pytest tests/ -v --tb=short -x
```

Supported CI Python versions are 3.11 and 3.12.

When a public contract changes, update `ms-core` tests first, then bump and
verify each consumer project separately.

# rat-registry: model refresh command

**date**: 2026-02-20
**status**: implemented
**extends**: [2026-02-13-registry-tool-design.md](2026-02-13-registry-tool-design.md)

## overview

adds `rat-registry model refresh` — a bulk re-fetch command that updates all
model entries in the registry file by calling `fetch_model_metadata()` for each
and merging the results in-place.

## command

```
rat-registry model refresh    # re-fetch metadata for all models, merge in-place
```

no extra flags. uses the same `--path` / `REGISTRY_PATH` as other commands.

## behaviour

1. load registry file (existing path logic)
2. build `EmbeddedGateway` from env vars (`OPENROUTER_API_KEY`, `HF_API_KEY`)
3. for each model in `registry.models` (in order):
   - print `fetching <model-id>...`
   - call `gateway.fetch_model_metadata(model_id)`
   - success → merge into in-memory registry (per-field, same as `ModelRegistry::merge()`)
   - failure → record to failure list, print warning, continue
4. if any successes: write registry atomically (same atomic write as other commands)
5. print summary: `refreshed N/M models`; list failed model IDs if any
6. exit non-zero if any failures

## merge semantics

fetched metadata is merged field-by-field, not whole-entry replacement:
- `parameters` — per-key override (fetched keys overwrite, absent keys preserved)
- `pricing`, `max_output_tokens`, `info.context_window` — replace if present in fetched result
- `info.capabilities` — replace if non-empty in fetched result

this matches `ModelRegistry::merge()` and preserves hand-curated data that
providers don't return (e.g. NLI/embedding capability flags, custom parameters).

note: `ModelRegistry::merge()` currently operates on an in-memory registry, not
directly on `Vec<ModelMetadata>`. the implementation applies equivalent semantics
manually over the vec, or loads into a `ModelRegistry` for the merge then
serialises back.

## implementation location

fits within the existing `model` subcommand handler in `src/bin/rat_registry.rs`.
no new files needed unless the binary is already split into modules.

## what this does NOT do

- no parallel fetching (sequential is sufficient for a maintainer tool)
- no partial writes (write only after all fetches complete)
- no git operations
- no filtering by provider or capability (refresh all or nothing)

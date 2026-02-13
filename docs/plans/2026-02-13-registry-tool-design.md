# rat-registry: registry management tool

**date**: 2026-02-13
**status**: design

## overview

`rat-registry` is a maintainer-only CLI tool for managing the live model
registry (`../registry/registry.json`). it embeds `EmbeddedGateway` to fetch
model metadata directly from providers (no ratd dependency), and reads/writes
the `RemoteRegistry` JSON format using the same serde types as ratatoskr proper.

the tool lives in the ratatoskr repo to stay in sync with type definitions but
is gated behind a `registry-tool` feature flag so it never affects normal builds.

## binary & feature gating

```toml
# Cargo.toml additions

[features]
registry-tool = ["dep:clap", "dep:dialoguer", "dep:tracing-subscriber",
                 "openrouter", "huggingface"]

[[bin]]
name = "rat-registry"
path = "src/bin/rat_registry.rs"
required-features = ["registry-tool"]
```

**build**: `cargo build --bin rat-registry --features registry-tool`

`dialoguer` provides y/n confirmation prompts. `clap` is already an optional dep
(shared with server/client). the tool pulls in openrouter + huggingface providers
for metadata fetching.

## commands

### model management

```
rat-registry model add <model-id>       # fetch metadata, add to registry
rat-registry model list                  # tabular summary of registered models
rat-registry model remove <model-id>    # remove model entry
```

**`model add`**:
1. read registry file
2. if model already exists → prompt "overwrite? [y/n]"
3. build `EmbeddedGateway` from env vars (`OPENROUTER_API_KEY`, `HF_API_KEY`)
4. call `gateway.fetch_model_metadata(model_id)`
5. append `ModelMetadata` to models list, sort by id
6. write registry file

**`model list`**: tabular output — id, provider, capabilities, context window, pricing.

**`model remove`**:
1. read registry file
2. check if model is referenced by any preset → warn with details
3. prompt "remove? [y/n]" (or "remove? model is used by preset `premium.agentic` [y/n]")
4. remove entry, write file

### preset management

```
rat-registry preset set <tier> <slot> <model-id>   # set one preset entry
rat-registry preset list                            # display preset table
rat-registry preset remove <tier>                   # remove entire tier
```

**`preset set`**:
1. read registry file
2. infer known cost tiers and capability slots from existing presets
3. if `<tier>` is novel → warn "tier `ultra` doesn't exist yet, add it? [y/n]"
4. if `<slot>` is novel → warn "slot `reasoning` doesn't exist yet, add it? [y/n]"
5. if `<model-id>` is not in the models list → error "model not in registry, add it first"
6. set the entry, write file

**`preset remove`**: remove entire tier with confirmation.

## registry I/O

the tool reads and writes `RemoteRegistry` directly:

```rust
fn load_registry(path: &Path) -> Result<RemoteRegistry> {
    let content = fs::read_to_string(path)?;
    serde_json::from_str(&content)
}

fn save_registry(path: &Path, registry: &RemoteRegistry) -> Result<()> {
    let content = serde_json::to_string_pretty(registry)?;
    fs::write(path, content)
}
```

single source of truth — same types the library uses for parsing.

## smart warnings (convention-based validation)

no schema changes needed. the tool infers "known" values from the existing
registry contents:

- **known cost tiers**: keys of the `presets` map
- **known capability slots**: union of all value-map keys across presets

when the user tries to introduce a novel tier or slot, the tool warns and
asks for confirmation. this catches typos without imposing rigid schema.

## gateway construction

```rust
let mut builder = Ratatoskr::builder();

if let Ok(key) = env::var("OPENROUTER_API_KEY") {
    builder = builder.openrouter(key);
}
if let Ok(key) = env::var("HF_API_KEY") {
    builder = builder.huggingface(key);
}

let gateway = builder.build()?;
```

minimal — just enough providers to fetch metadata. no retry, no caching, no
routing config. the tool is interactive and single-shot.

## file structure

```
src/bin/rat_registry.rs    # CLI entry point, clap command definitions
src/bin/registry_tool/     # if it grows, split into modules
  mod.rs
  io.rs                    # load/save registry
  commands.rs              # add/list/remove logic
```

start as a single file; split only if it gets unwieldy.

## default registry path

`../registry/registry.json` relative to the repo root, overridable via
`--path <file>`. detected by walking up from the binary or from `$PWD`.

pragmatic default: `--path` defaults to
`env::var("REGISTRY_PATH").unwrap_or("../registry/registry.json")`.

## what this tool does NOT do

- no interactive json editing (edit the file by hand)
- no git operations (commit/push the registry yourself)
- no daemon dependency (embedded gateway only)
- no default-feature inclusion (never built unless explicitly requested)

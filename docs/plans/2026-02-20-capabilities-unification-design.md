# capabilities unification design

**issue**: #20
**date**: 2026-02-20
**status**: approved

## problem

two parallel representations of the same concept exist:

- `Capabilities` struct (10 bool fields) in `src/types/capabilities.rs`
- `ModelCapability` enum (10 variants) in `src/types/model.rs`

every new capability requires updating both types plus bridge logic, with no compiler enforcement of consistency. bridge impls (`From<&Capabilities> for HashSet<ModelCapability>` and vice versa) are pure boilerplate.

## solution: `Capabilities(HashSet<ModelCapability>)` newtype

replace the struct with a newtype wrapping `HashSet<ModelCapability>`. single source of truth: the enum is the canonical list of capabilities; `Capabilities` is just a typed set of them.

### rust type

```rust
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities(HashSet<ModelCapability>);

impl Capabilities {
    pub fn has(&self, cap: ModelCapability) -> bool
    pub fn insert(&mut self, cap: ModelCapability)
    pub fn merge(&self, other: &Self) -> Self   // set union
    pub fn iter(&self) -> impl Iterator<Item = &ModelCapability>

    // factory methods
    pub fn chat_only() -> Self
    pub fn full() -> Self
    pub fn local_only() -> Self
    pub fn huggingface_only() -> Self
}

impl From<HashSet<ModelCapability>> for Capabilities
impl From<Capabilities> for HashSet<ModelCapability>
impl FromIterator<ModelCapability> for Capabilities
```

`Default` = empty set. serde serialises as a JSON array of strings (e.g. `["Chat", "ChatStreaming"]`), which is more expressive than the old bool map and requires no custom serde impl.

### callsite changes

| before | after |
|---|---|
| `caps.chat` | `caps.has(ModelCapability::Chat)` |
| `caps.embed` | `caps.has(ModelCapability::Embed)` |
| `Capabilities { chat: true, .. }` | `Capabilities::from_iter([Chat, ChatStreaming])` |
| `caps.merge(&other)` | `caps.merge(&other)` (unchanged) |

### proto wire format

`CapabilitiesResponse` changes from 10 bool fields to a `repeated` enum. a `ModelCapability` proto enum is added alongside (distinct from the rust enum; mapped in `server::convert`).

```proto
enum ProtoModelCapability {
    MODEL_CAPABILITY_UNSPECIFIED = 0;
    MODEL_CAPABILITY_CHAT = 1;
    MODEL_CAPABILITY_CHAT_STREAMING = 2;
    MODEL_CAPABILITY_GENERATE = 3;
    MODEL_CAPABILITY_TOOL_USE = 4;
    MODEL_CAPABILITY_EMBED = 5;
    MODEL_CAPABILITY_NLI = 6;
    MODEL_CAPABILITY_CLASSIFY = 7;
    MODEL_CAPABILITY_STANCE = 8;
    MODEL_CAPABILITY_TOKEN_COUNTING = 9;
    MODEL_CAPABILITY_LOCAL_INFERENCE = 10;
}

message CapabilitiesResponse {
    repeated ProtoModelCapability capabilities = 1;
}
```

note: proto enum values use the `SCREAMING_SNAKE_CASE` convention with the type prefix, as required by proto3. the rust-side `ModelCapability` enum is unaffected.

## affected files

| file | change |
|---|---|
| `src/types/capabilities.rs` | full rewrite to newtype |
| `src/types/model.rs` | no change to enum; remove bridge impls if moved |
| `proto/ratatoskr.proto` | replace bool fields + add `ProtoModelCapability` enum |
| `src/server/service.rs` | `get_capabilities` simplifies to iterator map |
| `src/client/service_client.rs` | `capabilities()` simplifies to iterator collect |
| `src/gateway/embedded.rs` | `capabilities()` builds set via `from_iter` |
| `src/server/convert.rs` | update `ModelCapability` ↔ proto conversions |
| `tests/capabilities_test.rs` | update assertions to `.has()`, test new helpers |
| `tests/traits_test.rs` | update `MockGateway::capabilities()` |
| `tests/gateway_test.rs` | update any capability assertions |

## non-goals

- adding new capabilities (out of scope)
- changing the `ModelCapability` enum variants
- changing `ModelGateway::capabilities()` signature (return type stays `Capabilities`)

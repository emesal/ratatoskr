# Phase 1 Diagrams

> Mermaid diagrams for the Phase 1 implementation plan.

## System Context

Where ratatoskr fits in the ecosystem:

```mermaid
C4Context
    title System Context - Phase 1

    Person(user, "User", "Developer using chibi")

    System(chibi, "chibi", "CLI chat application")
    System(ratatoskr, "ratatoskr", "Model gateway crate")
    System_Ext(openrouter, "OpenRouter API", "LLM provider")

    Rel(user, chibi, "Uses")
    Rel(chibi, ratatoskr, "Embeds")
    Rel(ratatoskr, openrouter, "HTTPS/SSE")
```

## Crate Structure

```mermaid
graph TB
    subgraph ratatoskr["ratatoskr crate"]
        lib[lib.rs<br/>Public API]

        subgraph types["types/"]
            message[message.rs<br/>Message, Role]
            tool[tool.rs<br/>ToolDefinition, ToolCall]
            options[options.rs<br/>ChatOptions]
            response[response.rs<br/>ChatResponse, ChatEvent]
            model[model.rs<br/>ModelInfo, Capabilities]
            future[future.rs<br/>Embedding, NliResult<br/>stubbed]
        end

        traits[traits.rs<br/>ModelGateway]
        error[error.rs<br/>RatatoskrError]

        subgraph embedded["embedded/"]
            emb_mod[mod.rs<br/>EmbeddedGateway]
            builder[builder.rs<br/>Builder pattern]
        end

        subgraph providers["providers/"]
            prov_mod[mod.rs<br/>Provider trait]
            openrouter[openrouter.rs<br/>OpenRouter impl]
        end
    end

    lib --> traits
    lib --> types
    lib --> error
    lib --> embedded

    traits --> types
    embedded --> providers
    embedded --> traits
    providers --> types

    style ratatoskr fill:#1a1a2e,stroke:#16213e
    style types fill:#0f3460,stroke:#16213e
    style embedded fill:#0f3460,stroke:#16213e
    style providers fill:#0f3460,stroke:#16213e
```

## Type Relationships

```mermaid
classDiagram
    class ModelGateway {
        <<trait>>
        +chat_stream(messages, tools, options) Stream~ChatEvent~
        +chat(messages, tools, options) ChatResponse
        +list_models() Vec~ModelInfo~
        +capabilities() Capabilities
        +embed(text) Embedding
        +infer_nli(premise, hypothesis) NliResult
    }

    class EmbeddedGateway {
        -providers: Vec~Provider~
        +new(providers) Self
    }

    class OpenRouterProvider {
        -client: reqwest::Client
        -api_key: String
        -base_url: String
        +chat_stream() Stream
        +chat() ChatResponse
    }

    class Message {
        +role: Role
        +content: MessageContent
        +tool_calls: Option~Vec~ToolCall~~
        +name: Option~String~
        +system(content) Message
        +user(content) Message
        +assistant(content) Message
        +tool_result(id, content) Message
    }

    class Role {
        <<enum>>
        System
        User
        Assistant
        Tool
    }

    class ChatOptions {
        +model: String
        +temperature: Option~f32~
        +max_tokens: Option~usize~
        +tool_choice: Option~ToolChoice~
        +provider_options: Option~Value~
    }

    class ChatEvent {
        <<enum>>
        Content(String)
        Reasoning(String)
        ToolCallStart
        ToolCallDelta
        Usage
        Done
    }

    class ChatResponse {
        +content: String
        +tool_calls: Vec~ToolCall~
        +usage: Option~Usage~
    }

    class ToolDefinition {
        +name: String
        +description: String
        +parameters: Value
    }

    class ToolCall {
        +id: String
        +name: String
        +arguments: String
    }

    ModelGateway <|.. EmbeddedGateway
    EmbeddedGateway o-- OpenRouterProvider
    Message --> Role
    Message --> ToolCall
    ChatResponse --> ToolCall
    ChatResponse --> Usage
```

## Request Flow

### Non-Streaming Chat

```mermaid
sequenceDiagram
    participant App as Application
    participant GW as EmbeddedGateway
    participant OR as OpenRouterProvider
    participant API as OpenRouter API

    App->>GW: chat(messages, tools, options)
    GW->>OR: chat(messages, tools, options)

    OR->>OR: Build request JSON
    OR->>API: POST /chat/completions
    API-->>OR: JSON response

    OR->>OR: Parse response
    OR-->>GW: ChatResponse
    GW-->>App: ChatResponse
```

### Streaming Chat

```mermaid
sequenceDiagram
    participant App as Application
    participant GW as EmbeddedGateway
    participant OR as OpenRouterProvider
    participant API as OpenRouter API

    App->>GW: chat_stream(messages, tools, options)
    GW->>OR: chat_stream(messages, tools, options)

    OR->>OR: Build request JSON (stream: true)
    OR->>API: POST /chat/completions

    loop SSE Stream
        API-->>OR: data: {"choices":[{"delta":...}]}
        OR->>OR: Parse SSE event
        OR-->>GW: ChatEvent::Content / ToolCallDelta / etc
        GW-->>App: ChatEvent
    end

    API-->>OR: data: [DONE]
    OR-->>GW: ChatEvent::Done
    GW-->>App: ChatEvent::Done
```

## SSE Parsing State Machine

```mermaid
stateDiagram-v2
    [*] --> ReadingLine

    ReadingLine --> ProcessingEvent: Line complete
    ReadingLine --> ReadingLine: More bytes

    ProcessingEvent --> ReadingLine: Empty line
    ProcessingEvent --> ParseData: data prefix
    ProcessingEvent --> ReadingLine: Other prefix

    ParseData --> EmitContent: Has delta.content
    ParseData --> EmitToolStart: Has tool_calls.id
    ParseData --> EmitToolDelta: Has tool_calls.args
    ParseData --> EmitUsage: Has usage
    ParseData --> StreamDone: DONE sentinel

    EmitContent --> ReadingLine
    EmitToolStart --> ReadingLine
    EmitToolDelta --> ReadingLine
    EmitUsage --> ReadingLine

    StreamDone --> [*]
```

## Tool Call Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant GW as Gateway
    participant API as OpenRouter

    Note over App,API: Initial request with tools

    App->>GW: chat_stream(messages, tools, options)
    GW->>API: POST (with tool definitions)

    API-->>GW: ToolCallStart{id, name}
    GW-->>App: ChatEvent::ToolCallStart

    loop Argument chunks
        API-->>GW: ToolCallDelta{arguments}
        GW-->>App: ChatEvent::ToolCallDelta
    end

    API-->>GW: [DONE]
    GW-->>App: ChatEvent::Done

    Note over App: App executes tool, gets result

    App->>GW: chat_stream([...prev, tool_result], tools, options)
    GW->>API: POST (with tool result message)

    API-->>GW: Content chunks
    GW-->>App: ChatEvent::Content

    API-->>GW: [DONE]
    GW-->>App: ChatEvent::Done
```

## Builder Pattern

```mermaid
flowchart LR
    subgraph Construction
        A[Ratatoskr::embedded] --> B[with_openrouter]
        B --> C[base_url optional]
        C --> D[build]
    end

    D --> E[EmbeddedGateway]

    subgraph Usage
        E --> F[chat / chat_stream]
        E --> G[list_models]
        E --> H[capabilities]
    end
```

## Chibi Integration

### Before Extraction

```mermaid
flowchart TB
    subgraph chibi["chibi crate"]
        main[main.rs]
        api_mod[api/mod.rs<br/>~1000 lines]
        request[api/request.rs<br/>build_request_body]
        compact[api/compact.rs<br/>6 HTTP call sites]
        llm[llm.rs<br/>send_streaming_request]
        config[config.rs<br/>ToolChoice, ReasoningConfig]

        main --> api_mod
        api_mod --> request
        api_mod --> llm
        api_mod --> compact
        compact --> llm
        request --> config
    end

    llm --> OR[OpenRouter API]

    style llm fill:#e74c3c,stroke:#c0392b
    style request fill:#e74c3c,stroke:#c0392b
```

### After Extraction

```mermaid
flowchart TB
    subgraph chibi["chibi crate"]
        main[main.rs]
        api_mod[api/mod.rs<br/>~900 lines]
        adapter[api/adapter.rs<br/>type conversion]
        compact[api/compact.rs<br/>uses gateway.chat]
        config[config.rs<br/>chibi-specific only]

        main --> api_mod
        api_mod --> adapter
        api_mod --> compact
        compact --> adapter
    end

    subgraph ratatoskr["ratatoskr crate"]
        gateway[EmbeddedGateway]
        types[Message, ChatOptions, etc.]
        provider[OpenRouterProvider]

        gateway --> provider
    end

    adapter --> gateway
    provider --> OR[OpenRouter API]

    style adapter fill:#27ae60,stroke:#1e8449
    style gateway fill:#3498db,stroke:#2980b9
```

## Error Handling

```mermaid
flowchart TD
    Request[Make Request] --> HTTPError{HTTP Error?}

    HTTPError -->|Yes| MapHTTP[RatatoskrError::Http]
    HTTPError -->|No| StatusCheck{Status OK?}

    StatusCheck -->|401| AuthError[RatatoskrError::AuthenticationFailed]
    StatusCheck -->|429| RateLimit[RatatoskrError::RateLimited]
    StatusCheck -->|404 + model| ModelNotFound[RatatoskrError::ModelNotFound]
    StatusCheck -->|4xx/5xx| APIError[RatatoskrError::Api]
    StatusCheck -->|200| ParseResponse[Parse Response]

    ParseResponse --> JSONError{JSON Error?}
    JSONError -->|Yes| MapJSON[RatatoskrError::Json]
    JSONError -->|No| Success[Return Result]

    subgraph "Chibi Boundary"
        MapHTTP --> IOError[io::Error::other]
        AuthError --> IOError
        RateLimit --> IOError
        ModelNotFound --> IOError
        APIError --> IOError
        MapJSON --> IOError
    end
```

## Migration Steps

```mermaid
flowchart LR
    subgraph Phase1["Step 1: Build Ratatoskr"]
        A1[Create crate] --> A2[Implement types]
        A2 --> A3[OpenRouter provider]
        A3 --> A4[Tests]
    end

    subgraph Phase2["Step 2: Integrate"]
        B1[Add dependency] --> B2[Create adapter]
        B2 --> B3[Refactor api/mod.rs]
        B3 --> B4[Refactor compact.rs]
        B4 --> B5[Delete llm.rs]
    end

    subgraph Phase3["Step 3: Cleanup"]
        C1[Remove dead code] --> C2[Update tests]
        C2 --> C3[Update docs]
    end

    Phase1 --> Phase2 --> Phase3
```

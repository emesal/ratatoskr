# Streaming Responses

Streaming allows you to process LLM output as it's generated, providing better UX for long responses.

## Basic Streaming

Use `chat_stream` instead of `chat`:

```rust
use futures_util::StreamExt;
use ratatoskr::{ChatEvent, ChatOptions, Message, ModelGateway, Ratatoskr};

let gateway = Ratatoskr::builder()
    .openrouter(api_key)
    .build()?;

let messages = vec![Message::user("Write a short poem about Rust.")];
let options = ChatOptions::default().model("anthropic/claude-sonnet-4");

let mut stream = gateway.chat_stream(&messages, None, &options).await?;

while let Some(event) = stream.next().await {
    match event? {
        ChatEvent::Content(text) => {
            print!("{}", text);  // Print as it arrives
            std::io::Write::flush(&mut std::io::stdout())?;
        }
        ChatEvent::Done => break,
        _ => {}
    }
}
println!();  // Final newline
```

## Chat Events

The stream yields `ChatEvent` variants:

| Event | Description |
|-------|-------------|
| `Content(String)` | A chunk of response text |
| `Reasoning(String)` | A chunk of thinking/reasoning (extended thinking models) |
| `ToolCallStart { index, id, name }` | Model is starting a tool call |
| `ToolCallDelta { index, arguments }` | Partial JSON arguments for a tool call |
| `Usage(Usage)` | Token usage statistics |
| `Done` | Stream complete |

## Handling All Events

```rust
use ratatoskr::ChatEvent;

let mut content = String::new();
let mut reasoning = String::new();
let mut tool_calls = Vec::new();

while let Some(event) = stream.next().await {
    match event? {
        ChatEvent::Content(text) => {
            print!("{}", text);
            content.push_str(&text);
        }
        ChatEvent::Reasoning(thought) => {
            // Extended thinking output
            eprintln!("[thinking] {}", thought);
            reasoning.push_str(&thought);
        }
        ChatEvent::ToolCallStart { index, id, name } => {
            println!("\n[calling tool: {}]", name);
            tool_calls.push((id, name, String::new()));
        }
        ChatEvent::ToolCallDelta { index, arguments } => {
            // Accumulate JSON arguments
            if let Some((_, _, args)) = tool_calls.get_mut(index) {
                args.push_str(&arguments);
            }
        }
        ChatEvent::Usage(usage) => {
            eprintln!("\n[tokens: {}]", usage.total_tokens);
        }
        ChatEvent::Done => break,
    }
}
```

## Streaming with Tools

When using tools, the stream will emit tool call events:

```rust
use ratatoskr::{ChatEvent, ToolDefinition, ToolChoice};
use serde_json::json;

let tool = ToolDefinition::new(
    "calculate",
    "Perform a calculation",
    json!({
        "type": "object",
        "properties": {
            "expression": { "type": "string" }
        },
        "required": ["expression"]
    }),
);

let options = ChatOptions::default()
    .model("anthropic/claude-sonnet-4")
    .tool_choice(ToolChoice::Auto);

let mut stream = gateway.chat_stream(&messages, Some(&[tool]), &options).await?;

let mut current_tool_args = String::new();

while let Some(event) = stream.next().await {
    match event? {
        ChatEvent::ToolCallStart { name, id, .. } => {
            println!("Tool call started: {} ({})", name, id);
            current_tool_args.clear();
        }
        ChatEvent::ToolCallDelta { arguments, .. } => {
            current_tool_args.push_str(&arguments);
        }
        ChatEvent::Done => {
            if !current_tool_args.is_empty() {
                println!("Tool arguments: {}", current_tool_args);
            }
            break;
        }
        _ => {}
    }
}
```

## Collecting Stream to Response

If you want streaming output but also need the final `ChatResponse`:

```rust
use ratatoskr::{ChatEvent, ChatResponse, FinishReason, Usage};

let mut content = String::new();
let mut usage = None;

while let Some(event) = stream.next().await {
    match event? {
        ChatEvent::Content(text) => {
            print!("{}", text);
            content.push_str(&text);
        }
        ChatEvent::Usage(u) => usage = Some(u),
        ChatEvent::Done => break,
        _ => {}
    }
}

// Now you have both streamed output and the full content
println!("\n\nFull response length: {} chars", content.len());
```

## Error Handling in Streams

Errors can occur mid-stream:

```rust
while let Some(event) = stream.next().await {
    match event {
        Ok(ChatEvent::Content(text)) => print!("{}", text),
        Ok(ChatEvent::Done) => break,
        Ok(_) => {}
        Err(e) => {
            eprintln!("\nStream error: {}", e);
            break;
        }
    }
}
```

## Performance Considerations

- Streaming adds minimal overhead but provides better perceived latency
- First token arrives much faster than waiting for full response
- Use streaming for user-facing output, non-streaming for background processing
- Token usage is reported at the end of the stream

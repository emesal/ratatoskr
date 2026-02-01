# Tool Calling

Tool calling (also known as function calling) allows LLMs to request execution of functions you define. The model decides when to call tools based on the conversation context.

## Defining Tools

Tools are defined with a name, description, and JSON Schema for parameters:

```rust
use ratatoskr::ToolDefinition;
use serde_json::json;

let weather_tool = ToolDefinition::new(
    "get_weather",
    "Get the current weather for a location",
    json!({
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g. 'London' or 'New York, NY'"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units"
            }
        },
        "required": ["location"]
    }),
);
```

## Making Requests with Tools

Pass tools to `chat` or `chat_stream`:

```rust
use ratatoskr::{ChatOptions, Message, ModelGateway, ToolChoice};

let messages = vec![
    Message::user("What's the weather like in Tokyo?"),
];

let options = ChatOptions::default()
    .model("anthropic/claude-sonnet-4")
    .tool_choice(ToolChoice::Auto);

let response = gateway.chat(&messages, Some(&[weather_tool]), &options).await?;
```

## Handling Tool Calls

Check the response for tool calls:

```rust
use serde::Deserialize;

#[derive(Deserialize)]
struct WeatherArgs {
    location: String,
    units: Option<String>,
}

if !response.tool_calls.is_empty() {
    for tool_call in &response.tool_calls {
        match tool_call.name.as_str() {
            "get_weather" => {
                // Parse the JSON arguments
                let args: WeatherArgs = tool_call.parse_arguments()?;

                // Execute your function
                let weather = fetch_weather(&args.location, args.units.as_deref())?;

                // Continue the conversation with the result
                // (see "Tool Call Loop" below)
            }
            _ => println!("Unknown tool: {}", tool_call.name),
        }
    }
}
```

## Tool Choice

Control how the model uses tools:

```rust
use ratatoskr::ToolChoice;

// Let the model decide (default)
let options = ChatOptions::default()
    .tool_choice(ToolChoice::Auto);

// Force the model to use a tool
let options = ChatOptions::default()
    .tool_choice(ToolChoice::Required);

// Prevent tool use
let options = ChatOptions::default()
    .tool_choice(ToolChoice::None);

// Force a specific tool
let options = ChatOptions::default()
    .tool_choice(ToolChoice::Function { name: "get_weather".to_string() });
```

## Tool Call Loop

A complete tool-using conversation requires multiple turns:

```rust
use ratatoskr::{ChatOptions, Message, ModelGateway, ToolChoice, FinishReason};

async fn chat_with_tools(
    gateway: &impl ModelGateway,
    initial_message: &str,
    tools: &[ToolDefinition],
) -> ratatoskr::Result<String> {
    let mut messages = vec![
        Message::system("You are a helpful assistant with access to tools."),
        Message::user(initial_message),
    ];

    let options = ChatOptions::default()
        .model("anthropic/claude-sonnet-4")
        .tool_choice(ToolChoice::Auto);

    loop {
        let response = gateway.chat(&messages, Some(tools), &options).await?;

        // If no tool calls, we're done
        if response.tool_calls.is_empty() {
            return Ok(response.content);
        }

        // Add assistant's response with tool calls
        messages.push(Message::assistant_with_tool_calls(
            if response.content.is_empty() { None } else { Some(&response.content) },
            response.tool_calls.clone(),
        ));

        // Execute each tool and add results
        for tool_call in &response.tool_calls {
            let result = execute_tool(&tool_call.name, &tool_call.arguments)?;
            messages.push(Message::tool_result(&tool_call.id, &result));
        }

        // Continue the loop - model will process tool results
    }
}

fn execute_tool(name: &str, arguments: &str) -> Result<String, Box<dyn std::error::Error>> {
    match name {
        "get_weather" => {
            let args: serde_json::Value = serde_json::from_str(arguments)?;
            let location = args["location"].as_str().unwrap_or("unknown");
            // Your actual implementation here
            Ok(format!("Weather in {}: 22°C, sunny", location))
        }
        _ => Ok(format!("Unknown tool: {}", name)),
    }
}
```

## Multiple Tools

Define multiple tools for complex tasks:

```rust
let tools = vec![
    ToolDefinition::new(
        "search_web",
        "Search the web for information",
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query" }
            },
            "required": ["query"]
        }),
    ),
    ToolDefinition::new(
        "read_file",
        "Read contents of a file",
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path" }
            },
            "required": ["path"]
        }),
    ),
    ToolDefinition::new(
        "write_file",
        "Write content to a file",
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path" },
                "content": { "type": "string", "description": "Content to write" }
            },
            "required": ["path", "content"]
        }),
    ),
];
```

## Best Practices

1. **Clear descriptions** — Help the model understand when to use each tool
2. **Precise schemas** — Use `enum` for fixed choices, `description` for context
3. **Required fields** — Mark essential parameters as required
4. **Error handling** — Return meaningful error messages as tool results
5. **Validation** — Validate arguments before execution
6. **Idempotency** — Design tools to be safely retryable when possible

## Streaming with Tools

See [Streaming](./streaming.md) for handling tool calls in streams.

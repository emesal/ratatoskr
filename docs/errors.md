# Error Handling

Ratatoskr provides a unified error type that normalizes errors across all providers.

## Error Types

```rust
use ratatoskr::RatatoskrError;
```

### Network & API Errors

| Error | Description |
|-------|-------------|
| `Http(String)` | Network-level failure (connection, DNS, TLS) |
| `Api { status, message }` | Provider returned an error response |
| `RateLimited { retry_after }` | Too many requests; may include retry timing |
| `AuthenticationFailed` | Invalid or expired API key |
| `ModelNotFound(String)` | Requested model doesn't exist |

### Streaming Errors

| Error | Description |
|-------|-------------|
| `Stream(String)` | Error during streaming response |

### Data Errors

| Error | Description |
|-------|-------------|
| `Json(serde_json::Error)` | JSON parsing failed |
| `InvalidInput(String)` | Bad request parameters |

### Configuration Errors

| Error | Description |
|-------|-------------|
| `NoProvider` | No provider configured for the requested model |
| `NotImplemented(&'static str)` | Feature not yet available |
| `Unsupported` | Operation not supported by this provider |

### Soft Errors

| Error | Description |
|-------|-------------|
| `EmptyResponse` | Model returned no content |
| `ContentFiltered { reason }` | Response blocked by content policy |
| `ContextLengthExceeded { limit }` | Input too long for model |

### Wrapped Errors

| Error | Description |
|-------|-------------|
| `Llm(String)` | Error from underlying llm crate |

## Basic Error Handling

```rust
use ratatoskr::RatatoskrError;

match gateway.chat(&messages, None, &options).await {
    Ok(response) => {
        println!("{}", response.content);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## Handling Specific Errors

```rust
match gateway.chat(&messages, None, &options).await {
    Ok(response) => println!("{}", response.content),

    // Retryable errors
    Err(RatatoskrError::RateLimited { retry_after }) => {
        if let Some(duration) = retry_after {
            println!("Rate limited. Retry after {:?}", duration);
            tokio::time::sleep(duration).await;
            // Retry the request
        }
    }
    Err(RatatoskrError::Http(msg)) => {
        eprintln!("Network error: {}. Retrying...", msg);
        // Implement retry logic
    }

    // Configuration errors (fix and retry)
    Err(RatatoskrError::AuthenticationFailed) => {
        eprintln!("Invalid API key. Check your configuration.");
    }
    Err(RatatoskrError::ModelNotFound(model)) => {
        eprintln!("Model '{}' not found. Check the model name.", model);
    }
    Err(RatatoskrError::NoProvider) => {
        eprintln!("No provider configured for this model.");
    }

    // Input errors (fix the request)
    Err(RatatoskrError::InvalidInput(msg)) => {
        eprintln!("Bad request: {}", msg);
    }
    Err(RatatoskrError::ContextLengthExceeded { limit }) => {
        eprintln!("Input too long. Max {} tokens.", limit);
        // Truncate or summarize input
    }

    // Content policy
    Err(RatatoskrError::ContentFiltered { reason }) => {
        eprintln!("Content blocked: {}", reason);
    }

    // Other errors
    Err(e) => {
        eprintln!("Unexpected error: {}", e);
    }
}
```

## Retry Logic

Example exponential backoff for transient errors:

```rust
use std::time::Duration;
use ratatoskr::RatatoskrError;

async fn chat_with_retry(
    gateway: &impl ModelGateway,
    messages: &[Message],
    options: &ChatOptions,
    max_retries: u32,
) -> ratatoskr::Result<ChatResponse> {
    let mut attempt = 0;

    loop {
        match gateway.chat(messages, None, options).await {
            Ok(response) => return Ok(response),

            Err(RatatoskrError::RateLimited { retry_after }) => {
                if attempt >= max_retries {
                    return Err(RatatoskrError::RateLimited { retry_after });
                }
                let delay = retry_after.unwrap_or(Duration::from_secs(1 << attempt));
                tokio::time::sleep(delay).await;
                attempt += 1;
            }

            Err(RatatoskrError::Http(_)) if attempt < max_retries => {
                let delay = Duration::from_millis(100 * (1 << attempt));
                tokio::time::sleep(delay).await;
                attempt += 1;
            }

            Err(e) => return Err(e),
        }
    }
}
```

## Error Context

Errors implement `std::error::Error` and `Display`:

```rust
if let Err(e) = result {
    // Display message
    println!("Error: {}", e);

    // Source chain
    if let Some(source) = e.source() {
        println!("Caused by: {}", source);
    }
}
```

## The Result Type

Ratatoskr exports a convenient `Result` type alias:

```rust
use ratatoskr::Result;

async fn my_function() -> Result<String> {
    let response = gateway.chat(&messages, None, &options).await?;
    Ok(response.content)
}
```

This is equivalent to `std::result::Result<T, RatatoskrError>`.

## Converting Errors

`RatatoskrError` implements `From` for common error types:

```rust
// serde_json errors convert automatically
let value: serde_json::Value = serde_json::from_str(bad_json)?;
// Returns RatatoskrError::Json(...)
```

## Logging Errors

For production, log errors with context:

```rust
use tracing::{error, warn};

match gateway.chat(&messages, None, &options).await {
    Ok(response) => response,
    Err(RatatoskrError::RateLimited { .. }) => {
        warn!(model = %options.model, "Rate limited");
        // Handle...
    }
    Err(e) => {
        error!(error = %e, model = %options.model, "Chat request failed");
        return Err(e);
    }
}
```

## Best Practices

1. **Handle retryable errors** — RateLimited, Http errors are often transient
2. **Log with context** — Include model, request ID for debugging
3. **Graceful degradation** — Fall back to alternative models if available
4. **User-friendly messages** — Translate technical errors for end users
5. **Don't ignore errors** — Even "soft" errors like EmptyResponse need handling

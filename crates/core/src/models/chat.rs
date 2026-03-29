use kbolt_types::KboltError;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::config::TextInferenceOutputMode;
use crate::models::completion::CompletionClient;
use crate::models::http::{HttpJsonClient, HttpOperation};
use crate::models::text::strip_json_fences;
use crate::Result;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ChatCompletionRequestOptions {
    pub output_mode: TextInferenceOutputMode,
    pub max_tokens: Option<usize>,
    pub seed: Option<u32>,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

impl ChatCompletionRequestOptions {
    pub(super) fn json_object() -> Self {
        Self {
            output_mode: TextInferenceOutputMode::JsonObject,
            max_tokens: None,
            seed: None,
            temperature: Some(0.0),
            top_k: None,
            top_p: None,
            min_p: None,
            repeat_last_n: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }

    #[cfg(test)]
    pub(super) fn text() -> Self {
        Self {
            output_mode: TextInferenceOutputMode::Text,
            max_tokens: None,
            seed: None,
            temperature: Some(0.0),
            top_k: None,
            top_p: None,
            min_p: None,
            repeat_last_n: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct HttpChatClient {
    http: HttpJsonClient,
    model: String,
    options: ChatCompletionRequestOptions,
}

impl HttpChatClient {
    pub(super) fn new(
        base_url: &str,
        api_key_env: Option<&str>,
        timeout_ms: u64,
        max_retries: u32,
        model: &str,
        options: ChatCompletionRequestOptions,
        provider_name: &'static str,
    ) -> Self {
        Self {
            http: HttpJsonClient::new(
                base_url,
                api_key_env,
                timeout_ms,
                max_retries,
                "inference",
                provider_name,
            ),
            model: model.to_string(),
            options,
        }
    }
}

impl CompletionClient for HttpChatClient {
    fn complete(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let payload = build_chat_payload(&self.model, system_prompt, user_prompt, &self.options);

        let response: ChatCompletionResponse =
            self.http
                .post_json("chat/completions", &payload, HttpOperation::ChatCompletion)?;
        let content = response.into_text()?;
        let normalized = match self.options.output_mode {
            TextInferenceOutputMode::JsonObject => content.trim(),
            TextInferenceOutputMode::Text => strip_json_fences(&content),
        };
        Ok(normalized.to_string())
    }
}

pub(super) fn build_chat_payload(
    model: &str,
    system_prompt: &str,
    user_prompt: &str,
    options: &ChatCompletionRequestOptions,
) -> Value {
    let mut payload = json!({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
    });
    if let Some(max_tokens) = options.max_tokens {
        payload["max_tokens"] = json!(max_tokens);
    }
    if let Some(seed) = options.seed {
        payload["seed"] = json!(seed);
    }
    if let Some(temperature) = options.temperature {
        payload["temperature"] = json!(temperature);
    }
    if let Some(top_k) = options.top_k {
        payload["top_k"] = json!(top_k);
    }
    if let Some(top_p) = options.top_p {
        payload["top_p"] = json!(top_p);
    }
    if let Some(min_p) = options.min_p {
        payload["min_p"] = json!(min_p);
    }
    if let Some(repeat_last_n) = options.repeat_last_n {
        payload["repeat_last_n"] = json!(repeat_last_n);
    }
    if let Some(repeat_penalty) = options.repeat_penalty {
        payload["repeat_penalty"] = json!(repeat_penalty);
    }
    if let Some(frequency_penalty) = options.frequency_penalty {
        payload["frequency_penalty"] = json!(frequency_penalty);
    }
    if let Some(presence_penalty) = options.presence_penalty {
        payload["presence_penalty"] = json!(presence_penalty);
    }
    if options.output_mode == TextInferenceOutputMode::JsonObject {
        payload["response_format"] = json!({ "type": "json_object" });
    }
    payload
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

impl ChatCompletionResponse {
    fn into_text(self) -> Result<String> {
        let Some(choice) = self.choices.into_iter().next() else {
            return Err(KboltError::Inference(
                "chat completion response is missing choices".to_string(),
            )
            .into());
        };

        extract_text(choice.message.content).ok_or_else(|| {
            KboltError::Inference(
                "chat completion response did not contain text content".to_string(),
            )
            .into()
        })
    }
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Value,
}

fn extract_text(value: Value) -> Option<String> {
    match value {
        Value::String(content) => Some(content),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                match part {
                    Value::String(segment) => text.push_str(&segment),
                    Value::Object(map) => {
                        if let Some(Value::String(segment)) = map.get("text") {
                            text.push_str(segment);
                        }
                    }
                    _ => {}
                }
            }

            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        Value::Object(map) => map
            .get("text")
            .and_then(|item| item.as_str())
            .map(ToString::to_string),
        _ => None,
    }
}

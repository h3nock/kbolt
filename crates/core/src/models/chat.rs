use kbolt_types::KboltError;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::config::TextInferenceOutputMode;
use crate::models::completion::CompletionClient;
use crate::models::http::{HttpJsonClient, HttpOperation};
use crate::models::text::strip_json_fences;
use crate::Result;

#[derive(Debug, Clone)]
pub(super) struct HttpChatClient {
    http: HttpJsonClient,
    model: String,
    output_mode: TextInferenceOutputMode,
}

impl HttpChatClient {
    pub(super) fn new(
        base_url: &str,
        api_key_env: Option<&str>,
        timeout_ms: u64,
        max_retries: u32,
        model: &str,
        output_mode: TextInferenceOutputMode,
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
            output_mode,
        }
    }
}

impl CompletionClient for HttpChatClient {
    fn complete(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let payload = build_chat_payload(
            &self.model,
            system_prompt,
            user_prompt,
            self.output_mode.clone(),
        );

        let response: ChatCompletionResponse =
            self.http
                .post_json("chat/completions", &payload, HttpOperation::ChatCompletion)?;
        let content = response.into_text()?;
        let normalized = match self.output_mode {
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
    output_mode: TextInferenceOutputMode,
) -> Value {
    let mut payload = json!({
        "model": model,
        "temperature": 0,
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
    if output_mode == TextInferenceOutputMode::JsonObject {
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
            KboltError::Inference("chat completion response did not contain text content".to_string()).into()
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

            if text.is_empty() { None } else { Some(text) }
        }
        Value::Object(map) => map
            .get("text")
            .and_then(|item| item.as_str())
            .map(ToString::to_string),
        _ => None,
    }
}

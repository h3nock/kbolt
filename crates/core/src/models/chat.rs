use kbolt_types::KboltError;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::models::completion::CompletionClient;
use crate::models::http::{HttpJsonClient, HttpOperation};
use crate::models::text::strip_json_fences;
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChatCompletionOutputMode {
    JsonObject,
    Text,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ChatCompletionRequestOptions {
    pub output_mode: ChatCompletionOutputMode,
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
    pub llama_cpp: Option<LlamaCppChatRequestOptions>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct LlamaCppChatRequestOptions {
    pub grammar: Option<String>,
    pub chat_template_kwargs: Option<Value>,
}

impl LlamaCppChatRequestOptions {
    pub(super) fn non_thinking() -> Self {
        Self {
            grammar: None,
            chat_template_kwargs: Some(json!({ "enable_thinking": false })),
        }
    }
}

impl ChatCompletionRequestOptions {
    pub(super) fn json_object() -> Self {
        Self {
            output_mode: ChatCompletionOutputMode::JsonObject,
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
            llama_cpp: None,
        }
    }

    #[cfg(test)]
    pub(super) fn text() -> Self {
        Self {
            output_mode: ChatCompletionOutputMode::Text,
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
            llama_cpp: None,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct HttpChatClient {
    http: HttpJsonClient,
    model: String,
    endpoint_suffix: &'static str,
    options: ChatCompletionRequestOptions,
}

impl HttpChatClient {
    pub(super) fn new(
        base_url: &str,
        api_key_env: Option<&str>,
        timeout_ms: u64,
        max_retries: u32,
        model: &str,
        endpoint_suffix: &'static str,
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
            endpoint_suffix,
            options,
        }
    }
}

impl CompletionClient for HttpChatClient {
    fn complete(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let payload = build_chat_payload(&self.model, system_prompt, user_prompt, &self.options);

        let response: ChatCompletionResponse = self.http.post_json(
            self.endpoint_suffix,
            &payload,
            HttpOperation::ChatCompletion,
        )?;
        let content = response.into_text()?;
        let normalized = match self.options.output_mode {
            ChatCompletionOutputMode::JsonObject => content.trim(),
            ChatCompletionOutputMode::Text => strip_json_fences(&content),
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
    if let Some(llama_cpp) = options.llama_cpp.as_ref() {
        if let Some(grammar) = llama_cpp.grammar.as_ref() {
            payload["grammar"] = json!(grammar);
        }
        if let Some(chat_template_kwargs) = llama_cpp.chat_template_kwargs.as_ref() {
            payload["chat_template_kwargs"] = chat_template_kwargs.clone();
        }
    }
    if options.output_mode == ChatCompletionOutputMode::JsonObject {
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

#[cfg(test)]
mod tests {
    use super::{
        build_chat_payload, ChatCompletionOutputMode, ChatCompletionRequestOptions,
        LlamaCppChatRequestOptions,
    };

    #[test]
    fn build_chat_payload_adds_llama_non_thinking_controls_when_requested() {
        let mut options = ChatCompletionRequestOptions::text();
        options.llama_cpp = Some(LlamaCppChatRequestOptions::non_thinking());

        let payload = build_chat_payload("model", "system", "user", &options);
        assert_eq!(payload["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[test]
    fn build_chat_payload_adds_llama_grammar_when_requested() {
        let mut options = ChatCompletionRequestOptions::text();
        options.llama_cpp = Some(LlamaCppChatRequestOptions {
            grammar: Some("root ::= \"ok\"".to_string()),
            chat_template_kwargs: None,
        });

        let payload = build_chat_payload("model", "system", "user", &options);
        assert_eq!(payload["grammar"], "root ::= \"ok\"");
    }

    #[test]
    fn build_chat_payload_omits_llama_controls_by_default() {
        let options = ChatCompletionRequestOptions::text();
        let payload = build_chat_payload("model", "system", "user", &options);
        assert!(payload.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn text_request_options_keep_text_output_mode() {
        let options = ChatCompletionRequestOptions::text();
        assert_eq!(options.output_mode, ChatCompletionOutputMode::Text);
    }
}

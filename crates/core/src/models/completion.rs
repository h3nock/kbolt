use crate::Result;

pub(super) trait CompletionClient: Send + Sync {
    fn complete(&self, system_prompt: &str, user_prompt: &str) -> Result<String>;
}

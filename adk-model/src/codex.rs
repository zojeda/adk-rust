//! Codex ChatGPT-subscription model integration.
//!
//! This module provides a dedicated ADK client for Codex's ChatGPT-backed
//! Responses API surface. It reuses the OpenAI Responses conversion layer but
//! applies Codex-specific authentication headers and base URL defaults.

use crate::openai::{
    OpenAIResponsesClient, OpenAIResponsesConfig, ReasoningEffort, ReasoningSummary,
};
use adk_core::{AdkError, Llm, LlmRequest, LlmResponseStream};
use async_trait::async_trait;

/// Default Codex Responses API base URL used for ChatGPT-backed access.
pub const CODEX_API_BASE: &str = "https://chatgpt.com/backend-api/codex";

/// Header that identifies the active ChatGPT workspace/account.
pub const CHATGPT_ACCOUNT_ID_HEADER: &str = "ChatGPT-Account-ID";

/// Configuration for a Codex ChatGPT-backed Responses client.
///
/// # Example
///
/// ```rust,ignore
/// use adk_model::codex::{CodexResponsesClient, CodexResponsesConfig};
///
/// let config = CodexResponsesConfig::new(
///     "chatgpt-access-token",
///     "workspace_123",
///     "gpt-5.2-codex",
/// );
/// let client = CodexResponsesClient::new(config)?;
/// ```
#[derive(Debug, Clone)]
pub struct CodexResponsesConfig {
    /// ChatGPT access token used as bearer auth.
    pub access_token: String,
    /// ChatGPT workspace/account id sent via `ChatGPT-Account-ID`.
    pub account_id: String,
    /// Model name exposed by Codex.
    pub model: String,
    /// Optional Codex base URL override for testing or self-hosted proxies.
    pub base_url: Option<String>,
    /// Optional reasoning effort for reasoning-capable models.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Optional reasoning summary mode for reasoning-capable models.
    pub reasoning_summary: Option<ReasoningSummary>,
}

impl CodexResponsesConfig {
    /// Create a new Codex config from a ChatGPT access token and account id.
    pub fn new(
        access_token: impl Into<String>,
        account_id: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            access_token: access_token.into(),
            account_id: account_id.into(),
            model: model.into(),
            base_url: None,
            reasoning_effort: None,
            reasoning_summary: None,
        }
    }

    /// Override the default Codex base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set reasoning effort for reasoning-capable models.
    #[must_use]
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Set reasoning summary mode for reasoning-capable models.
    #[must_use]
    pub fn with_reasoning_summary(mut self, summary: ReasoningSummary) -> Self {
        self.reasoning_summary = Some(summary);
        self
    }

    fn into_openai_config(self) -> OpenAIResponsesConfig {
        let mut config = OpenAIResponsesConfig::new(self.access_token, self.model)
            .with_provider_name("codex")
            .with_base_url(self.base_url.unwrap_or_else(|| CODEX_API_BASE.to_string()))
            .with_header(CHATGPT_ACCOUNT_ID_HEADER, self.account_id);

        if let Some(effort) = self.reasoning_effort {
            config = config.with_reasoning_effort(effort);
        }
        if let Some(summary) = self.reasoning_summary {
            config = config.with_reasoning_summary(summary);
        }

        config
    }
}

/// Dedicated `Llm` implementation for Codex ChatGPT-backed access.
///
/// This client uses the same ADK request/response mapping as
/// [`OpenAIResponsesClient`](crate::openai::OpenAIResponsesClient) but targets
/// the Codex backend and sends the required ChatGPT account header.
pub struct CodexResponsesClient {
    inner: OpenAIResponsesClient,
}

impl CodexResponsesClient {
    /// Create a new Codex Responses client.
    ///
    /// # Errors
    ///
    /// Returns [`AdkError`] if the token, account header, or client headers are invalid.
    pub fn new(config: CodexResponsesConfig) -> Result<Self, AdkError> {
        Ok(Self { inner: OpenAIResponsesClient::new(config.into_openai_config())? })
    }

    /// Apply retry configuration to the underlying Responses client.
    #[must_use]
    pub fn with_retry_config(mut self, retry_config: crate::RetryConfig) -> Self {
        self.inner = self.inner.with_retry_config(retry_config);
        self
    }
}

#[async_trait]
impl Llm for CodexResponsesClient {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn generate_content(
        &self,
        request: LlmRequest,
        stream: bool,
    ) -> Result<LlmResponseStream, AdkError> {
        self.inner.generate_content(request, stream).await
    }
}

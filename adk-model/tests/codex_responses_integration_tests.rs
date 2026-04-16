#[cfg(feature = "openai")]
mod codex_responses_client_integration {
    use adk_core::{Content, Llm, LlmRequest};
    use adk_model::codex::{CHATGPT_ACCOUNT_ID_HEADER, CodexResponsesClient, CodexResponsesConfig};
    use adk_model::retry::RetryConfig;
    use futures::StreamExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn make_client(base_url: &str) -> CodexResponsesClient {
        let config = CodexResponsesConfig::new("chatgpt-token", "workspace_123", "gpt-5.2-codex")
            .with_base_url(base_url);
        CodexResponsesClient::new(config)
            .expect("client creation should succeed")
            .with_retry_config(RetryConfig::disabled())
    }

    #[tokio::test]
    async fn request_parameters_chatgpt_account_header() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_1",
                "model": "gpt-5.2-codex",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "ok",
                                "annotations": []
                            }
                        ]
                    }
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = make_client(&server.uri());
        let request =
            LlmRequest::new("gpt-5.2-codex", vec![Content::new("user").with_text("hello")]);

        let mut stream =
            client.generate_content(request, false).await.expect("generate_content should succeed");

        while (stream.next().await).is_some() {}

        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);

        let account_header = received[0]
            .headers
            .get(CHATGPT_ACCOUNT_ID_HEADER)
            .expect("ChatGPT-Account-ID header should be present");
        assert_eq!(
            account_header.to_str().unwrap(),
            "workspace_123",
            "account header should match"
        );
    }
}

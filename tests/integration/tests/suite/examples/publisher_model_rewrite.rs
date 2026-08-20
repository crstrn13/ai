// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the publisher model rewrite example configuration.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_send, json_post, parse_body, parse_status, start_echo_backend, start_proxy};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn publisher_model_rewrite_config_parses() {
    let config = super::load_example_config(
        "inference/publisher-model-rewrite.yaml",
        29920,
        HashMap::from([
            ("10.0.1.1:8080", 29921_u16),
            ("10.0.2.1:8080", 29922_u16),
            ("10.0.3.1:8080", 29923_u16),
        ]),
    );

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
    assert_eq!(
        &*config.listeners[0].name, "inference-gateway",
        "listener name should be inference-gateway"
    );
}

#[test]
fn publisher_model_rewrite_strips_prefix() {
    let backend_guard = start_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "inference/publisher-model-rewrite.yaml",
        proxy_port,
        HashMap::from([
            ("10.0.1.1:8080", backend_port),
            ("10.0.2.1:8080", backend_port),
            ("10.0.3.1:8080", backend_port),
        ]),
    );

    let proxy = start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/chat/completions",
            r#"{"model":"publishers/llm-internal/models/facebook/opt-125m","messages":[{"role":"user","content":"Hello"}]}"#,
        ),
    );

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("backend should echo valid JSON");
    assert_eq!(
        parsed["model"], "facebook/opt-125m",
        "publisher prefix should be stripped from model field"
    );
    assert!(parsed["messages"].is_array(), "messages should be preserved");
}

#[test]
fn publisher_model_rewrite_passes_through_non_publisher_model() {
    let backend_guard = start_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "inference/publisher-model-rewrite.yaml",
        proxy_port,
        HashMap::from([
            ("10.0.1.1:8080", backend_port),
            ("10.0.2.1:8080", backend_port),
            ("10.0.3.1:8080", backend_port),
        ]),
    );

    let proxy = start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}"#,
        ),
    );

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("backend should echo valid JSON");
    assert_eq!(
        parsed["model"], "gpt-4o",
        "non-publisher model should pass through unchanged"
    );
}

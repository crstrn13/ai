// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `inflight-tracking` example configuration.
//!
//! The per-model counts live in an in-memory `InFlightRegistry` with no HTTP
//! surface, so these tests verify the filter is transparent: requests proxy
//! cleanly and responses pass through unchanged with `inflight_tracker` in the
//! pipeline, whether or not the body carries a trackable `model`.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_proxy,
};

const JSON_BODY: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","choices":[]}"#;

/// Build the example-config proxy fronting a fixed-response backend.
fn proxy_for(backend_port: u16, proxy_port: u16) -> praxis_core::config::Config {
    load_example_config(
        "inflight-tracking.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port)]),
    )
}

#[test]
fn example_config_inflight_tracking_passthrough() {
    let backend = Backend::fixed(JSON_BODY)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let proxy = start_proxy(&proxy_for(backend.port(), proxy_port));

    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/chat/completions",
            r#"{"model":"gpt-4","max_tokens":256,"messages":[{"role":"user","content":"hi"}]}"#,
        ),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "a tracked request should proxy with inflight_tracker in the pipeline"
    );
    assert_eq!(
        parse_body(&raw),
        JSON_BODY,
        "response body should pass through unchanged"
    );
}

#[test]
fn example_config_inflight_tracking_untracked_request_passthrough() {
    // A request carrying no top-level model is attributed to default_model; the
    // filter must still forward the request and leave the response untouched.
    let backend = Backend::fixed(JSON_BODY)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let proxy = start_proxy(&proxy_for(backend.port(), proxy_port));

    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        ),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "an untracked request should still proxy cleanly"
    );
    assert_eq!(
        parse_body(&raw),
        JSON_BODY,
        "response body should pass through unchanged"
    );
}

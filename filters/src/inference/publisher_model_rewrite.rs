// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Publisher model name rewrite filter for `MaaS` body-based routing.
//!
//! Strips the `publishers/{namespace}/models/` prefix from the
//! request body's `model` field before the request reaches the
//! upstream. The full model path is preserved in the routing
//! header (set by `model_to_header`) so `HTTPRoute` matching is
//! unaffected.

use async_trait::async_trait;
use bytes::Bytes;
use praxis_ai_apis::json_body::replace_json_body;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext,
    body::DEFAULT_JSON_BODY_MAX_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes,
    parse_filter_config,
};
use serde::Deserialize;
use tracing::debug;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Filter name used in config and tracing.
const FILTER_NAME: &str = "publisher_model_rewrite";
/// Prefix that identifies a publisher-scoped model path.
const PUBLISHERS_PREFIX: &str = "publishers/";
/// Separator between the publisher namespace and the model name.
const MODELS_SEPARATOR: &str = "/models/";

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the publisher model rewrite filter.
///
/// ```yaml
/// filter: publisher_model_rewrite
/// max_body_bytes: 10485760  # optional, defaults to 10 MiB
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublisherModelRewriteConfig {
    /// Maximum request body size to buffer before parsing.
    #[serde(default = "default_max_body_bytes")]
    max_body_bytes: usize,
}

/// Default for `max_body_bytes`.
fn default_max_body_bytes() -> usize {
    DEFAULT_JSON_BODY_MAX_BYTES
}

// -----------------------------------------------------------------------------
// PublisherModelRewriteFilter
// -----------------------------------------------------------------------------

/// Strips the `publishers/{namespace}/models/` prefix from the
/// request body's `model` field.
///
/// When a client sends `"model": "publishers/llm-internal/models/facebook/opt-125m"`,
/// this filter rewrites the body to `"model": "facebook/opt-125m"` so the
/// upstream (e.g. vLLM) receives the model name it expects.
///
/// Place this filter after `model_to_header` in the chain so the
/// full publisher path is available for routing before the body
/// is rewritten.
///
/// # YAML configuration
///
/// ```yaml
/// filter: publisher_model_rewrite
/// max_body_bytes: 10485760  # optional, defaults to 10 MiB
/// ```
///
/// # Example
///
/// ```rust
/// use praxis_ai_filters::PublisherModelRewriteFilter;
///
/// let yaml = serde_yaml::Value::Null;
/// let filter = PublisherModelRewriteFilter::from_config(&yaml).unwrap();
/// assert_eq!(filter.name(), "publisher_model_rewrite");
/// ```
pub struct PublisherModelRewriteFilter {
    /// Maximum request body size to buffer.
    max_body_bytes: usize,
}

impl PublisherModelRewriteFilter {
    /// Create from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or validation fails.
    ///
    /// [`FilterError`]: praxis_filter::FilterError
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: PublisherModelRewriteConfig = parse_filter_config(FILTER_NAME, config)?;
        validate_max_body_bytes(FILTER_NAME, cfg.max_body_bytes)?;
        Ok(Box::new(Self {
            max_body_bytes: cfg.max_body_bytes,
        }))
    }
}

/// Strip the `publishers/{namespace}/models/` prefix from a model name.
///
/// Returns `Some(suffix)` when the value matches the pattern, `None` otherwise.
fn strip_publisher_prefix(model: &str) -> Option<&str> {
    let without_prefix = model.strip_prefix(PUBLISHERS_PREFIX)?;
    let (_namespace, suffix) = without_prefix.split_once(MODELS_SEPARATOR.trim_start_matches('/'))?;
    if suffix.is_empty() {
        return None;
    }
    Some(suffix)
}

#[async_trait]
impl HttpFilter for PublisherModelRewriteFilter {
    fn name(&self) -> &'static str {
        "publisher_model_rewrite"
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    async fn on_request_body(
        &self,
        _ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let Some(raw) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };

        let mut value: serde_json::Value = match serde_json::from_slice(raw) {
            Ok(v) => v,
            Err(_) => return Ok(FilterAction::Continue),
        };

        let Some(model_str) = value.get("model").and_then(serde_json::Value::as_str) else {
            return Ok(FilterAction::Continue);
        };

        let rewritten = match strip_publisher_prefix(model_str) {
            Some(suffix) => suffix.to_owned(),
            None => return Ok(FilterAction::Continue),
        };

        debug!(
            original = model_str,
            rewritten = %rewritten,
            "stripping publisher prefix from model field"
        );

        if let Some(field) = value.get_mut("model") {
            *field = serde_json::Value::String(rewritten);
        }

        replace_json_body(body, &value, FILTER_NAME, "model")
            .map_err(|e| -> FilterError { format!("{FILTER_NAME}: {e}").into() })?;

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use super::*;

    // -- strip_publisher_prefix --------------------------------------------------

    #[test]
    fn strips_standard_publisher_path() {
        assert_eq!(
            strip_publisher_prefix("publishers/llm-internal/models/facebook/opt-125m"),
            Some("facebook/opt-125m")
        );
    }

    #[test]
    fn strips_single_segment_model() {
        assert_eq!(
            strip_publisher_prefix("publishers/corp/models/gpt-4o"),
            Some("gpt-4o")
        );
    }

    #[test]
    fn strips_deeply_nested_model() {
        assert_eq!(
            strip_publisher_prefix("publishers/ns/models/org/sub/model-v2"),
            Some("org/sub/model-v2")
        );
    }

    #[test]
    fn ignores_non_publisher_model() {
        assert_eq!(strip_publisher_prefix("gpt-4o"), None);
    }

    #[test]
    fn ignores_model_without_models_separator() {
        assert_eq!(strip_publisher_prefix("publishers/ns/gpt-4o"), None);
    }

    #[test]
    fn ignores_empty_suffix_after_models() {
        assert_eq!(strip_publisher_prefix("publishers/ns/models/"), None);
    }

    #[test]
    fn ignores_empty_string() {
        assert_eq!(strip_publisher_prefix(""), None);
    }

    #[test]
    fn ignores_publishers_prefix_only() {
        assert_eq!(strip_publisher_prefix("publishers/"), None);
    }

    // -- Config ------------------------------------------------------------------

    #[test]
    fn default_config_parses() {
        let filter = PublisherModelRewriteFilter::from_config(&serde_yaml::Value::Null).unwrap();
        assert_eq!(filter.name(), FILTER_NAME);
    }

    #[test]
    fn custom_max_body_bytes_parses() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 1048576").unwrap();
        let filter = PublisherModelRewriteFilter::from_config(&yaml).unwrap();
        assert_eq!(filter.name(), FILTER_NAME);
    }

    #[test]
    fn zero_max_body_bytes_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 0").unwrap();
        assert!(PublisherModelRewriteFilter::from_config(&yaml).is_err());
    }

    #[test]
    fn unknown_fields_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("unknown_field: true").unwrap();
        assert!(PublisherModelRewriteFilter::from_config(&yaml).is_err());
    }

    #[test]
    fn body_access_is_read_write() {
        let filter = PublisherModelRewriteFilter::from_config(&serde_yaml::Value::Null).unwrap();
        assert_eq!(filter.request_body_access(), BodyAccess::ReadWrite);
    }

    #[test]
    fn body_mode_is_stream_buffer() {
        let filter = PublisherModelRewriteFilter::from_config(&serde_yaml::Value::Null).unwrap();
        assert!(matches!(
            filter.request_body_mode(),
            BodyMode::StreamBuffer { max_bytes: Some(_) }
        ));
    }

    // -- on_request_body ---------------------------------------------------------

    #[tokio::test]
    async fn rewrites_publisher_model_in_body() {
        let filter = PublisherModelRewriteFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"publishers/llm-internal/models/facebook/opt-125m","messages":[]}"#;
        let mut body = Some(Bytes::from_static(json));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        let parsed: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
        assert_eq!(parsed["model"], "facebook/opt-125m");
        assert!(parsed["messages"].is_array(), "other fields should be preserved");
    }

    #[tokio::test]
    async fn passes_through_non_publisher_model() {
        let filter = PublisherModelRewriteFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"gpt-4o","messages":[]}"#;
        let mut body = Some(Bytes::from(json.to_vec()));
        let original = body.clone();

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(body, original, "body should be unchanged for non-publisher model");
    }

    #[tokio::test]
    async fn passes_through_missing_model_field() {
        let filter = PublisherModelRewriteFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"messages":[]}"#;
        let mut body = Some(Bytes::from(json.to_vec()));
        let original = body.clone();

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(body, original, "body should be unchanged when model field is absent");
    }

    #[tokio::test]
    async fn passes_through_non_string_model() {
        let filter = PublisherModelRewriteFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":42,"messages":[]}"#;
        let mut body = Some(Bytes::from(json.to_vec()));
        let original = body.clone();

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(body, original, "body should be unchanged for non-string model");
    }

    #[tokio::test]
    async fn passes_through_invalid_json() {
        let filter = PublisherModelRewriteFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let mut body = Some(Bytes::from_static(b"not json"));
        let original = body.clone();

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(body, original, "body should be unchanged for invalid JSON");
    }

    #[tokio::test]
    async fn passes_through_empty_body() {
        let filter = PublisherModelRewriteFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let mut body: Option<Bytes> = None;

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert!(body.is_none());
    }

    #[tokio::test]
    async fn continues_before_end_of_stream() {
        let filter = PublisherModelRewriteFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"publishers/ns/models/m","messages":[]}"#;
        let mut body = Some(Bytes::from(json.to_vec()));

        let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
    }

    #[tokio::test]
    async fn on_request_is_noop() {
        let filter = PublisherModelRewriteFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
    }
}

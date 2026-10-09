// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Request metadata extractor: in-flight tracking per model (issue #374).
//!
//! On request start this filter increments a per-model in-flight count and adds
//! the request's `max_tokens`/`max_output_tokens` to a per-model reservation.
//! Both figures live in [`InFlightRegistry`], a pipeline extension a scorer can
//! read to bias model selection toward the least-loaded backend.
//!
//! The decrement is driven by [`Drop`], not a response hook. On request start
//! the filter parks an `InFlightGuard` in `ctx.filter_state`; the per-request
//! context is dropped on *every* terminal path — success, client abort, upstream
//! error, timeout — so the guard's `Drop` always runs exactly once and the
//! counters can never leak upward. Response hooks are not reliable on aborted
//! streams, so the filter has none.

mod config;
mod registry;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
pub use registry::{InFlightRegistry, ModelMetrics};

use self::config::{InFlightConfig, validate_config};
use crate::json_scan::TopLevelKeyScanner;

/// Public filter type name used in YAML and registration.
pub const FILTER_NAME: &str = "inflight_tracker";

/// Tracks live request counts and reserved tokens per model.
pub struct InFlightTrackerFilter {
    /// Model attributed when the request body reveals none.
    default_model: String,
}

impl InFlightTrackerFilter {
    /// Build from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or validation fails.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: InFlightConfig = parse_filter_config(FILTER_NAME, config)?;
        validate_config(&cfg)?;
        Ok(Box::new(Self {
            default_model: cfg.default_model,
        }))
    }
}

/// RAII guard parked in `ctx.filter_state` for the request's lifetime.
///
/// Its [`Drop`] decrements the registry by exactly what request start reserved.
/// Because the per-request context owns this guard and is dropped on every
/// terminal path, the decrement runs once no matter how the request ends. The
/// guard carries its own [`InFlightRegistry`] handle — a cheap `Arc` clone —
/// because when it drops there is no `ctx` left to reach the shared table
/// through.
struct InFlightGuard {
    /// Shared handle to the counter table (clones share one inner map).
    registry: InFlightRegistry,

    /// Model this request was attributed to, so completion decrements the same
    /// key that start incremented.
    model: String,

    /// Exact token amount added on start, subtracted verbatim on completion.
    reserved: u64,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.registry.on_request_complete(&self.model, self.reserved);
    }
}

#[async_trait]
impl HttpFilter for InFlightTrackerFilter {
    fn name(&self) -> &'static str {
        "inflight_tracker"
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        // Stream, never StreamBuffer: a passive tracker must not cap request
        // size. A bounded StreamBuffer rejects bodies over its limit with 413,
        // and an unbounded one buffers every request in full. The top-level
        // `model` and token cap sit at the front of the body, so a per-chunk
        // scan reads them without buffering or rejecting anything.
        BodyMode::Stream
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        _end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        // Count exactly once: once the guard is parked, later chunks are no-ops.
        if ctx.get_filter_state::<InFlightGuard>().is_some() {
            return Ok(FilterAction::Continue);
        }

        // Wait for the first non-empty chunk, where the leading top-level fields
        // live; a bodyless request is never counted.
        let Some(bytes) = body.as_ref().filter(|b| !b.is_empty()) else {
            return Ok(FilterAction::Continue);
        };

        // Clone the registry out of extensions so the borrow on `ctx` ends
        // before the mutable insert below.
        let Some(registry) = ctx.extensions.get::<InFlightRegistry>().cloned() else {
            return Ok(FilterAction::Continue); // extension not installed; no-op
        };

        let (model, cap) = extract_model_and_cap(bytes);
        let model = model.unwrap_or_else(|| self.default_model.clone());
        let reserved = registry.on_request_start(&model, cap.unwrap_or(0));

        ctx.insert_filter_state(InFlightGuard {
            registry,
            model,
            reserved,
        });
        Ok(FilterAction::Continue)
    }
}

/// Extract the top-level `model` and token cap from a JSON request body chunk.
///
/// Returns the top-level `model` string and the reservation taken from the first
/// of `max_tokens` (Anthropic / Chat Completions) or `max_output_tokens` (OpenAI
/// Responses) to appear. Both are read in a single pass; a key nested inside
/// `messages` content is never surfaced, so a decoy there cannot misattribute
/// the request. Either component is `None` when its key is absent, carries the
/// wrong JSON type, or falls past the end of the chunk — these fields sit at the
/// front of the body, so the first chunk carries them in practice.
fn extract_model_and_cap(bytes: &[u8]) -> (Option<String>, Option<u64>) {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return (None, None);
    };
    let mut scanner = TopLevelKeyScanner::new(text);
    let (mut model, mut cap) = (None, None);

    while let Some(key) = scanner.next_top_level_key() {
        match key {
            "model" if model.is_none() => model = scanner.string_value().map(str::to_owned),
            "max_tokens" | "max_output_tokens" if cap.is_none() => cap = scanner.u64_value(),
            _ => {},
        }
        if model.is_some() && cap.is_some() {
            break;
        }
    }

    (model, cap)
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn start_increments_then_complete_decrements() {
        let registry = InFlightRegistry::new();
        let reserved = registry.on_request_start("gpt-4", 100);
        assert_eq!(
            registry.get("gpt-4"),
            ModelMetrics {
                in_flight: 1,
                reserved_tokens: 100
            }
        );
        registry.on_request_complete("gpt-4", reserved);
        assert_eq!(registry.get("gpt-4"), ModelMetrics::default());
    }

    #[test]
    fn complete_floors_at_zero() {
        let registry = InFlightRegistry::new();
        // A stray completion with no matching start must not underflow.
        registry.on_request_complete("gpt-4", 100);
        assert_eq!(registry.get("gpt-4"), ModelMetrics::default());
    }

    #[test]
    fn guard_drop_decrements_on_every_path() {
        // The decrement is welded to the guard's lifetime: whatever ends the
        // request, dropping the guard nets the reservation back out.
        let registry = InFlightRegistry::new();
        let reserved = registry.on_request_start("gpt-4", 100);
        {
            let _guard = InFlightGuard {
                registry: registry.clone(),
                model: "gpt-4".to_owned(),
                reserved,
            };
            assert_eq!(
                registry.get("gpt-4"),
                ModelMetrics {
                    in_flight: 1,
                    reserved_tokens: 100
                }
            );
        } // guard dropped here, as if the per-request context were torn down

        assert_eq!(registry.get("gpt-4"), ModelMetrics::default());
    }

    #[test]
    fn extract_reads_model_and_cap_together() {
        let body = br#"{"model":"gpt-4","max_tokens":512,"messages":[]}"#;
        let (model, cap) = extract_model_and_cap(body);
        assert_eq!(model.as_deref(), Some("gpt-4"));
        assert_eq!(cap, Some(512));
    }

    #[test]
    fn extract_reads_responses_max_output_tokens() {
        let body = br#"{"model":"o3","max_output_tokens":2048}"#;
        let (model, cap) = extract_model_and_cap(body);
        assert_eq!(model.as_deref(), Some("o3"));
        assert_eq!(cap, Some(2048));
    }

    #[test]
    fn extract_missing_fields_are_none() {
        let body = br#"{"messages":[{"role":"user","content":"hi"}]}"#;
        let (model, cap) = extract_model_and_cap(body);
        assert_eq!(model, None);
        assert_eq!(cap, None);
    }
}

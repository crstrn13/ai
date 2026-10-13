// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Streaming scanner over the top-level keys of a JSON request body.
//!
//! Request bodies arrive in chunks and can be megabytes long, so this scans
//! with a string- and depth-aware state machine instead of buffering and
//! deserializing the whole document. Only keys belonging to the top-level
//! object are surfaced, so a `"model"` (or any other) key nested inside
//! `messages` content can never misattribute the request.
//!
//! Shared by the `external_metering` and `inflight_tracker` filters; the
//! former scans per chunk for `model`, the latter carries a small resumable
//! scan state across chunks for `model` plus the token cap, never buffering
//! the body.

/// Extract the top-level `model` field from a JSON body fragment.
///
/// Returns `None` when the fragment does not contain the complete top-level
/// `"model": "..."` pair (missing, cut off, or a non-string value).
pub(crate) fn extract_model_from_bytes(bytes: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut scanner = TopLevelKeyScanner::new(text);

    while let Some(key) = scanner.next_top_level_key() {
        if key == "model" {
            return scanner.string_value().map(str::to_owned);
        }
    }

    None
}

/// Incremental scanner over the top-level keys of a JSON object fragment.
///
/// Tracks nesting depth and string boundaries (including escapes) so keys
/// inside nested objects, arrays, or string values are never surfaced.
pub(crate) struct TopLevelKeyScanner<'a> {
    /// Remaining unscanned input.
    rest: &'a str,

    /// Current object/array nesting depth; the document object is depth 1.
    depth: u32,
}

impl<'a> TopLevelKeyScanner<'a> {
    /// Start scanning at the beginning of a JSON document fragment.
    pub(crate) fn new(text: &'a str) -> Self {
        Self { rest: text, depth: 0 }
    }

    /// Advance to the next key of the top-level object and return it.
    pub(crate) fn next_top_level_key(&mut self) -> Option<&'a str> {
        loop {
            let mut chars = self.rest.char_indices();
            let (pos, ch) = chars.next()?;
            match ch {
                '{' | '[' => {
                    self.depth = self.depth.checked_add(1)?;
                    self.rest = self.rest.get(pos + 1..)?;
                },
                '}' | ']' => {
                    self.depth = self.depth.checked_sub(1)?;
                    self.rest = self.rest.get(pos + 1..)?;
                },
                '"' => {
                    let start = pos + 1;
                    let end = find_string_end(self.rest, start)?;
                    let content = self.rest.get(start..end)?;
                    let after = self.rest.get(end + 1..)?;
                    let is_key = after.trim_start().starts_with(':');
                    self.rest = after;
                    if self.depth == 1 && is_key {
                        return Some(content);
                    }
                },
                _ => {
                    self.rest = self.rest.get(pos + ch.len_utf8()..)?;
                },
            }
        }
    }

    /// Read the string value following the key just returned.
    ///
    /// Returns `None` when the value is not a string (e.g. `null`) or the
    /// fragment is cut off before the closing quote.
    pub(crate) fn string_value(&self) -> Option<&'a str> {
        let after_colon = self.rest.trim_start().strip_prefix(':')?;
        let value = after_colon.trim_start().strip_prefix('"')?;
        let end = find_string_end(value, 0)?;
        value.get(..end)
    }
}

/// Find the byte offset of the unescaped closing quote for the string
/// starting at `from` (which must point just past the opening quote).
fn find_string_end(text: &str, from: usize) -> Option<usize> {
    let mut escaped = false;
    for (offset, ch) in text.get(from..)?.char_indices() {
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(from + offset);
        }
    }
    None
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn extracts_model_from_compact_json() {
        let body = br#"{"model":"gpt-4","messages":[]}"#;
        assert_eq!(extract_model_from_bytes(body).as_deref(), Some("gpt-4"));
    }

    #[test]
    fn extracts_model_with_whitespace_around_colon() {
        let body = br#"{ "model" : "claude-sonnet-4" , "stream": true }"#;
        assert_eq!(extract_model_from_bytes(body).as_deref(), Some("claude-sonnet-4"));
    }

    #[test]
    fn extracts_model_when_not_first_field() {
        let body = br#"{"stream":true,"max_tokens":100,"model":"gpt-4o-mini"}"#;
        assert_eq!(extract_model_from_bytes(body).as_deref(), Some("gpt-4o-mini"));
    }

    #[test]
    fn extract_model_ignores_nested_decoy() {
        // A "model" key inside a message object must not win over the real
        // top-level field, regardless of field order.
        let body = br#"{"messages":[{"role":"user","content":"hi","model":"decoy"}],"model":"gpt-4"}"#;
        assert_eq!(
            extract_model_from_bytes(body).as_deref(),
            Some("gpt-4"),
            "nested decoy must not shadow the top-level model"
        );
    }

    #[test]
    fn extract_model_ignores_decoy_inside_string_value() {
        let body = br#"{"prompt":"please say \"model\": \"decoy\" back","model":"gpt-4o"}"#;
        assert_eq!(
            extract_model_from_bytes(body).as_deref(),
            Some("gpt-4o"),
            "a quoted decoy inside a string value must be skipped"
        );
    }

    #[test]
    fn extract_model_returns_none_when_only_nested() {
        let body = br#"{"messages":[{"model":"decoy","content":"hi"}]}"#;
        assert!(
            extract_model_from_bytes(body).is_none(),
            "a nested-only model key must not be attributed"
        );
    }

    #[test]
    fn extract_model_returns_none_when_absent() {
        let body = br#"{"messages":[{"role":"user","content":"hi"}]}"#;
        assert!(extract_model_from_bytes(body).is_none());
    }

    #[test]
    fn extract_model_returns_none_on_truncated_chunk() {
        // A streamed first chunk may cut off mid-value.
        let body = br#"{"model":"gpt-4"#;
        assert!(extract_model_from_bytes(body).is_none());
    }

    #[test]
    fn extract_model_returns_none_on_non_utf8() {
        let body = &[0xFF_u8, 0xFE, 0x00, 0x01];
        assert!(extract_model_from_bytes(body).is_none());
    }

    #[test]
    fn extract_model_returns_none_on_non_string_value() {
        let body = br#"{"model":null}"#;
        assert!(extract_model_from_bytes(body).is_none());
    }
}

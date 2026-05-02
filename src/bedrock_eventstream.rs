//! Decoder for AWS Event Stream binary framing, scoped to what Bedrock's
//! ConverseStream emits. We do not validate CRCs (AWS's own SDKs make that
//! optional); any corrupt frame would fail JSON parsing downstream anyway.
//!
//! Frame layout:
//! ```text
//!   [total_len:     u32 BE]
//!   [headers_len:   u32 BE]
//!   [prelude_crc:   u32 BE]        -- ignored
//!   [headers:       headers_len bytes]
//!   [payload:       total_len - headers_len - 16 bytes]
//!   [message_crc:   u32 BE]        -- ignored
//! ```
//!
//! Header layout (repeat):
//! ```text
//!   [name_len: u8][name: bytes]
//!   [type: u8]
//!   [value bytes depend on type]
//! ```
//!
//! Bedrock only sends string-typed (type=7) headers, encoded as
//! `[len: u16 BE][bytes]`. We decode just those; other types return an error.

use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;

const PRELUDE_LEN: usize = 12; // total_len + headers_len + prelude_crc
const CRC_TRAILER_LEN: usize = 4;
const HEADER_TYPE_STRING: u8 = 7;

/// A single decoded event-stream message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub headers: HashMap<String, String>,
    pub payload: Vec<u8>,
}

impl Frame {
    /// Return the `:event-type` header, which Bedrock uses to discriminate
    /// semantic event kinds (messageStart, contentBlockDelta, etc.).
    pub fn event_type(&self) -> Option<&str> {
        self.headers.get(":event-type").map(String::as_str)
    }

    /// Return the `:message-type` header. Bedrock sets this to "event" for
    /// normal events and "exception" for errors.
    pub fn message_type(&self) -> Option<&str> {
        self.headers.get(":message-type").map(String::as_str)
    }
}

/// Incremental byte buffer that yields complete `Frame`s as bytes arrive.
/// HTTP streaming delivers chunks that don't line up with frame boundaries,
/// so we buffer bytes and parse whenever a complete frame is available.
pub struct FrameDecoder {
    buf: Vec<u8>,
}

impl Default for FrameDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameDecoder {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn feed(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Try to parse one complete frame from the buffered bytes. Returns
    /// `Ok(None)` if more bytes are needed to complete a frame, or
    /// `Err(_)` on malformed framing (non-recoverable — caller should abort
    /// the stream).
    pub fn next_frame(&mut self) -> Result<Option<Frame>> {
        if self.buf.len() < PRELUDE_LEN {
            return Ok(None);
        }

        let total_len = u32::from_be_bytes(self.buf[0..4].try_into().unwrap()) as usize;
        let headers_len = u32::from_be_bytes(self.buf[4..8].try_into().unwrap()) as usize;

        if total_len < PRELUDE_LEN + CRC_TRAILER_LEN + headers_len {
            bail!(
                "event-stream frame lengths out of range (total={}, headers={})",
                total_len,
                headers_len
            );
        }

        if self.buf.len() < total_len {
            return Ok(None);
        }

        let headers_start = PRELUDE_LEN;
        let headers_end = headers_start + headers_len;
        let payload_end = total_len - CRC_TRAILER_LEN;

        let headers = parse_headers(&self.buf[headers_start..headers_end])?;
        let payload = self.buf[headers_end..payload_end].to_vec();

        // Consume the parsed frame.
        self.buf.drain(..total_len);

        Ok(Some(Frame { headers, payload }))
    }

    /// True when the internal buffer has been fully consumed. Useful for
    /// asserting that the stream ended cleanly (no trailing partial frame).
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

fn parse_headers(mut bytes: &[u8]) -> Result<HashMap<String, String>> {
    let mut headers = HashMap::new();
    while !bytes.is_empty() {
        let (name, rest) = read_header_name(bytes)?;
        bytes = rest;
        let (value, rest) = read_header_value(bytes)?;
        bytes = rest;
        headers.insert(name, value);
    }
    Ok(headers)
}

fn read_header_name(bytes: &[u8]) -> Result<(String, &[u8])> {
    if bytes.is_empty() {
        bail!("truncated header: missing name length");
    }
    let name_len = bytes[0] as usize;
    if bytes.len() < 1 + name_len {
        bail!("truncated header: name length {} exceeds remaining {}", name_len, bytes.len() - 1);
    }
    let name = std::str::from_utf8(&bytes[1..1 + name_len])
        .map_err(|_| anyhow!("non-UTF8 header name"))?
        .to_string();
    Ok((name, &bytes[1 + name_len..]))
}

fn read_header_value(bytes: &[u8]) -> Result<(String, &[u8])> {
    if bytes.is_empty() {
        bail!("truncated header: missing type");
    }
    let type_byte = bytes[0];
    if type_byte != HEADER_TYPE_STRING {
        bail!(
            "unsupported header type {} (Bedrock only sends strings/type=7)",
            type_byte
        );
    }
    if bytes.len() < 3 {
        bail!("truncated header: missing value length");
    }
    let val_len = u16::from_be_bytes([bytes[1], bytes[2]]) as usize;
    let value_start = 3;
    let value_end = value_start + val_len;
    if bytes.len() < value_end {
        bail!(
            "truncated header: value length {} exceeds remaining {}",
            val_len,
            bytes.len() - value_start
        );
    }
    let value = std::str::from_utf8(&bytes[value_start..value_end])
        .map_err(|_| anyhow!("non-UTF8 header value"))?
        .to_string();
    Ok((value, &bytes[value_end..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal string header segment: [name_len:u8][name][type=7][val_len:u16 BE][val]
    fn encode_header(name: &str, value: &str) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(name.len() as u8);
        out.extend_from_slice(name.as_bytes());
        out.push(HEADER_TYPE_STRING);
        out.extend_from_slice(&(value.len() as u16).to_be_bytes());
        out.extend_from_slice(value.as_bytes());
        out
    }

    fn encode_frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
        let mut header_bytes = Vec::new();
        for (n, v) in headers {
            header_bytes.extend_from_slice(&encode_header(n, v));
        }
        let total_len = PRELUDE_LEN + header_bytes.len() + payload.len() + CRC_TRAILER_LEN;
        let mut frame = Vec::with_capacity(total_len);
        frame.extend_from_slice(&(total_len as u32).to_be_bytes());
        frame.extend_from_slice(&(header_bytes.len() as u32).to_be_bytes());
        frame.extend_from_slice(&0u32.to_be_bytes()); // prelude_crc (ignored)
        frame.extend_from_slice(&header_bytes);
        frame.extend_from_slice(payload);
        frame.extend_from_slice(&0u32.to_be_bytes()); // message_crc (ignored)
        frame
    }

    #[test]
    fn parses_single_frame() {
        let frame = encode_frame(
            &[(":event-type", "messageStart"), (":message-type", "event")],
            br#"{"role":"assistant"}"#,
        );
        let mut dec = FrameDecoder::new();
        dec.feed(&frame);
        let out = dec.next_frame().unwrap().unwrap();
        assert_eq!(out.event_type(), Some("messageStart"));
        assert_eq!(out.message_type(), Some("event"));
        assert_eq!(out.payload, br#"{"role":"assistant"}"#);
        assert!(dec.is_empty());
    }

    #[test]
    fn needs_more_bytes_returns_none() {
        let frame = encode_frame(&[(":event-type", "x")], b"payload");
        let mut dec = FrameDecoder::new();
        dec.feed(&frame[..frame.len() / 2]);
        assert!(dec.next_frame().unwrap().is_none());
        dec.feed(&frame[frame.len() / 2..]);
        let out = dec.next_frame().unwrap().unwrap();
        assert_eq!(out.payload, b"payload");
    }

    #[test]
    fn parses_two_frames_back_to_back() {
        let mut buf = encode_frame(&[(":event-type", "a")], b"one");
        buf.extend_from_slice(&encode_frame(&[(":event-type", "b")], b"two"));
        let mut dec = FrameDecoder::new();
        dec.feed(&buf);
        let a = dec.next_frame().unwrap().unwrap();
        let b = dec.next_frame().unwrap().unwrap();
        assert_eq!(a.event_type(), Some("a"));
        assert_eq!(a.payload, b"one");
        assert_eq!(b.event_type(), Some("b"));
        assert_eq!(b.payload, b"two");
        assert!(dec.next_frame().unwrap().is_none());
    }

    #[test]
    fn byte_drip_feed_reassembles_correctly() {
        let frame = encode_frame(&[(":event-type", "contentBlockDelta")], b"hello");
        let mut dec = FrameDecoder::new();
        for byte in frame {
            dec.feed(&[byte]);
            // It's fine to call next_frame repeatedly; None until complete.
            let _ = dec.next_frame().unwrap();
        }
        // After the final byte, one final call should surface the frame —
        // but because we already called next_frame on the last iteration,
        // the frame was consumed there. Test with a fresh decoder instead:
        let frame = encode_frame(&[(":event-type", "x")], b"y");
        let mut dec = FrameDecoder::new();
        for (i, byte) in frame.iter().enumerate() {
            dec.feed(std::slice::from_ref(byte));
            let res = dec.next_frame().unwrap();
            if i + 1 == frame.len() {
                assert_eq!(res.unwrap().event_type(), Some("x"));
            } else {
                assert!(res.is_none(), "should not complete at byte {}", i);
            }
        }
    }

    #[test]
    fn rejects_non_string_header_type() {
        // Craft a header with type=2 (byte) instead of 7 (string).
        let header_bytes: Vec<u8> = vec![
            1u8,    // name_len
            b'x',   // name
            2u8,    // unsupported type
            0u8,    // value byte
        ];
        let total_len = PRELUDE_LEN + header_bytes.len() + CRC_TRAILER_LEN;
        let mut frame = Vec::new();
        frame.extend_from_slice(&(total_len as u32).to_be_bytes());
        frame.extend_from_slice(&(header_bytes.len() as u32).to_be_bytes());
        frame.extend_from_slice(&0u32.to_be_bytes());
        frame.extend_from_slice(&header_bytes);
        frame.extend_from_slice(&0u32.to_be_bytes());

        let mut dec = FrameDecoder::new();
        dec.feed(&frame);
        assert!(dec.next_frame().is_err());
    }

    /// Decode a real captured ConverseStream response end-to-end. Fixture
    /// is a raw binary response from Bedrock's /converse-stream endpoint for
    /// a "count from 1 to 5" prompt. This exercises real prelude sizes,
    /// varying frame lengths, and messageStart/contentBlockDelta/messageStop
    /// event types in a single run.
    #[test]
    fn decodes_real_bedrock_stream_fixture() {
        let bytes = include_bytes!("../tests/fixtures/bedrock_converse_stream.bin");
        let mut dec = FrameDecoder::new();
        dec.feed(bytes);
        let mut events = Vec::new();
        while let Some(frame) = dec.next_frame().unwrap() {
            events.push(frame);
        }
        assert!(dec.is_empty(), "fixture had trailing partial frame");
        assert!(!events.is_empty(), "expected at least one frame");

        let types: Vec<&str> = events.iter().filter_map(|f| f.event_type()).collect();
        assert!(types.contains(&"messageStart"), "missing messageStart event");
        assert!(
            types.contains(&"contentBlockDelta"),
            "missing contentBlockDelta events"
        );
        assert!(types.contains(&"messageStop"), "missing messageStop event");

        // Reassemble the text from contentBlockDelta payloads. The text
        // should at least contain the digit "1" from "1 to 5".
        let mut reassembled = String::new();
        for frame in &events {
            if frame.event_type() == Some("contentBlockDelta") {
                let v: serde_json::Value = serde_json::from_slice(&frame.payload).unwrap();
                if let Some(t) = v["delta"]["text"].as_str() {
                    reassembled.push_str(t);
                }
            }
        }
        assert!(
            reassembled.contains('1'),
            "expected reassembled text to contain digits: got {:?}",
            reassembled
        );
    }
}

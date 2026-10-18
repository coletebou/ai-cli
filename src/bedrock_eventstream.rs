//! Decoder for AWS Event Stream binary framing, scoped to what Bedrock's
//! ConverseStream emits. Both CRCs are validated: a corrupted length field
//! would otherwise mis-frame everything after it rather than fail cleanly.
//!
//! Frame layout:
//! ```text
//!   [total_len:     u32 BE]
//!   [headers_len:   u32 BE]
//!   [prelude_crc:   u32 BE]        -- CRC32 of the 8 bytes above
//!   [headers:       headers_len bytes]
//!   [payload:       total_len - headers_len - 16 bytes]
//!   [message_crc:   u32 BE]        -- CRC32 of everything before it
//! ```
//!
//! Header layout (repeat):
//! ```text
//!   [name_len: u8][name: bytes]
//!   [type: u8]
//!   [value bytes depend on type]
//! ```
//!
//! Bedrock only sends string-typed (type=7) headers today. Other value
//! types are length-decoded and skipped so a new header kind can't abort
//! the stream.

use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;

const PRELUDE_LEN: usize = 12; // total_len + headers_len + prelude_crc
const CRC_TRAILER_LEN: usize = 4;
const HEADER_TYPE_STRING: u8 = 7;
/// AWS caps an event-stream message at 16 MiB. Anything larger is a corrupt
/// (or hostile) length prefix; refuse it instead of buffering without bound.
const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;

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

    /// Return the `:message-type` header: "event" for normal events,
    /// "exception" for modeled server errors, "error" for unmodeled ones.
    pub fn message_type(&self) -> Option<&str> {
        self.headers.get(":message-type").map(String::as_str)
    }

    /// For `exception` frames, the exception name (e.g.
    /// `throttlingException`) lives in `:exception-type`, not `:event-type`.
    pub fn exception_type(&self) -> Option<&str> {
        self.headers.get(":exception-type").map(String::as_str)
    }

    /// Code carried by `error` frames (paired with `error_message`).
    pub fn error_code(&self) -> Option<&str> {
        self.headers.get(":error-code").map(String::as_str)
    }

    pub fn error_message(&self) -> Option<&str> {
        self.headers.get(":error-message").map(String::as_str)
    }
}

/// Incremental byte buffer that yields complete `Frame`s as bytes arrive.
/// HTTP streaming delivers chunks that don't line up with frame boundaries,
/// so we buffer bytes and parse whenever a complete frame is available.
pub struct FrameDecoder {
    buf: Vec<u8>,
    /// Offset of the first unconsumed byte. Consumed bytes are compacted
    /// away lazily (on the next `feed`) so a chunk holding many frames isn't
    /// shifted once per frame.
    start: usize,
}

impl Default for FrameDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameDecoder {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            start: 0,
        }
    }

    pub fn feed(&mut self, bytes: &[u8]) {
        if self.start > 0 {
            self.buf.drain(..self.start);
            self.start = 0;
        }
        self.buf.extend_from_slice(bytes);
    }

    /// Try to parse one complete frame from the buffered bytes. Returns
    /// `Ok(None)` if more bytes are needed to complete a frame, or
    /// `Err(_)` on malformed framing (non-recoverable — caller should abort
    /// the stream).
    pub fn next_frame(&mut self) -> Result<Option<Frame>> {
        let pending = &self.buf[self.start..];
        if pending.len() < PRELUDE_LEN {
            return Ok(None);
        }

        let total_len = read_u32(pending, 0) as usize;
        let headers_len = read_u32(pending, 4) as usize;
        let prelude_crc = read_u32(pending, 8);

        // Check the prelude before trusting its lengths: a corrupt length
        // would otherwise make us wait for bytes that never come.
        let actual = crc32(&pending[..8]);
        if actual != prelude_crc {
            bail!(
                "event-stream prelude CRC mismatch (expected {:#010x}, got {:#010x})",
                prelude_crc,
                actual
            );
        }

        if total_len > MAX_FRAME_LEN
            || headers_len > total_len
            || total_len < PRELUDE_LEN + CRC_TRAILER_LEN + headers_len
        {
            bail!(
                "event-stream frame lengths out of range (total={}, headers={})",
                total_len,
                headers_len
            );
        }

        if pending.len() < total_len {
            return Ok(None);
        }

        let payload_end = total_len - CRC_TRAILER_LEN;
        let message_crc = read_u32(pending, payload_end);
        let actual = crc32(&pending[..payload_end]);
        if actual != message_crc {
            bail!(
                "event-stream message CRC mismatch (expected {:#010x}, got {:#010x})",
                message_crc,
                actual
            );
        }

        let headers_end = PRELUDE_LEN + headers_len;
        let headers = parse_headers(&pending[PRELUDE_LEN..headers_end])?;
        let payload = pending[headers_end..payload_end].to_vec();

        // Consume the parsed frame.
        self.start += total_len;
        if self.start == self.buf.len() {
            self.buf.clear();
            self.start = 0;
        }

        Ok(Some(Frame { headers, payload }))
    }

    /// True when the internal buffer has been fully consumed. Used to
    /// detect a stream that ended mid-frame (connection cut short).
    pub fn is_empty(&self) -> bool {
        self.start == self.buf.len()
    }
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// CRC-32 (IEEE 802.3, the zlib/PNG variant) as used by AWS event streams.
fn crc32(bytes: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
                k += 1;
            }
            table[i] = c;
            i += 1;
        }
        table
    };
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc = TABLE[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

fn parse_headers(mut bytes: &[u8]) -> Result<HashMap<String, String>> {
    let mut headers = HashMap::new();
    while !bytes.is_empty() {
        let (name, rest) = read_header_name(bytes)?;
        bytes = rest;
        let (value, rest) = read_header_value(bytes)?;
        bytes = rest;
        if let Some(value) = value {
            headers.insert(name, value);
        }
    }
    Ok(headers)
}

fn read_header_name(bytes: &[u8]) -> Result<(String, &[u8])> {
    if bytes.is_empty() {
        bail!("truncated header: missing name length");
    }
    let name_len = bytes[0] as usize;
    if bytes.len() < 1 + name_len {
        bail!(
            "truncated header: name length {} exceeds remaining {}",
            name_len,
            bytes.len() - 1
        );
    }
    let name = std::str::from_utf8(&bytes[1..1 + name_len])
        .map_err(|_| anyhow!("non-UTF8 header name"))?
        .to_string();
    Ok((name, &bytes[1 + name_len..]))
}

/// Decode one header value. String values are returned; every other type
/// is skipped (`None`) using its fixed or length-prefixed size.
fn read_header_value(bytes: &[u8]) -> Result<(Option<String>, &[u8])> {
    let Some((&type_byte, rest)) = bytes.split_first() else {
        bail!("truncated header: missing type");
    };
    let fixed_len = match type_byte {
        0 | 1 => Some(0), // bool true / bool false: no value bytes
        2 => Some(1),     // byte
        3 => Some(2),     // short
        4 => Some(4),     // int
        5 | 8 => Some(8), // long, timestamp
        9 => Some(16),    // uuid
        6 | HEADER_TYPE_STRING => None,
        other => bail!("unknown event-stream header type {}", other),
    };
    if let Some(len) = fixed_len {
        if rest.len() < len {
            bail!("truncated header: type {} needs {} bytes", type_byte, len);
        }
        return Ok((None, &rest[len..]));
    }

    // Byte array (6) or string (7): u16 length prefix.
    if rest.len() < 2 {
        bail!("truncated header: missing value length");
    }
    let val_len = u16::from_be_bytes([rest[0], rest[1]]) as usize;
    let value_end = 2 + val_len;
    if rest.len() < value_end {
        bail!(
            "truncated header: value length {} exceeds remaining {}",
            val_len,
            rest.len() - 2
        );
    }
    if type_byte != HEADER_TYPE_STRING {
        return Ok((None, &rest[value_end..]));
    }
    let value = std::str::from_utf8(&rest[2..value_end])
        .map_err(|_| anyhow!("non-UTF8 header value"))?
        .to_string();
    Ok((Some(value), &rest[value_end..]))
}

/// Test helper shared with `bedrock.rs` tests: encode a well-formed frame
/// (with real CRCs) from string headers and a payload.
#[cfg(test)]
pub fn encode_test_frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
    let mut header_bytes = Vec::new();
    for (n, v) in headers {
        header_bytes.push(n.len() as u8);
        header_bytes.extend_from_slice(n.as_bytes());
        header_bytes.push(HEADER_TYPE_STRING);
        header_bytes.extend_from_slice(&(v.len() as u16).to_be_bytes());
        header_bytes.extend_from_slice(v.as_bytes());
    }
    encode_raw_frame(&header_bytes, payload)
}

#[cfg(test)]
fn encode_raw_frame(header_bytes: &[u8], payload: &[u8]) -> Vec<u8> {
    let total_len = PRELUDE_LEN + header_bytes.len() + payload.len() + CRC_TRAILER_LEN;
    let mut frame = Vec::with_capacity(total_len);
    frame.extend_from_slice(&(total_len as u32).to_be_bytes());
    frame.extend_from_slice(&(header_bytes.len() as u32).to_be_bytes());
    let prelude_crc = crc32(&frame);
    frame.extend_from_slice(&prelude_crc.to_be_bytes());
    frame.extend_from_slice(header_bytes);
    frame.extend_from_slice(payload);
    let message_crc = crc32(&frame);
    frame.extend_from_slice(&message_crc.to_be_bytes());
    frame
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
        encode_test_frame(headers, payload)
    }

    #[test]
    fn crc32_matches_reference_vector() {
        // The standard CRC-32 check value.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
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
        assert!(!dec.is_empty(), "a partial frame is still buffered");
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
        assert!(dec.is_empty());
    }

    #[test]
    fn byte_drip_feed_reassembles_correctly() {
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
        assert!(dec.is_empty());
    }

    #[test]
    fn rejects_corrupted_payload() {
        let mut frame = encode_frame(&[(":event-type", "x")], b"payload");
        let payload_byte = frame.len() - CRC_TRAILER_LEN - 1;
        frame[payload_byte] ^= 0xFF;
        let mut dec = FrameDecoder::new();
        dec.feed(&frame);
        let err = dec.next_frame().unwrap_err().to_string();
        assert!(err.contains("message CRC"), "got: {}", err);
    }

    #[test]
    fn rejects_corrupted_length_before_waiting_for_bytes() {
        let mut frame = encode_frame(&[(":event-type", "x")], b"payload");
        // Inflate total_len; without the prelude CRC we'd just wait forever.
        frame[0] = 0x7F;
        let mut dec = FrameDecoder::new();
        dec.feed(&frame);
        let err = dec.next_frame().unwrap_err().to_string();
        assert!(err.contains("prelude CRC"), "got: {}", err);
    }

    #[test]
    fn rejects_oversized_frames() {
        // A well-formed prelude claiming a 1 GiB frame.
        let mut prelude = Vec::new();
        prelude.extend_from_slice(&(1u32 << 30).to_be_bytes());
        prelude.extend_from_slice(&0u32.to_be_bytes());
        let crc = crc32(&prelude);
        prelude.extend_from_slice(&crc.to_be_bytes());
        let mut dec = FrameDecoder::new();
        dec.feed(&prelude);
        assert!(dec.next_frame().is_err());
    }

    #[test]
    fn skips_non_string_header_types() {
        // A byte header (type 2), a timestamp (type 8), then a string.
        let mut header_bytes: Vec<u8> = vec![1, b'b', 2, 0x2A];
        header_bytes.extend_from_slice(&[2, b't', b's', 8]);
        header_bytes.extend_from_slice(&0u64.to_be_bytes());
        header_bytes.extend_from_slice(&[11]);
        header_bytes.extend_from_slice(b":event-type");
        header_bytes.push(HEADER_TYPE_STRING);
        header_bytes.extend_from_slice(&3u16.to_be_bytes());
        header_bytes.extend_from_slice(b"abc");
        let frame = encode_raw_frame(&header_bytes, b"{}");

        let mut dec = FrameDecoder::new();
        dec.feed(&frame);
        let out = dec.next_frame().unwrap().unwrap();
        assert_eq!(out.event_type(), Some("abc"));
        assert_eq!(out.headers.len(), 1);
    }

    #[test]
    fn rejects_unknown_header_type() {
        let header_bytes: Vec<u8> = vec![1, b'x', 42, 0];
        let frame = encode_raw_frame(&header_bytes, b"");
        let mut dec = FrameDecoder::new();
        dec.feed(&frame);
        assert!(dec.next_frame().is_err());
    }

    /// Decode a real captured ConverseStream response end-to-end. Fixture
    /// is a raw binary response from Bedrock's /converse-stream endpoint for
    /// a "count from 1 to 5" prompt. This exercises real prelude sizes,
    /// real CRCs, varying frame lengths, and
    /// messageStart/contentBlockDelta/messageStop event types in one run.
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
        assert!(
            types.contains(&"messageStart"),
            "missing messageStart event"
        );
        assert!(
            types.contains(&"contentBlockDelta"),
            "missing contentBlockDelta events"
        );
        assert!(types.contains(&"messageStop"), "missing messageStop event");

        let mut reassembled = String::new();
        for frame in &events {
            if frame.event_type() == Some("contentBlockDelta") {
                let v: serde_json::Value = serde_json::from_slice(&frame.payload).unwrap();
                if let Some(t) = v["delta"]["text"].as_str() {
                    reassembled.push_str(t);
                }
            }
        }
        for digit in ['1', '2', '3', '4', '5'] {
            assert!(
                reassembled.contains(digit),
                "expected {} in reassembled text: got {:?}",
                digit,
                reassembled
            );
        }
    }
}

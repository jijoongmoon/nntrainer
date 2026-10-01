use libc::{c_char, c_int, c_void, size_t};
use std::ptr;
use std::slice;
use tokenizers::models::bpe::{Vocab, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

type TokenizerHandle = *mut c_void;

#[repr(C)]
pub struct TokenizerEncodeResult {
    token_ids: *mut c_int,
    len: size_t,
}

struct TokenizerState {
    tokenizer: Tokenizer,
    decoded: Vec<u8>,
    token: Vec<u8>,
}

unsafe fn bytes_from_raw<'a>(data: *const c_char, len: size_t) -> Option<&'a [u8]> {
    if data.is_null() {
        return if len == 0 { Some(&[]) } else { None };
    }
    Some(slice::from_raw_parts(data as *const u8, len))
}

unsafe fn state_from_handle<'a>(handle: TokenizerHandle) -> Option<&'a mut TokenizerState> {
    if handle.is_null() {
        return None;
    }
    Some(&mut *(handle as *mut TokenizerState))
}

unsafe fn fill_result(result: *mut TokenizerEncodeResult, ids: &[u32]) {
    if result.is_null() {
        return;
    }

    if ids.is_empty() {
        (*result).token_ids = ptr::null_mut();
        (*result).len = 0;
        return;
    }

    let byte_len = ids.len() * std::mem::size_of::<c_int>();
    let out = libc::malloc(byte_len) as *mut c_int;
    if out.is_null() {
        (*result).token_ids = ptr::null_mut();
        (*result).len = 0;
        return;
    }

    for (idx, id) in ids.iter().enumerate() {
        *out.add(idx) = *id as c_int;
    }

    (*result).token_ids = out;
    (*result).len = ids.len();
}

fn parse_merges(data: &[u8]) -> Option<Vec<(String, String)>> {
    let text = std::str::from_utf8(data).ok()?;
    let mut merges = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("#version") {
            continue;
        }

        let mut parts = line.split(' ');
        let first = parts.next()?;
        let second = parts.next()?;
        if parts.next().is_some() {
            return None;
        }

        merges.push((first.to_string(), second.to_string()));
    }

    Some(merges)
}

fn parse_added_tokens(data: &[u8]) -> Option<Vec<AddedToken>> {
    if data.is_empty() {
        return Some(Vec::new());
    }

    if let Ok(tokens) = serde_json::from_slice::<Vec<AddedToken>>(data) {
        return Some(tokens);
    }

    let tokens = serde_json::from_slice::<Vec<String>>(data).ok()?;
    Some(
        tokens
            .into_iter()
            .map(|token| AddedToken::from(token, false))
            .collect(),
    )
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_new_from_str(
    json: *const c_char,
    len: size_t,
) -> TokenizerHandle {
    let Some(bytes) = bytes_from_raw(json, len) else {
        return ptr::null_mut();
    };

    match Tokenizer::from_bytes(bytes) {
        Ok(tokenizer) => Box::into_raw(Box::new(TokenizerState {
            tokenizer,
            decoded: Vec::new(),
            token: Vec::new(),
        })) as TokenizerHandle,
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn byte_level_bpe_tokenizers_new_from_str(
    vocab: *const c_char,
    vocab_len: size_t,
    merges: *const c_char,
    merges_len: size_t,
    added_tokens: *const c_char,
    added_tokens_len: size_t,
) -> TokenizerHandle {
    let Some(vocab_bytes) = bytes_from_raw(vocab, vocab_len) else {
        return ptr::null_mut();
    };
    let Some(merges_bytes) = bytes_from_raw(merges, merges_len) else {
        return ptr::null_mut();
    };
    let Some(added_tokens_bytes) = bytes_from_raw(added_tokens, added_tokens_len) else {
        return ptr::null_mut();
    };
    let Ok(vocab) = serde_json::from_slice::<Vocab>(vocab_bytes) else {
        return ptr::null_mut();
    };
    let Some(merges) = parse_merges(merges_bytes) else {
        return ptr::null_mut();
    };
    let Some(added_tokens) = parse_added_tokens(added_tokens_bytes) else {
        return ptr::null_mut();
    };

    let Ok(model) = BPE::builder().vocab_and_merges(vocab, merges).build() else {
        return ptr::null_mut();
    };

    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
    tokenizer.with_post_processor(Some(ByteLevel::default()));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer.add_tokens(&added_tokens);

    Box::into_raw(Box::new(TokenizerState {
        tokenizer,
        decoded: Vec::new(),
        token: Vec::new(),
    })) as TokenizerHandle
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_encode(
    handle: TokenizerHandle,
    data: *const c_char,
    len: size_t,
    add_special_token: c_int,
    result: *mut TokenizerEncodeResult,
) {
    let Some(state) = state_from_handle(handle) else {
        fill_result(result, &[]);
        return;
    };
    let Some(bytes) = bytes_from_raw(data, len) else {
        fill_result(result, &[]);
        return;
    };
    let Ok(text) = std::str::from_utf8(bytes) else {
        fill_result(result, &[]);
        return;
    };

    match state.tokenizer.encode(text, add_special_token != 0) {
        Ok(encoding) => fill_result(result, encoding.get_ids()),
        Err(_) => fill_result(result, &[]),
    }
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_encode_batch(
    handle: TokenizerHandle,
    data: *const *const c_char,
    len: *mut size_t,
    num_seqs: size_t,
    add_special_token: c_int,
    results: *mut TokenizerEncodeResult,
) {
    if data.is_null() || len.is_null() || results.is_null() {
        return;
    }

    for idx in 0..num_seqs {
        tokenizers_encode(
            handle,
            *data.add(idx),
            *len.add(idx),
            add_special_token,
            results.add(idx),
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_free_encode_results(
    results: *mut TokenizerEncodeResult,
    num_seqs: size_t,
) {
    if results.is_null() {
        return;
    }

    for idx in 0..num_seqs {
        let result = results.add(idx);
        if !(*result).token_ids.is_null() {
            libc::free((*result).token_ids as *mut c_void);
        }
        (*result).token_ids = ptr::null_mut();
        (*result).len = 0;
    }
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_decode(
    handle: TokenizerHandle,
    data: *const u32,
    len: size_t,
    skip_special_token: c_int,
) {
    let Some(state) = state_from_handle(handle) else {
        return;
    };
    if data.is_null() {
        state.decoded.clear();
        return;
    }

    let ids = slice::from_raw_parts(data, len);
    match state.tokenizer.decode(ids, skip_special_token != 0) {
        Ok(decoded) => state.decoded = decoded.into_bytes(),
        Err(_) => state.decoded.clear(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_get_decode_str(
    handle: TokenizerHandle,
    data: *mut *const c_char,
    len: *mut size_t,
) {
    if data.is_null() || len.is_null() {
        return;
    }
    let Some(state) = state_from_handle(handle) else {
        *data = ptr::null();
        *len = 0;
        return;
    };

    *data = state.decoded.as_ptr() as *const c_char;
    *len = state.decoded.len();
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_get_vocab_size(handle: TokenizerHandle, size: *mut size_t) {
    if size.is_null() {
        return;
    }
    let Some(state) = state_from_handle(handle) else {
        *size = 0;
        return;
    };

    *size = state.tokenizer.get_vocab_size(true) as size_t;
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_id_to_token(
    handle: TokenizerHandle,
    id: u32,
    data: *mut *const c_char,
    len: *mut size_t,
) {
    if data.is_null() || len.is_null() {
        return;
    }
    let Some(state) = state_from_handle(handle) else {
        *data = ptr::null();
        *len = 0;
        return;
    };

    state.token = state
        .tokenizer
        .id_to_token(id)
        .unwrap_or_default()
        .into_bytes();
    *data = state.token.as_ptr() as *const c_char;
    *len = state.token.len();
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_token_to_id(
    handle: TokenizerHandle,
    token: *const c_char,
    len: size_t,
    id: *mut i32,
) {
    if id.is_null() {
        return;
    }
    let Some(state) = state_from_handle(handle) else {
        *id = -1;
        return;
    };
    let Some(bytes) = bytes_from_raw(token, len) else {
        *id = -1;
        return;
    };
    let Ok(text) = std::str::from_utf8(bytes) else {
        *id = -1;
        return;
    };

    *id = state
        .tokenizer
        .token_to_id(text)
        .map(|token_id| token_id as i32)
        .unwrap_or(-1);
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_free(handle: TokenizerHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut TokenizerState));
    }
}

// ---------------------------------------------------------------------------
// Tokenizer snapshot: a persistent post-parse cache payload.
//
// Rationale: Tokenizer::from_bytes on a ~32MB BPE tokenizer.json costs
// ~500-600ms per process on an idle desktop host (more under contention with
// load workers), and the crate's own serde path cannot be made
// faster by switching serde formats -- the cost is dominated by the internal
// Content/Value buffering (ModelWrapper's untagged+flatten deserialize) and
// by BPE construction, not by JSON text parsing. (Measured: a full-object
// msgpack round-trip loads in ~505ms vs ~525ms JSON -- no win. Also, BPE's
// custom Serialize declares 8 struct fields but writes 10, which truncates
// vocab/merges in any length-prefixed format; only the JSON-object route is
// safe.)
//
// The snapshot therefore bypasses serde for the heavy sections. Layout
// (little-endian):
//
//   u32 magic 'NTKS' | u32 version
//   u32 comp_len   | components JSON (verbatim copies of every non-model
//                    section of tokenizer.json + the BPE option fields)
//   u32 vocab_cnt  | vocab_cnt x { u32 id, u32 len, bytes }
//   u32 merges_cnt | merges_cnt x { u32 len_a, bytes_a, u32 len_b, bytes_b }
//
// Reconstruction uses only public tokenizers API (BpeBuilder + component
// deserializers + add_tokens in the same order as the crate's own
// TokenizerVisitor), measured at ~175ms for the same file (~3x). Encode /
// decode / vocab-size / id<->token identity against the parse path is the
// caller's acceptance gate; ANY structural surprise here returns
// null/failure so the caller can fall back to the parse path.
//
// Only BPE models are snapshotted (vocab/merges are the only heavy sections
// worth bypassing); everything else returns failure => no cache.
// ---------------------------------------------------------------------------

const SNAPSHOT_MAGIC: u32 = 0x4E544B53; // 'NTKS'
const SNAPSHOT_VERSION: u32 = 1;

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn u32(&mut self) -> Option<u32> {
        let end = self.pos.checked_add(4)?;
        if end > self.buf.len() {
            return None;
        }
        let v = u32::from_le_bytes(self.buf[self.pos..end].try_into().ok()?);
        self.pos = end;
        Some(v)
    }
    fn bytes(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(len)?;
        if end > self.buf.len() {
            return None;
        }
        let b = &self.buf[self.pos..end];
        self.pos = end;
        Some(b)
    }
    fn str_field(&mut self) -> Option<&'a str> {
        let len = self.u32()? as usize;
        std::str::from_utf8(self.bytes(len)?).ok()
    }
}

/// Build the snapshot payload from the ORIGINAL tokenizer.json bytes.
/// Conservative: any shape this writer does not fully understand => None
/// (the caller simply does not cache). The caller must only invoke this
/// after the parse path succeeded on the same bytes, so the file is known
/// to be loadable by the crate.
fn snapshot_from_json_impl(json: &[u8]) -> Option<Vec<u8>> {
    let v: serde_json::Value = serde_json::from_slice(json).ok()?;
    let obj = v.as_object()?;

    // The crate's TokenizerVisitor only accepts version "1.0"; mirror it.
    if let Some(ver) = obj.get("version") {
        if ver.as_str() != Some("1.0") {
            return None;
        }
    }

    let model = obj.get("model")?.as_object()?;
    if model.get("type").and_then(|t| t.as_str()) != Some("BPE") {
        return None; // only BPE is worth (and safe for) bypassing
    }

    // added_tokens: require the exact field set the reconstruction consumes,
    // so a novel added-token shape falls back to the parse path instead of
    // being silently mis-reconstructed.
    if let Some(added) = obj.get("added_tokens") {
        let arr = added.as_array()?;
        for t in arr {
            let o = t.as_object()?;
            if !(o.get("content").map_or(false, |x| x.is_string())
                && o.get("id").map_or(false, |x| x.is_u64())
                && o.get("special").map_or(false, |x| x.is_boolean())
                && o.get("single_word").map_or(false, |x| x.is_boolean())
                && o.get("lstrip").map_or(false, |x| x.is_boolean())
                && o.get("rstrip").map_or(false, |x| x.is_boolean())
                && o.get("normalized").map_or(false, |x| x.is_boolean()))
            {
                return None;
            }
        }
    }

    // Components JSON: verbatim copies, so reconstruction feeds the exact
    // same JSON values to the exact same component deserializers.
    let mut comp = serde_json::Map::new();
    for key in [
        "normalizer",
        "pre_tokenizer",
        "post_processor",
        "decoder",
        "padding",
        "truncation",
        "added_tokens",
    ] {
        if let Some(val) = obj.get(key) {
            comp.insert(key.to_string(), val.clone());
        }
    }
    let mut bpe_opts = serde_json::Map::new();
    for key in [
        "dropout",
        "unk_token",
        "continuing_subword_prefix",
        "end_of_word_suffix",
        "fuse_unk",
        "byte_fallback",
        "ignore_merges",
    ] {
        if let Some(val) = model.get(key) {
            bpe_opts.insert(key.to_string(), val.clone());
        }
    }
    comp.insert("bpe".to_string(), serde_json::Value::Object(bpe_opts));
    let comp_json = serde_json::to_string(&serde_json::Value::Object(comp)).ok()?;

    let vocab = model.get("vocab")?.as_object()?;
    let merges = model.get("merges")?.as_array()?;

    let mut out = Vec::with_capacity(json.len() / 2 + comp_json.len() + 64);
    put_u32(&mut out, SNAPSHOT_MAGIC);
    put_u32(&mut out, SNAPSHOT_VERSION);
    put_str(&mut out, &comp_json);

    put_u32(&mut out, u32::try_from(vocab.len()).ok()?);
    for (token, id) in vocab {
        let id = u32::try_from(id.as_u64()?).ok()?;
        put_u32(&mut out, id);
        put_str(&mut out, token);
    }

    put_u32(&mut out, u32::try_from(merges.len()).ok()?);
    for m in merges {
        if let Some(pair) = m.as_array() {
            if pair.len() != 2 {
                return None;
            }
            put_str(&mut out, pair[0].as_str()?);
            put_str(&mut out, pair[1].as_str()?);
        } else {
            // legacy "a b" string form
            let s = m.as_str()?;
            let mut it = s.splitn(2, ' ');
            let a = it.next()?;
            let b = it.next()?;
            put_str(&mut out, a);
            put_str(&mut out, b);
        }
    }
    Some(out)
}

/// Rebuild the Tokenizer from a snapshot payload. Bounds-checked throughout;
/// any inconsistency => None (caller falls back to the parse path). The
/// assembly order mirrors the crate's own TokenizerVisitor: model FIRST,
/// then components, then add_tokens against the fully-populated model so
/// added-token id resolution is identical.
fn tokenizer_from_snapshot_impl(blob: &[u8]) -> Option<Tokenizer> {
    let mut cur = Cursor { buf: blob, pos: 0 };
    if cur.u32()? != SNAPSHOT_MAGIC || cur.u32()? != SNAPSHOT_VERSION {
        return None;
    }
    let comp_json = cur.str_field()?;
    let comp: serde_json::Value = serde_json::from_str(comp_json).ok()?;
    let comp = comp.as_object()?;

    let vocab_cnt = cur.u32()? as usize;
    // 8 bytes minimum per entry: reject counts the remaining bytes cannot hold.
    if vocab_cnt > (blob.len() - cur.pos) / 8 {
        return None;
    }
    let mut vocab = Vocab::with_capacity(vocab_cnt);
    for _ in 0..vocab_cnt {
        let id = cur.u32()?;
        let token = cur.str_field()?;
        vocab.insert(token.to_string(), id);
    }

    let merges_cnt = cur.u32()? as usize;
    if merges_cnt > (blob.len() - cur.pos) / 8 {
        return None;
    }
    let mut merges: Vec<(String, String)> = Vec::with_capacity(merges_cnt);
    for _ in 0..merges_cnt {
        let a = cur.str_field()?;
        let b = cur.str_field()?;
        merges.push((a.to_string(), b.to_string()));
    }
    if cur.pos != blob.len() {
        return None; // trailing bytes => not something this version wrote
    }

    let empty = serde_json::Map::new();
    let opts = comp
        .get("bpe")
        .and_then(|b| b.as_object())
        .unwrap_or(&empty);
    let mut b = BPE::builder().vocab_and_merges(vocab, merges);
    // Mirrors BPEVisitor: absent or null => builder default, exactly like
    // `if let Some(x) = map.next_value::<Option<_>>()`.
    if let Some(d) = opts.get("dropout").and_then(|x| x.as_f64()) {
        b = b.dropout(d as f32);
    }
    if let Some(u) = opts.get("unk_token").and_then(|x| x.as_str()) {
        b = b.unk_token(u.to_string());
    }
    if let Some(p) = opts
        .get("continuing_subword_prefix")
        .and_then(|x| x.as_str())
    {
        b = b.continuing_subword_prefix(p.to_string());
    }
    if let Some(s) = opts.get("end_of_word_suffix").and_then(|x| x.as_str()) {
        b = b.end_of_word_suffix(s.to_string());
    }
    if let Some(f) = opts.get("fuse_unk").and_then(|x| x.as_bool()) {
        b = b.fuse_unk(f);
    }
    if let Some(f) = opts.get("byte_fallback").and_then(|x| x.as_bool()) {
        b = b.byte_fallback(f);
    }
    if let Some(f) = opts.get("ignore_merges").and_then(|x| x.as_bool()) {
        b = b.ignore_merges(f);
    }
    let bpe = b.build().ok()?;

    let mut tk = Tokenizer::new(bpe);
    if let Some(n) = comp.get("normalizer") {
        if !n.is_null() {
            tk.with_normalizer(Some(
                serde_json::from_value::<tokenizers::NormalizerWrapper>(n.clone()).ok()?,
            ));
        }
    }
    if let Some(p) = comp.get("pre_tokenizer") {
        if !p.is_null() {
            tk.with_pre_tokenizer(Some(
                serde_json::from_value::<tokenizers::PreTokenizerWrapper>(p.clone()).ok()?,
            ));
        }
    }
    if let Some(p) = comp.get("post_processor") {
        if !p.is_null() {
            tk.with_post_processor(Some(
                serde_json::from_value::<tokenizers::PostProcessorWrapper>(p.clone()).ok()?,
            ));
        }
    }
    if let Some(d) = comp.get("decoder") {
        if !d.is_null() {
            tk.with_decoder(Some(
                serde_json::from_value::<tokenizers::DecoderWrapper>(d.clone()).ok()?,
            ));
        }
    }
    if let Some(p) = comp.get("padding") {
        if !p.is_null() {
            tk.with_padding(Some(serde_json::from_value(p.clone()).ok()?));
        }
    }
    if let Some(t) = comp.get("truncation") {
        if !t.is_null() {
            tk.with_truncation(Some(serde_json::from_value(t.clone()).ok()?))
                .ok()?;
        }
    }
    if let Some(a) = comp.get("added_tokens") {
        let arr = a.as_array()?;
        let mut toks: Vec<AddedToken> = Vec::with_capacity(arr.len());
        for t in arr {
            let o = t.as_object()?;
            toks.push(
                AddedToken::from(
                    o.get("content")?.as_str()?.to_string(),
                    o.get("special")?.as_bool()?,
                )
                .single_word(o.get("single_word")?.as_bool()?)
                .lstrip(o.get("lstrip")?.as_bool()?)
                .rstrip(o.get("rstrip")?.as_bool()?)
                .normalized(o.get("normalized")?.as_bool()?),
            );
        }
        tk.add_tokens(&toks);
    }
    Some(tk)
}

/// C ABI: build a snapshot payload from tokenizer.json bytes. On success
/// *out_data (malloc'd, free with tokenizers_snapshot_free) and *out_len are
/// set; on ANY failure both are zeroed. Never panics across the boundary.
#[no_mangle]
pub unsafe extern "C" fn tokenizers_snapshot_from_json(
    json: *const c_char,
    len: size_t,
    out_data: *mut *mut c_char,
    out_len: *mut size_t,
) {
    if out_data.is_null() || out_len.is_null() {
        return;
    }
    *out_data = ptr::null_mut();
    *out_len = 0;
    let Some(bytes) = bytes_from_raw(json, len) else {
        return;
    };
    let blob = std::panic::catch_unwind(|| snapshot_from_json_impl(bytes))
        .ok()
        .flatten();
    let Some(blob) = blob else {
        return;
    };
    let out = libc::malloc(blob.len()) as *mut u8;
    if out.is_null() {
        return;
    }
    ptr::copy_nonoverlapping(blob.as_ptr(), out, blob.len());
    *out_data = out as *mut c_char;
    *out_len = blob.len();
}

/// C ABI: free a buffer returned by tokenizers_snapshot_from_json.
#[no_mangle]
pub unsafe extern "C" fn tokenizers_snapshot_free(data: *mut c_char) {
    if !data.is_null() {
        libc::free(data as *mut c_void);
    }
}

/// C ABI: rebuild a tokenizer from a snapshot payload. Returns NULL on ANY
/// failure (bad magic/version/bounds/deserialize) -- the caller falls back
/// to the JSON parse path. Never panics across the boundary.
#[no_mangle]
pub unsafe extern "C" fn tokenizers_new_from_snapshot(
    data: *const c_char,
    len: size_t,
) -> TokenizerHandle {
    let Some(bytes) = bytes_from_raw(data, len) else {
        return ptr::null_mut();
    };
    let tok = std::panic::catch_unwind(|| tokenizer_from_snapshot_impl(bytes))
        .ok()
        .flatten();
    match tok {
        Some(tokenizer) => Box::into_raw(Box::new(TokenizerState {
            tokenizer,
            decoded: Vec::new(),
            token: Vec::new(),
        })) as TokenizerHandle,
        None => ptr::null_mut(),
    }
}

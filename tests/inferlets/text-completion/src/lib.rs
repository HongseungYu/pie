//! Greedy text completion, **host-driven** — the fixture the serving door is
//! gated on.
//!
//! # Why this is not `naive-baseline` with the sampler swapped
//!
//! Every other decode inferlet in this directory carries its next token ON THE
//! DEVICE: the epilogue writes the sampled token straight back into the channel
//! the `embed` port reads, and the host never sees it. That is the fast shape,
//! and it needs an engine that resolves the `EmbedTokens`/`KvLen` descriptor
//! ports at kernel time — `GeometryClass::DecodeEnvelope`, or the pooled
//! device-geometry class above it.
//!
//! The CUDA shell does not, and says so: its load answers `ports:
//! PortMask::NONE` and `geometry: GeometryClass::Host`, so every geometry
//! vector a fire runs on is staged from the host. Against that shell a
//! device-carried token is a value the runtime cannot know, and the fire is
//! refused by name (`EmbedTokens is not host-derivable`) rather than run on a
//! guess.
//!
//! So this one brings the TOKEN back to the host and sends it down again as a
//! host-writer cell — and only the token. Everything else the fire reads is
//! DERIVED from the KV length by pure arithmetic, so the epilogue still
//! carries it on the device exactly as `naive-baseline` does, and the runtime's
//! host shadow folds the same arithmetic to know what each fire will read.
//! One host round trip per token, which is the honest depth for a shell with
//! no descriptor-port plane, and it is entirely within what the contract
//! serves today.
//!
//! Greedy (`reduce_argmax`) rather than sampled, because the point of the gate
//! is that the same prompt produces the same continuation on every run.

use inferlet::eta::hybrid::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Optional teacher-forced continuation: comma-separated token ids. When
    /// given, each step feeds the next id here instead of the greedy pick, so
    /// two engines route the same experts (measurement only).
    #[serde(default)]
    force: String,
    /// Sampling temperature. 0.0 (the default) is exact greedy decoding, which
    /// is what the serving gate checks; a positive value switches to nucleus
    /// sampling so a long decode does not collapse into a repetition loop.
    #[serde(default)]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    /// Seed for the in-graph Gumbel noise, so a sampled run is reproducible.
    #[serde(default = "default_seed")]
    seed: u32,
}

fn default_top_p() -> f32 {
    0.95
}

fn default_seed() -> u32 {
    0x51ed
}

fn default_prompt() -> String {
    "The capital of France is".into()
}

fn default_max_tokens() -> usize {
    8
}

#[derive(Serialize)]
struct Output {
    /// The continuation, decoded.
    text: String,
    /// How many tokens it is.
    count: usize,
    /// The greedy pick at every step, before any `force` substitution: with
    /// `force` given this is the model's own prediction of each forced token.
    picked: Vec<u32>,
    /// The continuation's token ids.
    ///
    /// **LAST, AND THAT IS LOAD-BEARING.** `pie::sweep::fleet` — the fleet
    /// runner behind `pie sweep`, `pie config tune` and the contention gate —
    /// reads a lane's answer with `parse_tokens`, which takes the LAST `[` in
    /// the document. A guest that returned only prose came back as "returned
    /// no tokens", which reads as a broken program and is a program that
    /// worked; the field the runner has always looked for is this one.
    tokens: Vec<u32>,
}

/// The greedy pick over a logits row, as a one-lane `[1]` i32 cell.
fn greedy(logits: Tensor) -> Tensor {
    reshape(reduce_argmax(&logits), [1])
}

/// The pick for this step, as a one-lane `[1]` i32 cell: greedy at temperature
/// 0, nucleus sampling above it. `r` is the taken `[2]` u32 rng state
/// (`[key, ctr]`) that drives the Gumbel noise.
fn pick(r: &Tensor, temperature: f32, top_p: f32) -> Tensor {
    let logits = intrinsics::logits();
    if temperature == 0.0 {
        return greedy(logits);
    }
    reshape(nucleus_sample(&(&logits / temperature.max(1e-4)), top_p, r), [1])
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let max_tokens = input.max_tokens;
    let temperature = input.temperature;
    let top_p = input.top_p;
    let seed = input.seed;
    if !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
        return Err("top_p must be greater than 0 and at most 1".into());
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err("temperature must be finite and not negative".into());
    }
    let force: Vec<i32> = input
        .force
        .split(',')
        .filter_map(|w| w.trim().parse::<i32>().ok())
        .collect();
    let forced = |i: usize, sampled: i32| -> i32 { force.get(i).copied().unwrap_or(sampled) };
    let ws = WorkingSet::new();
    // The recurrent state, on a model that folds one: a row per sequence,
    // and this program is one sequence. On a pure-attention model there is
    // none, and an empty binding IS the attention pass (the hybrid interface
    // with `rs = []`; the host accepts that on an attention model and
    // refuses it on a folding one). Fold policy below: everything, every
    // fire — this program never buffers.
    let rs_ws: Vec<RsWorkingSet> = match model::pass_kind() {
        model::ForwardKind::Attention => Vec::new(),
        model::ForwardKind::Hybrid => vec![RsWorkingSet::new()],
        model::ForwardKind::Recurrent => {
            return Err("this program has no recurrent-only path (it needs a KV cache)".into());
        }
        model::ForwardKind::Diffusion => {
            return Err("this program decodes a token at a time; a diffusion model wants a canvas loop".into());
        }
    };
    let page_size = kv_page_size();

    if max_tokens == 0 {
        return Ok(Output {
            text: String::new(),
            count: 0,
            picked: Vec::new(),
            tokens: Vec::new(),
        });
    }
    let mut picked: Vec<u32> = Vec::with_capacity(max_tokens);

    // The model's opening (`<bos>` where it has one) before the raw text: a
    // gemma without it answers noise.
    let mut prompt = inferlet::chat::prefix();
    prompt.extend(model::encode(&input.prompt));
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("reserve KV")?;

    let pipe = Pipeline::new();
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // ── PREFILL (chunked, C-wide) ─────────────────────────────────────────
    //
    // `prefill_chunks` is the SDK's split, for the same reason every other
    // inferlet here uses it: a prompt longer than the engine's per-launch
    // token capacity has to be split, and the obvious split leaves a
    // one-token last chunk.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let mut first = 0i32;
    for &(base, end) in &prefill_chunks(n, None) {
        let len = end - base;
        let toks = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let embed_indptr = Channel::from([0u32, len]).named("embed_indptr_p");
        let positions = Channel::from_iter(base..end).named("positions_p");
        let pages = Channel::from_iter(0..max_pages).named("pages_p");
        let page_indptr = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        let w_slot = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_p");
        let w_off = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_p");
        let kv_len = Channel::from([end]).named("kv_len_p");
        let tok_out = Channel::new([1], dtype::i32).named("tok_out_p");
        let rng_p = Channel::from([seed, base]).named("rng_p");

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &embed_indptr)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            }),
            &rs_ws,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )?;
        fwd.epilogue(move || {
            let r = rng_p.take();
            tok_out.put(&pick(&r, temperature, top_p));
            rng_p.put(&(&r + iota(2)));
        });
        fwd.submit(&pipe)
            .with_context(|| format!("prefill submit @{base}"))?;
        // Every chunk samples and every sample must be drained, even the ones
        // whose token is thrown away: an epilogue put that is never taken
        // fills the ring.
        first = tok_out
            .take_host::<i32>()
            .await
            .with_context(|| format!("prefill drain @{base}"))?;
    }
    picked.push(first as u32);
    let first = forced(0, first);
    generated.push(first as u32);

    // ── DECODE (1-wide, host-driven token) ────────────────────────────────
    //
    // ONE channel is host-driven, and it is the token. Everything else the
    // fire reads — the position, the write slot and offset, the page CSR and
    // the readable extent — is DERIVED from the KV length by pure
    // arithmetic, so the epilogue carries it on the device and the runtime's
    // host shadow folds the same arithmetic
    // (`eta_compiler::eval::pareval`) to know what each fire will read.
    // That is `naive-baseline`'s decode exactly, minus the one put that makes
    // it undecidable: `tok_in.put(&token)`.
    //
    // The token cannot go the same way, and that is not a gap in this
    // program. A sampled token is device-DECIDED — the shadow commits it
    // unknown rather than guessing — so a fire that reads it needs an engine
    // resolving the `EmbedTokens` port at kernel time. The CUDA shell answers
    // `ports: PortMask::NONE` and `geometry: GeometryClass::Host`: it stages
    // every geometry vector from the host and resolves no descriptor port on
    // the device. So the token comes back to the host, and goes down again as
    // a host-writer cell — one round trip per token, which is the honest
    // depth here.
    if generated.len() < max_tokens {
        let tok_in = Channel::from([first]).named("tok_in");
        let embed_indptr = Channel::from([0u32, 1]).named("embed_indptr");
        let positions = Channel::from([n]).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from([n / page_size]).named("w_slot");
        let w_off = Channel::from([n % page_size]).named("w_off");
        let kv_len = Channel::from([n + 1]).named("kv_len");
        let tok_out = Channel::new([1], dtype::i32).named("tok_out");
        let rng = Channel::from([seed, n]).named("rng");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &embed_indptr)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            }),
            &rs_ws,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )?;
        fwd.epilogue(move || {
            // `length` is the readable extent this fire runs at, so it is
            // also the position the NEXT fire's token sits at.
            let length = kv_len.take();
            let next_length = &length + 1u32;
            let page_count = next_length.div_ceil(page_size);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page_size);
            w_off.put(&length % page_size);
            page_indptr.put(indptr(1, &page_count));
            let r = rng.take();
            tok_out.put(&pick(&r, temperature, top_p));
            rng.put(&(&r + iota(2)));
        });

        loop {
            fwd.submit(&pipe).context("decode submit")?;
            let token = tok_out.take_host::<i32>().await.context("decode drain")?;
            picked.push(token as u32);
            let token = forced(generated.len(), token);
            generated.push(token as u32);
            if generated.len() >= max_tokens {
                break;
            }
            tok_in.put([token]);
        }
    }
    pipe.close();

    Ok(Output {
        count: generated.len(),
        text: model::decode(&generated)?,
        picked,
        tokens: generated,
    })
}

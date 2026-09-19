//! Token decoding loop for GLM-OCR.
//!
//! Consumes the assembled vision-prefix `input_embeds` and the GLM-4 decoder: prefills the KV
//! cache, then takes one token per forward pass from the crate's shared token selector, which
//! applies the repetition penalty and samples, until an EOS token or `max_new_tokens`.
//! `generate_mrope` threads
//! explicit M-RoPE position ids through prefill and each decode step; `generate` uses plain
//! sequence-length offsets.
//!
//! Despite the `MtpConfig` name, no multi-token prediction happens — see `num_tokens_per_step`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MtpConfig {
    /// Intended number of tokens predicted per decoder forward pass. Currently inert: neither
    /// `generate` nor `generate_mrope` reads it, and decoding is always one token per pass.
    pub num_tokens_per_step: usize,
    /// Greedy when `false`; nucleus sampling when `true`.
    pub sample: bool,
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
}

#[cfg(not(target_arch = "wasm32"))]
impl MtpConfig {
    /// Express GLM-OCR's decoding knobs as the crate's shared decode configuration.
    ///
    /// `repeat_last_n` is 0, so the penalty reads the whole output, and the policy is the
    /// frequency form, so a token is penalised once for every time it has appeared. GLM-OCR has
    /// decoded that way since the backend landed. On a dense page the presence form leaves it
    /// repeating one phrase, because the pressure on the repeated token stops growing.
    fn decode_config(&self, max_new_tokens: usize) -> crate::models::decode::DecodeConfig {
        crate::models::decode::DecodeConfig {
            max_new_tokens,
            repeat_penalty: self.repetition_penalty,
            repeat_last_n: 0,
            repeat_penalty_policy: crate::models::decode::RepeatPenaltyPolicy::Frequency,
            temperature: if self.sample { f64::from(self.temperature) } else { 0.0 },
            top_p: f64::from(self.top_p),
            ..crate::models::decode::DecodeConfig::default()
        }
    }
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            num_tokens_per_step: 4,
            sample: false,
            top_p: 0.9,
            temperature: 0.1,
            repetition_penalty: 1.1,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use candle_core::{Device, Tensor};

    use super::super::decoder::Glm4Decoder;
    use super::MtpConfig;
    use crate::CandleOcrError;
    use crate::error::Result;
    use crate::models::decode::TokenSelector;

    /// Build a `(3, 1, 1)` M-RoPE position tensor where all three axes share the
    /// same scalar value `pos`. Used for per-step autoregressive decoding once
    /// the vision region has been consumed.
    fn make_text_step_positions(pos: u32, dev: &Device) -> Result<Tensor> {
        let buf = vec![pos, pos, pos];
        let tensor = Tensor::from_vec(buf, (3, 1, 1), dev)
            .map_err(|e| CandleOcrError::InferenceFailed(format!("Step position tensor: {}", e)))?;
        Ok(tensor)
    }

    /// Run the decoding loop with explicit M-RoPE position_ids for the prefill
    /// pass and an incrementing `(t, h, w) = (next, next, next)` triple for
    /// each generated text token.
    ///
    /// `prefill_position_ids` must be shape `(3, 1, prefix_len)` — built by the
    /// engine to encode the vision-prefixed sequence's per-token positions.
    /// `next_text_pos_start` is the position assigned to the first decoded
    /// token (== `max_position_in_prefill + 1`, computed by the engine).
    pub fn generate_mrope(
        decoder: &mut Glm4Decoder,
        input_embeds: &Tensor,
        prefill_position_ids: &Tensor,
        next_text_pos_start: u32,
        config: &MtpConfig,
        max_new_tokens: usize,
        eos_token_ids: &[u32],
    ) -> Result<Vec<u32>> {
        decoder.clear_kv_cache();
        let mut output_ids = Vec::new();

        let mut logits = decoder
            .forward_embeds(input_embeds, prefill_position_ids)
            .map_err(|e| CandleOcrError::InferenceFailed(format!("Prefill forward: {}", e)))?;
        super::super::glm_debug_tensor("prefill_logits", &logits);

        let mut next_text_pos = next_text_pos_start;
        let dev = input_embeds.device().clone();

        let mut selector = TokenSelector::new(&config.decode_config(max_new_tokens));

        while output_ids.len() < max_new_tokens {
            let last_logits = logits
                .squeeze(0)
                .map_err(|e| CandleOcrError::InferenceFailed(format!("Squeeze batch: {}", e)))?;

            let token_id = selector.next_token(&last_logits, &output_ids)?;

            if output_ids.len() < 5 && tracing::enabled!(tracing::Level::TRACE) {
                super::super::glm_debug_tensor(&format!("logits_step{}", output_ids.len()), &last_logits);
                tracing::trace!(
                    "[glm-debug] step{}: token_id={} is_eos={}",
                    output_ids.len(),
                    token_id,
                    eos_token_ids.contains(&token_id)
                );
            }

            output_ids.push(token_id);

            if eos_token_ids.contains(&token_id) {
                return Ok(output_ids);
            }

            let token_tensor = Tensor::new(&[token_id as i64], &dev)
                .map_err(|e| CandleOcrError::InferenceFailed(format!("Token tensor: {}", e)))?
                .unsqueeze(0)
                .map_err(|e| CandleOcrError::InferenceFailed(format!("Add batch: {}", e)))?;

            let token_embeds = decoder
                .embed_tokens(&token_tensor)
                .map_err(|e| CandleOcrError::InferenceFailed(format!("Embed tokens: {}", e)))?;

            let step_positions = make_text_step_positions(next_text_pos, &dev)?;

            logits = decoder
                .forward_embeds(&token_embeds, &step_positions)
                .map_err(|e| CandleOcrError::InferenceFailed(format!("Decode forward: {}", e)))?;

            next_text_pos += 1;
        }

        Ok(output_ids)
    }

    /// Run the MTP decoding loop and return generated token IDs (excluding the prefix).
    ///
    /// Algorithm:
    /// 1. Prefill: forward `input_embeds` at seqlen_offset=0 to seed KV cache
    /// 2. Per-token: sample using greedy or nucleus sampling
    /// 3. For each token: embed and forward at the current seqlen_offset
    /// 4. Stop on EOS or `max_new_tokens`
    ///
    /// Note: `config.num_tokens_per_step` is never read. Every forward pass emits exactly one
    /// token regardless of the configured value (default 4), so throughput is that of plain
    /// autoregressive decoding, not multi-token prediction.
    pub fn generate(
        decoder: &mut Glm4Decoder,
        input_embeds: &Tensor,
        config: &MtpConfig,
        max_new_tokens: usize,
        eos_token_ids: &[u32],
    ) -> Result<Vec<u32>> {
        decoder.clear_kv_cache();
        let mut output_ids = Vec::new();
        let prefix_len = input_embeds.dim(1)?;

        let mut logits = decoder
            .forward_embeds_with_offset(input_embeds, 0)
            .map_err(|e| CandleOcrError::InferenceFailed(format!("Prefill forward: {}", e)))?;

        let mut seqlen_offset = prefix_len;

        let mut selector = TokenSelector::new(&config.decode_config(max_new_tokens));

        while output_ids.len() < max_new_tokens {
            let last_logits = logits
                .squeeze(0)
                .map_err(|e| CandleOcrError::InferenceFailed(format!("Squeeze batch: {}", e)))?;

            let token_id = selector.next_token(&last_logits, &output_ids)?;

            output_ids.push(token_id);

            if eos_token_ids.contains(&token_id) {
                return Ok(output_ids);
            }

            let token_tensor = Tensor::new(&[token_id as i64], logits.device())
                .map_err(|e| CandleOcrError::InferenceFailed(format!("Token tensor: {}", e)))?
                .unsqueeze(0)
                .map_err(|e| CandleOcrError::InferenceFailed(format!("Add batch: {}", e)))?;

            let token_embeds = decoder
                .embed_tokens(&token_tensor)
                .map_err(|e| CandleOcrError::InferenceFailed(format!("Embed tokens: {}", e)))?;

            logits = decoder
                .forward_embeds_with_offset(&token_embeds, seqlen_offset)
                .map_err(|e| CandleOcrError::InferenceFailed(format!("Decode forward: {}", e)))?;

            seqlen_offset += 1;
        }

        Ok(output_ids)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use imp::{generate, generate_mrope};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::MtpConfig;
    use crate::models::decode::TokenSelector;
    use candle_core::{Device, Tensor};

    /// Scores for a three-token vocabulary, on the CPU.
    fn row(values: &[f32]) -> Tensor {
        Tensor::from_slice(values, values.len(), &Device::Cpu).expect("logits row")
    }

    #[test]
    fn glm_escapes_a_token_it_has_already_repeated() {
        // The defect behind the GLM-OCR page regression, at the seam where it happens.
        //
        // Token 1 leads token 2 by more than the penalty divides away, so one application of
        // the penalty leaves token 1 ahead and the decoder emits it again. GLM-OCR needs the
        // pressure to grow with each repeat, otherwise the step below returns token 1 however
        // long the loop has already run and the page fills with one phrase.
        let config = MtpConfig::default();
        let mut selector = TokenSelector::new(&config.decode_config(2048));
        let history = vec![1u32; 40];

        assert_eq!(
            selector.next_token(&row(&[0.1, 5.0, 3.8]), &history).expect("token"),
            2,
            "a token repeated 40 times must lose the step"
        );
    }

    #[test]
    fn glm_keeps_a_token_that_leads_and_has_not_repeated() {
        // The other side of the same step: pressure that grows with repeats must not push the
        // decoder off a leading token it has only just produced.
        let config = MtpConfig::default();
        let mut selector = TokenSelector::new(&config.decode_config(2048));

        assert_eq!(
            selector.next_token(&row(&[0.1, 5.0, 3.8]), &[1]).expect("token"),
            1,
            "one occurrence must not cost token 1 the step"
        );
    }
}

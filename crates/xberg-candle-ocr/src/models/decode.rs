//! One token-selection step, shared by the VLM OCR decode loops.
//!
//! Each backend keeps its own KV cache, position bookkeeping, forward pass and stop condition.
//! Choosing the next token from a row of logits is the same problem in all of them, and candle
//! already owns it: [`LogitsProcessor`] samples the row and [`apply_repeat_penalty`] lowers the
//! score of a token that has already appeared.
//!
//! Before this module GLM-OCR carried its own repetition penalty and nucleus sampler, while
//! DeepSeek-OCR and PaddleOCR-VL took a bare argmax with no repetition control at all. A greedy
//! decoder with no repetition control repeats one phrase until it reaches its token limit, which
//! is what DeepSeek-OCR did on a dense page once its limit rose off 128 (GH#1674).

use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::apply_repeat_penalty;
use serde::{Deserialize, Serialize};

use crate::error::{CandleOcrError, Result};

fn default_max_new_tokens() -> usize {
    4096
}

fn default_repeat_penalty() -> f32 {
    1.1
}

fn default_repeat_last_n() -> usize {
    64
}

fn default_seed() -> u64 {
    1337
}

fn default_top_p() -> f64 {
    0.9
}

/// Decoding limits and sampling controls for one OCR backend.
///
/// Every field carries a serde default, so a model `config.json` that names none of them still
/// loads and gets these values.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct DecodeConfig {
    /// Upper bound on the tokens generated for one image.
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
    /// Divides the score of a token already seen in the window. 1.0 disables the penalty.
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    /// How many of the most recent tokens the penalty reads. 0 means the whole output.
    #[serde(default = "default_repeat_last_n")]
    pub repeat_last_n: usize,
    /// Greedy when 0.0, nucleus sampling above it.
    #[serde(default)]
    pub temperature: f64,
    /// Nucleus mass. Read only when `temperature` is above 0.0.
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    /// Seed for the sampling RNG. Read only when `temperature` is above 0.0.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: default_max_new_tokens(),
            repeat_penalty: default_repeat_penalty(),
            repeat_last_n: default_repeat_last_n(),
            temperature: 0.0,
            top_p: default_top_p(),
            seed: default_seed(),
        }
    }
}

/// Chooses the next token from a row of logits, applying the repetition penalty first.
pub struct TokenSelector {
    processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TokenSelector {
    /// Build a selector for one image. Keep it for the whole decode loop: the sampling RNG is
    /// seeded once, here.
    #[must_use]
    pub fn new(config: &DecodeConfig) -> Self {
        let sampling = if config.temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            Sampling::TopP {
                p: config.top_p,
                temperature: config.temperature,
            }
        };
        Self {
            processor: LogitsProcessor::from_sampling(config.seed, sampling),
            repeat_penalty: config.repeat_penalty,
            repeat_last_n: config.repeat_last_n,
        }
    }

    /// Choose the next token.
    ///
    /// `logits` holds the scores for one position: the last dimension is the vocabulary and
    /// every dimension before it must be 1, so `(vocab)`, `(1, vocab)` and `(1, 1, vocab)` all
    /// work. `history` is the tokens generated so far, most recent last.
    ///
    /// # Errors
    ///
    /// Returns [`CandleOcrError::InvalidTensorShape`] if `logits` covers more than one position,
    /// and [`CandleOcrError::InferenceFailed`] if candle cannot apply the penalty or sample.
    pub fn next_token(&mut self, logits: &Tensor, history: &[u32]) -> Result<u32> {
        let row = Self::single_row(logits)?;
        let penalised = if self.repeat_penalty == 1.0 || history.is_empty() {
            row
        } else {
            let start = if self.repeat_last_n == 0 {
                0
            } else {
                history.len().saturating_sub(self.repeat_last_n)
            };
            apply_repeat_penalty(&row, self.repeat_penalty, &history[start..])
                .map_err(|e| CandleOcrError::InferenceFailed(format!("Repetition penalty: {e}")))?
        };
        self.processor
            .sample(&penalised)
            .map_err(|e| CandleOcrError::InferenceFailed(format!("Sampling: {e}")))
    }

    /// Flatten the scores for one position to the 1-D row candle's helpers read.
    fn single_row(logits: &Tensor) -> Result<Tensor> {
        let dims = logits.dims();
        let leading = dims.split_last().map_or(&[][..], |(_, rest)| rest);
        if dims.is_empty() || leading.iter().any(|&d| d != 1) {
            return Err(CandleOcrError::InvalidTensorShape {
                expected: "scores for one position, such as (vocab) or (1, 1, vocab)".to_string(),
                got: format!("{dims:?}"),
            });
        }
        logits
            .flatten_all()
            .map_err(|e| CandleOcrError::InferenceFailed(format!("Flatten logits: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn row(values: &[f32]) -> Tensor {
        Tensor::from_slice(values, values.len(), &Device::Cpu).expect("logits row")
    }

    fn selector(config: DecodeConfig) -> TokenSelector {
        TokenSelector::new(&config)
    }

    #[test]
    fn greedy_selection_takes_the_highest_score_when_nothing_repeats() {
        let mut s = selector(DecodeConfig::default());
        assert_eq!(s.next_token(&row(&[0.1, 0.9, 0.3]), &[]).expect("token"), 1);
    }

    #[test]
    fn a_token_in_the_window_loses_to_one_that_is_not() {
        // Token 1 leads on raw score. Having just produced it, the penalty must hand the step to
        // token 2. This is the step that stops a greedy decoder repeating one token forever.
        let mut s = selector(DecodeConfig {
            repeat_penalty: 2.0,
            ..DecodeConfig::default()
        });
        assert_eq!(s.next_token(&row(&[0.1, 0.9, 0.8]), &[1]).expect("token"), 2);
    }

    #[test]
    fn a_negative_score_is_pushed_further_down_by_the_penalty() {
        // Both scores are negative, the branch where the penalty multiplies instead of dividing.
        // Token 1 wins untouched, and must lose once it is in the history.
        let mut plain = selector(DecodeConfig {
            repeat_penalty: 1.0,
            ..DecodeConfig::default()
        });
        assert_eq!(plain.next_token(&row(&[-0.5, -0.4]), &[1]).expect("token"), 1);

        let mut penalised = selector(DecodeConfig {
            repeat_penalty: 2.0,
            ..DecodeConfig::default()
        });
        assert_eq!(penalised.next_token(&row(&[-0.5, -0.4]), &[1]).expect("token"), 0);
    }

    #[test]
    fn a_penalty_of_one_leaves_the_choice_alone() {
        let mut s = selector(DecodeConfig {
            repeat_penalty: 1.0,
            ..DecodeConfig::default()
        });
        assert_eq!(s.next_token(&row(&[0.1, 0.9, 0.8]), &[1]).expect("token"), 1);
    }

    #[test]
    fn the_window_forgets_a_token_that_left_it() {
        // Token 1 was generated, then pushed out of a two-token window, so it is eligible again.
        let mut s = selector(DecodeConfig {
            repeat_penalty: 2.0,
            repeat_last_n: 2,
            ..DecodeConfig::default()
        });
        assert_eq!(s.next_token(&row(&[0.1, 0.9, 0.8]), &[1, 5, 6]).expect("token"), 1);
    }

    #[test]
    fn a_window_of_zero_penalises_the_whole_history() {
        // GLM-OCR's policy. The same history under a two-token window leaves token 1 eligible,
        // which is the test above; here token 1 is still penalised and token 2 wins.
        let mut s = selector(DecodeConfig {
            repeat_penalty: 2.0,
            repeat_last_n: 0,
            ..DecodeConfig::default()
        });
        assert_eq!(s.next_token(&row(&[0.1, 0.9, 0.8]), &[1, 5, 6]).expect("token"), 2);
    }

    #[test]
    fn scores_from_a_batched_model_select_the_same_token_as_a_flat_row() {
        // GLM-OCR hands over (vocab); DeepSeek-OCR and PaddleOCR-VL hand over (1, 1, vocab).
        let flat = row(&[0.1, 0.9, 0.8]);
        let batched = flat.reshape((1, 1, 3)).expect("reshape");
        let mut from_flat = selector(DecodeConfig {
            repeat_penalty: 2.0,
            ..DecodeConfig::default()
        });
        let mut from_batched = selector(DecodeConfig {
            repeat_penalty: 2.0,
            ..DecodeConfig::default()
        });
        assert_eq!(
            from_flat.next_token(&flat, &[1]).expect("token"),
            from_batched.next_token(&batched, &[1]).expect("token")
        );
    }

    #[test]
    fn scores_for_more_than_one_position_are_rejected() {
        // A caller that forgets to narrow a prefill block to its last position gets an error
        // rather than a token chosen from the wrong row.
        let block = Tensor::from_slice(&[0.1f32, 0.9, 0.8, 0.2, 0.3, 0.4], (2, 3), &Device::Cpu).expect("block");
        let mut s = selector(DecodeConfig::default());
        let err = s.next_token(&block, &[]).expect_err("two positions must be rejected");
        assert!(
            matches!(err, CandleOcrError::InvalidTensorShape { .. }),
            "expected an invalid-shape error, got {err:?}"
        );
    }

    #[test]
    fn a_config_naming_no_fields_deserialises_to_the_defaults() {
        let parsed: DecodeConfig = serde_json::from_str("{}").expect("empty config");
        assert_eq!(parsed, DecodeConfig::default());
        assert_eq!(parsed.max_new_tokens, 4096);
        assert_eq!(parsed.repeat_penalty, 1.1);
        assert_eq!(parsed.repeat_last_n, 64);
        assert_eq!(parsed.temperature, 0.0);
    }
}

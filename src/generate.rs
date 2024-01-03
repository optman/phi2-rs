use crate::{
    cache::Cache,
    model::{Dtype, Params, PhiLM},
};
use dfdx::prelude::*;
use rand::{rngs::StdRng, Rng};
use std::{collections::HashSet, fmt::Debug, io::Write};
use tokenizers::tokenizer::Tokenizer;

pub struct GenerateOption {
    pub greedy: bool,
    pub use_cache: bool,
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub max_seq_len: usize,
    pub pos_scale: usize,
    pub verbose: bool,
    pub cache_size: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for GenerateOption {
    fn default() -> Self {
        Self {
            greedy: false,
            use_cache: true,
            top_k: 40,
            top_p: 0.95,
            temperature: 0.8,
            max_seq_len: 1_000_000,
            pos_scale: 1,
            verbose: false,
            cache_size: 256,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

pub fn generate<E: Dtype, P: Params, D: Device<E>>(
    tokenizer: &Tokenizer,
    rng: &mut StdRng,
    dev: &D,
    m: &PhiLM<E, P, D>,
    prompt: &str,
    gen_num: usize,
    opt: &GenerateOption,
    eos_token: &str,
) -> (usize, String)
where
    D: Device<f32>,
{
    if opt.verbose {
        print!("{:}", prompt);
        std::io::stdout().flush().unwrap();
    }

    let eos_token = *tokenizer.get_vocab(true).get(eos_token).unwrap();

    let prompt = tokenizer.encode(prompt, true).unwrap();
    let mut seq: Vec<usize> = prompt.get_ids().iter().map(|c| *c as usize).collect();

    let mut pos = 0;
    let seq_len = seq.len();
    let x = dev.tensor_from_vec(seq.clone(), (seq_len,));

    let cache = if opt.use_cache {
        Some(Cache::new(m.params().layers(), opt.cache_size))
    } else {
        None
    };

    let mut x_len = seq_len;
    let (mut y, mut cache) = m.try_forward(x, pos, opt.pos_scale, cache).unwrap();
    pos += if cache.is_some() { x_len } else { 0 };

    let mut early_break = None;
    for i in 0..gen_num {
        if seq.len() >= opt.max_seq_len {
            early_break = Some(i);
            break;
        }
        let probs = y.select(dev.tensor(x_len - 1)).to_dtype::<f32>();
        let probs = if opt.repeat_penalty == 1.0 {
            probs
        } else {
            apply_penalty(
                probs,
                opt.repeat_penalty,
                &seq[seq.len().saturating_sub(opt.repeat_last_n)..],
            )
        };
        let next_idx = if opt.greedy {
            greedy(probs.as_vec())
        } else {
            let probs = (probs / opt.temperature).softmax().to_dtype();
            topk(probs.as_vec(), opt.top_p, opt.top_k, rng)
        };

        if next_idx as u32 == eos_token {
            early_break = Some(i);
            break;
        }

        seq.push(next_idx);

        if opt.verbose {
            if let Some(text) = tokenizer.id_to_token(next_idx as u32) {
                let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                print!("{text}");
            }
            std::io::stdout().flush().unwrap();
        }

        //next round
        let (x, pos_inc) = if cache.is_some() {
            (dev.tensor_from_vec(vec![next_idx], (1,)), 1)
        } else {
            (dev.tensor_from_vec(seq.clone(), (seq.len(),)), 0)
        };
        x_len = x.shape().0;
        (y, cache) = m.try_forward(x, pos, opt.pos_scale, cache).unwrap();
        pos += pos_inc;
    }

    (
        early_break.unwrap_or(gen_num),
        tokenizer
            .decode(&seq.into_iter().map(|c| c as u32).collect::<Vec<_>>(), true)
            .unwrap(),
    )
}

fn greedy<E: PartialOrd + Debug>(probs: Vec<E>) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .map(|x| x.0)
        .unwrap()
}

fn topk(probs: Vec<f32>, top_p: f32, top_k: usize, rng: &mut StdRng) -> usize {
    let mut probs: Vec<_> = probs.into_iter().enumerate().collect();

    probs.sort_unstable_by(|(_, a), (_, b)| b.total_cmp(a));

    let mut choices = top_k;
    let mut total = 0.0;
    for (i, &(_, p)) in probs.iter().enumerate().take(top_k) {
        total += p;
        if total >= top_p {
            choices = i + 1;
            break;
        }
    }

    let prob: f32 = rng.gen_range(0.0..total);
    let mut accum = 0.0;
    for &(i, p) in probs.iter().take(choices) {
        accum += p;
        if accum >= prob {
            return i;
        }
    }

    unreachable!()
}

pub fn apply_penalty<S: Shape, D: Device<f32>>(
    probs: Tensor<S, f32, D>,
    penalty: f32,
    context: &[usize],
) -> Tensor<S, f32, D> {
    let shape = *probs.shape();
    let dev = probs.dev();
    let mut probs = probs.as_vec();

    let context: HashSet<_> = context.iter().collect();

    for (token_id, p) in probs.iter_mut().enumerate() {
        if context.contains(&token_id) {
            if *p >= 0. {
                *p /= penalty;
            } else {
                *p *= penalty;
            }
        }
    }

    dev.tensor_from_vec(probs, shape)
}

pub fn print_metrics(elapsed: std::time::Duration, num_tokens_generated: usize) {
    let elapsed_s = elapsed.as_secs_f64();
    let tokens_per_s = num_tokens_generated as f64 / elapsed_s;
    let ms_per_token = 1000.0 * elapsed_s / num_tokens_generated as f64;

    println!();
    println!(
        "*Generated {} tokens in {:.3?} ({tokens_per_s:.3} tokens/s, {ms_per_token:.0} ms/token)*",
        num_tokens_generated, elapsed
    );
}

use anyhow::Result;
use dfdx::prelude::*;
use rand::prelude::{SeedableRng, StdRng};

mod model;
use model::PhiLM;

mod nn_loader;

mod tensor_loader;
use tensor_loader::SafeTensorLoader;
use tokenizers::Tokenizer;

mod cache;
mod config;
mod rotary;
use config::ConfigV2;
mod generate;
use generate::{generate, print_metrics, GenerateOption};

use clap::Parser;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(long, default_value_t = false)]
    greedy: bool,

    #[arg(short, default_value_t = 4096)]
    num_tokens: usize,

    #[arg(
        long,
        short,
        default_value = "Write a detailed analogy between mathematics and a lighthouse."
    )]
    prompt: String,

    #[arg(long, short)]
    model_path: String,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let mut rng = StdRng::seed_from_u64(args.seed);

    let root = args.model_path;
    let paths = vec![
        format!("{root}/model-00001-of-00002.safetensors"),
        format!("{root}/model-00002-of-00002.safetensors"),
    ];

    let tokenizer_model = format!("{root}/tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_model).map_err(anyhow::Error::msg)?;

    let loader = SafeTensorLoader::new(paths.into_iter().map(|f| f.to_owned()).collect())?;

    let dev = AutoDevice::default();

    let cfg = ConfigV2 {};

    let m = PhiLM::<f32, _, _>::load_model(cfg, &dev, &loader)?;

    let bench = true;

    let gen_opt = GenerateOption {
        use_cache: true,
        verbose: true,
        greedy: args.greedy,
        cache_size: 4096,
        ..Default::default()
    };

    let start = std::time::Instant::now();

    let (gen_num, _) = generate(
        &tokenizer,
        &mut rng,
        &dev,
        &m,
        &args.prompt,
        args.num_tokens,
        &gen_opt,
    );
    if bench {
        print_metrics(start.elapsed(), gen_num);
    }

    Ok(())
}

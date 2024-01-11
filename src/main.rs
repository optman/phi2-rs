use anyhow::Result;
use dfdx::dtypes::f16;
use dfdx::prelude::*;
use rand::prelude::{SeedableRng, StdRng};

mod model;
use model::Mistral;

mod nn_loader;

mod rmsnorm;

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

    #[arg(short, default_value_t = 2048)]
    num_tokens: usize,

    #[arg(long, default_value_t = false)]
    disable_cache: bool,

    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    #[arg(long, default_value_t = 40)]
    top_k: usize,

    #[arg(long)]
    temperature: Option<f32>,

    #[arg(long, default_value_t = false)]
    bench: bool,

    #[arg(long, default_value_t = 1)] //cache_size / seq_len(training)
    pos_scale: usize,

    #[arg(long, default_value_t = 4096)]
    cache_size: usize,

    #[arg(long, short, default_value = "Once upon a time,")]
    prompt: String,

    #[arg(long, short)]
    model_path: String,

    #[arg(long, default_value_t = 16)]
    split: usize,
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

    let eos_token = "</s>";

    let dev = AutoDevice::default();
    let dev2 = Cpu::default();

    let cfg = ConfigV2 {};

    let m = Mistral::<f16, _, _, _>::load_model(cfg, &dev, &dev2, &loader, args.split)?;

    let gen_opt = GenerateOption {
        use_cache: !args.disable_cache,
        verbose: true,
        top_k: args.top_k,
        top_p: args.top_p,
        temperature: args.temperature,
        pos_scale: args.pos_scale,
        cache_size: args.cache_size,
        ..Default::default()
    };

    let start = std::time::Instant::now();

    let (gen_num, _) = generate(
        &tokenizer,
        &mut rng,
        &dev,
        &dev2,
        &m,
        &args.prompt,
        args.num_tokens,
        &gen_opt,
        eos_token,
    );
    if args.bench {
        print_metrics(start.elapsed(), gen_num);
    }

    Ok(())
}

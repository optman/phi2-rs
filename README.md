## phi2-rs

load Mistral 7B LLM model with [dfdx](https://github.com/coreylowman/dfdx).

Because dfdx don't support bf16 type yet, we load model weighs in mix precision f16 and f32, then it can be fit in 16G gpu memory. 


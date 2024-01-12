## phi2-rs

load Mixtral 7Bx2 LLM model with [dfdx](https://github.com/coreylowman/dfdx).

Because dfdx not support bf16 type yet, we load model weights in f32, that would not fit in 16G gup memory.

So we split the model, first half on gpu, the other on cpu. It is slow, but at least it can be loaded.




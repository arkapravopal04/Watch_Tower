# DoRA protocol

The implementation uses **DoRA** in place of traditional LoRA. Kaggle is
the intended GPU runtime; local CPU runs check engineering correctness.

For a PyTorch weight `W` of shape `(out, in)`, define:

```text
V = W + (alpha / rank) B A
g = m / row_norm(V)
D = diag(g) V
ordinary DoRA weight = D
protected DoRA weight = W + (D - W)(I - Q Q^T)
```

`A`, `B`, and the output-row magnitude vector `m` are trainable FP32 parameters.
`m` starts at the base row norms; `A` is Kaiming initialized; `B` starts at zero.
Base weights, bias and Q are frozen. Empty Q is ordinary DoRA. Protected and
random-Q methods project **the entire update**, including magnitude changes.
Projecting A alone would leave `(diag(g)-I) W Q` unconstrained.

This is an input-projected DoRA variant, not a claim that standard DoRA preserves
old-task activations. Its final row norms need not equal `m` after projection.
The complete update is no longer constrained to rank `r`: the input-space bound
is `rank(delta_W) <= min(out, in-k)`. The original LoRA rank bound does not apply.
Local protection is not a guarantee that model answers remain unchanged.

Row norms are computed with gradients detached, following the reduced-overhead
DoRA formulation. Dense direction temporaries are chunked by output row and
discarded. The forward path projects inputs and uses factorized directional
updates, never a dense input projector. It also needs an FP32 base projection
for the magnitude contribution. This costs more than ordinary LoRA: measure
memory, throughput and checkpointing on Kaggle before scheduling the study.

Exports contain `A_eff`, `B`, scale, Q, **and row gains**. They must be applied
to the exact recorded base. Resume files instead preserve raw A/B/m, Q,
optimizer/scaler, data order and RNG state. A stock raw-factor LoRA merge cannot
reconstruct these artifacts. Merging accumulates the complete update in FP32
before storage casting and reports both intended and rounded residuals.

A verified merge is published as a **weight patch**: the stored q/v weights of
the merged model (relative to the pinned M0, cumulative across stages), its
full tensor hashes, the effective adapter and the merge validation. Loading
copies the stored tensors onto M0 and re-checks every hash, so M_A and every
candidate are bit-identical across Kaggle sessions without 8 GB model copies.

The tests cover the standard DoRA equation, initial identity, A/B/m gradients,
frozen bases, magnitude-only protection changes, Adam steps, effective export,
FP16 rounding and exact CPU resume, and match a version-pinned PEFT 0.18.0
`use_dora=True` layer (forward, gradients, AdamW steps, merge). GPU precision
and runtime are measured by the pilot (NB2) before any study run.

Sources: [DoRA paper](https://arxiv.org/abs/2402.09353),
[PEFT DoRA option](https://huggingface.co/docs/peft/package_reference/lora).

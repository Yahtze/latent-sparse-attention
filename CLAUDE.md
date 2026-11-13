@AGENTS.md

Canonical repo instructions live in `AGENTS.md` (imported above), shared with all coding agents.
Claude-specific instructions go below this line.

## Working on This Machine

The dev laptop is macOS with no GPU, and the experiments run on a rented GPU box.
Do not install dependencies here - `uv run --extra training` fails because torch is pinned to a cu129 build with no macOS wheel, and the user asked for no installs.
`make format` and `make check` therefore cannot run here, so say so when reporting.
Compile-check with `uv run --no-project --python 3.12 python -m py_compile <files>`, since the system `python3` is 3.7 and rejects `type` statements.
Delete the `__pycache__` directories that leaves behind.
Hand-wrap code to 110 columns, since ruff cannot run.

The repo is jj in colocated mode and the `main` bookmark tracks `main@origin`.
To push: `jj bookmark set main -r @-`, then `jj git push --bookmark main`.
The user has pushed straight to `main` when they said to, so push only when they say so.

## The Top-k Inference Sweep

The experiment reads the released N-LSA checkpoints at a range of `attention.top_k` values and plots accuracy against inference `top_k`.
All ten released checkpoints are one model - the 0.8B GPT-2-vocabulary hybrid, three Gated DeltaNet layers to one attention layer, at `step-5866` of a 131,072-token decay.
There is no smaller released N-LSA, and `340M-gpt2` is a dense flavor for the DSA ablations only.

### Training Facts

- The shared trunk `lsam8-fineweb` trained at `seq_len: 4096` and `top_k: 1024`, for 95,368 steps of about 1M tokens, and its stable-stage checkpoint is `step-94414`.
- Each decay branches from that trunk, with `seq_len: 131072`, 5,866 steps of about 2M tokens, `decay_ratio: 0.97` and its own `top_k`.
- The number in a checkpoint name is its decay `top_k` - `decay128k-lsa-{1024,2048,4096}`.
- `decay_fineweb128k_lsa.yaml` sets `top_k: 2048` and the `evaluate_released_lsa_k*.yaml` files override it.
- `top_k` is a config value read at run time and not stored in the weights, and `top_p` defaults to `2 * top_k // bucket_size` when unset.
- The defaults are window 512, sink 4, bucket size 8, bucket cap 16, block 512 and pool 4.
- Which corpus the decay reads, and whether the paper's trunk is shared by every baseline, were not checked.

### Commands and Files

- `configs/training/native/qwen35_0_8b_gpt2/sweep_released_lsa_k{1024,2048,4096}.yaml` sweep `attention.top_k` over 256 to 8,192 in powers of 2.
- The dense baseline is `evaluate_released_dense.yaml`, which files under `decay128k-dense`.
- `lsa native evaluate <yaml> --suite benchmark --tasks ruler-32k --gpus N` runs one job per sweep point, filed under `<records>/<run.name>/<preset>/` and refusing `--label`.
- `lsa native curve <records> --presets ruler-32k,ruler-64k,ruler-128k` writes one `top_k_<preset>.png` each, with a ring at the trained `top_k` read off the run name's trailing number.
- The RULER files are rendered with `lsa native prepare-ruler --lengths 32768 65536 131072`, which needs internet and uses the GPT-2 tokenizer, so only the configs under `qwen35_0_8b_gpt2/` can score them.
- Presets are `ruler` (4k, 8k, 16k), `ruler-4k`, `ruler-32k`, `ruler-64k` and `ruler-128k`.

### Things That Surprised

- The benchmark suite took exactly one experiment per command before the sweep support, because it unpacked `[experiment] = load(...)`.
- The default context is `max(seq_len, longest RULER cell)`, which is 131,072 for these checkpoints even at `ruler-4k`.
- Generation has no decode cache and recomputes the whole prefix per token, and `build_ruler` sets `max_gen_toks: 128` with no stop strings.
- A point therefore costs about 1,300 documents times 128 forwards, roughly 1.7e14 times T FLOPs at least, which is hours on an H100 at 32k and days on an A40 at 128k.
- The levers are `--max-gen-toks`, `--limit`, fewer `top_k` values and fewer lengths, and they must be the same for every point.
- At `ruler-4k` any `top_k` of about 3,600 or more reads every key, so 4,096 and 8,192 coincide and the long presets are where `top_k` binds.
- The `ruler-32k` preset comment says "past the length the models trained at", which is true of the 4k trunk and not of the 131k decay.

### Kernels and GPUs

- The released configs set `fast_read: true`, and the CUDA read and rank kernels run only on compute capability exactly (9, 0).
- Every other GPU, an A40 included, falls back to the Triton `gather_sparse` and `lsa_select` with nothing to configure.
- The read-table path that builds a sequence-by-sequence bool mask is taken only when the fast read is skipped, for example for non-bf16 queries, and at 128k that mask alone is about 17 GB.
- Each run needs one `attention.top_k`, set per run, and an `early_budget` is a separate flat-design-only knob the sweep does not use.

### Rented Machines

- Every evaluate command submits through submitit, so a box with no Slurm needs a site file with `launcher: local`, copied from `configs/site/local.yaml`.
- The local launcher refuses multi-node jobs and reservations.
- Whether a sweep's six jobs run one after another or all at once on one GPU was not checked, so run one point at a time with an override such as `attention.top_k=2048`.
- Jobs read the hub caches offline, so fetch the checkpoints and datasets first.

### Not Verified

- None of the new code or tests has run, including the `make check` gates.
- Sweep run names are probably `released-decay128k-lsa-2048_seq_len131072_top_k256`, since the base decay yaml also sweeps `data.seq_len`, and the tests match only on `_top_k<k>`.
- The Triton fallback has not been checked against the paper's `ruler-4k` score of 52.2, which the repo reproduces as 52.5 for the `k2048` checkpoint, so run that first on the new box.
- Memory at 128k, evaluation query dtype, and the Gated DeltaNet dependencies on a non-Hopper GPU were not measured.
- Throughput and rental price figures in the discussion were estimates from spec sheets, not measurements.

# Native 0.8B hybrids at the GPT-2 vocabulary

The runs of the paper's native comparison (Section 4.4), and the configs that score the released checkpoints.
Every file here overlays `base.yaml`, and `phase/` holds the pre-train and decay overlays the methods share.
The README at the repository root lists which file trains which method.

## Presets

`lsa native evaluate --suite benchmark --tasks <preset>` takes a preset, a task, or several of either joined by commas.

| Preset | Tasks |
|---|---|
| `core` | hellaswag, arc_easy, arc_challenge, piqa, winogrande, lambada_openai, wikitext, sciq |
| `ruler` | the 13 RULER tasks at 4,096, 8,192 and 16,384 tokens |
| `ruler-4k` | the same 13 at 4,096 tokens, the paper's RULER column |
| `ruler-32k` | the same 13 at 32,768, past the length the models trained at |
| `ruler-64k`, `ruler-128k` | the same 13 at 65,536 and 131,072, the length the released 128k decays finished at |

A RULER cell is also a task of its own, `ruler_<task>_<length>`, for example `ruler_niah_single_3_4096`.
The 13 tasks are `niah_single_1`, `niah_single_2`, `niah_single_3`, `niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3`, `niah_multiquery`, `niah_multivalue`, `vt`, `cwe`, `fwe`, `qa_hotpot` and `qa_squad`, 100 documents each.
The paper's RULER column is the mean of all 13 at 4,096 tokens.

## RULER data

The benchmark reads fixed prompt files, so prepare them once.
Pass nothing to choose a tokenizer: `lsa native prepare-ruler` counts every prompt in the models' own GPT-2 tokenizer, which is the tokenizer the paper's evaluation sized them by.
The files are therefore for the GPT-2 family only, and a model with another tokenizer, such as the Qwen3.5 family under `configs/training/native/qwen35_0_8b`, would see mis-sized prompts.
Score RULER with the configs in this folder.

```bash
lsa --site mycluster native prepare-ruler --lengths 4096 8192 16384
```

The command fetches its sources where it is typed, because a compute node has no egress, and then submits one CPU job.
It needs lm-eval's `ruler` extra, which the `training` extra carries.

The files land at `$LSA_ROOT/data/benchmarks-gpt2/ruler/<task>_<length>.jsonl`.
Set `LSA_RULER_ROOT` to read and write another root, which holds the same `ruler/` folder.
An existing file is kept, and `--force` regenerates it.

Do not use `lsa data prepare ruler` for this.
It renders NVIDIA's own generators, which draw other needles and haystacks, and over the seven needle tasks other than `niah_multikey_3`, the same checkpoint scores about 5 points lower on their files, mostly on `niah_single_3`.
Those files are what the served accuracy suite reads, under the tokenizer of the model it serves.

Grading is RULER's string match, and every task scores the fraction of its references found in the reply, as the paper's native column did.
The served accuracy suite scores the two QA tasks by official RULER instead, as 1 when any one reference is found.
Under that rule the released checkpoint's `qa_squad` rises from .371 to .410 and the mean of 13 from 52.5 to 52.8.

## Reproducing the paper's RULER column

```bash
lsa --site mycluster native prepare-ruler --lengths 4096
lsa --site mycluster native evaluate configs/training/native/qwen35_0_8b_gpt2/evaluate_released_lsa_k2048.yaml \
  --suite benchmark --tasks ruler-4k --gpus 1
```

The paper reports 52.2 for the mean of the 13 tasks.
The released `decay128k-lsa-2048` scores 52.5 here, graded as the paper graded it, and 52.8 with any reference counting for the QA tasks.
A cell holds 100 documents, so its standard error is up to 5 points, and only the mean is worth comparing.
The preset is the run's folder name, so it stays short: a run is filed under the `--tasks` value as typed, and a list of the 13 cells spelled out is over the 255 characters a file name holds.

## Reading at Another top_k

The released N-LSA checkpoints were each trained at one `top_k`, and `top_k` is a read-time size, so a checkpoint can be read at others.
`sweep_released_lsa_k{1024,2048,4096}.yaml` read one checkpoint at 256 to 8,192 keys, and the dense checkpoint is scored once.

```bash
for k in 1024 2048 4096; do
  lsa --site mycluster native evaluate configs/training/native/qwen35_0_8b_gpt2/sweep_released_lsa_k$k.yaml \
    --suite benchmark --tasks ruler-4k --gpus 1
done
lsa --site mycluster native evaluate configs/training/native/qwen35_0_8b_gpt2/evaluate_released_dense.yaml \
  --suite benchmark --tasks ruler-4k --gpus 1
lsa native curve $LSA_ROOT/training/native/benchmark --presets ruler-4k --baseline decay128k-dense
```

A sweep submits one job per point and files each under its run name, such as `released-decay128k-lsa-2048_top_k512`, so `--label` is refused.
`top_p` stays unset, so every point selects twice the buckets its keys fill, 64 buckets at 256 keys and 2,048 at 8,192.
`lsa native curve` writes one figure for each of `--presets`, as `top_k_<preset>.png` beside the records.
A figure has one line for each checkpoint, named for the `top_k` it trained at and ringed at that point, and the dense checkpoint as a dashed line.
It needs matplotlib, which the `analysis` dependency group carries.

A `ruler-4k` prompt is 4,096 keys, so past about 3,580 long-range keys its last query already reads every key and the points at 4,096 and 8,192 coincide.
The released checkpoints finished a 131,072-token decay, so the long presets are the ones where a budget of 256 to 8,192 keys binds.
Prepare their files with `lsa native prepare-ruler --lengths 32768 65536 131072` and swap `--tasks ruler-4k` for `ruler-32k`, `ruler-64k` or `ruler-128k`, and list them in `--presets` of `lsa native curve`, which draws a figure for each.

## Not provided

SWDE and FDA, the two extraction tasks of the paper's Table 4, have no task or preset here, and their prompt files are not part of this repository.

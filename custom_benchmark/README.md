# custom_benchmark

`custom_benchmark` is a dataset plugin registered in BusterX's
`DATASET_REGISTRY`. It lets you build your own video real/fake evaluation set
**purely by organizing files into directories** — no code changes required.

The implementation lives in `busterx/datasetpp/custom_benchmark.py` and is
registered under the name `custom_benchmark`.

______________________________________________________________________

## 1. Directory layout

Treat this directory (`./custom_benchmark/`) as the dataset root. **Each first-level subdirectory under the root is treated as one
subclass**, and the directory name becomes the subclass name. All valid video
files found recursively inside it are used as samples of that subclass.

```
custom_benchmark/
├── real/          # subclass name: real     -> solution = "A" (real)
│   ├── <hash1>.mp4
│   └── <hash2>.mov
├── sora/          # subclass name: sora     -> solution = "B" (AI generated)
│   ├── <hash1>.mp4
│   └── <hash2>.mkv
├── kling/         # subclass name: kling    -> solution = "B" (AI generated)
│   └── ...
└── your_model/    # any custom subclass
    └── ...
```

Conventions:

- A subdirectory named `real` (case-insensitive) is labeled `A` (real).
  See `REAL_SUBCAT_NAMES` in `busterx/datasetpp/custom_benchmark.py`.

- Any other subdirectory name is labeled `B` (fake / AI generated).

- Subdirectories whose names start with `.` or `_` (e.g. `.cache`, `_tmp`)
  are ignored.

- A subdirectory that contains **no valid video file** is skipped silently and
  is not registered as a subclass.

- Video files are detected by extension (case-insensitive). Currently
  supported extensions:

  ```
  .mp4 .mkv .mov .avi .webm .flv .wmv .m4v
  .mpg .mpeg .ts .3gp .vp9 .vp8 .av1
  ```

  To extend this list, edit `VIDEO_EXTS` at the top of
  `busterx/datasetpp/custom_benchmark.py`.

### Recommended: name videos by their hash

It is **strongly recommended** to rename every video to its content hash
(e.g. SHA-256 / MD5 / BLAKE3) before placing it under a subclass directory,
for example `9f86d081884c7d65...mp4`. Reasons:

- **Stable & deterministic IDs.** The filename uniquely identifies the
  content; renaming, copying or moving across machines does not change the
  identity of a sample.
- **Built-in deduplication.** The same video accidentally added twice (or
  appearing across different subclasses) is trivial to spot — identical
  hashes mean identical bytes.
- **Frame-cache friendliness.** When `--input_mode image` is used, frames
  are extracted into `cache_dir`. Hash-based filenames keep cache entries
  collision-free and reusable across runs / machines.
- **Safe filenames.** Hash strings contain only `[0-9a-f]`, avoiding issues
  with spaces, unicode characters, or shell-unfriendly symbols that some
  source filenames carry.
- **Privacy / leak-safety.** Original filenames often leak the source
  (YouTube IDs, model prompts, internal project names). A hash reveals
  nothing about provenance.

A minimal one-liner to rename a directory of videos to their SHA-256 hash:

```bash
for f in *.mp4; do
  h=$(sha256sum "$f" | awk '{print $1}')
  mv -- "$f" "${h}.mp4"
done
```

______________________________________________________________________

## 2. Build the dataset (`build_dataset`)

See the `build_custom_benchmark` target in `Makefile`:

```bash
make build_custom_benchmark
```

`--input_mode` accepts `video` or `image`:

- `video`: feed video paths directly, no frame extraction.
- `image`: extract frames at `--sample_fps` into `cache_dir`, then feed them
  as images.

Each sample carries a `subcat` field that records the subclass it belongs to,
which is used later for grouped metrics.

______________________________________________________________________

## 3. Inference (`infer`)

See the `infer_custom_benchmark` target in `Makefile`. After inference you
will get a jsonl file in which each line contains at least `response`,
`solution` and `subcat` — that is what the evaluator consumes.

______________________________________________________________________

## 4. Metrics (`calc_metric`)

```bash
uv run --no-sync busterx eval \
    --dataset_pp custom_benchmark \
    --result_file ./result/infer_custom_benchmark/<TIME>.jsonl
```

`calc_metric` will:

1. Group samples by the `subcat` field.

1. Compute and print **per-subclass accuracy**.

1. Aggregate every non-`real` subclass into a single **Fake ACC** column.

1. Compute the **Overall ACC** across all samples.

1. Render the result with `tabulate`. Real subclasses are listed first
   (alphabetically), followed by fake subclasses (alphabetically), then the
   `Fake ACC` and `Overall ACC` columns. Example:

   ```
   +--------+--------+--------+-------+------------+---------------+
   |  real  |  kling |  sora  | ...   |  Fake ACC  |  Overall ACC  |
   +========+========+========+=======+============+===============+
   | 92.50  | 76.10  | 81.30  | ...   |   78.70    |     83.20     |
   +--------+--------+--------+-------+------------+---------------+
   ```

   Headers are generated dynamically from the subclasses actually scanned —
   add or remove a subdirectory and the next evaluation reflects it
   automatically, no code changes required.

1. Also print a LaTeX-friendly one-liner:

   ```
   & 92.5 & 76.1 & 81.3 & 78.7 & 83.2 \\
   ```

______________________________________________________________________

## 5. Quick recipe for adding new data

1. Create a new subdirectory under this directory; the directory name becomes
   the subclass name (`real` for real videos, anything else is treated as
   fake).
1. Drop video files into it (any of the extensions listed above).
   **Recommended:** rename each file to its content hash first.
1. Re-run `make build_custom_benchmark` to regenerate the jsonl.
1. Run inference + `make` / `busterx eval` to see metrics that include the
   new subclass.

No Python code changes are needed.

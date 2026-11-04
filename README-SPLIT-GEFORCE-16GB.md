# Running the all-cards split engine on 2x GeForce 16 GB (host-memory expert cache)

This branch (`split-geforce-16gb`, based on `v0.2.10` / `ffa80d7`) carries two small, opt-in patches that let the
all-cards split engine start and answer on two 16 GB GeForce cards, where the routed experts do **not** fit in VRAM.
With none of the new environment variables set, behaviour is exactly as upstream. The license (MIT) and the original
authors' credits are unchanged.

Tested once, on one machine, with one model (see "Numbers"). Treat this as a field report, not a supported configuration.

## What changed

1. **`strata_control/split_engine.py`: environment switches for the launcher rule (default unchanged).**
   - `STRATA_SPLIT_SKIP_VRAM_CHECK=1`: `decide()` no longer refuses when the experts do not fit in the cards' memory;
     it records "experts larger than the cards' memory (host-memory expert cache)" as untested instead
     (still needs `STRATA_SPLIT_ALLOW_UNTESTED=1` as before).
   - `STRATA_SPLIT_FNX_ARENA` and `STRATA_SPLIT_FNX_EXPERT_RAM_GIB`: replace the values `split_config()` forces
     (`FNX_ARENA=0`, `FNX_EXPERT_RAM_GIB=off`), so the engine's existing host-RAM expert cache and arena can be used.
2. **`src/split/src/model.cpp`: `FNX_POOL_MIN_RESIDENT=N` (default 0 = upstream).** `choose_residency()` never takes a
   size pool below N resident experts.

## Why it was needed

- **Launcher rule.** The engine already supports keeping a pinned host copy of the experts and filling VRAM slots from
  it (`choose_residency()`, the "VRAM short" loop), but the launcher refuses to start when the experts do not fit in
  VRAM, and forces the arena and the expert-RAM cache off.
- **Pool floor.** IQ3_S mixes quantization types per layer, so the experts fall into 7 size pools. The expert cache needs
  `2*MAXT*TOPK` (320) slots in *every* pool. `choose_residency()` moves the coldest-per-byte experts to host memory first,
  which empties the small pools first (here pool 3 = 3 layers, 1,536 experts), and `place_experts()` then throws:
  `expert cache: pool 3 has 235 slots, a window needs 320 (fewer experts in RAM)` for every `FNX_EXPERT_RAM_GIB` from 12
  up, while the default 3 GiB runs out of VRAM (`expert cache: slots: out of memory`). On larger cards the required
  `R` is small and this probably never shows. The floor keeps every pool above the slot count the cache needs. The
  "VRAM short" loop goes through the same function, so the floor applies there too.
- **mlock.** The engine pins about 80 GiB here (29.5 GB of embeddings plus the pinned expert copy). A stock Linux desktop
  has `ulimit -l` around 16 GiB, so the load stops with
  `fnx: mlock of per_layer_token_embd.weight (27465 MiB) failed: Cannot allocate memory; run with RLIMIT_MEMLOCK raised`.
  No code change is needed, but the limit has to be raised for the engine process (see below). Workstations with an
  unlimited memlock will not see this.

## Hardware and software used

- GIGABYTE X570 AORUS PRO WIFI, Ryzen 5 5600, 128 GB DDR4 (125 GiB seen by the OS)
- 2x GeForce RTX 4060 Ti 16 GB (sm_89), both CPU-attached PCIe Gen4 x8, no P2P
- NVMe on Gen3 x4
- Ubuntu 22.04.4, kernel 6.8, NVIDIA driver 615.71.09 (CUDA 13.4)
- CUDA toolkit from pip (CUDA 13.4, nvcc 13.4.92); cmake and ninja from a Python venv
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S (2 files, checksums verified), 46.8 GiB of routed experts, plus the MTP draft (Q2_0)

## Build

1. `git clone --branch split-geforce-16gb https://github.com/nabe2030/Strata.git`
2. Unpack the llama.cpp commit pinned in `setup.py` (3cf0325) and point `GG` at it.
3. Build the split engine for sm_89 with two cards:

   ```
   SRC=<Strata checkout> B=<build dir> CUDA=<cuda-root> GG=<llama.cpp> ARCH=89 CARDS=2 JOBS=8 \
     packaging/build_split_cuda.sh
   ```

   There were no sm_89-specific compile errors (warnings only), and the peer-function check at the end of the script passed.
4. Install the binary as `strata-split-2-sm89` and put `libcudart`, `libcublas` and `libcublasLt` (`.so.13`) next to it
   (the binary uses `RUNPATH $ORIGIN`).

## Run

Environment for the engine:

```
FNX_ARENA=1
FNX_EXPERT_RAM_GIB=20
FNX_POOL_MIN_RESIDENT=384      # 2*MAXT*TOPK = 320, plus a margin
```

Engine arguments: `--spec 6 --lookup 15 --batch 1 --max-context 8192 --conversation-cache-mib 1024`

Raise the memlock limit for the engine process only, for example:

```
sudo prlimit --memlock=unlimited --pid <engine pid>
# or start the engine under:  sudo prlimit --memlock=unlimited -- <command>
```

Server: `serve/server.py --engine strata --config cfg.json`, where `cfg.json` sets `exe` to the engine binary and
`drafting` to `mtp`. Leave `split_fallback` out of the config on purpose, so a failed start is not hidden by a fallback.
Confirm that the split engine really runs from the engine log (`engine started: ...strata-split-2-sm89`, then the
`fnx:` lines).

If you start through the launcher instead, the matching switches are
`STRATA_SPLIT_SKIP_VRAM_CHECK=1 STRATA_SPLIT_ALLOW_UNTESTED=1` (and optionally `STRATA_SPLIT_FNX_ARENA`,
`STRATA_SPLIT_FNX_EXPERT_RAM_GIB`). **The launcher path (`strata launch`) was not run in this test**; the server was
given the run configuration directly.

## Numbers

Settings: `FNX_ARENA=1 FNX_EXPERT_RAM_GIB=20 FNX_POOL_MIN_RESIDENT=384`, `--spec 6 --lookup 15 --batch 1`, context 8192,
temperature 0, through `serve/server.py`.

| `FNX_EXPERT_RAM_GIB` | Experts in RAM at the end | Short-prompt decode (tok/s, 3 runs) | 3.5K-token prompt (tok/s) | VRAM free per card | Load |
|---|---|---|---|---|---|
| 16 | n/a | fails to start (`prefill: cudaMalloc ... out of memory`) | n/a | n/a | n/a |
| 20 | 17,502 of 24,576 | 51.0 / 59.6 / 53.4 | 769 | about 0.8 GiB | 99 s |
| 30 | 17,258 of 24,576 | 51.4 / 60.8 / 54.9 | 806 | about 0.55 GiB | 97 s |

- With R=20, 7,074 experts (29 %) stay in VRAM and 17,502 (71 %) are in host memory; 7,453 of those were moved by the
  "VRAM short" loop. About 15.2 GB of VRAM is used per card.
- MTP draft acceptance was about 45 to 55 % (K=6). One tool call through the OpenAI route worked.
- GPU temperature stayed below 60 C.
- R=20 and R=30 end almost at the same residency because the "VRAM short" loop decides the final count. R of 16 or less
  runs out of memory before that loop.

Caveats:

- **A large `rsync` was reading the same NVMe during these measurements**, so the numbers are indicative only.
- No comparison against the normal layer split on the same machine, no quality suite, no context beyond 8192,
  no `--batch` above 1, and a single model/quant were tried.
- `FNX_POOL_MIN_RESIDENT=384` is a value that worked here, not a tuned one.

## Files

- `strata_control/split_engine.py`: patch 1 (launcher switches)
- `src/split/src/model.cpp`: patch 2 (pool floor)

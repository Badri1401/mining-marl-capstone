# CPU Parallel Environment Rollout

## Overview

Implemented SubprocVecEnv pattern for parallel episode collection in MADDPG V3 training. Each environment runs in an isolated process with its own PyBullet instance, eliminating GIL contention and enabling true parallel rollout.

## Architecture

```
Main Process (Neural Network Training)
   ├── Worker 1 (PyBullet Env 1) ──[pipe]──┐
   ├── Worker 2 (PyBullet Env 2) ──[pipe]──├─→ Transition Buffer → Training
   ├── Worker 3 (PyBullet Env 3) ──[pipe]──┤
   └── Worker N (PyBullet Env N) ──[pipe]──┘
```

## Key Features

1. **Process Isolation**: Each environment in separate process (no GIL, no PyBullet conflicts)
2. **Non-blocking Collection**: Workers run independently, main process polls for results
3. **Episode-level Parallelism**: Each worker runs full episodes
4. **Graceful Shutdown**: Proper cleanup of processes and pipes
5. **Backward Compatible**: Default num_parallel_envs=1 uses original single-env path

## Files

- `parallel_env_manager.py`: Core parallel rollout implementation
  - `ParallelEnvManager`: Manages worker processes and communication
  - `_worker_process()`: Worker function running in isolated process
  - `EpisodeCollector`: High-level interface for episode collection

- `train_maddpg.py`: Updated training loop with parallel support
  - Dual-path: single-env (original) vs parallel rollout
  - Phase tracking, curriculum advancement preserved
  - All metrics and logging maintained

- `test_parallel_rollout.py`: Validation script

## Usage

### Basic (Single Environment - Original Behavior)
```bash
python3 train_maddpg.py \
    --episodes 100000 \
    --reward-mode v3_clean \
    --curriculum-mode v3_smooth
```

### Parallel Rollout (4 Workers)
```bash
python3 train_maddpg.py \
    --episodes 100000 \
    --reward-mode v3_clean \
    --curriculum-mode v3_smooth \
    --num-parallel-envs 4 \
    --torch-threads 4
```

### Parallel Rollout with Ablation Flags
```bash
python3 train_maddpg.py \
    --episodes 100000 \
    --reward-mode v2_baseline \
    --curriculum-mode v2_original \
    --num-parallel-envs 6 \
    --torch-threads 2 \
    --disable-plateau-noise \
    --disable-waiting-shaping
```

### Max Throughput on 8-Core CPU (Single Job)
```bash
./run_single_max.sh                   # auto: 6 envs + 2 torch threads
ENVS=7 TORCH=1 ./run_single_max.sh   # manual override
```

### Dual-Mode Comparison (auto-splits CPU budget)
```bash
./run_parallel.sh                     # auto: (cores/2-1) envs per pane
./run_parallel.sh 100000 3 1          # explicit: 3 envs + 1 torch per pane
```

## Performance Expectations

### Speedup Analysis

**Single-Env Breakdown (per episode):**
- Environment step: ~15ms/step × 300 steps = 4.5s
- Neural network forward: ~2ms/step × 300 steps = 0.6s
- Training (4 steps): ~50ms × 4 = 0.2s
- **Total**: ~5.3s/episode

**With N=4 Parallel Workers:**
- Environment steps parallelized across 4 processes
- Neural network + training still centralized (no change)
- Expected speedup: ~2.5-3x (environment is 85% of time)
- Theoretical: 5.3s / 3 ≈ 1.8s/episode
- Accounting for overhead: ~2.0-2.2s/episode

**With N=8 Parallel Workers:**
- Further parallelization of environment rollout
- Diminishing returns due to centralized training
- Expected speedup: ~3.5-4x
- Theoretical: 5.3s / 4 ≈ 1.3s/episode

### Recommended Worker Counts

The core budget rule is: `env_workers + torch_threads ≤ cpu_count`

- **8-core CPU, single job**: 6 env workers + 2 torch threads (auto-default)
- **8-core CPU, dual comparison**: 3 env workers + 1 torch thread per pane (auto-default)
- **16-core CPU, single job**: 14 env workers + 2 torch threads
- **GPU training**: 8-16 workers, torch threads irrelevant (GPU handles compute)
- **Memory constraint**: Each worker needs ~200MB (PyBullet + env state)

## Implementation Details

### Worker Communication Protocol

Commands:
- `('reset',)` → returns (obs, info)
- `('step', actions)` → returns (next_obs, rewards, terminations, truncations, infos)
- `('get_attr', name)` → returns attribute value
- `('set_attr', name, value)` → sets attribute
- `('call_method', name, args, kwargs)` → calls method
- `('close',)` → shutdown worker

### Error Handling

- Worker crashes: Main process detects and marks worker inactive
- Pipe errors: Graceful degradation, continues with remaining workers
- Cleanup: All processes terminated on exit, pipes closed

### Curriculum Integration

- Each worker tracks its own phase
- Phase advancement triggered by main process
- `call_method_all('set_phase', phase)` updates all workers atomically
- Mastery calculation uses aggregated history from all workers

## Testing

Run validation test:
```bash
python3 test_parallel_rollout.py
```

Expected output:
- 4 workers spawned successfully
- Random actions for 50 steps
- All transitions collected
- Clean shutdown

## Design Decisions

### Why SubprocVecEnv over alternatives?

1. **vs DummyVecEnv (sequential)**: No parallelism, just abstraction
2. **vs Threading**: Python GIL limits CPU parallelism
3. **vs Ray**: Overkill for single-machine, adds complexity
4. **vs AsyncVectorEnv**: Less stable with PyBullet

### Why process-level isolation?

- PyBullet uses OpenGL/GPU resources that conflict across threads
- Each process has independent memory space (no state leaks)
- GIL avoided completely
- Matches industry standard (Stable-Baselines3, OpenAI Baselines)

### Why episode-level parallelism?

- Simpler than step-level (no synchronization)
- Better cache locality (full episode in one worker)
- Matches MADDPG replay buffer design (store full episodes)

## Performance Tuning

### CPU Thread Budget Rule

The total CPU load from one training process is:

```
env_workers + torch_threads ≤ cpu_count
```

This is enforced automatically: if `--torch-threads` is not specified, the trainer
computes `torch_threads = cpu_count - num_parallel_envs`.  When the budget is
exceeded, a `[WARNING]` is printed at startup.

**Single-job examples (8-core machine):**

| env_workers | torch_threads | total | status |
|:-----------:|:-------------:|:-----:|:------:|
| 6           | 2             | 8     | ✅ OK  |
| 8           | 1             | 9     | ⚠️     |
| 4           | 4             | 8     | ✅ OK  |

**Dual-job (run_parallel.sh) on 8 cores** — the budget is shared across both
panes.  The launcher auto-detects core count and sets per-pane defaults to
`(cores/2 - 1)` env workers + 1 torch thread so the combined load stays within
the machine limit.

### CLI Flags

```
--num-parallel-envs N       Number of env worker processes     (default: 1)
--torch-threads N           PyTorch intra-op threads           (default: auto)
--torch-interop-threads N   PyTorch inter-op threads           (default: 1)
```

### Launchers

| Script              | Purpose                | Default budget           |
|---------------------|------------------------|--------------------------|
| `run_single_max.sh` | Max throughput (1 job) | cores-2 envs + 2 torch   |
| `run_parallel.sh`   | Dual-mode comparison   | (cores/2-1) envs + 1 torch per pane |
| `run_ablations.sh`  | Sequential ablations   | cores-2 envs + 2 torch   |

### CPU Thread Limiting
```python
torch.set_num_threads(N)  # Set via --torch-threads (auto-detected by default)
```
With N workers + T torch threads = cpu_count total, prevents oversubscription.

### Batch Collection
Parallel path collects N episodes before training, improving GPU utilization.

### Pipe Buffer Size
Default pipe buffer (OS-dependent) sufficient for typical transitions (~1KB each).

## Future Enhancements (Not Implemented)

1. **Prioritized Worker Scheduling**: Fast workers get more episodes
2. **Adaptive Worker Count**: Scale based on CPU usage
3. **GPU-based Environment**: Replace PyBullet with GPU physics
4. **Distributed Training**: Multi-node rollout with centralized trainer

## Troubleshooting

### Workers hang at startup
- Check PyBullet installation: `pip install pybullet`
- Verify OpenGL support: `xvfb-run python3 test_parallel_rollout.py`

### Memory errors with many workers
- Reduce num_parallel_envs (each worker ~200MB)
- Monitor with `htop` during training

### Slower than single-env
- Check CPU core count: `nproc`
- Verify no other heavy processes running
- Try different worker counts (4, 8, 16)

## Validation

Test completed successfully:
```
✓ 4 workers spawned
✓ 200 transitions collected (50 steps × 4 workers)
✓ Phase setting works
✓ Clean shutdown
```

## References

- OpenAI Baselines SubprocVecEnv: https://github.com/openai/baselines
- Stable-Baselines3 VecEnv: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
- Python multiprocessing: https://docs.python.org/3/library/multiprocessing.html

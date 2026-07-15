"""
CPU Parallel Environment Rollout Manager
File: parallel_env_manager.py

Implements SubprocVecEnv pattern for parallel episode collection:
- Each worker runs in its own process with isolated PyBullet instance
- Communication via multiprocessing.Pipe
- Non-blocking collection of transitions from multiple environments
- Graceful shutdown and error handling

Usage:
    manager = ParallelEnvManager(
        env_fn=lambda: MiningEnvironmentMADDPG_V3(...),
        num_workers=4
    )
    manager.reset_all()
    for step in range(max_steps):
        actions = get_actions(manager.get_observations())
        transitions = manager.step_all(actions)
        # Store transitions in replay buffer
    manager.close()
"""

import multiprocessing as mp
import numpy as np
from typing import Callable, Dict, List, Tuple, Any, Optional
import traceback


def _worker_process(worker_id: int, pipe, env_fn: Callable):
    """
    Worker process function — runs in isolated process with own PyBullet instance.
    
    Commands:
        ('reset',) -> (obs, info)
        ('step', actions) -> (next_obs, rewards, terminations, truncations, infos)
        ('get_attr', name) -> attribute value
        ('set_attr', name, value) -> None
        ('call_method', name, args, kwargs) -> method result
        ('close',) -> None (exits loop)
    """
    try:
        env = env_fn()
        
        while True:
            try:
                cmd = pipe.recv()
                
                if cmd[0] == 'reset':
                    obs, info = env.reset()
                    pipe.send(('success', (obs, info)))
                
                elif cmd[0] == 'step':
                    actions = cmd[1]
                    result = env.step(actions)
                    pipe.send(('success', result))
                
                elif cmd[0] == 'get_attr':
                    attr_name = cmd[1]
                    value = getattr(env, attr_name)
                    pipe.send(('success', value))
                
                elif cmd[0] == 'set_attr':
                    attr_name, value = cmd[1], cmd[2]
                    setattr(env, attr_name, value)
                    pipe.send(('success', None))
                
                elif cmd[0] == 'call_method':
                    method_name = cmd[1]
                    args = cmd[2] if len(cmd) > 2 else ()
                    kwargs = cmd[3] if len(cmd) > 3 else {}
                    method = getattr(env, method_name)
                    result = method(*args, **kwargs)
                    pipe.send(('success', result))
                
                elif cmd[0] == 'close':
                    env.close()
                    pipe.send(('success', None))
                    break
                
                else:
                    pipe.send(('error', f"Unknown command: {cmd[0]}"))
            
            except Exception as e:
                tb = traceback.format_exc()
                pipe.send(('error', (str(e), tb)))
    
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Worker {worker_id}] Fatal error during initialization: {e}\n{tb}")
    finally:
        pipe.close()


class ParallelEnvManager:
    """
    Manages multiple environment workers for parallel rollout collection.
    
    Key features:
    - Process isolation: Each env in separate process (avoids GIL, PyBullet conflicts)
    - Non-blocking collection: Workers run independently
    - Episode-level parallelism: Each worker runs full episodes
    - Transition batching: Accumulates transitions from all workers
    """
    
    def __init__(self, env_fn: Callable, num_workers: int = 4):
        """
        Args:
            env_fn: Callable that creates environment instance (must be picklable)
            num_workers: Number of parallel worker processes
        """
        self.num_workers = num_workers
        self.env_fn = env_fn
        self.workers = []
        self.pipes = []
        self.worker_obs = [None] * num_workers
        self.worker_active = [False] * num_workers
        
        # Start worker processes
        ctx = mp.get_context('spawn')  # 'spawn' is safest for complex envs like PyBullet
        for i in range(num_workers):
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(
                target=_worker_process,
                args=(i, child_pipe, env_fn),
                daemon=True
            )
            process.start()
            self.workers.append(process)
            self.pipes.append(parent_pipe)
            self.worker_active[i] = True
    
    def reset_all(self) -> List[Dict]:
        """Reset all workers and return initial observations."""
        observations = []
        for i, pipe in enumerate(self.pipes):
            pipe.send(('reset',))
        
        for i, pipe in enumerate(self.pipes):
            status, result = pipe.recv()
            if status == 'error':
                raise RuntimeError(f"Worker {i} reset error: {result}")
            obs, _ = result
            self.worker_obs[i] = obs
            self.worker_active[i] = True
            observations.append(obs)
        
        return observations
    
    def step_all(self, actions_list: List[Dict]) -> List[Tuple]:
        """
        Send actions to all workers and collect transitions.
        
        Args:
            actions_list: List of action dicts, one per worker
        
        Returns:
            List of (worker_id, obs, actions, rewards, next_obs, terminations,
                     truncations, infos) tuples — worker_id identifies source.
        """
        if len(actions_list) != self.num_workers:
            raise ValueError(f"Expected {self.num_workers} action dicts, got {len(actions_list)}")
        
        # Send step commands to all workers
        for i, (pipe, actions) in enumerate(zip(self.pipes, actions_list)):
            if self.worker_active[i]:
                pipe.send(('step', actions))
        
        # Collect results
        transitions = []
        for i, pipe in enumerate(self.pipes):
            if not self.worker_active[i]:
                continue
            
            status, result = pipe.recv()
            if status == 'error':
                error_msg, tb = result
                print(f"[Worker {i}] Error: {error_msg}\n{tb}")
                self.worker_active[i] = False
                continue
            
            next_obs, rewards, terminations, truncations, infos = result
            
            # Store transition WITH worker_id so caller can attribute correctly
            transition = (
                i,  # worker_id
                self.worker_obs[i],
                actions_list[i],
                rewards,
                next_obs,
                terminations,
                truncations,
                infos
            )
            transitions.append(transition)
            
            # Update worker state
            self.worker_obs[i] = next_obs
            
            # Check if episode ended
            if all(terminations.values()) or any(truncations.values()):
                self.worker_active[i] = False
        
        return transitions
    
    def get_active_workers(self) -> List[int]:
        """Return indices of workers that are still running episodes."""
        return [i for i in range(self.num_workers) if self.worker_active[i]]
    
    def reset_worker(self, worker_id: int) -> Dict:
        """Reset a specific worker and return initial observation."""
        pipe = self.pipes[worker_id]
        pipe.send(('reset',))
        status, result = pipe.recv()
        if status == 'error':
            raise RuntimeError(f"Worker {worker_id} reset error: {result}")
        obs, _ = result
        self.worker_obs[worker_id] = obs
        self.worker_active[worker_id] = True
        return obs
    
    def get_attr_all(self, attr_name: str) -> List[Any]:
        """Get attribute value from all workers."""
        for pipe in self.pipes:
            pipe.send(('get_attr', attr_name))
        
        results = []
        for i, pipe in enumerate(self.pipes):
            status, result = pipe.recv()
            if status == 'error':
                raise RuntimeError(f"Worker {i} get_attr error: {result}")
            results.append(result)
        
        return results
    
    def set_attr_all(self, attr_name: str, value: Any):
        """Set attribute value on all workers."""
        for pipe in self.pipes:
            pipe.send(('set_attr', attr_name, value))
        
        for i, pipe in enumerate(self.pipes):
            status, _ = pipe.recv()
            if status == 'error':
                raise RuntimeError(f"Worker {i} set_attr error")
    
    def call_method_all(self, method_name: str, *args, **kwargs) -> List[Any]:
        """Call method on all workers and return results."""
        for pipe in self.pipes:
            pipe.send(('call_method', method_name, args, kwargs))
        
        results = []
        for i, pipe in enumerate(self.pipes):
            status, result = pipe.recv()
            if status == 'error':
                raise RuntimeError(f"Worker {i} call_method error: {result}")
            results.append(result)
        
        return results
    
    def close(self):
        """Gracefully shutdown all workers."""
        for pipe in self.pipes:
            try:
                pipe.send(('close',))
                if pipe.poll(2.0):
                    pipe.recv()  # Wait for confirmation if the worker responds
            except:
                pass
        
        for process in self.workers:
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        
        for pipe in self.pipes:
            pipe.close()
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.close()
        except:
            pass


class EpisodeCollector:
    """
    High-level interface for collecting full episodes from parallel workers.
    
    Handles:
    - Episode-level collection (run until all done)
    - Per-worker noise scaling
    - Automatic reset after episode completion
    - Transition accumulation
    """
    
    def __init__(self, manager: ParallelEnvManager, max_steps: int = 1000):
        self.manager = manager
        self.max_steps = max_steps
        self.num_workers = manager.num_workers
    
    def collect_episodes(self, 
                        get_actions_fn: Callable,
                        noise_scales: Optional[List[float]] = None) -> List[List[Tuple]]:
        """
        Collect full episodes from all workers in parallel.
        
        Args:
            get_actions_fn: Function (obs, noise_scale) -> actions
            noise_scales: List of noise scales per worker (optional)
        
        Returns:
            List of episode transitions, one list per worker
        """
        if noise_scales is None:
            noise_scales = [0.0] * self.num_workers
        
        # Reset all workers
        observations = self.manager.reset_all()
        
        # Per-worker transition storage
        episode_transitions = [[] for _ in range(self.num_workers)]
        worker_steps = [0] * self.num_workers
        
        # Run until all workers complete or reach max steps
        while any(self.manager.worker_active):
            active_workers = self.manager.get_active_workers()
            if not active_workers:
                break
            
            # Get actions for active workers
            actions_list = []
            for i in range(self.num_workers):
                if self.manager.worker_active[i]:
                    obs = self.manager.worker_obs[i]
                    actions = get_actions_fn(obs, noise_scales[i])
                    actions_list.append(actions)
                    worker_steps[i] += 1
                else:
                    actions_list.append({})  # Placeholder for inactive workers
            
            # Step all active workers
            transitions = self.manager.step_all(actions_list)
            
            # Store transitions — use worker_id from tuple
            for trans in transitions:
                wid = trans[0]  # worker_id is first element
                episode_transitions[wid].append(trans)
            
            # Check max steps
            for i in active_workers:
                if worker_steps[i] >= self.max_steps:
                    self.manager.worker_active[i] = False
        
        return episode_transitions

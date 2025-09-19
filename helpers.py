# Copyright 2023 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""Helpers for CSIL – local experts only (no RLDS/TFDS), minimal env glue."""

from __future__ import annotations

import glob
import os
import pickle
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import dm_env
import gym  # legacy Gym for MuJoCo v2 tasks
import jax
import jax.numpy as jnp
import numpy as np
import tree
from acme import specs, types
from acme import wrappers
from gym.spaces import Box

# -- Optional: keep D4RL side-effects (env registrations) if present.
try:
    import d4rl  # pylint: disable=unused-import
except Exception:
    pass

# -- Gymnasium Adroit v1 support (optional but preferred for v1 tasks).
try:
    import gymnasium as gymn
    import gymnasium_robotics as gymn_robotics
    _HAVE_GYMN = True
    try:
        gymn.register_envs(gymn_robotics)  # no-op if already registered
    except Exception:
        pass
except Exception:
    _HAVE_GYMN = False

# =========================
# Data loading (local files)
# =========================

# --- ADD just above _np_dict_from_pkl ----------------------------------------
from typing import Sequence

def _is_episode_list_obj(obj: Mapping) -> bool:
    # IQ-Learn / many lab formats: lists (or tuples) of per-episode sequences
    req = ("states" in obj) and ("actions" in obj) and ("rewards" in obj)
    if not req: return False
    s = obj["states"]
    return isinstance(s, Sequence) and len(s) > 0 and isinstance(s[0], (list, tuple, np.ndarray))

def _flatten_episode_list_obj(obj: Mapping) -> Dict[str, np.ndarray]:
    """Convert {'states':[ep1,...], 'actions':[...], ...} into flat step-wise arrays."""
    # states / next_states
    states_eps = obj["states"]
    next_states_eps = obj.get("next_states", states_eps)
    actions_eps = obj["actions"]
    rewards_eps = obj["rewards"]
    dones_eps = obj.get("dones", [[False]*len(rewards_eps[i]) for i in range(len(rewards_eps))])

    # stack each episode to (T, dim) then concat across episodes
    def _stack_eps(eps):
        mats = []
        for ep in eps:
            a = np.asarray(ep)
            # (T,) of vectors -> (T,dim)
            if a.ndim == 1:
                a = np.stack(list(a), axis=0) if a.dtype == object else a[:, None]
            elif a.ndim > 2:
                a = a.reshape(a.shape[0], -1)
            mats.append(a.astype(np.float32))
        return np.concatenate(mats, axis=0)

    obs  = _stack_eps(states_eps)
    nobs = _stack_eps(next_states_eps)
    acts = _stack_eps(actions_eps)
    rews = _stack_eps(rewards_eps).reshape(-1)

    dones_list = []
    for ep in dones_eps:
        d = np.asarray(ep).astype(bool)
        if d.ndim > 1: d = d.reshape(-1)
        dones_list.append(d)
    dones = np.concatenate(dones_list, axis=0)
    disc = (1.0 - dones.astype(np.float32))

    return dict(
        obs=obs, next_obs=nobs, action=acts,
        reward=rews.astype(np.float32), discount=disc, done=dones.astype(np.float32)
    )
# -----------------------------------------------------------------------------

def _np_dict_from_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # (A) Episodic IQ-Learn / D4RL-style: flatten to step-level arrays.
    if isinstance(obj, Mapping) and _is_episode_list_obj(obj):
        return _flatten_episode_list_obj(obj)

    # (B) Already flat dict of arrays (e.g., D4RL): pass through.
    if isinstance(obj, Mapping):
        return dict(obj)

    raise ValueError(f"Unsupported pickle structure in {path}: {type(obj)}")




def _np_dict_from_npz(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {k: np.asarray(v) for k, v in d.items()}

# def _np_dict_from_pkl(path: str) -> Dict[str, Any]:
#     with open(path, "rb") as f:
#         obj = pickle.load(f)
#     if isinstance(obj, Mapping):
#         return dict(obj)
#     raise ValueError(f"Unsupported pickle format in {path}: {type(obj)}")

def _flatten_obs_if_dict(obs: Any) -> np.ndarray:
    if isinstance(obs, Mapping):
        leaves = []
        for k in sorted(obs.keys()):
            v = np.asarray(obs[k])
            leaves.append(v.reshape(v.shape[0], -1))
        return np.concatenate(leaves, axis=-1)
    return np.asarray(obs)

def _normalize_keys(d: Dict[str, Any]) -> Dict[str, np.ndarray]:
    alias = {
        # observations
        "observation": "obs", "observations": "obs", "obs": "obs", "state": "obs", "states": "obs",
        # next observations
        "next_observation": "next_obs", "next_observations": "next_obs", "next_obs": "next_obs",
        "next_state": "next_obs", "next_states": "next_obs",
        # actions / rewards / done / discount
        "action": "action", "actions": "action", "act": "action",
        "reward": "reward", "rewards": "reward", "rew": "reward",
        "done": "done", "dones": "done", "terminal": "done",
        "terminals": "terminals", "timeouts": "timeouts",
        "discount": "discount", "discounts": "discount",
        # optional mujoco state
        "qpos": "qpos", "qvel": "qvel",
    }
    out: Dict[str, np.ndarray] = {}
    for k, v in d.items():
        k2 = alias.get(k, k)
        if k2 in ("obs", "next_obs"):
            out[k2] = _flatten_obs_if_dict(v)
        else:
            out[k2] = np.asarray(v)
    # infer done/discount if needed
    if "done" not in out:
        term = out.get("terminals")
        tout = out.get("timeouts")
        if term is not None and tout is not None:
            out["done"] = (term.astype(bool) | tout.astype(bool)).astype(np.float32)
        elif term is not None:
            out["done"] = term.astype(np.float32)
        elif tout is not None:
            out["done"] = tout.astype(np.float32)
    if "discount" not in out and "done" in out:
        out["discount"] = (1.0 - out["done"].astype(np.float32))
    return out

def _compute_next_obs_if_missing(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if "next_obs" in d:
        return d
    obs = np.asarray(d["obs"])
    next_obs = obs[1:]
    keep = slice(0, next_obs.shape[0])
    out = {k: np.asarray(v)[keep] for k, v in d.items() if k != "obs"}
    out["obs"] = obs[keep]
    out["next_obs"] = next_obs
    if "discount" in out:
        out["discount"] = out["discount"][: next_obs.shape[0]]
    if "done" in out and "discount" not in out:
        out["discount"] = (1.0 - out["done"].astype(np.float32))[: next_obs.shape[0]]
    return out

def _compute_next_action(d: Dict[str, np.ndarray]) -> np.ndarray:
    acts = np.asarray(d["action"])
    return acts[1:]

def _concat_dicts(ds: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = sorted({k for d in ds for k in d.keys()})
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        vals = [np.asarray(d[k]) for d in ds if k in d]
        out[k] = np.concatenate(vals, axis=0)
    return out

def _episode_bounds(done: Optional[np.ndarray]) -> List[Tuple[int, int]]:
    if done is None:
        return []
    done = np.asarray(done).astype(bool)
    idx = np.flatnonzero(done)
    if idx.size == 0:
        return [(0, len(done))]
    starts = np.concatenate([[0], idx[:-1] + 1])
    ends = idx + 1
    return list(zip(starts.tolist(), ends.tolist()))

def _slice_first_n_episodes(raw: Dict[str, np.ndarray], n_eps: int) -> Dict[str, np.ndarray]:
    if n_eps <= 0 or "done" not in raw:
        return raw
    bounds = _episode_bounds(raw["done"])
    sls = [slice(s, e) for (s, e) in bounds[:n_eps]]
    N = raw[next(iter(raw))].shape[0]
    mask = np.zeros((N,), dtype=bool)
    for sl in sls:
        mask[sl] = True
    return {k: v[mask] for k, v in raw.items()}

def _gather_files(root_or_path: str) -> List[str]:
    if os.path.isdir(root_or_path):
        files = sorted([
            *glob.glob(os.path.join(root_or_path, "*.npz")),
            *glob.glob(os.path.join(root_or_path, "*.pkl")),
        ])
    else:
        files = [root_or_path]
    if not files:
        raise FileNotFoundError(f"No .npz/.pkl under {root_or_path}")
    return files

def _load_local_arrays(root_or_path: str) -> Dict[str, np.ndarray]:
    parts: List[Dict[str, np.ndarray]] = []
    for f in _gather_files(root_or_path):
        raw = _np_dict_from_npz(f) if f.endswith(".npz") else _np_dict_from_pkl(f)
        norm = _normalize_keys(raw)
        comp = _compute_next_obs_if_missing(norm)
        parts.append(comp)
    return _concat_dicts(parts) if len(parts) > 1 else parts[0]

@dataclass
class _Arrays:
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    discount: np.ndarray
    next_obs: np.ndarray
    next_action: Optional[np.ndarray] = None

def _to_arrays(d: Dict[str, np.ndarray], want_next_action: bool) -> _Arrays:
    # def _as_2d(a) -> np.ndarray:
    #     a = np.asarray(a)
    #     # If this is an object array of per-step vectors, stack them
    #     if a.dtype == object:
    #         a = np.stack(list(a), axis=0)
    #     if a.ndim == 1:
    #         a = a[:, None]
    #     elif a.ndim > 2:
    #         a = a.reshape(a.shape[0], -1)
    #     return a.astype(np.float32)

    def _as_2d(a) -> np.ndarray:
        a = np.asarray(a)
        if a.dtype == object:
            # Flatten any nested object-y rows into numeric vectors
            flat = [np.asarray(x).reshape(-1) for x in a.reshape(-1)]
            a = np.stack(flat, axis=0)
        if a.ndim == 1:
            a = a[:, None]
        elif a.ndim > 2:
            a = a.reshape(a.shape[0], -1)
        return a.astype(np.float32, copy=False)

    obs = _as_2d(d["obs"])
    next_obs = _as_2d(d["next_obs"])
    action = _as_2d(d["action"])

    reward = np.asarray(d.get("reward", np.zeros((obs.shape[0],), np.float32)), dtype=np.float32).reshape(-1)
    discount = np.asarray(d.get("discount", np.ones((obs.shape[0],), np.float32)), dtype=np.float32).reshape(-1)

    out = _Arrays(obs=obs, action=action, reward=reward, discount=discount, next_obs=next_obs)

    if want_next_action and "action" in d:
        na = _as_2d(_compute_next_action(d))
        L = min(out.obs.shape[0], na.shape[0])
        out = _Arrays(
            obs=out.obs[:L], action=out.action[:L], reward=out.reward[:L],
            discount=out.discount[:L], next_obs=out.next_obs[:L], next_action=na[:L]
        )
    else:
        L = min(out.obs.shape[0], out.next_obs.shape[0])
        out = _Arrays(
            obs=out.obs[:L], action=out.action[:L], reward=out.reward[:L],
            discount=out.discount[:L], next_obs=out.next_obs[:L]
        )
    return out


class _ArrayBatchIterator:
    def __init__(self, arr: _Arrays, batch_size: int, seed: int, with_next_action: bool):
        self._arr = arr
        self._bsz = int(batch_size)
        self._rng = random.Random(int(seed))
        self._with_na = with_next_action and (arr.next_action is not None)
        self._N = arr.obs.shape[0]
        self._perm: List[int] = []

    def __iter__(self) -> "_ArrayBatchIterator":
        return self

    def __next__(self) -> types.Transition:
        if self._N == 0:
            raise ValueError("Empty dataset: no transitions loaded.")
    
        # Sample indices (with replacement if dataset is smaller than batch).
        if self._N >= self._bsz:
            if len(self._perm) < self._bsz:
                self._perm = list(range(self._N))
                self._rng.shuffle(self._perm)
            idx = [self._perm.pop() for _ in range(self._bsz)]
        else:
            idx = [self._rng.randrange(self._N) for _ in range(self._bsz)]
    
        idx = np.asarray(idx, dtype=np.int32)
    
        # Helper: ensure a proper batch axis (B, ...)
        def _ensure_batch(x):
            x = np.asarray(x)
            x = x[idx]
            # Sometimes x[idx] can still be 1D if original was (N,) of objects
            if x.ndim == 1:
                x = np.stack(list(x), axis=0) if x.dtype == object else x[:, None]
            return x
    
        obs = _ensure_batch(self._arr.obs)
        act = _ensure_batch(self._arr.action)
        rew = np.asarray(self._arr.reward)[idx].reshape(-1)  # (B,)
        dis = np.asarray(self._arr.discount)[idx].reshape(-1)  # (B,)
        nxt = _ensure_batch(self._arr.next_obs)
    
        extras = {}
        if self._with_na:
            extras["next_action"] = _ensure_batch(self._arr.next_action)
   
        return types.Transition(
            observation=obs, action=act, reward=rew, discount=dis,
            next_observation=nxt, extras=extras
        )


class MixedIterator(Iterator[types.Transition]):
    """Combine two streams 50/50 (interleaved) — unchanged API."""
    def __init__(self, first_iterator: Iterator[types.Transition], second_iterator: Iterator[types.Transition], with_extras: bool = False):
        self._first = first_iterator
        self._second = second_iterator
        self.with_extras = with_extras

    def __iter__(self):
        return self

    def __next__(self) -> types.Transition:
        a = next(self._first)
        b = next(self._second)
        combined = tree.map_structure(lambda x, y: jnp.concatenate((x, y), axis=0), a, b)
        return combined

# =========================
# Environments
# =========================

class _Spec:
    def __init__(self, env_id: str):
        self.id = env_id

class GymnasiumOldAPIAdapter:
    """Expose old Gym API on top of Gymnasium for Acme wrappers."""
    def __init__(self, gymn_env, env_id: str):
        self._env = gymn_env
        self.spec = _Spec(env_id)
        # best-effort spaces for wrappers
        try:
            obs_s = self._env.observation_space
            act_s = self._env.action_space
            obs_shape = tuple(getattr(obs_s, "shape", ()))
            act_shape = tuple(getattr(act_s, "shape", ()))
            if Box is not None and obs_shape:
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
            if Box is not None and act_shape:
                self.action_space = Box(low=-1.0, high=1.0, shape=act_shape, dtype=np.float32)
        except Exception:
            pass

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        return obs, float(reward), done, info

    def render(self, *a, **k):
        return self._env.render(*a, **k)

    def close(self):
        return self._env.close()

    def seed(self, seed=None):
        try:
            self._env.reset(seed=seed)
        except Exception:
            pass
        return [seed]

    def __getattr__(self, attr):
        return getattr(self._env, attr)

def make_environment(task: str) -> Callable[[], dm_env.Environment]:
    """Return a thunk creating the requested env wrapped for Acme."""
    def _from_gymnasium(env_id: str):
        env = gymn.make(env_id)
        env = GymnasiumOldAPIAdapter(env, env_id)
        env = wrappers.GymWrapper(env)
        env = wrappers.SinglePrecisionWrapper(env)
        return env

    def _from_gym(env_id: str):
        env_ = gym.make(env_id)
        env_ = wrappers.GymWrapper(env_)
        env_ = wrappers.SinglePrecisionWrapper(env_)
        return env_

    def _make():
        # Adroit uses Gymnasium v1 ids; MuJoCo control uses Gym v2 ids
        if _HAVE_GYMN and ("Adroit" in task or task.endswith("-v1")):
            return _from_gymnasium(task)
        return _from_gym(task)

    return _make

def _default_expert_dir_for(task: str) -> str:
    # simple default: ./experts/<task>
    return os.path.join("experts", task)

def _coerce_to_spec(arr: _Arrays, env_spec: specs.EnvironmentSpec) -> _Arrays:
    """Ensure obs/action match env spec dims by slicing extras off the right."""
    def _flat_dim(space):
        shape = tuple(space.shape) if hasattr(space, "shape") else tuple(space)
        return int(np.prod(shape, dtype=int)) if shape else 0

    obs_dim = _flat_dim(env_spec.observations)
    act_dim = _flat_dim(env_spec.actions)

    obs = arr.obs
    nxt = arr.next_obs
    act = arr.action

    if obs.ndim != 2:
        obs = obs.reshape(obs.shape[0], -1)
    if nxt.ndim != 2:
        nxt = nxt.reshape(nxt.shape[0], -1)
    if act.ndim != 2:
        act = act.reshape(act.shape[0], -1)

    if obs_dim and obs.shape[1] != obs_dim:
        # If dataset packs extra features (e.g. images), keep only what the env expects.
        obs = obs[:, :obs_dim]
        nxt = nxt[:, :obs_dim]
    if act_dim and act.shape[1] != act_dim:
        act = act[:, :act_dim]

    next_action = arr.next_action
    if next_action is not None and next_action.ndim != 2:
        next_action = next_action.reshape(next_action.shape[0], -1)
    if next_action is not None and act_dim and next_action.shape[1] != act_dim:
        next_action = next_action[:, :act_dim]

    L = min(obs.shape[0], nxt.shape[0], act.shape[0], arr.reward.shape[0], arr.discount.shape[0])
    return _Arrays(
        obs=obs[:L],
        action=act[:L],
        reward=arr.reward[:L],
        discount=arr.discount[:L],
        next_obs=nxt[:L],
        next_action=None if next_action is None else next_action[:L],
    )

# =========================
# Public API
# =========================

def get_env_and_demonstrations(
    task: str,
    num_demonstrations: int,
    expert: bool = True,
    use_sarsa: bool = False,
    in_memory: bool = True,
    *,
    expert_backend: str = "local",
    expert_path: Optional[str] = None,
) -> Tuple[Callable[[], dm_env.Environment], specs.EnvironmentSpec, Callable[[int, int], Iterator[types.Transition]]]:
    if expert_backend != "local":
        raise ValueError("Only expert_backend='local' is supported (no RLDS/TFDS).")

    make_env = make_environment(task)
    env_spec = specs.make_environment_spec(make_env())

    exp_dir = expert_path or _default_expert_dir_for(task)
    raw = _load_local_arrays(exp_dir)
    raw = _slice_first_n_episodes(raw, num_demonstrations)
    arrays = _to_arrays(raw, want_next_action=use_sarsa)
    arrays = _coerce_to_spec(arrays, env_spec)

    def make_demonstrations(batch_size: int, seed: int = 0) -> Iterator[types.Transition]:
        return _ArrayBatchIterator(arrays, batch_size, seed, with_next_action=use_sarsa)

    return make_env, env_spec, make_demonstrations

def get_offline_dataset(
    task: str,
    environment_spec: specs.EnvironmentSpec,
    expert_num_demonstration: int,
    offline_num_demonstrations: int,
    expert_offline_data: bool = False,
    use_sarsa: bool = False,
    in_memory: bool = True,
    *,
    expert_backend: str = "local",
    expert_path: Optional[str] = None,
    offline_path: Optional[str] = None,
) -> Tuple[Callable[[int, int], Iterator[Any]], Callable[[int, jax.Array], Iterator[types.Transition]]]:
    if expert_backend != "local":
        raise ValueError("Only expert_backend='local' is supported (no RLDS/TFDS).")

    exp_dir = expert_path or _default_expert_dir_for(task)
    off_dir = offline_path or exp_dir

    # slice episodes on raw dicts first
    exp_raw = _slice_first_n_episodes(_load_local_arrays(exp_dir), expert_num_demonstration)
    off_raw = _slice_first_n_episodes(_load_local_arrays(off_dir), offline_num_demonstrations)

    exp_arrays = _to_arrays(exp_raw, want_next_action=use_sarsa)
    off_arrays = _to_arrays(off_raw, want_next_action=use_sarsa)
    exp_arrays = _coerce_to_spec(exp_arrays, environment_spec)
    off_arrays = _coerce_to_spec(off_arrays, environment_spec)

    def make_offline_dataset(batch_size: int, key: jax.Array) -> Iterator[types.Transition]:
        seed = int(jax.random.bits(key, 32))
        return _ArrayBatchIterator(off_arrays, batch_size, seed, with_next_action=use_sarsa)

    def make_imitation_dataset(batch_size: int, key: jax.Array) -> Iterator[Any]:
        seed = int(jax.random.bits(key, 32))
        expert_it = _ArrayBatchIterator(exp_arrays, batch_size, seed, with_next_action=use_sarsa)
        offline_it = _ArrayBatchIterator(off_arrays, batch_size, seed + 1, with_next_action=use_sarsa)
        return MixedIterator(expert_it, offline_it)

    return make_imitation_dataset, make_offline_dataset

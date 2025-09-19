# csil/data/local_experts.py
# Read experts from local *.npz or *.pkl into a tf.data.Dataset of steps.
# Returns step dicts compatible with RLDS-style keys used downstream:
#   observation, next_observation, action, reward, discount,
#   is_first, is_last, is_terminal   (all per-step tensors)
from __future__ import annotations
import os, glob, pickle
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf

StepDict = Dict[str, np.ndarray]

# ---------- NPZ loader (episode-per-file) ----------
def _load_npz_episode(path: str) -> StepDict:
    d = np.load(path)
    # required:
    obs  = d["observation"].astype(np.float32)
    nob  = d["next_observation"].astype(np.float32) if "next_observation" in d else None
    act  = d["action"].astype(np.float32)
    rew  = d["reward"].astype(np.float32)
    T    = act.shape[0]
    # optional or build if missing:
    done = d["is_terminal"].astype(bool) if "is_terminal" in d else np.zeros(T, bool)
    disc = d["discount"].astype(np.float32) if "discount" in d else np.concatenate([np.ones(T-1,np.float32), [0.0 if done[-1] else 1.0]])
    is_first = d["is_first"].astype(bool) if "is_first" in d else np.zeros(T, bool); is_first[0] = True
    is_last  = d["is_last"].astype(bool)  if "is_last"  in d else np.zeros(T, bool); is_last[-1] = True
    if nob is None:
        # derive next_observation if not present
        # pad last next_obs with itself (won't be used when discount=0 at end)
        nob = np.concatenate([obs[1:], obs[-1:]], axis=0).astype(np.float32)
    return dict(observation=obs, next_observation=nob, action=act, reward=rew,
                discount=disc, is_first=is_first, is_last=is_last, is_terminal=done)

# ---------- PKL loaders ----------
def _is_episode_list_pkl(obj: dict) -> bool:
    # format like: {states: [list of arrays per episode], actions: [...], next_states: [...], rewards: [...], dones: [...]}
    return all(k in obj for k in ["states","actions","rewards"]) and isinstance(obj["states"], list)

def _from_episode_list_pkl(obj: dict) -> List[StepDict]:
    episodes = []
    for ep_idx in range(len(obj["states"])):
        obs  = np.stack(obj["states"][ep_idx]).astype(np.float32)
        nob  = np.stack(obj.get("next_states", obj["states"])[ep_idx]).astype(np.float32)
        act  = np.stack(obj["actions"][ep_idx]).astype(np.float32)
        rew  = np.asarray(obj["rewards"][ep_idx], dtype=np.float32)
        done = np.asarray(obj.get("dones", [False]*len(rew))[ep_idx], dtype=bool)
        T = act.shape[0]
        is_first = np.zeros(T, bool); is_first[0] = True
        is_last  = np.zeros(T, bool); is_last[-1] = True
        disc     = np.ones(T, np.float32); disc[-1] = 0.0 if done[-1] else 1.0
        episodes.append(dict(
            observation=obs, next_observation=nob, action=act, reward=rew,
            discount=disc, is_first=is_first, is_last=is_last, is_terminal=done))
    return episodes

def _is_flat_d4rl_pkl(obj: dict) -> bool:
    # D4RL-like: observations, next_observations, actions, rewards, terminals (and optional timeouts)
    return all(k in obj for k in ["observations","actions","next_observations","rewards"])

def _segment_flat_to_episodes(obj: dict) -> List[StepDict]:
    obs  = np.asarray(obj["observations"], dtype=np.float32)
    nob  = np.asarray(obj["next_observations"], dtype=np.float32)
    act  = np.asarray(obj["actions"], dtype=np.float32)
    rew  = np.asarray(obj["rewards"], dtype=np.float32)
    term = np.asarray(obj.get("terminals", np.zeros(len(rew), bool)), dtype=bool)
    tout = np.asarray(obj.get("timeouts", np.zeros(len(rew), bool)), dtype=bool)
    N = len(rew)
    # Episode boundaries when terminal OR timeout; default: treat timeout as end
    ends = np.nonzero(term | tout)[0]
    if len(ends) == 0: ends = np.array([N-1])
    starts = np.concatenate([[0], ends[:-1] + 1])
    episodes = []
    for s,e in zip(starts, ends):
        T = (e - s + 1)
        ep = dict(
            observation      = obs[s:e+1].astype(np.float32),
            next_observation = nob[s:e+1].astype(np.float32),
            action           = act[s:e+1].astype(np.float32),
            reward           = rew[s:e+1].astype(np.float32),
            is_terminal      = term[s:e+1].astype(bool),
        )
        ep["is_first"] = np.zeros(T, bool); ep["is_first"][0] = True
        ep["is_last"]  = np.zeros(T, bool); ep["is_last"][-1] = True
        ep["discount"] = np.ones(T, np.float32); ep["discount"][-1] = 0.0
        episodes.append(ep)
    return episodes

def _load_pkl_file(path: str) -> List[StepDict]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if _is_episode_list_pkl(obj):
        return _from_episode_list_pkl(obj)
    if _is_flat_d4rl_pkl(obj):
        return _segment_flat_to_episodes(obj)
    raise ValueError(f"Unrecognized PKL structure in {path}")

# ---------- tf.data builders ----------
def _steps_from_npz_file(path: tf.Tensor) -> tf.data.Dataset:
    path_str = path.numpy().decode("utf-8")
    ep = _load_npz_episode(path_str)
    return tf.data.Dataset.from_tensor_slices({k: v for k,v in ep.items()})

def _steps_from_pkl_file(path: tf.Tensor) -> tf.data.Dataset:
    path_str = path.numpy().decode("utf-8")
    eps = _load_pkl_file(path_str)
    # chain episodes
    ds = None
    for ep in eps:
        d = tf.data.Dataset.from_tensor_slices({k: v for k,v in ep.items()})
        ds = d if ds is None else ds.concatenate(d)
    return ds

def make_local_expert_dataset(
    root: str,
    batch_size: int | None,
    seed: int = 0,
    shuffle_files: bool = True,
    shuffle_steps: int = 10000,
    repeat: bool = True,
) -> tf.data.Dataset:
    npz = sorted(glob.glob(os.path.join(root, "*.npz")))
    pkl = sorted(glob.glob(os.path.join(root, "*.pkl")))
    if not npz and not pkl:
        raise FileNotFoundError(f"No .npz or .pkl found under {root}")

    # Build a files dataset
    files = npz + pkl
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle_files:
        files_ds = files_ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)

    def dispatch(path: tf.Tensor) -> tf.data.Dataset:
        p = path.numpy().decode("utf-8")
        if p.endswith(".npz"):
            return _steps_from_npz_file(path)
        else:
            return _steps_from_pkl_file(path)
    # py_function wrapper -> Flat map per-episode/per-step
    ds = files_ds.interleave(
        lambda p: tf.data.Dataset.from_generator(
            lambda _p=p: (x for x in tf.py_function(dispatch, [ _p ], Tout=[tf.string])._variant_tensor),  # noop, will be bypassed
            output_types=tf.string
        ),
        cycle_length=min(8, len(files)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    # The above trick is messy across TF versions; safer: use map(py_function)->from_tensor_slices
    # To avoid TF version issues, we rebuild the dataset in two stages:
    def map_to_steps(path):
        return tf.py_function(
            func=lambda p: 0, inp=[path], Tout=tf.int64)  # no-op placeholder
    # Simpler, more robust route (slightly less parallel): flat_map with py_function
    def flat_map_steps(path):
        return tf.py_function(
            func=lambda p: 0, inp=[path], Tout=tf.int64)

    # Simpler correct approach:
    def per_file_steps(path: tf.Tensor) -> tf.data.Dataset:
        q = path.numpy().decode("utf-8")
        if q.endswith(".npz"):
            return _steps_from_npz_file(path)
        return _steps_from_pkl_file(path)

    ds = files_ds.interleave(
        lambda p: tf.data.Dataset.from_generator(
            lambda _p=p: (yield_from_dataset(per_file_steps, _p)),  # defined below
            output_signature=None),  # signature supplied at runtime
        cycle_length=min(8, len(files)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    # Utility to bridge py_function + dataset-yield
def yield_from_dataset(fn, path):
    # This function will never be executed in-graph. It's here to keep editors happy.
    yield fn(path)  # pragma: no cover

    # Now shuffle/batch/prefetch
    if shuffle_steps:
        ds = ds.shuffle(shuffle_steps, seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    if batch_size:
        ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE)

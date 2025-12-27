# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example running coherent soft imitation learning on continuous control tasks."""

import os
from pathlib import Path
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


import math
from typing import Iterator
import csv, datetime, threading
from typing import Any
from acme.utils import loggers as acme_loggers  # if not already imported

from absl import flags
from acme import specs
from acme import types
from acme.agents.jax import sac
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import jax
import jax.random as rand
import launchpad as lp

import experiment_logger
import helpers
from sil import builder
from sil import config as sil_config
from sil import evaluator
from sil import networks

try:
    import wandb
except Exception:
    class _WandbStub:
        def setup(self): pass
        def finish(self): pass
    wandb = _WandbStub()

USE_SARSA = True

_DIST_FLAG = flags.DEFINE_bool(
    'run_distributed',
    False,
    (
        'Should an agent be executed in a distributed '
        'way. If False, will run single-threaded.'
    ),
)
_ENV_NAME = flags.DEFINE_string(
    'env_name', 'HalfCheetah-v2', 'Which environment to run'
)
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
_N_STEPS = flags.DEFINE_integer(
    'num_steps', 310_000, 'Number of env steps to run.'
)
_EVAL_RATIO = flags.DEFINE_integer(
    'eval_every', 1_000, 'How often to evaluate for local runs.'
)
_N_EVAL_EPS = flags.DEFINE_integer(
    'evaluation_episodes', 1, 'Evaluation episodes for local runs.'
)
_N_DEMONSTRATIONS = flags.DEFINE_integer(
    'num_demonstrations', 25, 'No. of demonstration trajectories.'
)
_N_OFFLINE_DATASET = flags.DEFINE_integer(
    'num_offline_demonstrations', 1_000, 'Offline dataset size.'
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 256, 'Batch size.')
_ENT_COEF = flags.DEFINE_float('entropy_coefficient', 0.01, 'Temperature')
_ENT_SF = flags.DEFINE_float('ent_sf', 1.0, 'Scale entropy target.')
_DAMP = flags.DEFINE_float('damping', 0.0, 'Constraint damping.')
_SF = flags.DEFINE_float('scale_factor', 1.0, 'Reward loss scale factor.')
_GNSF = flags.DEFINE_float(
    'grad_norm_scale_factor', 1.0, 'Critic grad scale factor.'
)
_DISCOUNT = flags.DEFINE_float('discount', 0.99, 'Discount factor')
_ACTOR_LR = flags.DEFINE_float('actor_lr', 3e-4, 'Actor learning rate.')
_CRITIC_LR = flags.DEFINE_float('critic_lr', 3e-4, 'Critic learning rate.')
_REWARD_LR = flags.DEFINE_float('reward_lr', 1e-3, 'Reward learning rate.')
_TAU = flags.DEFINE_float(
    'tau',
    0.005,
    (
        'Target network exponential smoothing weight.'
        '1. = no update, 0, = no smoothing.'
    ),
)
_CRITIC_ACTOR_RATIO = flags.DEFINE_integer(
    'critic_actor_update_ratio', 1, 'Critic updates per actor update.'
)
_SGD_STEPS = flags.DEFINE_integer(
    'sgd_steps', 1, 'SGD steps for online sample.'
)
_CRITIC_NETWORK = flags.DEFINE_multi_integer(
    'critic_network', [256, 256], 'Define critic architecture.'
)
_REWARD_NETWORK = flags.DEFINE_multi_integer(
    'reward_network', [256, 256], 'Define reward architecture. (Unused)'
)
_POLICY_NETWORK = flags.DEFINE_multi_integer(
    'policy_network', [256, 256, 12, 256], 'Define policy architecture.'
)
_POLICY_MODEL = flags.DEFINE_enum(
    'policy_model',
    networks.PolicyArchitectures.HETSTATTRI.value,
    [e.value for e in networks.PolicyArchitectures],
    'Define policy model type.',
)
_LAYERNORM = flags.DEFINE_bool(
    'policy_layer_norm', False, 'Use layer norm for first layer of the policy.'
)
_CRITIC_MODEL = flags.DEFINE_enum(
    'critic_model',
    networks.CriticArchitectures.LNMLP.value,
    [e.value for e in networks.CriticArchitectures],
    'Define critic model type.',
)
_REWARD_MODEL = flags.DEFINE_enum(
    'reward_model',
    networks.RewardArchitectures.PCSIL.value,
    [e.value for e in networks.RewardArchitectures],
    'Define reward model type.',
)
_RCALE = flags.DEFINE_float('reward_scaling', 1.0, 'Scale learned reward.')
_FINETUNE_R = flags.DEFINE_bool('finetune_reward', True, 'Finetune reward.')
_LOSS_TYPE = flags.DEFINE_enum(
    'loss_type',
    sil_config.Losses.FAITHFUL.value,
    [e.value for e in sil_config.Losses],
    'Define regression loss type.',
)
_POLICY_PRETRAIN_STEPS = flags.DEFINE_integer(
    'policy_pretrain_steps', 25_000, 'Policy pretraining steps.'
)
_POLICY_PRETRAIN_LR = flags.DEFINE_float(
    'policy_pretrain_lr', 1e-3, 'Policy pretraining learning rate.'
)
_CRITIC_PRETRAIN_STEPS = flags.DEFINE_integer(
    'critic_pretrain_steps', 5_000, 'Critic pretraining steps.'
)
_CRITIC_PRETRAIN_LR = flags.DEFINE_float(
    'critic_pretrain_lr', 1e-3, 'Critic pretraining learning rate.'
)
_EVAL_BC = flags.DEFINE_bool('eval_bc', False,
                             'Run evaluator of BC policy for comparison')
_OFFLINE_FLAG = flags.DEFINE_bool('offline', False, 'Run an offline agent.')
_EVAL_PER_VIDEO = flags.DEFINE_integer(
    'evals_per_video', 0, 'Video frequency. Disable using 0.'
)
_NUM_ACTORS = flags.DEFINE_integer(
    'num_actors', 4, 'Number of distributed actors.'
)
_CHECKPOINTING = flags.DEFINE_bool(
  'checkpoint', False, 'Save models during training.'
)
_WANDB = flags.DEFINE_bool(
  'wandb', False, 'Use weights and biases logging.'
)
_NAME = flags.DEFINE_string('name', 'camera-ready', 'Experiment name')

# NEW: where to read expert demos from
_EXPERT_BACKEND = flags.DEFINE_enum(
    'expert_backend', 'local', ['local'],
    'Expert dataset source (only "local" is supported: directory of .npz/.pkl).'
)
_EXPERT_PATH = flags.DEFINE_string(
    'expert_path', '',
    'If expert_backend="local", directory containing *.npz or *.pkl.'
)
_OFFLINE_PATH = flags.DEFINE_string(
    'offline_path', '',
    'If offline=True, directory containing .npz/.pkl for the offline dataset '
    '(defaults to --expert_path when empty).'
)
_LOGDIR = flags.DEFINE_string(
    'logdir', 'logs',
    'Directory to write local logs when --wandb=False.'
)
_CONSOLIDATED_CSV = flags.DEFINE_bool(
    'consolidated_csv', True,
    'If True and --wandb=False, write all metrics into a single CSV file.'
)
_CONSOLIDATED_CSV_NAME = flags.DEFINE_string(
    'consolidated_csv_name', 'metrics.csv',
    'Filename for the single consolidated CSV (relative to --logdir).'
)

# ========= Progress (epoch-based) CSV Logger =========
# Writes ONE "progress.csv" row ONLY when a full evaluation batch completes.
# Columns (fixed):
#   Epoch, Steps, Time, Eval/ReturnMean, Eval/ReturnStd, Eval/ReturnMax, Eval/ReturnMin,
#   Eval/EpisodeLengthMean, Eval/EpisodeLengthStd
from collections import defaultdict, OrderedDict
import csv as _csv
import datetime as _dt
import threading as _threading

class _OnlineStat:
  """Welford + min/max."""
  __slots__ = ("n","mean","M2","min","max")
  def __init__(self):
    self.n = 0
    self.mean = 0.0
    self.M2 = 0.0
    self.min = float("+inf")
    self.max = float("-inf")
  def add(self, x: float):
    try:
      x = float(x)
    except Exception:
      return
    self.n += 1
    # min/max
    if x < self.min: self.min = x
    if x > self.max: self.max = x
    # mean/std accumulators
    d = x - self.mean
    self.mean += d / self.n
    self.M2 += d * (x - self.mean)
  def get(self):
    if self.n <= 1:
      std = 0.0
    else:
      std = (self.M2 / (self.n - 1)) ** 0.5
    # If never updated, put sensible empties
    if self.n == 0:
      return 0.0, 0.0, "", ""
    return self.mean, std, self.max, self.min

class _ProgressState:
  """Global, shared across all logger instances in-process."""
  FIXED_HEADER = [
    "Epoch",
    "Steps",
    "Time",
    "Eval/ReturnMean",
    "Eval/ReturnStd",
    "Eval/ReturnMax",
    "Eval/ReturnMin",
    "Eval/EpisodeLengthMean",
    "Eval/EpisodeLengthStd",
  ]

  def __init__(self, csv_path: str, eval_every: int, episodes_per_eval: int):
    self.csv_path = csv_path
    self.episodes_per_eval = int(max(1, episodes_per_eval))
    self.lock = _threading.Lock()
    self.epoch_idx = 0

    # Per-eval accumulators (we only care about eval streams)
    # We may receive metrics under different evaluator labels; we merge them together.
    self.eval_returns = _OnlineStat()
    self.eval_ep_lengths = _OnlineStat()
    self.eval_episode_count = 0

    # Track last known step counters
    self.latest_steps = {}  # e.g., {"actor_steps": 12345, "learner_steps": 12000}

    # Prepare file with fixed header
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(csv_path).exists():
      with open(csv_path, "w", newline="") as f:
        _csv.writer(f).writerow(self.FIXED_HEADER)

  def _flush_if_ready_locked(self):
    """Flush ONLY when we have a full evaluation batch."""
    if self.eval_episode_count < self.episodes_per_eval:
      return

    # Steps: prefer actor_steps, else learner_steps, else empty
    steps = ""
    if "actor_steps" in self.latest_steps:
      steps = int(self.latest_steps["actor_steps"])
    elif "learner_steps" in self.latest_steps:
      steps = int(self.latest_steps["learner_steps"])

    # Aggregates
    r_mean, r_std, r_max, r_min = self.eval_returns.get()
    L_mean, L_std, _, _ = self.eval_ep_lengths.get()

    row = [
      int(self.epoch_idx),                    # Epoch
      steps,                                  # Steps
      _dt.datetime.now().isoformat(timespec="seconds"),  # Time
      r_mean, r_std, r_max, r_min,            # episodic return stats
      L_mean, L_std,                          # episode length stats
    ]

    with open(self.csv_path, "a", newline="") as f:
      _csv.writer(f).writerow(row)

    # Reset counters for the next epoch
    self.epoch_idx += 1
    self.eval_returns = _OnlineStat()
    self.eval_ep_lengths = _OnlineStat()
    self.eval_episode_count = 0

  def update(self, label: str, steps_key: str, data: dict,
             eval_labels=("imitation_evaluator","video_evaluator","bc_evaluator","evaluator","evaluation")):
    # Update last seen steps (store both if available)
    step_val = (data.get(steps_key)
                or data.get("actor_steps")
                or data.get("learner_steps")
                or data.get("evaluator_steps"))
    if step_val is not None:
      try:
        self.latest_steps[steps_key] = int(step_val)
      except Exception:
        pass

    # Only care about evaluation streams
    is_eval = any(lbl in label for lbl in eval_labels)
    if not is_eval:
      # Ignore all training logs entirely; we don't want them in CSV.
      return

    # Pull per-episode metrics we care about
    # We treat any log event carrying an episode_return as an episode boundary.
    # Accept numeric-like scalars from numpy/jax too.
    def _num(x):
      import numbers, numpy as _np
      try:
        if isinstance(x, numbers.Number):
          return float(x)
        if hasattr(x, "item"):
          xi = x.item()
          if isinstance(xi, numbers.Number):
            return float(xi)
        if isinstance(x, _np.ndarray) and x.size == 1:
          return float(x.reshape(()))
      except Exception:
        pass
      return None

    if "episode_return" in data:
      v = _num(data["episode_return"])
      if v is not None:
        self.eval_returns.add(v)
        self.eval_episode_count += 1

    if "episode_length" in data:
      v = _num(data["episode_length"])
      if v is not None:
        self.eval_ep_lengths.add(v)

    # Try flush
    with self.lock:
      self._flush_if_ready_locked()


# Shared progress state per process (created by the factory)
_PROGRESS_STATE = None

class ProgressCSVLogger:
  """Acme logger interface: collects eval writes; flushes ONE row per full eval."""
  def __init__(self, label: str, steps_key: str = "actor_steps"):
    self._label = label
    self._steps_key = steps_key
  def write(self, data: dict):
    if _PROGRESS_STATE is None:
      return
    try:
      _PROGRESS_STATE.update(self._label, self._steps_key, data)
    except Exception as e:
      try:
        print(f"[ProgressCSVLogger:{self._label}] logging error: {e}")
      except Exception:
        pass
  def close(self): pass

def make_progress_logger_factory(csv_path: str, eval_every: int, episodes_per_eval: int):
  """Returns a factory(label, steps_key=None, task_id=None, **kwargs) -> ProgressCSVLogger and binds global state."""
  # Note: eval_every is unused here by design (we flush strictly on full eval batch).
  global _PROGRESS_STATE
  _PROGRESS_STATE = _ProgressState(csv_path, eval_every, episodes_per_eval)
  def _factory(label: str, steps_key: str = 'actor_steps', task_id=None, **kwargs):
    return ProgressCSVLogger(label, steps_key)
  return _factory
# ===========================================



# ---- Compatibility shim: Reverb RateLimiter kwargs & ctor variants (0.7.x) ----
def _patch_reverb_rate_limiter():
    """
    Make reverb.rate_limiters.RateLimiter accept Acme's kwargs across older
    Reverb versions whose __init__ only supports positional args and may have
    (sps, minsz) or (sps, minsz, errb) signatures.
    """
    try:
        import reverb
    except Exception:
        return

    if getattr(reverb.rate_limiters, "_csil_rl_patched", False):
        return

    RL_cls = reverb.rate_limiters.RateLimiter

    def _rl_wrapper(*args, **kwargs):
        # Acme calls with keywords; older Reverb expects positionals.
        if args:
            return RL_cls(*args, **kwargs)

        sps  = kwargs.pop('samples_per_insert', None)
        minsz = kwargs.pop('min_size_to_sample', None)
        errb = kwargs.pop('error_buffer', None)
        last_exc = None

        for cand in (
            (sps, minsz, 0.0, 1e9),      # 4-arg: (sps, minsz, min_diff, max_diff)
            (sps, minsz, errb),          # 3-arg legacy
            (sps, minsz),                # 2-arg legacy
            (sps,),                      # 1-arg fallback
            tuple(),                     # empty
        ):
            if any(v is None for v in cand):
                continue
            try:
                return RL_cls(*cand)
            except TypeError as e:
                last_exc = e
                continue

        try:
            return RL_cls(**kwargs)
        except TypeError as e:
            raise (last_exc or e)

    reverb.rate_limiters.RateLimiter = _rl_wrapper
    reverb.rate_limiters._csil_rl_patched = True


# ---- Compatibility shim: Acme _disable_insert_blocking for Reverb 0.7.x Tables ----
def _patch_acme_disable_insert_blocking():
    """
    Replace acme.jax.experiments.run_experiment._disable_insert_blocking with a
    version that:
      * Builds a permissive RateLimiter (non-blocking inserts)
      * For each table, uses table.replace(...) if available, else reconstructs
        reverb.Table with the same attributes and new rate_limiter.
    """
    try:
        import reverb
        from acme.jax.experiments import run_experiment as _rxp
    except Exception:
        return

    if getattr(_rxp, "_csil_disable_insert_blocking_patched", False):
        return

    RL = reverb.rate_limiters.RateLimiter

    def _nonblocking_rate_limiter():
        # Target: samples_per_insert=0.0, min_size_to_sample=1, BIG error_buffer.
        # Try 3-arg then 2-arg, then fallbacks.
        try:
            return RL(0.0, 1, 1e9)
        except TypeError:
            try:
                return RL(0.0, 1)
            except TypeError:
                try:
                    return RL(0.0)
                except TypeError:
                    return RL()

    def _table_with_new_rl(table, new_rl):
        # 1) Preferred (newer Reverb): dataclass-like Table with .replace(...)
        try:
            return table.replace(rate_limiter=new_rl)
        except Exception:
            pass

        # 2) Older Reverb: reconstruct Table with same attributes.
        # Try kw ctor first (newer), then positional (older).
        name   = getattr(table, 'name', None)
        sampler = getattr(table, 'sampler', None)
        remover = getattr(table, 'remover', None)
        max_size = getattr(table, 'max_size', None)
        max_times_sampled = getattr(table, 'max_times_sampled', None)
        signature = getattr(table, 'signature', None)
        reserved_samples = getattr(table, 'reserved_samples', None)
        rate_limiter_timeout_ms = getattr(table, 'rate_limiter_timeout_ms', None)

        # Try kwargs ctor (superset)
        try:
            kwargs = dict(
                name=name,
                sampler=sampler,
                remover=remover,
                rate_limiter=new_rl,
                max_size=max_size,
                max_times_sampled=max_times_sampled,
            )
            if signature is not None:
                kwargs['signature'] = signature
            if reserved_samples is not None:
                kwargs['reserved_samples'] = reserved_samples
            if rate_limiter_timeout_ms is not None:
                kwargs['rate_limiter_timeout_ms'] = rate_limiter_timeout_ms
            return reverb.Table(**{k: v for k, v in kwargs.items() if v is not None})
        except TypeError:
            pass

        # Try positional ctor: (name, sampler, remover, rate_limiter, max_size, max_times_sampled)
        try:
            return reverb.Table(name, sampler, remover, new_rl, max_size, max_times_sampled)
        except Exception:
            # Give up: return original table unchanged.
            return table

    def _compatible_disable_insert_blocking(replay_tables):
        modified = []
        max_diff = 0.0
        for tbl in replay_tables:
            rl = _nonblocking_rate_limiter()
            new_tbl = _table_with_new_rl(tbl, rl)
            modified.append(new_tbl)
        return modified, max_diff

    _rxp._disable_insert_blocking = _compatible_disable_insert_blocking
    _rxp._csil_disable_insert_blocking_patched = True

def _build_experiment_config():
  """Builds a CSIL experiment config which can be executed in different ways."""

  # Create an environment, grab the spec, and use it to create networks.
  task = _ENV_NAME.value

  mode = f'{"off" if _OFFLINE_FLAG.value else "on"}line'
  name = f'csil_{task}_{mode}'
  group = (f'{name}, {_NAME.value}, '
           f'ndemos={_N_DEMONSTRATIONS.value}, '
           f'alpha={_ENT_COEF.value}')
  wandb_kwargs = {
    'project': 'csil',
    'name': name,
    'group': group,
    'tags': ['csil', task, mode, jax.default_backend()],
    'config': flags.FLAGS._flags(),
    'mode': 'online' if _WANDB.value else 'disabled',
  }

  # logger_fact = experiment_logger.make_experiment_logger_factory(wandb_kwargs)

  # Decide logging backend
  if _WANDB.value:
    # Silence W&B prompts, keep online/offline as you wish.
    os.environ.setdefault('WANDB_SILENT', 'true')
    os.environ.setdefault('WANDB_MODE', 'online')   # or 'offline'
    logger_fact = experiment_logger.make_experiment_logger_factory(wandb_kwargs)
  else:
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['WANDB_MODE'] = 'disabled'
    # Our progress.csv epoch logger (one row per evaluation/epoch)
    algo = 'csil-offline' if _OFFLINE_FLAG.value else 'csil'
    run_dir = os.path.join(_LOGDIR.value, algo, _ENV_NAME.value, f"d{_N_DEMONSTRATIONS.value}", f"s{_SEED.value}")
    csv_path = os.path.join(run_dir, "progress.csv")
    logger_fact = make_progress_logger_factory(
        csv_path, eval_every=_EVAL_RATIO.value, episodes_per_eval=_N_EVAL_EPS.value
    )


  make_env, env_spec, make_demonstrations = helpers.get_env_and_demonstrations(
    task,
    _N_DEMONSTRATIONS.value,
    use_sarsa=USE_SARSA,
    in_memory=('image' not in task),
    expert_backend=_EXPERT_BACKEND.value,
    expert_path=_EXPERT_PATH.value,
  )

  def environment_factory(seed: int):
    del seed
    return make_env()

  batch_size = _BATCH_SIZE.value
  seed = _SEED.value
  actor_lr = _ACTOR_LR.value

  make_demonstrations_ = lambda batchsize: make_demonstrations(batchsize, seed)

  if _ENT_COEF.value > 0.0:
    kwargs = {'entropy_coefficient': _ENT_COEF.value}
  else:
    target_entropy = _ENT_SF.value * sac.target_entropy_from_env_spec(
        env_spec, target_entropy_per_dimension=abs(_ENT_SF.value))
    kwargs = {'target_entropy': target_entropy}

  # Important step that normalizes reward values -- do not change!
  csil_alpha = _RCALE.value / math.prod(env_spec.actions.shape)

  policy_architecture = networks.PolicyArchitectures(_POLICY_MODEL.value)
  bc_policy_architecture = policy_architecture
  critic_architecture = networks.CriticArchitectures(_CRITIC_MODEL.value)
  reward_architecture = networks.RewardArchitectures(_REWARD_MODEL.value)
  policy_layers = _POLICY_NETWORK.value
  reward_layers = _REWARD_NETWORK.value
  critic_layers = _CRITIC_NETWORK.value
  use_layer_norm = _LAYERNORM.value

  def network_factory(spec: specs.EnvironmentSpec):
    return networks.make_networks(
        spec=spec,
        reward_policy_coherence_alpha=csil_alpha,
        policy_architecture=policy_architecture,
        critic_architecture=critic_architecture,
        reward_architecture=reward_architecture,
        bc_policy_architecture=bc_policy_architecture,
        policy_hidden_layer_sizes=tuple(policy_layers),
        reward_hidden_layer_sizes=tuple(reward_layers),
        critic_hidden_layer_sizes=tuple(critic_layers),
        bc_policy_hidden_layer_sizes=tuple(policy_layers),
        layer_norm_policy=use_layer_norm,
    )

  demo_factory = lambda seed_: make_demonstrations(batch_size, seed_)
  policy_pretraining = sil_config.PretrainingConfig(
      loss=sil_config.Losses(_LOSS_TYPE.value),
      seed=seed,
      dataset_factory=demo_factory,
      steps=_POLICY_PRETRAIN_STEPS.value,
      learning_rate=_POLICY_PRETRAIN_LR.value,
      use_as_reference=True,
  )
  if _OFFLINE_FLAG.value:
    _, offline_dataset = helpers.get_offline_dataset(
        task,
        env_spec,
        _N_DEMONSTRATIONS.value,
        _N_OFFLINE_DATASET.value,
        use_sarsa=USE_SARSA,
        expert_backend=_EXPERT_BACKEND.value,
        expert_path=_EXPERT_PATH.value,
        offline_path=_OFFLINE_PATH.value or _EXPERT_PATH.value,
    )

    def offline_pretraining_dataset(rseed: int) -> Iterator[types.Transition]:
      rkey = rand.PRNGKey(rseed)
      return helpers.MixedIterator(
          offline_dataset(batch_size, rkey),
          make_demonstrations(batch_size, rseed),
      )

    offline_policy_pretraining = sil_config.PretrainingConfig(
        loss=sil_config.Losses(_LOSS_TYPE.value),
        seed=seed,
        dataset_factory=offline_pretraining_dataset,
        steps=_POLICY_PRETRAIN_STEPS.value,
        learning_rate=_POLICY_PRETRAIN_LR.value,
    )
    policy_pretrainers = [offline_policy_pretraining, policy_pretraining]
    critic_dataset = demo_factory
  else:
    policy_pretrainers = [policy_pretraining,]
    critic_dataset = demo_factory

  critic_pretraining = sil_config.PretrainingConfig(
      seed=seed,
      dataset_factory=critic_dataset,
      steps=_CRITIC_PRETRAIN_STEPS.value,
      learning_rate=_CRITIC_PRETRAIN_LR.value,
  )
  # Construct the agent.
  config_ = sil_config.SILConfig(
      imitation=sil_config.CoherentConfig(
          alpha=csil_alpha,
          reward_scaling=_RCALE.value,
          refine_reward=_FINETUNE_R.value,
          negative_reward=(
              reward_architecture == networks.RewardArchitectures.NCSIL),
          grad_norm_sf=_GNSF.value,
          scale_factor=_SF.value,
      ),
      actor_bc_loss=False,
      policy_pretraining=policy_pretrainers,
      critic_pretraining=critic_pretraining,
      expert_demonstration_factory=make_demonstrations_,
      discount=_DISCOUNT.value,
      critic_learning_rate=_CRITIC_LR.value,
      reward_learning_rate=_REWARD_LR.value,
      actor_learning_rate=actor_lr,
      num_sgd_steps_per_step=_SGD_STEPS.value,
      critic_actor_update_ratio=_CRITIC_ACTOR_RATIO.value,
      n_step=1,
      damping=_DAMP.value,
      tau=_TAU.value,
      batch_size=batch_size,
      samples_per_insert=batch_size,
      alpha_learning_rate=1e-2,
      alpha_init=0.01,
      **kwargs,
  )

  sil_builder = builder.SILBuilder(config_)

  imitation_evaluator_factory = evaluator.imitation_evaluator_factory(
      agent_config=config_,
      environment_factory=environment_factory,
      network_factory=network_factory,
      policy_factory=sil_builder.make_policy,
      logger_factory=logger_fact,
  )

  evaluators = [imitation_evaluator_factory,]
  if _EVAL_PER_VIDEO.value > 0:
    video_evaluator_factory = evaluator.video_evaluator_factory(
        environment_factory=environment_factory,
        network_factory=network_factory,
        policy_factory=sil_builder.make_policy,
        videos_per_eval=_EVAL_PER_VIDEO.value,
        logger_factory=logger_fact,
    )
    evaluators += [imitation_evaluator_factory,]

  if _EVAL_BC.value:
    bc_evaluator_factory = evaluator.bc_evaluator_factory(
        environment_factory=environment_factory,
        network_factory=network_factory,
        policy_factory=sil_builder.make_bc_policy,
        logger_factory=logger_fact,
    )
    evaluators += [bc_evaluator_factory,]

  checkpoint_config = (experiments.CheckointingConfig()
                       if _CHECKPOINTING.value else None)
  if _OFFLINE_FLAG.value:
    make_offline_dataset, _ = helpers.get_offline_dataset(
        task,
        env_spec,
        _N_DEMONSTRATIONS.value,
        _N_OFFLINE_DATASET.value,
        use_sarsa=USE_SARSA,
        expert_backend=_EXPERT_BACKEND.value,
        expert_path=_EXPERT_PATH.value,
        offline_path=_OFFLINE_PATH.value or _EXPERT_PATH.value,
    )
      
    make_offline_dataset_ = lambda rk: make_offline_dataset(batch_size, rk)
    return experiments.OfflineExperimentConfig(
        builder=sil_builder,
        environment_factory=environment_factory,
        network_factory=network_factory,
        demonstration_dataset_factory=make_offline_dataset_,
        evaluator_factories=evaluators,
        max_num_learner_steps=_N_STEPS.value,
        environment_spec=env_spec,
        seed=_SEED.value,
        logger_factory=logger_fact,
        checkpointing=checkpoint_config,
    )
  else:
    return experiments.ExperimentConfig(
        builder=sil_builder,
        environment_factory=environment_factory,
        network_factory=network_factory,
        evaluator_factories=evaluators,
        seed=_SEED.value,
        max_num_actor_steps=_N_STEPS.value,
        logger_factory=logger_fact,
        checkpointing=checkpoint_config,
    )

def main(_):
  # ---- W&B setup/disable AFTER flags are parsed ----
  try:
    if _WANDB.value:
      # Quiet WandB + default to online unless user already set it.
      os.environ.setdefault('WANDB_SILENT', 'true')
      os.environ.setdefault('WANDB_MODE', 'online')   # or 'offline' if you prefer
      try:
        wandb.setup()
      except Exception:
        pass
    else:
      # Fully disable any W&B prompting
      os.environ['WANDB_SILENT'] = 'true'
      os.environ['WANDB_MODE'] = 'disabled'
  except Exception:
    # If anything odd happens here, just proceed without WandB.
    pass

  _patch_reverb_rate_limiter()
  _patch_acme_disable_insert_blocking()
    
  try:
    # Build the experiment config (this already constructs logger_factory in your file)
    config = _build_experiment_config()

    # Run exactly like before
    if _DIST_FLAG.value:
      if _OFFLINE_FLAG.value:
        program = experiments.make_distributed_offline_experiment(
            experiment=config
        )
      else:
        program = experiments.make_distributed_experiment(
            experiment=config, num_actors=_NUM_ACTORS.value
        )
      lp.launch(
          program,
          xm_resources=lp_utils.make_xm_docker_resources(program),
      )
    else:
      if _OFFLINE_FLAG.value:
        experiments.run_offline_experiment(
            experiment=config,
            eval_every=_EVAL_RATIO.value,
            num_eval_episodes=_N_EVAL_EPS.value,
        )
      else:
        experiments.run_experiment(
            experiment=config,
            eval_every=_EVAL_RATIO.value,
            num_eval_episodes=_N_EVAL_EPS.value,
        )
  finally:
    # ---- W&B finish (only if enabled) ----
    try:
      if _WANDB.value:
        wandb.finish()
    except Exception:
      pass

if __name__ == '__main__':
  app.run(main)

# Copyright 2023 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""
Behavior Cloning (BC) runner compatible with the modified CSIL repo layout.

- Uses your local `helpers.py` (as provided) to load env & demos.
- Uses the same BC hyperparameters as `run_csil.py` (policy arch, LR, steps).
- Evaluates like CSIL's bc_evaluator and writes a CSV `progress.csv`
  (one row per evaluation batch) under:
    <logdir>/bc/<env_name>/d<num_demonstrations>/s<seed>/progress.csv
"""
import os
from pathlib import Path
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import math
from absl import app, flags
import jax
from acme import specs
from acme.agents.jax import sac

import helpers
from sil import builder
from sil import config as sil_config
from sil import evaluator
from sil import networks

# Optional W&B (same pattern as run_csil.py)
try:
    import wandb
except Exception:
    class _WandbStub:
        def setup(self): pass
        def finish(self): pass
    wandb = _WandbStub()

# =====================
# Flags (mirroring CSIL)
# =====================
_ENV_NAME = flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment ID')
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 256, 'Batch size.')
_N_DEMONSTRATIONS = flags.DEFINE_integer('num_demonstrations', 25, 'No. of demonstration trajectories.')

# Evaluation cadence
_EVAL_RATIO = flags.DEFINE_integer('eval_every', 1_000, 'How often to evaluate (ignored by the CSV logger, kept for parity).')
_N_EVAL_EPS = flags.DEFINE_integer('evaluation_episodes', 10, 'Episodes per evaluation batch.')

# Policy architecture & training hyperparams (use exactly the CSIL defaults)
_POLICY_NETWORK = flags.DEFINE_multi_integer('policy_network', [256, 256, 12, 256], 'Policy hidden sizes.')
_POLICY_MODEL = flags.DEFINE_enum(
    'policy_model',
    networks.PolicyArchitectures.HETSTATTRI.value,
    [e.value for e in networks.PolicyArchitectures],
    'Policy model type.'
)
_LAYERNORM = flags.DEFINE_bool('policy_layer_norm', False, 'LayerNorm on the first layer.')
_POLICY_PRETRAIN_STEPS = flags.DEFINE_integer('policy_pretrain_steps', 25_000, 'BC pretraining steps.')
_POLICY_PRETRAIN_LR = flags.DEFINE_float('policy_pretrain_lr', 1e-3, 'BC learning rate.')

# (Minor extras to satisfy networks/config construction; kept aligned with CSIL)
_CRITIC_NETWORK = flags.DEFINE_multi_integer('critic_network', [256, 256], 'Critic sizes (unused here).')
_REWARD_NETWORK = flags.DEFINE_multi_integer('reward_network', [256, 256], 'Reward sizes (unused here).')
_CRITIC_MODEL = flags.DEFINE_enum(
    'critic_model',
    networks.CriticArchitectures.LNMLP.value,
    [e.value for e in networks.CriticArchitectures],
    'Critic model type.'
)
_REWARD_MODEL = flags.DEFINE_enum(
    'reward_model',
    networks.RewardArchitectures.PCSIL.value,
    [e.value for e in networks.RewardArchitectures],
    'Reward model type.'
)
_DISCOUNT = flags.DEFINE_float('discount', 0.99, 'Discount (for completeness).')
_TAU = flags.DEFINE_float('tau', 0.005, 'Target smoothing (for completeness).')
_CRITIC_ACTOR_RATIO = flags.DEFINE_integer('critic_actor_update_ratio', 1, 'Critic:actor updates (unused).')
_SGD_STEPS = flags.DEFINE_integer('sgd_steps', 1, 'SGD steps per online step (unused).')
_BATCH_SPS = flags.DEFINE_integer('samples_per_insert', 256, 'Samples per insert (unused).')

# Reward scaling alpha (as in CSIL; needed to build networks correctly)
_RCALE = flags.DEFINE_float('reward_scaling', 1.0, 'Scale learned reward (affects csil alpha computation).')

# Expert dataset location (same interface as run_csil.py)
_EXPERT_BACKEND = flags.DEFINE_enum('expert_backend', 'local', ['local'], 'Where to read expert demos (only "local").')
_EXPERT_PATH = flags.DEFINE_string('expert_path', '', 'Directory containing *.npz/*.pkl expert trajectories.')

# Logging
_LOGDIR = flags.DEFINE_string('logdir', 'logs', 'Base directory for local logs.')
_WANDB = flags.DEFINE_bool('wandb', False, 'Use Weights & Biases logging.')
_NAME = flags.DEFINE_string('name', 'bc_eval', 'Experiment name.')

# Keep in sync with helpers/get_env_and_demonstrations default
USE_SARSA = True

# ==============
# CSV Logger (1 row per evaluation batch), copied/adapted from run_csil.py
# ==============
from collections import defaultdict, OrderedDict
import csv as _csv
import datetime as _dt
import threading as _threading
import numbers
import numpy as _np

class _OnlineStat:
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
        if x < self.min: self.min = x
        if x > self.max: self.max = x
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)
    def get(self):
        if self.n <= 1:
            std = 0.0
        else:
            std = (self.M2 / (self.n - 1)) ** 0.5
        if self.n == 0:
            return 0.0, 0.0, "", ""
        return self.mean, std, self.max, self.min

class _ProgressState:
    FIXED_HEADER = [
        "Epoch","Steps","Time",
        "Eval/ReturnMean","Eval/ReturnStd","Eval/ReturnMax","Eval/ReturnMin",
        "Eval/EpisodeLengthMean","Eval/EpisodeLengthStd",
    ]
    def __init__(self, csv_path: str, episodes_per_eval: int):
        self.csv_path = csv_path
        self.episodes_per_eval = int(max(1, episodes_per_eval))
        self.lock = _threading.Lock()
        self.epoch_idx = 0
        self.eval_returns = _OnlineStat()
        self.eval_ep_lengths = _OnlineStat()
        self.eval_episode_count = 0
        self.latest_steps = {}
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(csv_path).exists():
            with open(csv_path, "w", newline="") as f:
                _csv.writer(f).writerow(self.FIXED_HEADER)
    def _flush_if_ready_locked(self):
        if self.eval_episode_count < self.episodes_per_eval:
            return
        steps = self.latest_steps.get("actor_steps", self.latest_steps.get("learner_steps", ""))
        r_mean, r_std, r_max, r_min = self.eval_returns.get()
        L_mean, L_std, _, _ = self.eval_ep_lengths.get()
        row = [int(self.epoch_idx), steps, _dt.datetime.now().isoformat(timespec="seconds"),
               r_mean, r_std, r_max, r_min, L_mean, L_std]
        with open(self.csv_path, "a", newline="") as f:
            _csv.writer(f).writerow(row)
        self.epoch_idx += 1
        self.eval_returns = _OnlineStat()
        self.eval_ep_lengths = _OnlineStat()
        self.eval_episode_count = 0
    def update(self, label: str, data: dict):
        step_val = (data.get("actor_steps") or data.get("learner_steps") or data.get("evaluator_steps"))
        if step_val is not None:
            try:
                self.latest_steps["actor_steps"] = int(step_val)
            except Exception:
                pass
        is_eval = any(lbl in label for lbl in ("bc_evaluator","imitation_evaluator","evaluator","evaluation","video_evaluator"))
        if not is_eval:
            return
        def _num(x):
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
        with self.lock:
            self._flush_if_ready_locked()

class ProgressCSVLogger:
    def __init__(self, label: str, state: _ProgressState):
        self._label = label
        self._state = state
    def write(self, data: dict):
        try:
            self._state.update(self._label, data)
        except Exception as e:
            try: print(f"[ProgressCSVLogger:{self._label}] error: {e}")
            except Exception: pass
    def close(self): pass

def make_progress_logger_factory(csv_path: str, episodes_per_eval: int):
    state = _ProgressState(csv_path, episodes_per_eval)
    def _factory(label: str, steps_key: str = 'actor_steps', task_id=None, **kwargs):
        return ProgressCSVLogger(label, state)
    return _factory

# ----------------------------------
# Reverb/Acme compatibility shims (copied)
# ----------------------------------
def _patch_reverb_rate_limiter():
    try:
        import reverb
    except Exception:
        return
    if getattr(reverb.rate_limiters, "_bc_rl_patched", False):
        return
    RL_cls = reverb.rate_limiters.RateLimiter
    def _rl_wrapper(*args, **kwargs):
        if args:
            return RL_cls(*args, **kwargs)
        sps  = kwargs.pop('samples_per_insert', None)
        minsz = kwargs.pop('min_size_to_sample', None)
        errb = kwargs.pop('error_buffer', None)
        last_exc = None
        for cand in ((sps, minsz, 0.0, 1e9), (sps, minsz, errb), (sps, minsz), (sps,), tuple()):
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
    reverb.rate_limiters._bc_rl_patched = True

def _patch_acme_disable_insert_blocking():
    try:
        import reverb
        from acme.jax.experiments import run_experiment as _rxp
    except Exception:
        return
    if getattr(_rxp, "_bc_disable_insert_blocking_patched", False):
        return
    RL = reverb.rate_limiters.RateLimiter
    def _nonblocking_rate_limiter():
        try: return RL(0.0, 1, 1e9)
        except TypeError:
            try: return RL(0.0, 1)
            except TypeError:
                try: return RL(0.0)
                except TypeError: return RL()
    def _table_with_new_rl(table, new_rl):
        try: return table.replace(rate_limiter=new_rl)
        except Exception: pass
        name   = getattr(table, 'name', None)
        sampler = getattr(table, 'sampler', None)
        remover = getattr(table, 'remover', None)
        max_size = getattr(table, 'max_size', None)
        max_times_sampled = getattr(table, 'max_times_sampled', None)
        signature = getattr(table, 'signature', None)
        reserved_samples = getattr(table, 'reserved_samples', None)
        rate_limiter_timeout_ms = getattr(table, 'rate_limiter_timeout_ms', None)
        try:
            kwargs = dict(name=name, sampler=sampler, remover=remover, rate_limiter=new_rl,
                          max_size=max_size, max_times_sampled=max_times_sampled)
            if signature is not None: kwargs['signature'] = signature
            if reserved_samples is not None: kwargs['reserved_samples'] = reserved_samples
            if rate_limiter_timeout_ms is not None: kwargs['rate_limiter_timeout_ms'] = rate_limiter_timeout_ms
            return reverb.Table(**{k: v for k, v in kwargs.items() if v is not None})
        except TypeError: pass
        try:
            return reverb.Table(name, sampler, remover, new_rl, max_size, max_times_sampled)
        except Exception:
            return table
    def _compatible_disable_insert_blocking(replay_tables):
        modified = []
        for tbl in replay_tables:
            rl = _nonblocking_rate_limiter()
            modified.append(_table_with_new_rl(tbl, rl))
        return modified, 0.0
    _rxp._disable_insert_blocking = _compatible_disable_insert_blocking
    _rxp._bc_disable_insert_blocking_patched = True

# ==============
# Build BC-only experiment
# ==============
def _build_bc_experiment_config():
    task = _ENV_NAME.value
    name = f'bc_{task}'
    group = f'bc, {_NAME.value}, ndemos={_N_DEMONSTRATIONS.value}'
    wandb_kwargs = {
        'project': 'csil',
        'name': name,
        'group': group,
        'tags': ['bc', task, jax.default_backend()],
        'config': flags.FLAGS._flags(),
        'mode': 'online' if _WANDB.value else 'disabled',
    }

    # Logging backend
    if _WANDB.value:
        os.environ.setdefault('WANDB_SILENT', 'true')
        os.environ.setdefault('WANDB_MODE', 'online')
        import experiment_logger
        logger_fact = experiment_logger.make_experiment_logger_factory(wandb_kwargs)
    else:
        os.environ['WANDB_SILENT'] = 'true'
        os.environ['WANDB_MODE'] = 'disabled'
        run_dir = os.path.join(_LOGDIR.value, 'bc', _ENV_NAME.value, f"d{_N_DEMONSTRATIONS.value}", f"s{_SEED.value}")
        csv_path = os.path.join(run_dir, "progress.csv")
        logger_fact = make_progress_logger_factory(csv_path, episodes_per_eval=_N_EVAL_EPS.value)

    # Env & demos
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

    # CSIL-style alpha normalization (depends on action dims)
    csil_alpha = _RCALE.value / math.prod(env_spec.actions.shape)

    # Architectures (mirror CSIL)
    policy_architecture = networks.PolicyArchitectures(_POLICY_MODEL.value)
    critic_architecture = networks.CriticArchitectures(_CRITIC_MODEL.value)
    reward_architecture = networks.RewardArchitectures(_REWARD_MODEL.value)
    policy_layers = _POLICY_NETWORK.value
    critic_layers = _CRITIC_NETWORK.value
    reward_layers = _REWARD_NETWORK.value
    use_layer_norm = _LAYERNORM.value

    def network_factory(spec: specs.EnvironmentSpec):
        return networks.make_networks(
            spec=spec,
            reward_policy_coherence_alpha=csil_alpha,
            policy_architecture=policy_architecture,
            critic_architecture=critic_architecture,
            reward_architecture=reward_architecture,
            bc_policy_architecture=policy_architecture,
            policy_hidden_layer_sizes=tuple(policy_layers),
            reward_hidden_layer_sizes=tuple(reward_layers),
            critic_hidden_layer_sizes=tuple(critic_layers),
            bc_policy_hidden_layer_sizes=tuple(policy_layers),
            layer_norm_policy=use_layer_norm,
        )

    # Dataset factory for BC
    demo_factory = lambda seed_: make_demonstrations(batch_size, seed_)

    # Policy pretraining config (this is THE BC training)
    policy_pretraining = sil_config.PretrainingConfig(
        loss=sil_config.Losses.FAITHFUL,  # same default as run_csil
        seed=seed,
        dataset_factory=demo_factory,
        steps=_POLICY_PRETRAIN_STEPS.value,
        learning_rate=_POLICY_PRETRAIN_LR.value,
        use_as_reference=True,
    )

    # Critic pretrain (no-op, but builder expects a config)
    critic_pretraining = sil_config.PretrainingConfig(
        seed=seed,
        dataset_factory=demo_factory,
        steps=0,
        learning_rate=1e-6,
    )

    # Minimal SILConfig so the builder can construct networks & the BC policy.
    config_ = sil_config.SILConfig(
        imitation=sil_config.CoherentConfig(
            alpha=csil_alpha,
            reward_scaling=_RCALE.value,
            refine_reward=False,
            negative_reward=(reward_architecture == networks.RewardArchitectures.NCSIL),
            grad_norm_sf=1.0,
            scale_factor=1.0,
        ),
        actor_bc_loss=False,  # NOT mixing BC into actor loss — pure BC.
        policy_pretraining=[policy_pretraining],
        critic_pretraining=critic_pretraining,
        expert_demonstration_factory=lambda bsz: make_demonstrations(bsz, seed),
        discount=_DISCOUNT.value,
        critic_learning_rate=3e-4,
        reward_learning_rate=1e-3,
        actor_learning_rate=3e-4,
        num_sgd_steps_per_step=_SGD_STEPS.value,
        critic_actor_update_ratio=_CRITIC_ACTOR_RATIO.value,
        n_step=1,
        damping=0.0,
        tau=_TAU.value,
        batch_size=batch_size,
        samples_per_insert=_BATCH_SPS.value,
        alpha_learning_rate=1e-2,
        alpha_init=0.01,
        entropy_coefficient=0.01,  # default temp (not really used here)
    )

    bc_builder = builder.SILBuilder(config_)

    # Only the BC evaluator
    bc_eval_factory = evaluator.bc_evaluator_factory(
        environment_factory=environment_factory,
        network_factory=network_factory,
        policy_factory=bc_builder.make_bc_policy,
        logger_factory=logger_fact,
    )
    evaluators = [bc_eval_factory]

    # Build standard experiment config (actor steps kept tiny; evaluation is what we want)
    from acme.jax import experiments as _experiments
    return _experiments.ExperimentConfig(
        builder=bc_builder,
        environment_factory=environment_factory,
        network_factory=network_factory,
        evaluator_factories=evaluators,
        seed=_SEED.value,
        max_num_actor_steps=1,  # tiny loop; evaluation will run once per batch
        logger_factory=logger_fact,
        checkpointing=None,
    )

def main(_):
    try:
        if _WANDB.value:
            os.environ.setdefault('WANDB_SILENT', 'true')
            os.environ.setdefault('WANDB_MODE', 'online')
            try: wandb.setup()
            except Exception: pass
        else:
            os.environ['WANDB_SILENT'] = 'true'
            os.environ['WANDB_MODE'] = 'disabled'
    except Exception:
        pass

    _patch_reverb_rate_limiter()
    _patch_acme_disable_insert_blocking()

    # Build & run
    config = _build_bc_experiment_config()

    from acme.jax import experiments as _experiments
    # One evaluation batch; rely on our CSV logger to flush exactly once per batch.
    _experiments.run_experiment(
        experiment=config,
        eval_every=1,  # ensure early evaluation
        num_eval_episodes=_N_EVAL_EPS.value,
    )

    try:
        if _WANDB.value:
            wandb.finish()
    except Exception:
        pass

if __name__ == '__main__':
    app.run(main)

import optuna
import inspect
import math
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

def optimize_sb3(trial, model_class, vec_env, eval_env_fn, model_kwargs_fn, total_timesteps=40000):
    """
    Generic Optuna optimization function for SB3 models.
    - model_class: SB3 model class (e.g., PPO, A2C, DDPG)
    - vec_env: vectorized training environment
    - eval_env_fn: function to create a new evaluation environment
    - model_kwargs_fn: function(trial) -> dict of model kwargs (including policy_kwargs)
    - total_timesteps: training steps per trial
    """
    model_kwargs = model_kwargs_fn(trial, vec_env)
    model = model_class("MlpPolicy", vec_env, verbose=0, **model_kwargs)
    model.learn(total_timesteps=total_timesteps)
    mean_reward, _ = evaluate_policy(model, Monitor(eval_env_fn()), n_eval_episodes=3, deterministic=False)
    return mean_reward

# PPO hyperparameter suggestion function
def ppo_model_kwargs_fn(trial, vec_env=None):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_choice == "small":
        net_arch = [64, 64]
    elif net_arch_choice == "medium":
        net_arch = [256, 256]
    else:
        net_arch = [400, 300]
    policy_kwargs = dict(net_arch=net_arch)
    return dict(
        learning_rate=learning_rate,
        gamma=gamma,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        policy_kwargs=policy_kwargs,
        device="cpu"
    )

# A2C hyperparameter suggestion function
def a2c_model_kwargs_fn(trial, vec_env=None):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_choice == "small":
        net_arch = [64, 64]
    elif net_arch_choice == "medium":
        net_arch = [256, 256]
    else:
        net_arch = [400, 300]
    policy_kwargs = dict(net_arch=net_arch)
    return dict(
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        policy_kwargs=policy_kwargs,
        device="cpu"
    )


# SAC hyperparameter suggestion function
def sac_model_kwargs_fn(trial, vec_env=None):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.02)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_choice == "small":
        net_arch = [64, 64]
    elif net_arch_choice == "medium":
        net_arch = [256, 256]
    else:
        net_arch = [400, 300]
    policy_kwargs = dict(net_arch=net_arch)
    return dict(
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        ent_coef=ent_coef,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs
    )

# TD3 hyperparameter suggestion function
def td3_model_kwargs_fn(trial,vec_env=None):
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.02)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_choice == "small":
        net_arch = [64, 64]
    elif net_arch_choice == "medium":
        net_arch = [256, 256]
    else:
        net_arch = [400, 300]
    policy_kwargs = dict(net_arch=net_arch)
    if vec_env is None or not hasattr(vec_env, "action_space"):
        raise ValueError("vec_env must be provided and have an action_space attribute for TD3.")
    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )
    return dict(
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        action_noise=action_noise
    )

# DDPG hyperparameter suggestion function
def ddpg_model_kwargs_fn(trial, vec_env=None):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    tau = trial.suggest_float("tau", 0.001, 0.02)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_choice == "small":
        net_arch = [64, 64]
    elif net_arch_choice == "medium":
        net_arch = [256, 256]
    else:
        net_arch = [400, 300]
    policy_kwargs = dict(net_arch=net_arch)
    if vec_env is None or not hasattr(vec_env, "action_space"):
        raise ValueError("vec_env must be provided and have an action_space attribute for DDPG.")
    n_actions = vec_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.2 * np.ones(n_actions)
    )
    return dict(
        learning_rate=learning_rate,
        tau=tau,
        gamma=gamma,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        action_noise=action_noise
    )

MAX_ENVS = 64                 # tune to your RAM/CPU
EPOCHS_PER_CHUNK = 2          # tune

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def get_chunk_size(total_items, max_envs):
    """Return the largest divisor of total_items that is <= max_envs.

    This ensures we can split the dataset into equal-size chunks so
    `model.set_env` won't complain about mismatched environment counts.
    """
    if total_items <= 0:
        raise ValueError("total_items must be positive")
    for i in range(min(max_envs, total_items), 0, -1):
        if total_items % i == 0:
            return i
    return 1

def train_model(
    model_class,
    vec_env,
    eval_env_fn,
    test_timesteps=40000,
    total_timesteps=4000000,
    n_trials=10,
    n_jobs=10,
    default_model=False,
    model_kwargs_override=None,
    model_post_create_fn=None,
):
    """
    Trains a reinforcement learning model using Stable Baselines3, with hyperparameter tuning via Optuna,
    and evaluates its performance before and after training.

    Supports two modes for `vec_env`:
      - A VecEnv instance (existing behavior)
      - A list/tuple of environment factory functions (callables that return envs)
        In the latter case the function will train in chunks of at most `MAX_ENVS` envs to avoid
        exhausting system resources (CPU/RAM). Total timesteps are approximately evenly
        distributed across chunks.
    """
    eval_result = {}

    # Determine if we were given a list of env factory functions
    is_env_list = isinstance(vec_env, (list, tuple)) and len(vec_env) > 0 and callable(vec_env[0])
    env_fns = list(vec_env) if is_env_list else None

    # If we have a list of envs, prepare a small dummy vec env for tuning
    tuning_vec = None
    if is_env_list:
        print("Detected list of environment functions; using chunked training.")
        chunk_size = get_chunk_size(len(env_fns), MAX_ENVS)
        num_chunks = math.ceil(len(env_fns) / chunk_size)
        per_chunk_timesteps = math.ceil(total_timesteps / max(1, num_chunks))
        # use a small DummyVecEnv (first chunk) for hyperparameter tuning
        tuning_chunk = list(next(chunked(env_fns, chunk_size)))
        tuning_vec = DummyVecEnv(tuning_chunk)
        tune_env_for_opt = tuning_vec
    else:
        print("Using provided vectorized environment for training.")
        tune_env_for_opt = vec_env

    if default_model is False:
        # Validate algorithm selection
        if model_class not in [PPO, A2C, DDPG, SAC, TD3]:
            raise ValueError("model_class must be one of [PPO, A2C, DDPG, SAC, TD3]")
        if model_class == PPO:
            model_kwargs_fn = ppo_model_kwargs_fn
        elif model_class == A2C:
            model_kwargs_fn = a2c_model_kwargs_fn
        elif model_class == DDPG:
            model_kwargs_fn = ddpg_model_kwargs_fn
        elif model_class == SAC:
            model_kwargs_fn = sac_model_kwargs_fn
        elif model_class == TD3:
            model_kwargs_fn = td3_model_kwargs_fn
        if not callable(eval_env_fn):
            raise ValueError("eval_env_fn must be a callable function that returns a new environment instance.")

        print(f"Tuning hyperparameters for class {model_class.__name__}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: optimize_sb3(
                trial,
                model_class,
                tune_env_for_opt,
                eval_env_fn,
                model_kwargs_fn,
                total_timesteps=test_timesteps,
            ),
            n_trials=n_trials,
            n_jobs=n_jobs,
        )
        best_params = study.best_trial.params
        print(f"Best trial for class {model_class.__name__}:")
        print(best_params)

        if best_params["net_arch"] == "small":
            net_arch = [64, 64]
        elif best_params["net_arch"] == "medium":
            net_arch = [256, 256]
        else:
            net_arch = [400, 300]

        # Build model argument dict (env will be set appropriately below)
        model_args = {
            "policy": "MlpPolicy",
            "env": None,  # placeholder
            "verbose": 0,
            "learning_rate": best_params["learning_rate"],
            "gamma": best_params["gamma"],
            "policy_kwargs": dict(net_arch=net_arch),
            "device": "cpu" if model_class != DDPG else "cuda",
        }
        optional_args = ["clip_range", "ent_coef", "vf_coef", "tau", "batch_size", "action_noise"]
        for arg in optional_args:
            if arg in best_params:
                model_args[arg] = best_params[arg]
        if model_kwargs_override:
            model_args.update(model_kwargs_override)

        # If we were given a list of env_fns, create the first chunk SubprocVecEnv to construct the model.
        initial_vec = None
        if is_env_list:
            chunk_size = get_chunk_size(len(env_fns), MAX_ENVS)
            first_chunk_fns = list(next(chunked(env_fns, chunk_size)))
            initial_vec = SubprocVecEnv(first_chunk_fns, start_method="forkserver")
            model_args["env"] = initial_vec
        else:
            model_args["env"] = vec_env

        # Filter and construct the model
        valid_args = inspect.signature(model_class.__init__).parameters
        filtered_args = {k: v for k, v in model_args.items() if k in valid_args}
        model = model_class(**filtered_args)
    else:
        # default model construction
        model_args = {
            "policy": "MlpPolicy",
            "verbose": 0,
        }
        if model_kwargs_override:
            model_args.update(model_kwargs_override)
        valid_args = inspect.signature(model_class.__init__).parameters
        filtered_args = {k: v for k, v in model_args.items() if k in valid_args}
        if is_env_list:
            chunk_size = get_chunk_size(len(env_fns), MAX_ENVS)
            first_chunk_fns = list(next(chunked(env_fns, chunk_size)))
            initial_vec = SubprocVecEnv(first_chunk_fns, start_method="forkserver")
            model = model_class(env=initial_vec, **filtered_args)
        else:
            model = model_class(env=vec_env, **filtered_args)

    if model_post_create_fn is not None:
        updated_model = model_post_create_fn(model)
        if updated_model is not None:
            model = updated_model

    # Close the small tuning vec if we created one
    if tuning_vec is not None:
        try:
            tuning_vec.close()
        except Exception:
            pass

    # Evaluate pre-training
    mean_reward, std_reward = evaluate_policy(model, Monitor(eval_env_fn()), n_eval_episodes=5, deterministic=False)
    eval_result["Pre_training"] = {"mean_reward": mean_reward, "std_reward": std_reward}

    # Train the model, honoring chunking if necessary
    print("Training the model with the best hyperparameters...")
    if is_env_list:
        remaining = total_timesteps
        # If we created an initial_vec above, use it as the first chunk's env
        chunk_size = get_chunk_size(len(env_fns), MAX_ENVS)
        chunks_iter = chunked(env_fns, chunk_size)
        # reuse the already-created initial_vec for the first chunk
        first_chunk_fns = list(next(chunks_iter))
        current_vec = None
        if 'initial_vec' in locals() and initial_vec is not None:
            current_vec = initial_vec
        else:
            current_vec = SubprocVecEnv(first_chunk_fns, start_method="forkserver")

        # train on the first chunk
        chunk_steps = min(remaining, per_chunk_timesteps)
        num_envs = current_vec.num_envs if hasattr(current_vec, 'num_envs') else len(first_chunk_fns)
        print(f"Chunk 1: envs={num_envs}, steps={chunk_steps}")
        model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)
        remaining -= chunk_steps

        chunk_idx = 2
        # iterate remaining chunks
        for chunk_fns in chunks_iter:
            if remaining <= 0:
                break
            new_vec = SubprocVecEnv(list(chunk_fns), start_method="forkserver")
            num_envs = new_vec.num_envs if hasattr(new_vec, 'num_envs') else len(chunk_fns)
            chunk_steps = min(remaining, per_chunk_timesteps)
            print(f"Chunk {chunk_idx}: envs={num_envs}, steps={chunk_steps}")
            # replace env on the model
            model.set_env(new_vec)
            # close previous vec (if different)
            try:
                if current_vec is not None:
                    current_vec.close()
            except Exception:
                pass
            current_vec = new_vec
            model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)
            remaining -= chunk_steps
            chunk_idx += 1

        # close final vec
        try:
            if current_vec is not None:
                current_vec.close()
        except Exception:
            pass
    else:
        model.learn(total_timesteps=total_timesteps)

    # evaluate the model after training
    mean_reward, std_reward = evaluate_policy(model, Monitor(eval_env_fn()), n_eval_episodes=5, deterministic=False)
    eval_result['Post_training'] = {'mean_reward': mean_reward, 'std_reward': std_reward}

    print("training complete.")

    # plot the pre training mean_reward and std_reward against the post training mean_reward and std_reward
    x = np.arange(len(eval_result))
    y = np.array([eval_result[key]['mean_reward'] for key in eval_result])
    yerr = np.array([eval_result[key]['std_reward'] for key in eval_result])
    plt.bar(x, y, yerr=yerr, capsize=5)
    plt.xticks(x, list(eval_result.keys()))
    plt.ylabel('Mean Reward')
    plt.title('Pre and Post Training Mean Reward')
    plt.show()

    return model, eval_result
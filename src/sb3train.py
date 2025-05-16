import optuna
import inspect
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
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
    batch_size = trial.suggest_float("batch_size", [64, 128, 256])
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


def train_model(model_class, vec_env, eval_env_fn, test_timesteps=40000, total_timesteps=4000000, n_trials=10, n_jobs=10):
    """
    Trains a reinforcement learning model using Stable Baselines3, with hyperparameter tuning via Optuna,
    and evaluates its performance before and after training.
    Args:
        model_class (type): The RL algorithm class to use (must be one of [PPO, A2C, DDPG, SAC, TD3]).
        vec_env (VecEnv): The vectorized environment for training.
        eval_env_fn (callable): A function that returns a new evaluation environment instance.
        test_timesteps (int, optional): Number of timesteps for each Optuna trial during hyperparameter tuning. Default is 40,000.
        total_timesteps (int, optional): Total number of timesteps to train the final model. Default is 4,000,000.
        n_trials (int, optional): Number of Optuna trials for hyperparameter optimization. Default is 10.
        n_jobs (int, optional): Number of parallel jobs for Optuna optimization. Default is 10.
    Returns:
        model: The trained RL model instance.
        eval_result (dict): Dictionary containing mean and standard deviation of rewards before and after training.
    Raises:
        ValueError: If model_class is not one of the supported algorithms or eval_env_fn is not callable.
    Note:
        - The function performs hyperparameter tuning using Optuna.
        - It plots the model before and after training and plots the results.
    """
    # Check if model_class is one of the supported algorithms
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
    else:
        raise ValueError("model_class must be one of [PPO, A2C, DDPG, SAC, TD3]")
    if not callable(eval_env_fn):
        raise ValueError("eval_env_fn must be a callable function that returns a new environment instance.")
    
    eval_result = {}
    
    print("Tuning hyperparameters for class {}...".format(model_class.__name__))
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optimize_sb3(
            trial, 
            model_class, 
            vec_env, 
            eval_env_fn, 
            model_kwargs_fn,
            total_timesteps=test_timesteps
        ), n_trials=n_trials, n_jobs=n_jobs
    )
    best_params = study.best_trial.params
    print("Best trial for class {}:".format(model_class.__name__))
    print(best_params)

    if best_params["net_arch"] == "small":
        net_arch = [64, 64]
    elif best_params["net_arch"] == "medium":   
        net_arch = [256, 256]
    else:
        net_arch = [400, 300]   
    
# Build the argument dictionary
    model_args = {
        "policy": "MlpPolicy",
        "env": vec_env,
        "verbose": 0,
        "learning_rate": best_params["learning_rate"],
        "gamma": best_params["gamma"],
        "policy_kwargs": dict(net_arch=net_arch),
        "device": "cpu" if model_class != DDPG else "cuda"
    }
    # Optionally add arguments if present in best_params
    optional_args = ["clip_range", "ent_coef", "vf_coef", "tau", "batch_size", "action_noise"]
    for arg in optional_args:
        if arg in best_params:
            model_args[arg] = best_params[arg]

    # Filter out arguments not in the model's __init__ signature
    valid_args = inspect.signature(model_class.__init__).parameters
    filtered_args = {k: v for k, v in model_args.items() if k in valid_args}

    model = model_class(**filtered_args)

    # evaluate the model before training
    mean_reward, std_reward = evaluate_policy(model, Monitor(eval_env_fn()), n_eval_episodes=5, deterministic=False)
    eval_result['Pre_training']={'mean_reward': mean_reward, 'std_reward': std_reward}
    
    # Train the model with the best hyperparameters
    print("Training the model with the best hyperparameters...")
    model.learn(total_timesteps=total_timesteps)

    # evaluate the model after training
    mean_reward, std_reward = evaluate_policy(model, Monitor(eval_env_fn()), n_eval_episodes=5, deterministic=False)
    eval_result['Post_training']={'mean_reward': mean_reward, 'std_reward': std_reward}
    
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
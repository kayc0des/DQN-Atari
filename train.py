import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import ale_py


class BreakoutTrainer:
    """
    A class to encapsulate the training process for the Breakout game using DQN.
    """

    def __init__(self, model_dir="models", log_dir="logs", total_timesteps=50000):
        """
        Initializes the trainer with default directories and total timesteps.
        
        Args:
            model_dir (str): Directory for saving models.
            log_dir (str): Directory for saving logs.
            total_timesteps (int): Number of timesteps to train the model.
        """
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.total_timesteps = total_timesteps

        # Create necessary directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize environments
        self.env = self._create_wrapped_env()
        self.eval_env = self._create_wrapped_env()

        # Initialize the DQN model
        self.model = self._initialize_model()

    @staticmethod
    def _create_env(render_mode=None):
        """
        Creates the Breakout environment wrapped in a Monitor for logging.

        Args:
            render_mode (str, optional): Render mode for the environment.

        Returns:
            gym.Env: The wrapped Breakout environment.
        """
        env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
        env = Monitor(env)
        return env

    def _create_wrapped_env(self):
        """
        Creates a vectorized environment.

        Returns:
            DummyVecEnv: A vectorized environment.
        """
        return DummyVecEnv([lambda: self._create_env()])

    def _initialize_model(self):
        """
        Initializes the DQN model with a CNN policy.

        Returns:
            DQN: The initialized DQN model.
        """
        return DQN(
            "CnnPolicy",
            self.env,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            verbose=1,
            tensorboard_log=self.log_dir,
        )

    def train(self):
        """
        Trains the DQN agent using the specified environment and parameters.
        """
        # Set up callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, save_path=self.model_dir, name_prefix="dqn_breakout"
        )
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"{self.model_dir}/best_model",
            log_path=self.log_dir,
            eval_freq=10000,
            deterministic=True,
            render=False,
        )

        # Train the model
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )

        # Save the final model
        self.model.save(f"{self.model_dir}/policy.zip")
        print(f"Training completed! Model saved as '{self.model_dir}/policy.zip'")

    def evaluate(self, episodes=5):
        """
        Evaluates the trained model.

        Args:
            episodes (int): Number of episodes to evaluate the model.
        """
        print("Evaluating the trained model...")
        for episode in range(episodes):
            obs = self.eval_env.reset()
            done = False
            score = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action)
                score += reward
            print(f"Episode {episode + 1}: Score = {score}")


def main():
    """
    Main function to train and evaluate the Breakout DQN agent.
    """
    # Register ALE environments
    gym.register_envs(ale_py)

    # Initialize the trainer
    trainer = BreakoutTrainer(total_timesteps=50000)

    # Train the agent
    trainer.train()

    # Evaluate the agent
    trainer.evaluate(episodes=5)


if __name__ == "__main__":
    main()

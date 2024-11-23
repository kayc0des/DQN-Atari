# DQN Breakout Agent

This project utilizes Deep Q-Networks (DQN) to train an AI agent to play the Atari Breakout game. The project is structured into two main files:

- `train.py`: A script to train the DQN agent.
- `play.py`: A script to run the trained agent in the environment and evaluate its performance.

The project leverages **Stable-Baselines3**, a popular reinforcement learning library, and **ALE (Atari Learning Environment)** to simulate the Breakout game.

## Requirements

Before running the project, make sure you have the following installed:

- Python 3.7 or higher
- `pip` package manager

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## File Structure

```plaintext
DQN-Breakout-Project/
│
├── train.py         # Script for training the DQN agent
├── play.py          # Script for running the trained model and evaluating performance
├── models/          # Directory to save trained models
│   └── policy.zip   # Trained model file (in .zip format)
├── logs/            # Directory for logging and TensorBoard visualization
└── requirements.txt # File containing required Python packages
```

## train.py: Training the DQN Agent 

The train.py script is responsible for training the DQN agent to play the Breakout game. It creates an environment, initializes the DQN agent, trains the agent, and saves the trained model.

### How it works
1. Environment Creation: The Breakout environment is created and wrapped with monitoring capabilities.
2. DQN Agent Initialization: The DQN agent is created using Stable-Baselines3 with a Convolutional Neural Network (CNN) policy.
3. Callbacks: The script uses two callbacks:
    - CheckpointCallback: Saves the model every 10,000 steps.
    - EvalCallback: Evaluates the model's performance every 10,000 steps and saves the best model.
4. Training: The agent is trained for a specified number of timesteps (50,000 steps in this case).
5. Model Saving: The trained model is saved as a .zip file.

### Running the Training Script
To start training the agent, run the following command:

```bash
python3 train.py
```

After training, the model will be saved as models/policy.zip.

## play.py: Playing the Trained Model 

The play.py script is used to run the trained model in the Breakout environment. The agent plays a set number of episodes, and its performance (total reward and steps taken) is logged.

### How it works
1. Environment Setup: The Breakout environment is initialized with rendering enabled for human visualization.
2. Model Loading: The trained model is loaded from the .zip file.
3. Running the Game: The agent plays several episodes, with each episode consisting of multiple steps.
4. Logging: The total reward and number of steps for each episode are printed.

### Running the Play Script
To run the trained model and play the game for 5 episodes, use the following command:

```bash
python3 play.py
```
## Limitations

This project was created to explore the process of training and evaluating an RL agent. The setup is intentionally lightweight, which may limit the overall performance of the trained agent. To achieve better results, consider the following:

### 1. Increase Training Timesteps
Training the agent for only 50,000 timesteps (as in this setup) is insufficient for mastering the game. For meaningful results, training should run for millions of timesteps, which requires significantly more computational resources.

### 2. Better Hardware
The training process can be time-consuming on standard CPUs. Using a GPU with CUDA support will significantly accelerate training. For optimal performance, consider cloud platforms like AWS, GCP, or Azure, which provide high-performance GPU instances.

##  Conclusion

This project demonstrates how to use Stable-Baselines3 to train a reinforcement learning agent to play Atari Breakout. It provides functionality for both training the agent and evaluating its performance. By modifying the train.py and play.py files, you can experiment with different hyperparameters, evaluate different models, and save/load models in different formats.
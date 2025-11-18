# Deepstack: Reinforcement Learning Based Trading Strategy

Deepstack is an end-to-end reinforcement learning project that builds a Deep Q Network (DQN) to predict trading signals and automate decision making. It provides modules for model training, evaluation and inference using a simple and modular architecture.

## Features

- Deep Q Network (DQN) for trading decisions  
- Custom trading environment with buy, sell and hold actions  
- Training pipeline using experience replay  
- Saved trained weights (`deepstack_best_model.pth`)  
- Inference script for running predictions on new data  
- Jupyter notebook for experiments and visualization  
- Clean project structure for easy understanding

## Project Structure

Deepstack/
│── app.py # Inference script to use trained model
│── deepstack_core.py # DQN model, environment, utilities
│── train_model.py # Training pipeline
│── rl-project.ipynb # Notebook for experiments
│── deepstack_best_model.pth # Trained model file
│── requirements.txt # Dependencies


## Technologies Used

### Programming Language
- Python 3.x

### Machine Learning & RL Libraries
- PyTorch  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- tqdm

### Reinforcement Learning Concepts Used
- Deep Q Network (DQN)  
- Experience Replay  
- Epsilon-Greedy Exploration  
- Reward shaping  
- Target network updates  

## How Deepstack Works

### 1. Environment  
A custom trading environment simulates trading behavior.  
It returns:
- next state  
- reward  
- done flag  
- updated portfolio balance  

### 2. Model  
A deep neural network receives state data and outputs Q-values for:
- Buy  
- Sell  
- Hold  

The model learns the best action for each state based on historical experience replay.

### 3. Training  
The training script (`train_model.py`) performs:
- Replay buffer updates  
- Sampling mini-batches  
- Loss minimization with Adam optimizer  
- Periodic target model updates  
- Saving best-performing model  

### 4. Inference  
`app.py` loads `deepstack_best_model.pth` and produces predictions on new market data.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the Model:
```bash
python train_model.py
```

Run Inference:
```bash
python app.py
```

Using the Notebook:
```
jupyter notebook rl-project.ipynb
```

Future Improvements:

-Add LSTM-based agent for time-series modeling
-Improve reward shaping for more stable training
-Integrate with real-time stock APIs
-Add hyperparameter tuning
-Support multiple environments and asset classes

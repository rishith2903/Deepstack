"""
DeepStack Core Components
Extracted from rl-project.ipynb
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta
from collections import deque


class CryptoTradingEnv(gym.Env):
    """Cryptocurrency Trading Environment"""
    
    def __init__(self, cryptos=['BTC-USD', 'ETH-USD', 'ADA-USD', 'MATIC-USD'],
                 initial_balance=10000, lookback_window=30):
        super(CryptoTradingEnv, self).__init__()
        
        self.cryptos = cryptos
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.current_step = 0
        
        print("Downloading crypto data...")
        self.data = self.download_data()
        
        # Check if data was downloaded successfully
        if not self.data or all(len(df) == 0 for df in self.data.values()):
            raise ValueError("Failed to download crypto data. Please check your internet connection.")
        
        self.max_steps = len(list(self.data.values())[0]) - self.lookback_window - 1
        
        if self.max_steps <= 0:
            raise ValueError("Not enough data available. Please try again later.")
        
        self.action_space = spaces.MultiDiscrete([3] * len(cryptos))
        
        state_size = (len(cryptos) * self.lookback_window * 6) + len(cryptos) + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)
        
        print(f"Environment created!")
        print(f"Total trading days: {self.max_steps}")
        print(f"State size: {state_size}")

    def download_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)
        
        all_data = {}
        for crypto in self.cryptos:
            try:
                # Try downloading with better error handling
                ticker = yf.Ticker(crypto)
                data = ticker.history(start=start_date, end=end_date, timeout=10)
                
                if data.empty:
                    print(f"  ✗ {crypto}: No data available")
                    continue
                
                # Calculate technical indicators
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                data['RSI'] = self.calculate_rsi(data['Close'])
                
                data = data.dropna()
                
                if len(data) > 0:
                    all_data[crypto] = data
                    print(f"  ✓ {crypto}: {len(data)} days")
                else:
                    print(f"  ✗ {crypto}: Insufficient data after processing")
                    
            except Exception as e:
                print(f"  ✗ Error downloading {crypto}: {str(e)[:50]}")
                continue
        
        # Find minimum length across all cryptos to align data
        if all_data:
            min_length = min(len(df) for df in all_data.values())
            print(f"\n⚙ Aligning all data to {min_length} days (shortest dataset)")
            
            # Trim all dataframes to same length (use most recent data)
            aligned_data = {}
            for crypto, df in all_data.items():
                aligned_data[crypto] = df.iloc[-min_length:].reset_index(drop=True)
            
            return aligned_data
        
        return all_data

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def reset(self, seed=None):
        # Gymnasium requires seed parameter
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.holdings = {crypto: 0 for crypto in self.cryptos}
        self.total_profit = 0
        
        # Gymnasium requires returning (observation, info)
        return self._get_observation(), {}

    def _get_observation(self):
        state = []
        
        for crypto in self.cryptos:
            data_slice = self.data[crypto].iloc[
                self.current_step - self.lookback_window:self.current_step
            ]
            
            for col in ['Open', 'High', 'Low', 'Close', 'SMA_20', 'RSI']:
                values = data_slice[col].values
                normalized = (values - values.mean()) / (values.std() + 1e-8)
                state.extend(normalized)
        
        for crypto in self.cryptos:
            state.append(self.holdings[crypto])
        
        state.append(self.balance / self.initial_balance)
        
        return np.array(state, dtype=np.float32)

    def step(self, action):
        for i, crypto in enumerate(self.cryptos):
            current_price = self.data[crypto].iloc[self.current_step]['Close']
            act = action[i]
            
            if act == 1 and self.balance >= current_price:  # BUY
                shares_to_buy = self.balance * 0.1 / current_price
                self.holdings[crypto] += shares_to_buy
                self.balance -= shares_to_buy * current_price
                
            elif act == 2 and self.holdings[crypto] > 0:  # SELL
                self.balance += self.holdings[crypto] * current_price
                self.holdings[crypto] = 0
        
        self.current_step += 1
        
        portfolio_value = self.get_portfolio_value()
        reward = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # Gymnasium uses terminated and truncated instead of done
        terminated = self.current_step >= self.max_steps - 1
        truncated = False
        
        next_state = self._get_observation()
        info = {'portfolio_value': portfolio_value}
        
        return next_state, reward, terminated, truncated, info

    def get_portfolio_value(self):
        total_value = self.balance
        for crypto in self.cryptos:
            if self.holdings[crypto] > 0:
                current_price = self.data[crypto].iloc[self.current_step]['Close']
                total_value += self.holdings[crypto] * current_price
        return total_value

    def get_current_price(self, crypto):
        return self.data[crypto].iloc[self.current_step]['Close']


class DQNNetwork(nn.Module):
    """Deep Q-Network Architecture"""
    
    def __init__(self, input_size, output_size, hidden_sizes=[512, 256, 128]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.network(x)


class DeepStackAgent:
    """Deep Q-Learning Agent"""
    
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 memory_size=100000, batch_size=32):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.update_target_network()
        self.training_step = 0

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def act(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, 3, size=self.action_size // 3)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        q_values = q_values.cpu().numpy()[0].reshape(-1, 3)
        actions = np.argmax(q_values, axis=1)
        return actions

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        
        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[idx] for idx in batch_indices]
        
        states = torch.FloatTensor(np.array([exp[0] for exp in batch])).to(self.device)
        actions = [exp[1] for exp in batch]
        rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([exp[3] for exp in batch])).to(self.device)
        terminateds = torch.FloatTensor([exp[4] for exp in batch]).to(self.device)
        
        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        
        target_q_values = current_q_values.clone()
        
        for i in range(self.batch_size):
            action = actions[i]
            for j, act in enumerate(action):
                idx = j * 3 + act
                if terminateds[i]:
                    target_q_values[i][idx] = rewards[i]
                else:
                    next_action_values = next_q_values[i][j*3:(j+1)*3]
                    target_q_values[i][idx] = rewards[i] + self.gamma * torch.max(next_action_values)
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_step += 1
        
        return loss.item()

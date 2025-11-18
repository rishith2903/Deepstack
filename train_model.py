"""
Training Script to Generate deepstack_best_model.pth
Run this once before using the Streamlit interface
"""

import numpy as np
import torch
from deepstack_core import CryptoTradingEnv, DeepStackAgent

def train_deepstack(episodes=200, print_every=20):
    """Train DeepStack agent and save best model"""
    
    print("=" * 60)
    print("DEEPSTACK TRAINING STARTED")
    print("=" * 60)
    print(f"Training Episodes: {episodes}")
    print(f"This will take 15-25 minutes...")
    print("=" * 60)
    
    # Initialize environment and agent
    try:
        env = CryptoTradingEnv()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Wait a few minutes and try again (Yahoo Finance API may be rate-limited)")
        print("3. Try upgrading yfinance: pip install --upgrade yfinance")
        return None, None, None
    
    state_size = env.observation_space.shape[0]
    action_size = 3 * len(env.cryptos)
    
    agent = DeepStackAgent(state_size, action_size)
    
    scores = []
    portfolio_values = []
    losses = []
    best_score = -np.inf
    
    for episode in range(episodes):
        state, info = env.reset()  # Gymnasium returns (state, info)
        total_reward = 0
        episode_losses = []
        steps = 0
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)  # Gymnasium API
            
            agent.remember(state, action, reward, next_state, terminated)
            
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Update target network
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        # Record metrics
        portfolio_value = env.get_portfolio_value()
        scores.append(total_reward)
        portfolio_values.append(portfolio_value)
        
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses.append(avg_loss)
        
        # Save best model
        if total_reward > best_score:
            best_score = total_reward
            torch.save(agent.q_network.state_dict(), 'deepstack_best_model.pth')
        
        # Print progress
        if (episode + 1) % print_every == 0:
            avg_portfolio = np.mean(portfolio_values[-print_every:])
            profit_pct = ((avg_portfolio - env.initial_balance) / env.initial_balance) * 100
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Portfolio: ${avg_portfolio:,.2f}")
            print(f"  Avg Profit: {profit_pct:+.2f}%")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print("-" * 60)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    
    final_avg = np.mean(portfolio_values[-20:])
    final_profit = ((final_avg - env.initial_balance) / env.initial_balance) * 100
    
    print(f"Final Average Portfolio: ${final_avg:,.2f}")
    print(f"Final Profit: {final_profit:+.2f}%")
    print(f"Best Episode Score: {best_score:.4f}")
    print(f"\n✅ Model saved as 'deepstack_best_model.pth'")
    print("=" * 60)
    
    return scores, portfolio_values, losses


if __name__ == "__main__":
    train_deepstack(episodes=200, print_every=20)

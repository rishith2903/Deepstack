"""
DeepStack Trading Interface
Simple Streamlit interface for cryptocurrency trading AI demo
"""

import streamlit as st
import torch
import pandas as pd
import plotly.graph_objects as go
from deepstack_core import CryptoTradingEnv, DeepStackAgent
import time
import numpy as np

# Page config
st.set_page_config(
    page_title="DeepStack AI Trading",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 DeepStack AI Trading Demo")
st.markdown("### Autonomous Cryptocurrency Trading Using Deep Q-Networks")
st.markdown("---")

# Sidebar - User Input
with st.sidebar:
    st.header("⚙️ Trading Configuration")
    
    initial_balance = st.slider(
        "Initial Portfolio Balance ($)",
        min_value=5000,
        max_value=20000,
        value=10000,
        step=1000
    )
    
    st.markdown("---")
    st.markdown("**Trading Assets:**")
    st.markdown("✓ Bitcoin (BTC)")
    st.markdown("✓ Ethereum (ETH)")
    st.markdown("✓ Cardano (ADA)")
    st.markdown("✓ Polygon (MATIC)")
    
    st.markdown("---")
    st.markdown("**AI Model:** Deep Q-Network")
    st.markdown("**Training:** 200 episodes")
    st.markdown("**Period:** 2 years historical data")

# Main interface
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Starting Balance", f"${initial_balance:,}")
with col2:
    st.metric("Cryptocurrencies", "4")
with col3:
    st.metric("AI Model", "Pre-trained DQN")

st.markdown("---")

# Run button
if st.button("▶️ Run AI Trading Demo", type="primary", use_container_width=True):
    
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize environment and agent
    status_text.text("🔄 Initializing DeepStack environment...")
    progress_bar.progress(10)
    
    try:
        env = CryptoTradingEnv(initial_balance=initial_balance)
    except Exception as e:
        st.error(f"Error initializing environment: {e}")
        st.stop()
    
    state_size = env.observation_space.shape[0]
    action_size = 3 * len(env.cryptos)
    
    agent = DeepStackAgent(state_size, action_size)
    
    # Load trained model
    status_text.text("🔄 Loading pre-trained AI model...")
    progress_bar.progress(20)
    
    try:
        agent.q_network.load_state_dict(torch.load('deepstack_best_model.pth', map_location=agent.device))
        agent.epsilon = 0  # No exploration, pure exploitation
        status_text.text("✅ Model loaded successfully")
    except FileNotFoundError:
        st.error("⚠️ Model file 'deepstack_best_model.pth' not found!")
        st.info("Please run 'python train_model.py' first to generate the model file.")
        progress_bar.empty()
        status_text.empty()
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        progress_bar.empty()
        status_text.empty()
        st.stop()
    
    time.sleep(0.5)
    
        # Run trading simulation
    status_text.text("🤖 AI is analyzing market and trading...")
    progress_bar.progress(30)
    
    state, info = env.reset()  # Gymnasium API
    terminated = False
    truncated = False
    
    portfolio_history = [initial_balance]
    actions_log = []
    step_count = 0
    
    action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    
    while not (terminated or truncated):
        action = agent.act(state, training=False)
        next_state, reward, terminated, truncated, info = env.step(action)  # Gymnasium API
        
        portfolio_value = info['portfolio_value']
        portfolio_history.append(portfolio_value)
        
        # Log significant actions (non-HOLD)
        for i, crypto in enumerate(env.cryptos):
            if action[i] != 0:
                actions_log.append({
                    'Day': step_count,
                    'Crypto': crypto.replace('-USD', ''),
                    'Action': action_names[action[i]],
                    'Portfolio': f"${portfolio_value:,.2f}"
                })
        
        state = next_state
        step_count += 1
        
        # Update progress
        progress = min(30 + int((step_count / env.max_steps) * 60), 90)
        progress_bar.progress(progress)
    
    progress_bar.progress(100)
    status_text.text("✅ Trading completed!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    # Results
    final_balance = portfolio_history[-1]
    profit = final_balance - initial_balance
    profit_pct = (profit / initial_balance) * 100
    
    st.markdown("---")
    st.markdown("## 📊 Trading Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Portfolio", f"${final_balance:,.2f}")
    with col2:
        st.metric("Profit/Loss", f"${profit:,.2f}", f"{profit_pct:+.2f}%")
    with col3:
        st.metric("Total Trades", len(actions_log))
    with col4:
        st.metric("Trading Days", step_count)
    
    # Portfolio chart
    st.markdown("### 📈 Portfolio Value Over Time")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(portfolio_history))),
        y=portfolio_history,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00D9FF', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 217, 255, 0.1)'
    ))
    
    # Add starting line
    fig.add_hline(y=initial_balance, line_dash="dash", 
                  line_color="gray", annotation_text="Starting Balance")
    
    fig.update_layout(
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        height=450,
        hovermode='x unified',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trading log
    if actions_log:
        st.markdown("### 📝 Trading Actions Log")
        
        # Show last 15 actions
        display_log = actions_log[-15:] if len(actions_log) > 15 else actions_log
        df = pd.DataFrame(display_log)
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        if len(actions_log) > 15:
            st.caption(f"Showing last 15 of {len(actions_log)} total trades")
    else:
        st.info("ℹ️ AI maintained a conservative HOLD strategy throughout the period.")
    
    # Download results
    st.markdown("---")
    
    results_df = pd.DataFrame({
        'Day': range(len(portfolio_history)),
        'Portfolio_Value': portfolio_history,
        'Profit_Percentage': [((pv - initial_balance) / initial_balance) * 100 for pv in portfolio_history]
    })
    
    csv = results_df.to_csv(index=False)
    
    col1, col2 = st.columns([3, 1])
    with col2:
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=f"deepstack_results_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # Initial state - show instructions
    st.info("👆 Click the button above to start the AI trading demo!")
    
    st.markdown("---")
    st.markdown("### 🎯 What This Demo Does:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **The AI Will:**
        - 📊 Analyze 2 years of historical crypto data
        - 🧠 Use Deep Q-Learning to make trading decisions
        - 💰 Trade 4 major cryptocurrencies autonomously
        - 📈 Show you the portfolio performance
        """)
    
    with col2:
        st.markdown("""
        **You'll See:**
        - Real-time portfolio value changes
        - All buy/sell/hold decisions
        - Final profit or loss percentage
        - Detailed trading history
        """)
    
    st.markdown("---")
    st.markdown("**Note:** The AI uses a pre-trained Deep Q-Network model trained on 200 episodes.")

# Footer
st.markdown("---")
st.caption("DeepStack - Autonomous Cryptocurrency Trading Using Deep Q-Networks | VIT-AP University")

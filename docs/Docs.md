Here is a complete, runnable Python implementation using PyTorch. I've structured it in a modular way—separating the environment logic from the agent architecture—which keeps the overall system design clean and makes it easy to swap in your real equations later. 

This script implements the **Delta-Step mapping** we discussed, using synthetic placeholders for the data rates ($R_n, R_f$) and the sensing metric ($ASIR$).

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# ==========================================
# 1. The ISAC-NOMA Environment
# ==========================================
class RISEnvironment:
    def __init__(self):
        self.state_size = 6  # [h_n, h_f, h_t, a_n, a_f, a_T]
        self.action_size = 6 # 6 possible delta shifts
        self.delta = 0.05    # 5% shift per step
        self.reset()

    def reset(self):
        # Initialize with equal partitions
        self.a_n, self.a_f, self.a_T = 0.33, 0.33, 0.34
        
        # Synthetic channel gains (normalized 0 to 1)
        # In reality, these come from your Rayleigh/Nakagami-m fading models
        self.h_n = np.random.uniform(0.6, 1.0) 
        self.h_f = np.random.uniform(0.2, 0.5)
        self.h_t = np.random.uniform(0.4, 0.8)
        
        return self._get_state()

    def _get_state(self):
        return np.array([self.h_n, self.h_f, self.h_t, self.a_n, self.a_f, self.a_T], dtype=np.float32)

    def step(self, action):
        # --- ACTION SPACE MAPPING (Delta-Steps) ---
        if action == 0:   self.a_n -= self.delta; self.a_f += self.delta # Near -> Far
        elif action == 1: self.a_n -= self.delta; self.a_T += self.delta # Near -> Sensing
        elif action == 2: self.a_f -= self.delta; self.a_n += self.delta # Far -> Near
        elif action == 3: self.a_f -= self.delta; self.a_T += self.delta # Far -> Sensing
        elif action == 4: self.a_T -= self.delta; self.a_n += self.delta # Sensing -> Near
        elif action == 5: self.a_T -= self.delta; self.a_f += self.delta # Sensing -> Far

        # --- CONSTRAINT ENFORCEMENT ---
        # Ensure no zone drops below 5% to maintain minimum operation
        partitions = np.clip([self.a_n, self.a_f, self.a_T], 0.05, 1.0)
        partitions /= np.sum(partitions) # Re-normalize to ensure sum is exactly 1.0
        self.a_n, self.a_f, self.a_T = partitions

        # --- SYNTHETIC DATA GENERATION (Placeholders) ---
        # Replace these with Eq. 2, Eq. 3, and Eq. 18 from your paper
        # Base formula mock: Rate ~ bandwidth * log2(1 + SNR * elements * gain)
        R_n = 10 * np.log2(1 + 100 * self.a_n * self.h_n) 
        R_f = 10 * np.log2(1 + 100 * self.a_f * self.h_f)
        ASIR = 15 * np.log2(1 + 50 * self.a_T * self.h_t)

        # --- REWARD FUNCTION ---
        # 1. Jain's Fairness Index for the two communication users
        jfi = ((R_n + R_f)**2) / (2 * (R_n**2 + R_f**2) + 1e-5)
        
        # 2. Weighted Sum Reward (e.g., 60% Comm Fairness, 40% Sensing)
        # We normalize ASIR roughly to keep it on a similar scale to JFI (0 to 1)
        normalized_asir = min(ASIR / 50.0, 1.0) 
        reward = (0.6 * jfi) + (0.4 * normalized_asir)

        # In a dynamic environment, episodes might not strictly "end" unless a target is lost
        done = False 
        
        return self._get_state(), reward, done, {'R_n': R_n, 'R_f': R_f, 'ASIR': ASIR}

# ==========================================
# 2. The Deep Q-Network (DQN) Agent
# ==========================================
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        # Standard feed-forward architecture
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x) # Outputs Q-values for the 6 actions

# ==========================================
# 3. Training Loop Outline
# ==========================================
if __name__ == "__main__":
    env = RISEnvironment()
    agent = DQNAgent(env.state_size, env.action_size)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epsilon = 1.0      # Exploration rate
    epsilon_decay = 0.995
    gamma = 0.95       # Discount factor
    
    # Quick sanity check run for 100 steps
    state = env.reset()
    state_tensor = torch.FloatTensor(state)
    
    for step in range(100):
        # Epsilon-greedy action selection
        if np.random.rand() <= epsilon:
            action = random.randrange(env.action_size) # Explore
        else:
            with torch.no_grad():
                q_values = agent(state_tensor)
                action = torch.argmax(q_values).item() # Exploit
                
        # Take the action
        next_state, reward, done, metrics = env.step(action)
        
        # Print the results every 10 steps to see the partitions adjusting
        if step % 10 == 0:
            print(f"Step {step} | Reward: {reward:.3f}")
            print(f"Partitions -> Near: {env.a_n:.2f}, Far: {env.a_f:.2f}, Target: {env.a_T:.2f}")
            print(f"Metrics    -> R_n: {metrics['R_n']:.2f}, R_f: {metrics['R_f']:.2f}, ASIR: {metrics['ASIR']:.2f}\n")
            
        state_tensor = torch.FloatTensor(next_state)
        epsilon *= epsilon_decay
```

### Key Takeaways from the Code:

1.  **Constraint Enforcement:** Look at `np.clip` in the `step()` function. This prevents the DQL from completely shutting off a zone ($0\%$ elements), which would crash the system or cause a hard outage. It forces a minimum of $5\%$ elements per zone.
2.  **The Reward Optimization:** The `reward` variable scales both Jain's Fairness Index (JFI) and the ASIR. Because neural networks prefer normalized targets, converting raw rates into a bounded $[0, 1]$ scale ensures the gradients during optimization don't explode.
3.  **The Agent Model:** It uses two hidden layers of 64 nodes. For a small state space like 6 variables, you don't need a massive architecture to find the optimal Q-values. 

Would you like to start replacing those synthetic equations with the specific outage probability math (like $P_{out}^{D_n}$) from the ISAC-NOMA paper?

Here is the breakdown of the exact Reward Function logic, followed by the complete **DDPG (Deep Deterministic Policy Gradient)** implementation. 

This modular setup is highly visual and cleanly separated, making it a great candidate to showcase if you ever decide to break down an Actor-Critic architecture for a coding project walkthrough or screencast.

---

### 1. The Reward Function Logic

The reward function must act as the "referee" between the communication users and the sensing target. Since DDPG outputs continuous values, the reward needs to smoothly guide the gradients. 

We combine **Jain’s Fairness Index (JFI)** for the communication users ($R_n$ and $R_f$) with the **Normalized ASIR** for the sensing target.

$$r_t = w_c \cdot \frac{(R_n + R_f)^2}{2(R_n^2 + R_f^2) + \epsilon} + w_r \cdot \left( \frac{ASIR}{ASIR_{max}} \right)$$

* **$w_c$ and $w_r$**: Weights (e.g., $0.6$ and $0.4$) that let you prioritize communication over sensing, or vice versa.
* **$\epsilon$**: A tiny value (like $1e-5$) added to the denominator to prevent division by zero if both rates drop to zero.
* **$ASIR_{max}$**: A theoretical maximum to keep the sensing reward bound between $0$ and $1$, matching the scale of the JFI.

---

### 2. DDPG Implementation for 3-Zone Partitioning

Unlike DQL which picks an *index*, DDPG uses two neural networks:
1.  **The Actor:** Looks at the state and directly outputs the exact percentages $[a_n, a_f, a_T]$. We use a **Softmax** activation at the end so they naturally sum to $1.0$.
2.  **The Critic:** Looks at the state *and* the Actor's chosen partition, and outputs a single Q-value predicting how good that choice is.

Here is the PyTorch implementation:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ==========================================
# 1. DDPG Actor Network (The Decision Maker)
# ==========================================
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # Softmax ensures the 3 outputs (a_n, a_f, a_T) always sum to 1.0!
        action = torch.softmax(self.fc3(x), dim=-1) 
        return action

# ==========================================
# 2. DDPG Critic Network (The Evaluator)
# ==========================================
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        # The Critic looks at BOTH the state and the action
        self.fc1 = nn.Linear(state_size + action_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1) # Outputs a single Q-value

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# ==========================================
# 3. The Replay Buffer
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 4. The ISAC-NOMA Environment (Continuous)
# ==========================================
class ContinuousRISEnv:
    def __init__(self):
        self.state_size = 3  # Just the 3 channel gains for simplicity
        self.action_size = 3 # a_n, a_f, a_T
        self.reset()

    def reset(self):
        self.h_n = np.random.uniform(0.6, 1.0) 
        self.h_f = np.random.uniform(0.2, 0.5)
        self.h_t = np.random.uniform(0.4, 0.8)
        return np.array([self.h_n, self.h_f, self.h_t], dtype=np.float32)

    def step(self, action):
        a_n, a_f, a_T = action[0], action[1], action[2]

        # Synthetic Data (Replace with paper equations)
        R_n = 10 * np.log2(1 + 100 * a_n * self.h_n)
        R_f = 10 * np.log2(1 + 100 * a_f * self.h_f)
        ASIR = 15 * np.log2(1 + 50 * a_T * self.h_t)

        # The Reward Function
        jfi = ((R_n + R_f)**2) / (2 * (R_n**2 + R_f**2) + 1e-5)
        normalized_asir = min(ASIR / 50.0, 1.0)
        
        # 60% Comm Fairness, 40% Sensing
        reward = (0.6 * jfi) + (0.4 * normalized_asir) 
        
        next_state = self.reset() # In a static channel episode, we just draw new channels
        return next_state, reward, False

# ==========================================
# 5. DDPG Training Loop
# ==========================================
if __name__ == "__main__":
    env = ContinuousRISEnv()
    
    actor = Actor(env.state_size, env.action_size)
    critic = Critic(env.state_size, env.action_size)
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    
    replay_buffer = ReplayBuffer()
    batch_size = 64
    gamma = 0.99

    # Let's run a few steps to show the flow
    state = env.reset()
    
    for step in range(200):
        # 1. Select Action (with some random noise for exploration)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = actor(state_tensor).numpy()[0]
            
        # Add exploration noise (Dirichlet is good for probabilities that sum to 1)
        noise = np.random.dirichlet(np.ones(env.action_size)) * 0.1
        action = action + noise
        action = action / np.sum(action) # Re-normalize
        
        # 2. Step Environment
        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        
        # 3. Train Networks
        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            
            # Update Critic
            next_actions = actor(next_states)
            target_q = rewards + gamma * critic(next_states, next_actions).detach()
            current_q = critic(states, actions)
            critic_loss = nn.MSELoss()(current_q, target_q)
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            # Update Actor
            actor_loss = -critic(states, actor(states)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

        if step % 50 == 0:
            print(f"Step {step} | Reward: {reward:.3f} | Partitions: Near={action[0]:.2f}, Far={action[1]:.2f}, Target={action[2]:.2f}")
```

### Why DDPG Shines Here
The `torch.softmax` layer in the Actor is the "magic bullet" for this problem. It guarantees that no matter what crazy numbers the hidden layers produce, the final coefficients will *always* be positive and will *always* sum to $1.0$. This completely eliminates the need for manual clipping or complex constraint management.

Would you like me to help you integrate the specific Nakagami-m fading channel derivations (like $h_{D_n}$ and $g_T$) into the environment's `reset()` function next?
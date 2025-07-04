import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.puzzle_cube import PuzzleCube
import wandb
import random
from collections import deque
import math
from tqdm import tqdm

class RubiksTransformer(nn.Module):
    """Transformer model for Q-learning on Rubiks cube states"""
    
    def __init__(self, d_model=512, n_heads=8, n_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_actions = 12  # 12 possible moves
        
        # Input embedding: 6 colors per position
        self.input_embedding = nn.Linear(6, d_model)
        
        # Positional encoding for the 54 cube positions
        self.pos_encoding = nn.Parameter(torch.randn(54, d_model))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head for Q-values
        self.q_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, self.n_actions)
        )
        
        # Global pooling to aggregate position information
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x shape: (batch_size, 54, 6) - one-hot encoded cube state
        batch_size = x.size(0)
        
        # Reshape to flatten the one-hot encoding: (batch_size, 54, 6) -> (batch_size, 54, 6)
        x = x.view(batch_size, 54, 6)
        
        # Embed to model dimension: (batch_size, 54, 6) -> (batch_size, 54, d_model)
        x = self.input_embedding(x.reshape(batch_size, 54, 6))
        
        # Add positional encoding
        x = x + self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global pooling: (batch_size, 54, d_model) -> (batch_size, d_model)
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        
        # Get Q-values: (batch_size, d_model) -> (batch_size, n_actions)
        q_values = self.q_head(x)
        
        return q_values

class RubiksEnvironment:
    """Environment for Rubiks cube RL training"""
    
    def __init__(self):
        self.cube = PuzzleCube()
        self.actions = ["L", "L'", "R", "R'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]
        self.reset()
        
    def reset(self, scramble_steps=5):
        """Reset to solved state and scramble"""
        self.cube = PuzzleCube()
        self.initial_cube = self.cube.copy()
        
        # Scramble the cube
        self.scramble_actions = []
        for _ in range(scramble_steps):
            action = random.choice(self.actions)
            self.scramble_actions.append(action)
            self.cube = self.cube.move(action)
            
        return self.get_state()
    
    def get_state(self):
        """Get current state as one-hot encoded tensor"""
        # Convert cube state to bit array (54 x 6)
        batch_cube = self.cube._inner_cube
        bit_array = batch_cube.bit_array()[0]  # Get first (and only) cube
        return torch.tensor(bit_array, dtype=torch.float32)
    
    def step(self, action_idx):
        """Take action and return next state, reward, done"""
        action = self.actions[action_idx]
        self.cube = self.cube.move(action)
        
        next_state = self.get_state()
        done = self.cube.is_solved()
        
        # Reward structure
        if done:
            reward = 100.0  # Large reward for solving
        else:
            reward = -1.0   # Small penalty for each step
            
        return next_state, reward, done
    
    def get_valid_actions(self):
        """All actions are always valid"""
        return list(range(12))

class ReplayBuffer:
    """Experience replay buffer for Q-learning"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor([bool(d) for d in dones], dtype=torch.bool)
        )
    
    def __len__(self):
        return len(self.buffer)

class RubiksQAgent:
    """Q-learning agent for Rubiks cube solving"""
    
    def __init__(self, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        # Device selection: prioritize MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.q_network = RubiksTransformer().to(self.device)
        self.target_network = RubiksTransformer().to(self.device)
        self.update_target_network()
        
        # Training parameters
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Experience replay
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 32
        self.target_update_freq = 1000
        self.step_count = 0
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def select_action(self, state, valid_actions=None):
        """Select action using epsilon-greedy policy"""
        if valid_actions is None:
            valid_actions = list(range(12))
            
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # Mask invalid actions
            masked_q_values = q_values.clone()
            for i in range(12):
                if i not in valid_actions:
                    masked_q_values[0, i] = float('-inf')
                    
            return masked_q_values.argmax().item()
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

def evaluate_agent(agent, env, num_episodes=100, max_steps=30, scramble_steps=5):
    """Evaluate agent performance"""
    successes = 0
    total_steps = 0
    
    for _ in tqdm(range(num_episodes), desc=f"Evaluating {scramble_steps}-step scrambles", leave=False):
        state = env.reset(scramble_steps)
        episode_steps = 0
        
        for step in range(max_steps):
            action = agent.select_action(state, env.get_valid_actions())
            next_state, reward, done = env.step(action)
            
            episode_steps += 1
            
            if done:
                successes += 1
                total_steps += episode_steps
                break
                
            state = next_state
    
    success_rate = successes / num_episodes
    avg_steps = total_steps / max(successes, 1)
    
    return success_rate, avg_steps

def train_rubiks_agent():
    """Main training loop"""
    
    # Initialize wandb
    wandb.init(project="rubiks-rl", config={
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "batch_size": 32,
        "target_update_freq": 1000,
        "scramble_steps": 5,
        "max_episodes": 10000,
        "eval_freq": 500
    })
    
    # Initialize environment and agent
    env = RubiksEnvironment()
    agent = RubiksQAgent()
    
    # Training loop
    episode_rewards = []
    episode_steps = []
    
    # Create progress bar
    pbar = tqdm(range(wandb.config.max_episodes), desc="Training")
    
    for episode in pbar:
        state = env.reset(wandb.config.scramble_steps)
        episode_reward = 0
        episode_step = 0
        
        for step in range(20):  # Max steps per episode
            action = agent.select_action(state, env.get_valid_actions())
            next_state, reward, done = env.step(action)
            
            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            
            episode_reward += reward
            episode_step += 1
            
            if done:
                break
                
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        
        # Update progress bar
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_steps = np.mean(episode_steps[-100:]) if len(episode_steps) >= 100 else np.mean(episode_steps)
            
            pbar.set_postfix({
                'Avg Reward': f'{avg_reward:.2f}',
                'Avg Steps': f'{avg_steps:.2f}',
                'Epsilon': f'{agent.epsilon:.3f}',
                'Buffer': len(agent.replay_buffer)
            })
        
        # Logging
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_steps = np.mean(episode_steps[-100:]) if len(episode_steps) >= 100 else np.mean(episode_steps)
            
            wandb.log({
                "episode": episode,
                "avg_reward": avg_reward,
                "avg_steps": avg_steps,
                "epsilon": agent.epsilon,
                "buffer_size": len(agent.replay_buffer)
            })
        
        # Evaluation
        if episode % wandb.config.eval_freq == 0 and episode > 0:
            print(f"Evaluating at episode {episode}...")
            
            # Evaluate on original scramble distance
            success_rate, avg_solve_steps = evaluate_agent(agent, env, num_episodes=50, scramble_steps=5)
            
            wandb.log({
                "eval_success_rate_5": success_rate,
                "eval_avg_steps_5": avg_solve_steps,
            })
            
            print(f"Success rate (5 steps): {success_rate:.3f}, Avg solve steps: {avg_solve_steps:.2f}")
            
            # Test emergence: evaluate on longer scrambles
            emergence_results = {}
            for test_scramble in [10, 15, 20, 25]:
                success_rate, avg_solve_steps = evaluate_agent(
                    agent, env, num_episodes=50, scramble_steps=test_scramble
                )
                emergence_results[f"success_rate_{test_scramble}"] = success_rate
                emergence_results[f"avg_steps_{test_scramble}"] = avg_solve_steps
                
                print(f"Success rate ({test_scramble} steps): {success_rate:.3f}, Avg solve steps: {avg_solve_steps:.2f}")
            
            wandb.log(emergence_results)
            
            # Save model checkpoint
            if episode % (wandb.config.eval_freq * 4) == 0:
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.q_network.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'success_rate': success_rate,
                }, f'rubiks_model_episode_{episode}.pth')
                
                print(f"Model saved at episode {episode}")
    
    wandb.finish()
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    for test_scramble in [5, 10, 15, 20, 25, 30]:
        success_rate, avg_solve_steps = evaluate_agent(
            agent, env, num_episodes=100, scramble_steps=test_scramble
        )
        print(f"Scramble {test_scramble} steps: Success rate = {success_rate:.3f}, Avg solve steps = {avg_solve_steps:.2f}")

if __name__ == "__main__":
    train_rubiks_agent()
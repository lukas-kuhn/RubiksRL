import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.puzzle_cube import PuzzleCube
import wandb
import random
from tqdm import tqdm

class RubiksTransformer(nn.Module):
    """Small transformer for Q-learning on Rubiks cube"""
    
    def __init__(self, d_model=256, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_actions = 12  # 12 possible moves
        
        # Input embedding: 6 colors per position -> d_model
        self.input_embedding = nn.Linear(6, d_model)
        
        # Positional encoding for the 54 cube positions
        self.pos_encoding = nn.Parameter(torch.randn(54, d_model))
        
        # Small transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,  # 512
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Global pooling and output head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.q_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.n_actions)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 54, 6) - one-hot encoded cube state
        batch_size = x.size(0)
        
        # Embed each position: (batch_size, 54, 6) -> (batch_size, 54, d_model)
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer: (batch_size, 54, d_model) -> (batch_size, 54, d_model)
        x = self.transformer(x)
        
        # Global pooling: (batch_size, 54, d_model) -> (batch_size, d_model)
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        
        # Output Q-values: (batch_size, d_model) -> (batch_size, 12)
        return self.q_head(x)

class RubiksEnvironment:
    """Simple Rubiks cube environment"""
    
    def __init__(self):
        self.cube = PuzzleCube()
        self.actions = ["L", "L'", "R", "R'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]
        
    def reset(self, scramble_steps=5):
        """Reset to solved state and scramble"""
        self.cube = PuzzleCube()
        
        # Scramble the cube
        for _ in range(scramble_steps):
            action = random.choice(self.actions)
            self.cube = self.cube.move(action)
            
        return self.get_state()
    
    def get_state(self):
        """Get current state as one-hot encoded tensor"""
        batch_cube = self.cube._inner_cube
        bit_array = batch_cube.bit_array()[0]  # Get first (and only) cube
        return torch.tensor(bit_array, dtype=torch.float32)
    
    def step(self, action_idx):
        """Take action and return next state, reward, done"""
        action = self.actions[action_idx]
        self.cube = self.cube.move(action)
        
        next_state = self.get_state()
        done = self.cube.is_solved()
        
        # Simple reward structure
        if done:
            reward = 100.0  # Solve bonus
        else:
            reward = -1.0   # Step penalty
            
        return next_state, reward, done

class SimpleQAgent:
    """Simple Q-learning agent"""
    
    def __init__(self, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.1, max_episodes=20000):
        # Device selection
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Network
        self.q_network = RubiksTransformer().to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Q-learning parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.max_episodes = max_episodes
        self.episode_count = 0
        
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, 11)  # Random action
        
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        """Single-step Q-learning update"""
        state = state.unsqueeze(0).to(self.device)
        next_state = next_state.unsqueeze(0).to(self.device)
        
        # Current Q-value
        current_q = self.q_network(state)[0, action]
        
        # Target Q-value
        with torch.no_grad():
            if done:
                target_q = reward
            else:
                next_q_max = self.q_network(next_state).max().item()
                target_q = reward + self.gamma * next_q_max
        
        # Loss and update
        loss = F.mse_loss(current_q, torch.tensor(target_q, device=self.device))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Linear epsilon decay
        self.episode_count += 1
        if self.episode_count <= self.max_episodes:
            # Decay from 1.0 to epsilon_min over max_episodes
            self.epsilon = max(self.epsilon_min, 1.0 - (1.0 - self.epsilon_min) * (self.episode_count / self.max_episodes))
        else:
            self.epsilon = self.epsilon_min
        
        return loss.item()

def evaluate_agent(agent, env, num_episodes=50, max_steps=30, scramble_steps=5):
    """Evaluate agent performance"""
    successes = 0
    total_steps = 0
    
    # Temporarily disable exploration
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for _ in range(num_episodes):
        state = env.reset(scramble_steps)
        episode_steps = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            episode_steps += 1
            
            if done:
                successes += 1
                total_steps += episode_steps
                break
                
            state = next_state
    
    # Restore exploration
    agent.epsilon = old_epsilon
    
    success_rate = successes / num_episodes
    avg_steps = total_steps / max(successes, 1)
    
    return success_rate, avg_steps

def train_simple_q():
    """Simple Q-learning training loop"""
    
    # Initialize wandb
    wandb.init(project="rubiks-rl", config={
        "algorithm": "transformer_q_learning",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": "linear",
        "scramble_steps": 5,  # Start easier
        "max_episodes": 40000,
        "eval_freq": 1000,
        "max_episode_steps": 30
    })
    
    # Initialize environment and agent
    env = RubiksEnvironment()
    agent = SimpleQAgent(
        lr=wandb.config.learning_rate,
        max_episodes=wandb.config.max_episodes
    )
    
    # Training loop
    episode_rewards = []
    episode_steps = []
    losses = []
    
    pbar = tqdm(range(wandb.config.max_episodes), desc="Training")
    
    for episode in pbar:
        state = env.reset(wandb.config.scramble_steps)
        episode_reward = 0
        episode_step = 0
        episode_loss = 0
        
        for step in range(wandb.config.max_episode_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Single-step Q-learning update
            loss = agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_step += 1
            episode_loss += loss
            
            if done:
                print(f"🎉 SOLVED in {episode_step} steps at episode {episode}!")
                break
                
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        losses.append(episode_loss / episode_step)
        
        # Update progress bar more frequently
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_steps = np.mean(episode_steps[-100:]) if len(episode_steps) >= 100 else np.mean(episode_steps)
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            
            pbar.set_postfix({
                'Reward': f'{avg_reward:.1f}',
                'Steps': f'{avg_steps:.1f}',
                'Loss': f'{avg_loss:.4f}',
                'Epsilon': f'{agent.epsilon:.3f}'
            })
        
        # Logging
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_steps = np.mean(episode_steps[-100:]) if len(episode_steps) >= 100 else np.mean(episode_steps)
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            
            wandb.log({
                "episode": episode,
                "avg_reward": avg_reward,
                "avg_steps": avg_steps,
                "avg_loss": avg_loss,
                "epsilon": agent.epsilon
            })
        
        # Evaluation
        if episode % wandb.config.eval_freq == 0 and episode > 0:
            print(f"\nEvaluating at episode {episode}...")
            
            # Test on training difficulty
            success_rate, avg_solve_steps = evaluate_agent(agent, env, num_episodes=100, scramble_steps=5)
            
            wandb.log({
                "eval_success_rate_5": success_rate,
                "eval_avg_steps_5": avg_solve_steps,
            })
            
            print(f"Success rate (5 steps): {success_rate:.3f}, Avg solve steps: {avg_solve_steps:.2f}")
            
            # Test emergence on longer scrambles
            emergence_results = {}
            for test_scramble in [10, 15, 20]:
                success_rate, avg_solve_steps = evaluate_agent(
                    agent, env, num_episodes=50, scramble_steps=test_scramble
                )
                emergence_results[f"success_rate_{test_scramble}"] = success_rate
                emergence_results[f"avg_steps_{test_scramble}"] = avg_solve_steps
                
                print(f"Success rate ({test_scramble} steps): {success_rate:.3f}")
            
            wandb.log(emergence_results)
            
            # Save model
            if success_rate > 0.5:  # Save if doing well
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.q_network.state_dict(),
                    'success_rate': success_rate,
                }, f'simple_rubiks_model_episode_{episode}.pth')
                print(f"Model saved at episode {episode}")
    
    wandb.finish()
    print("Training completed!")

if __name__ == "__main__":
    train_simple_q()
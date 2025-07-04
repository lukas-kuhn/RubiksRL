#!/usr/bin/env python3
"""
Training script for Rubiks cube Q-learning agent
"""

from rubiks_transformer import train_rubiks_agent

if __name__ == "__main__":
    print("Starting Rubiks cube Q-learning training...")
    print("This will train a transformer model to solve the cube from 5 steps out")
    print("and test emergence on longer scrambles (10, 15, 20, 25+ steps)")
    print("Training progress will be logged to wandb project 'rubiks-rl'")
    print()
    
    try:
        train_rubiks_agent()
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
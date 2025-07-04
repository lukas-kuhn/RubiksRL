#!/usr/bin/env python3
"""
Q-learning training for Rubiks cube solving
"""

from simple_rubiks_q import train_simple_q

if __name__ == "__main__":
    print("🧩 Rubiks Cube Q-Learning Training")
    print("• Simple feedforward network")
    print("• Direct Q-learning updates (no replay buffer)")
    print("• Training on 5-step scrambles")
    print("• Testing emergence on 10, 15, 20+ step scrambles")
    print()
    
    train_simple_q()
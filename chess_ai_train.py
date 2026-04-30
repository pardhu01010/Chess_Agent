import pygame
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import timedelta

# Step 1: Define the Chess Environment
class ChessEnvironment:
    def __init__(self):
        self.board = chess.Board()
        self.game_over = False

    def reset(self):
        self.board = chess.Board()
        self.game_over = False
        return self.board.fen()

    def step(self, move):
        if move not in self.board.legal_moves:
            raise ValueError("Illegal Move Attempted!")

        self.board.push(move)
        reward = 0  # Default reward

        if self.board.is_game_over():
            self.game_over = True
            result = self.board.result()
            reward = 1 if result == "1-0" else -1 if result == "0-1" else 0

        return self.board.fen(), reward, self.game_over

    def get_legal_moves(self):
        return list(self.board.legal_moves)

# Helper function to convert FEN to tensor
def fen_to_tensor(fen):
    piece_map = {'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
                 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6}
    board_tensor = np.zeros((8, 8), dtype=np.float32)
    rows = fen.split()[0].split('/')
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                board_tensor[i, col] = piece_map.get(char, 0)
                col += 1
    return torch.tensor(board_tensor.flatten(), dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Define the Chess Model
class ChessModel(nn.Module):
    def __init__(self, output_size=64):
        super(ChessModel, self).__init__()
        self.fc1 = nn.Linear(8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Chess GUI
class ChessGUI:
    def __init__(self, size=600):
        self.size = size
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Chess Game")
        self.piece_images = self.load_piece_images()

    def load_piece_images(self):
        piece_images = {}
        pieces = ['K', 'Q', 'R', 'B', 'N', 'P']
        colors = ['w', 'b']
        for color in colors:
            for piece in pieces:
                image_path = os.path.join('assets', f'{color}{piece}.png')
                if os.path.exists(image_path):
                    piece_images[f'{color}{piece}'] = pygame.image.load(image_path)
        return piece_images

    def draw_board(self, board):
        square_size = self.size // 8
        colors = [(222, 184, 135), (139, 69, 19)]
        self.screen.fill((0, 0, 0))  # Clear screen
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(self.screen, color, pygame.Rect(col * square_size, row * square_size, square_size, square_size))
        for square, piece in board.piece_map().items():
            row, col = divmod(square, 8)
            piece_image = self.piece_images.get(f'{"w" if piece.color else "b"}{piece.symbol().upper()}', None)
            if piece_image:
                self.screen.blit(pygame.transform.scale(piece_image, (square_size, square_size)), (col * square_size, row * square_size))
        pygame.display.flip()

# DQN Agent
class DQNAgent:
    def __init__(self, model, optimizer, gamma=0.99):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01  # Lower minimum epsilon for better exploitation
        self.epsilon_decay = 0.9999  # Slower decay rate
        self.training_history = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'avg_moves': []
        }

    def get_move(self, state, legal_moves):
        state_tensor = fen_to_tensor(state)
        q_values = self.model(state_tensor).squeeze(0)  # Remove batch dimension

        # Only select from legal moves
        legal_q_values = [(move, q_values[move.from_square].item()) for move in legal_moves if move.from_square < 64]
        if not legal_q_values:
            return random.choice(legal_moves)  # Fallback for safety

        legal_q_values.sort(key=lambda x: x[1], reverse=True)

        # Epsilon-greedy strategy
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        return legal_q_values[0][0]  # Best move

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training function with model saving
def train_agent(episodes=10000, show_gui=False):
    pygame.init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    gui = ChessGUI() if show_gui else None
    model = ChessModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    agent = DQNAgent(model, optimizer)
    game_env = ChessEnvironment()
    max_moves = 200

    # Training metrics
    episode_rewards = []
    moving_avg_reward = []
    window_size = 100  # Window size for moving average
    checkpoint_interval = 1000
    start_time = time.time()

    try:
        for episode in range(episodes):
            state = game_env.reset()
            total_reward = 0
            done = False
            move_count = 0

            while not done and move_count < max_moves:
                legal_moves = game_env.get_legal_moves()
                if legal_moves:
                    action = agent.get_move(state, legal_moves)
                    next_state, reward, done = game_env.step(action)

                    total_reward += reward
                    state = next_state
                    move_count += 1

                    if show_gui and move_count % 10 == 0:
                        gui.draw_board(game_env.board)

                agent.update_epsilon()

            # Update training history
            if total_reward > 0:
                agent.training_history['wins'] += 1
            elif total_reward < 0:
                agent.training_history['losses'] += 1
            else:
                agent.training_history['draws'] += 1

            agent.training_history['avg_moves'].append(move_count)
            episode_rewards.append(total_reward)

            # Calculate and store moving average
            if len(episode_rewards) >= window_size:
                avg = sum(episode_rewards[-window_size:]) / window_size
                moving_avg_reward.append(avg)
            else:
                avg = sum(episode_rewards) / len(episode_rewards)
                moving_avg_reward.append(avg)

            # Show progress every 100 episodes
            if episode % 100 == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                avg_time_per_episode = elapsed / (episode + 1)
                estimated_remaining = avg_time_per_episode * (episodes - episode - 1)

                win_rate = agent.training_history['wins'] / (episode + 1) * 100
                avg_moves = sum(agent.training_history['avg_moves'][-100:]) / min(100, len(agent.training_history['avg_moves']))

                print(f"\nEpisode {episode + 1}/{episodes}")
                print(f"Win Rate: {win_rate:.2f}%")
                print(f"Average Moves: {avg_moves:.1f}")
                print(f"Epsilon: {agent.epsilon:.4f}")
                print(f"Current Moving Avg Reward: {moving_avg_reward[-1]:.3f}")
                print(f"Time per episode: {avg_time_per_episode:.2f}s")
                print(f"Elapsed time: {timedelta(seconds=int(elapsed))}")
                print(f"Estimated remaining: {timedelta(seconds=int(estimated_remaining))}")

            # Save checkpoint
            if (episode + 1) % checkpoint_interval == 0:
                checkpoint_path = f"chess_model_checkpoint_{episode + 1}.pth"
                torch.save({
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_history': agent.training_history,
                    'epsilon': agent.epsilon,
                    'moving_avg_reward': moving_avg_reward
                }, checkpoint_path)
                print(f"\nCheckpoint saved: {checkpoint_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final model...")

    finally:
        end_time = time.time()
        total_time = end_time - start_time

        # Save final model and plot training metrics
        torch.save(model.state_dict(), "chess_model_trained.pth")
        print("Final model saved as 'chess_model_trained.pth'")

        # Plot training metrics with more detail
        plt.figure(figsize=(12, 8))

        # Plot moving average reward
        plt.plot(moving_avg_reward, label='Moving Average', color='blue', linewidth=2)

        # Add episode rewards as light scatter points
        plt.scatter(range(len(episode_rewards)), episode_rewards,
                    alpha=0.1, color='gray', label='Episode Rewards')

        plt.title('Training Progress - Moving Average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add final stats as text
        stats_text = f'Final Stats:\nWin Rate: {agent.training_history["wins"] / episodes * 100:.1f}%\n'
        stats_text += f'Total Episodes: {episodes}\n'
        stats_text += f'Final ε: {agent.epsilon:.3f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Print final statistics
        total_games = sum([agent.training_history['wins'],
                           agent.training_history['losses'],
                           agent.training_history['draws']])
        print("\nTraining Complete!")
        print(f"Total training time: {timedelta(seconds=int(total_time))}")
        print(f"Average time per episode: {total_time / episodes:.2f} seconds")
        print(f"Total Games: {total_games}")
        print(f"Wins: {agent.training_history['wins']} ({agent.training_history['wins'] / total_games * 100:.1f}%)")
        print(f"Losses: {agent.training_history['losses']} ({agent.training_history['losses'] / total_games * 100:.1f}%)")
        print(f"Draws: {agent.training_history['draws']} ({agent.training_history['draws'] / total_games * 100:.1f}%)")
        print(f"Average Moves per Game: {sum(agent.training_history['avg_moves']) / len(agent.training_history['avg_moves']):.1f}")

    if show_gui:
        pygame.quit()

# Load trained model
def load_trained_model(model_path="chess_model_trained.pth"):
    model = ChessModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Trained model loaded successfully!")
    return model

# Run training without GUI for faster performance
if __name__ == "__main__":
    # Set show_gui=True if you want to visualize the board during training
    train_agent(episodes=10000, show_gui=False)

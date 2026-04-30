import pygame
import os
import torch
import torch.nn as nn

# Define the model architecture with matching layer names
class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)  # Match checkpoint dimensions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # The output shape will now be [1, 64]


# Instantiate the model
model = ChessModel()

# Load the state dictionary into the model
MODEL_PATH = "chess_model_trained.pth"
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()  # Set the model to evaluation mode

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
SQUARE_SIZE = WIDTH // 8

# Colors
WHITE = (222, 184, 135)
BLACK = (139, 69, 19)
SELECTED_COLOR = (255, 255, 0)
VALID_MOVE_COLOR = (0, 0, 255)
PANEL_COLOR = (50, 50, 50, 200)  # Dark gray with some transparency
GOLD_COLOR = (255, 215, 0)  # Gold color for the winner text
RED_COLOR = (220, 20, 60)  # Crimson red for "Game Over"

# Piece Images (Ensure these images are in the 'assets' folder)
PIECE_IMAGES = {
    'r': pygame.image.load(os.path.join('assets', 'bR.png')),
    'n': pygame.image.load(os.path.join('assets', 'bN.png')),
    'b': pygame.image.load(os.path.join('assets', 'bB.png')),
    'q': pygame.image.load(os.path.join('assets', 'bQ.png')),
    'k': pygame.image.load(os.path.join('assets', 'bK.png')),
    'p': pygame.image.load(os.path.join('assets', 'bP.png')),
    'R': pygame.image.load(os.path.join('assets', 'wR.png')),
    'N': pygame.image.load(os.path.join('assets', 'wN.png')),
    'B': pygame.image.load(os.path.join('assets', 'wB.png')),
    'Q': pygame.image.load(os.path.join('assets', 'wQ.png')),
    'K': pygame.image.load(os.path.join('assets', 'wK.png')),
    'P': pygame.image.load(os.path.join('assets', 'wP.png')),
}

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Two-Player Chess Game")

# Function to draw the chessboard
def draw_board(selected_square=None, valid_moves=None):
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            # Highlight the selected square
            if selected_square and selected_square == (row, col):
                pygame.draw.rect(screen, SELECTED_COLOR, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)

            # Highlight valid moves
            if valid_moves and (row, col) in valid_moves:
                pygame.draw.rect(screen, VALID_MOVE_COLOR, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)

# Function to draw pieces
def draw_pieces(board):
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece != ".":  # If there's a piece
                piece_image = PIECE_IMAGES.get(piece)
                if piece_image:
                    # Scale the image to fit the square size
                    piece_image = pygame.transform.scale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))
                    screen.blit(piece_image, (col * SQUARE_SIZE, row * SQUARE_SIZE))

# Initialize the chess board
def initialize_board():
    return [
        ["r", "n", "b", "q", "k", "b", "n", "r"],
        ["p", "p", "p", "p", "p", "p", "p", "p"],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        ["P", "P", "P", "P", "P", "P", "P", "P"],
        ["R", "N", "B", "Q", "K", "B", "N", "R"],
    ]

# Function to convert board to a tensor
def board_to_tensor(board):
    piece_map = {
        '.': 0, 'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6,
        'p': -1, 'r': -2, 'n': -3, 'b': -4, 'q': -5, 'k': -6
    }
    board_tensor = torch.tensor([piece_map[piece] for row in board for piece in row], dtype=torch.float32)
    return board_tensor.view(1, 64)  # Adjust to (1, 64) if the model expects this shape

# Function to get valid moves for a piece
def get_valid_moves(board, row, col, is_player_turn=True):
    piece = board[row][col]
    valid_moves = []

    # Return empty list if trying to move opponent's piece
    if (is_player_turn and not piece.isupper()) or (not is_player_turn and piece.isupper()):
        return []

    # Return empty list for empty squares
    if piece == ".":
        return []

    # Pawn movement logic
    if piece.lower() == 'p':
        direction = -1 if piece.isupper() else 1
        if 0 <= row + direction < 8 and board[row + direction][col] == ".":
            valid_moves.append((row + direction, col))
        for d in [-1, 1]:
            if 0 <= col + d < 8 and 0 <= row + direction < 8 and board[row + direction][col + d] != ".":
                if board[row + direction][col + d].isupper() != piece.isupper():
                    valid_moves.append((row + direction, col + d))

    # Rook and Queen movement logic
    elif piece.lower() == 'r' or piece.lower() == 'q':
        for d in [-1, 1]:
            for i in range(1, 8):
                new_row = row + d * i
                if 0 <= new_row < 8:
                    if board[new_row][col] == ".":
                        valid_moves.append((new_row, col))
                    else:
                        if board[new_row][col].isupper() != piece.isupper():
                            valid_moves.append((new_row, col))
                        break
                else:
                    break
            for i in range(1, 8):
                new_col = col + d * i
                if 0 <= new_col < 8:
                    if board[row][new_col] == ".":
                        valid_moves.append((row, new_col))
                    else:
                        if board[row][new_col].isupper() != piece.isupper():
                            valid_moves.append((row, new_col))
                        break
                else:
                    break

    # Bishop movement logic
    if piece.lower() == 'b' or piece.lower() == 'q':
        for d_row, d_col in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            for i in range(1, 8):
                new_row = row + d_row * i
                new_col = col + d_col * i
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    if board[new_row][new_col] == ".":
                        valid_moves.append((new_row, new_col))
                    else:
                        if board[new_row][new_col].isupper() != piece.isupper():
                            valid_moves.append((new_row, new_col))
                        break
                else:
                    break

    # Knight movement logic
    elif piece.lower() == 'n':
        knight_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        for move in knight_moves:
            new_row = row + move[0]
            new_col = col + move[1]
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                if board[new_row][new_col] == "." or board[new_row][new_col].isupper() != piece.isupper():
                    valid_moves.append((new_row,new_col))

    # King movement logic
    elif piece.lower() == 'k':
        king_moves = [
            (1, 0), (1, 1), (1,-1), (-1 ,0),
            (-1 ,1), (-1 ,-1), (0 ,1), (0 ,-1)
        ]
        for move in king_moves:
            new_row = row + move[0]
            new_col = col + move[1]
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                if board[new_row][new_col] == "." or board[new_row][new_col].isupper() != piece.isupper():
                    valid_moves.append((new_row,new_col))

    return valid_moves

# Function to calculate the AI's move
def get_black_move(board):
    # Convert board to tensor and get the model's move prediction
    board_tensor = board_to_tensor(board)
    with torch.no_grad():
        move = model(board_tensor)  # Model's output tensor

    # Find valid moves for all black pieces
    best_move = None
    best_score = float('-inf')

    for i in range(64):
        start_row, start_col = divmod(i, 8)
        piece = board[start_row][start_col]
        if piece.islower():  # Only consider black pieces
            valid_moves = get_valid_moves(board, start_row, start_col, is_player_turn=False)
            
            for end_row, end_col in valid_moves:
                score = evaluate_move(board, start_row, start_col, end_row, end_col)
                if score > best_score:
                    best_score = score
                    best_move = (start_row, start_col, end_row, end_col)
    
    return best_move

def evaluate_move(board, start_row, start_col, end_row, end_col):
    # Implement evaluation criteria, for example:
    # - Reward moves that capture opponent's pieces
    # - Penalize moves that place the king in danger
    piece = board[start_row][start_col]
    target_piece = board[end_row][end_col]
    
    score = 0
    if target_piece != '.':
        # Reward capturing an opponent's piece
        if (piece.islower() and target_piece.isupper()) or (piece.isupper() and target_piece.islower()):
            score += 10  # Arbitrary value for capturing an opponent's piece
    
    # Add other evaluation heuristics as needed
    
    return score

def draw_game_over_message(screen, winner):
    # Create a semi-transparent panel
    panel_surface = pygame.Surface((WIDTH, 200))
    panel_surface.fill(PANEL_COLOR)
    panel_surface.set_alpha(200)
    screen.blit(panel_surface, (0, HEIGHT//2 - 100))

    # Create and render "Game Over" text
    try:
        font_big = pygame.font.Font(None, 120)
        font_winner = pygame.font.Font(None, 90)
    except:
        # Fallback to default font if custom font fails
        font_big = pygame.font.SysFont('arial', 120, bold=True)
        font_winner = pygame.font.SysFont('arial', 90, bold=True)

    # Game Over text with shadow effect
    game_over_shadow = font_big.render("Game Over!", True, (0, 0, 0))
    game_over_text = font_big.render("Game Over!", True, RED_COLOR)
    
    # Winner text with shadow effect
    winner_shadow = font_winner.render(f"{winner} Wins!", True, (0, 0, 0))
    winner_text = font_winner.render(f"{winner} Wins!", True, GOLD_COLOR)

    # Position the text
    game_over_rect = game_over_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 50))
    winner_rect = winner_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 30))

    # Draw the shadow texts slightly offset
    screen.blit(game_over_shadow, (game_over_rect.x + 3, game_over_rect.y + 3))
    screen.blit(winner_shadow, (winner_rect.x + 3, winner_rect.y + 3))
    
    # Draw the main texts
    screen.blit(game_over_text, game_over_rect)
    screen.blit(winner_text, winner_rect)

# Main game loop
def main():
    board = initialize_board()
    clock = pygame.time.Clock()
    selected_square = None
    valid_moves = []
    player_turn = True  # True for White (uppercase), False for Black (lowercase)
    running = True
    game_over = False
    winner = None
    
    while running:
        draw_board(selected_square, valid_moves)
        draw_pieces(board)
        
        # Display game over message if game is over
        if game_over:
            draw_game_over_message(screen, winner)
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                pos = pygame.mouse.get_pos()
                row, col = pos[1] // SQUARE_SIZE, pos[0] // SQUARE_SIZE
                
                if player_turn:  # White's turn (player)
                    if selected_square:
                        if (row, col) in valid_moves:
                            # Check if capturing a king
                            if board[row][col].lower() == 'k':
                                game_over = True
                                winner = "White"
                            
                            # Make the move
                            board[row][col] = board[selected_square[0]][selected_square[1]]
                            board[selected_square[0]][selected_square[1]] = "."
                            selected_square = None
                            valid_moves = []
                            player_turn = False  # Switch to AI's turn
                        else:
                            selected_square = None
                            valid_moves = []
                    else:
                        piece = board[row][col]
                        if piece.isupper():  # Only select white pieces
                            selected_square = (row, col)
                            valid_moves = get_valid_moves(board, row, col, is_player_turn=True)
                
                elif not player_turn:  # Black's turn (AI)
                    black_move = get_black_move(board)
                    if black_move:
                        start_row, start_col, end_row, end_col = black_move
                        
                        # Check if capturing a king
                        if board[end_row][end_col].lower() == 'k':
                            game_over = True
                            winner = "Black"
                        
                        # Make the move
                        board[end_row][end_col] = board[start_row][start_col]
                        board[start_row][start_col] = "."
                        player_turn = True  # Switch back to player's turn

        clock.tick(60)

    pygame.quit()


# Run the game
if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 1f2cec354378ad665c1a2801ec974f153701cd18

# import pygame
# import os
# import chess

# class ChessGUI:
#     import pygame
# import os

# class ChessGUI:
#     def __init__(self, size=600):
#         self.size = size
#         self.screen = pygame.display.set_mode((size, size))
#         pygame.display.set_caption("Chess Game")
#         self.piece_images = self.load_piece_images()

#         # Custom colors for the chessboard squares
#         self.board_color_light = (222, 184, 135)  # Light brown (for white squares)
#         self.board_color_dark = (139, 69, 19)    # Dark brown (for black squares)

#     def load_piece_images(self):
#         piece_images = {}
#         pieces = ['K', 'Q', 'R', 'B', 'N', 'P']
#         colors = ['w', 'b']
        
#         for color in colors:
#             for piece in pieces:
#                 image_path = os.path.join('assets', f'{color}{piece}.png')
#                 if os.path.exists(image_path):
#                     piece_images[f'{color}{piece}'] = pygame.image.load(image_path)
#                 else:
#                     print(f"Image not found: {image_path}")
#         return piece_images

#     def draw_board(self, board):
#         self.screen.fill((255, 255, 255))  # Fill the screen with white background (for the border)
#         square_size = self.size // 8
        
#         for row in range(8):
#             for col in range(8):
#                 # Alternate between light and dark brown for the squares
#                 color = self.board_color_light if (row + col) % 2 == 0 else self.board_color_dark
#                 pygame.draw.rect(self.screen, color, pygame.Rect(col * square_size, row * square_size, square_size, square_size))
        
#         # Draw the pieces on the board
#         for row in range(8):
#             for col in range(8):
#                 piece = board.piece_at(row * 8 + col)
#                 if piece:
#                     piece_symbol = piece.symbol()
#                     piece_color = 'w' if piece_symbol.isupper() else 'b'
#                     piece_type = piece_symbol.upper()
#                     piece_image = self.piece_images.get(f'{piece_color}{piece_type}')
#                     if piece_image:
#                         piece_image = pygame.transform.scale(piece_image, (square_size, square_size))
#                         self.screen.blit(piece_image, (col * square_size, row * square_size))
        
#         pygame.display.flip()  # Update the screen with the new drawing

# # Example usage:
# if __name__ == "__main__":
#     pygame.init()
#     gui = ChessGUI()
#     board = chess.Board()
#     gui.draw_board(board)
    
#     # Main event loop to keep the window open until the user closes it
#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
                
#     pygame.quit()


import pygame
import os

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
SQUARE_SIZE = WIDTH // 8

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SELECTED_COLOR = (255, 255, 0)  # Color for selected square
VALID_MOVE_COLOR = (0, 0, 255)  # Color for valid moves

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

# Function to get valid moves for a piece
def get_valid_moves(board, row, col):
    piece = board[row][col]
    valid_moves = []

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



# Main loop function with turn handling
def main():
    board = initialize_board()
    selected_square = None
    valid_moves = []
    
    current_turn = True # True for white's turn; False for black's turn

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                row, col = mouse_y // SQUARE_SIZE, mouse_x // SQUARE_SIZE

                # Select or move a piece based on turn
                if selected_square is None: 
                    if current_turn and board[row][col].isupper(): # White's turn and selecting white pieces 
                        selected_square = (row,col)
                        valid_moves = get_valid_moves(board,row,col)
                    elif not current_turn and board[row][col].islower(): # Black's turn and selecting black pieces 
                        selected_square = (row,col)
                        valid_moves = get_valid_moves(board,row,col)
                else: 
                    # Move the piece only if it's a valid move 
                    if (row,col) in valid_moves: 
                        board[row][col] = board[selected_square[0]][selected_square[1]] 
                        board[selected_square[0]][selected_square[1]] = "." 
                        current_turn = not current_turn # Switch turns after a successful move 
                    
                    selected_square = None 
                    valid_moves.clear()

        # Draw the updated state of the game 
        draw_board(selected_square ,valid_moves) 
        draw_pieces(board) 
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

import numpy as np
import tkinter as tk
import environments.colors as c

import gym
from gym.spaces import Discrete, Box
from gym.utils import seeding


class GameEnv(object):
    """
        2048 (video game, implemented by using numpy)

        Description:
            2048 is a single-player sliding block puzzle game; see https://en.wikipedia.org/wiki/2048_(video_game).
            The goal of the game is to slide numbered tiles on a 4x4 grid to combine them to creat a tile with the
            number 2048. The game is played on a gray 4x4 grid, with numbered tiles that slide when a player moves
            them using the four arrow keys. Every turn, a new tile will randomly appear in an empty spot on the board
            with a value of either 2 or 4. Tiles slide as far as possible in the chosen direction until they are stopped
            by either another tile or the edge of the grid. If two tiles of the same number collide while moving, they
            will merge into a tile with the total value of the two tiles that collided. The resulting tile cannot merge
            with another tile again in the same move.

        Source:
            A version of this code with some errors can be found in https://www.youtube.com/watch?v=b4XP2IcI-Bg.

        Observation:
            Type: Box(low=0, high=2048, shape=(16,), type=np.int64)
                Observation (i.e., flatten 4x4 matrix, which describes the current board state)
            e.g., [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]

        Actions:
            Type: Discrete(4)
            Num   Action
            0     Move tiles placed on 4x4 grid to the left
            1     Move tiles placed on 4x4 grid to the right
            2     Move tiles placed on 4x4 grid to the up
            3     Move tiles placed on 4x4 grid to the down

        Reward:
            Reward is 1 if a tile with the number 2048 is formed; otherwise, reward is 0.

        Starting State:
            A board (gray 4x4 grid) consists of two randomly placed tiles with the number 2.

        Episode Termination:
            There is either a tile with the number 2048 on the board or no empty spot to place a tile on the board.
    """
    def __init__(self):
        # instantiate the render class
        self.render = self._render(self)
        # create different probability distributions and set a seed
        self.seed()
        # start a new game
        self.start_game()
        # define action and observation spaces
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=2048, shape=(16,), dtype=np.int64)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def start_game(self) -> None:
        """Initialize the game."""
        # create matrix of zeros
        self.state = np.zeros((4, 4), dtype=np.int64)
        # pick random coordinates
        row, col = self.np_random.randint(low=0, high=4, size=2, dtype=np.int64)
        # place the first tile on the board
        self.state[row, col] = 2
        # pick random coordinates
        while(self.state[row, col] != 0):
            row, col = self.np_random.randint(low=0, high=4, size=2, dtype=np.int64)
        # place the second tile on the board
        self.state[row, col] = 2
        # initialize the score
        self.score = 0

    def stack(self) -> None:
        # create matrix of zeros
        new_state = np.zeros((4, 4), dtype=np.int64)
        for i in range(4):
            k = 0
            for j in range(4):
                if (self.state[i, j] != 0):
                    new_state[i, k] = self.state[i, j]
                    k += 1
        self.state = new_state

    def combine(self) -> None:
        for i in range(4):
            for j in range(3):
                if (self.state[i, j] != 0) and (
                        self.state[i, j] == self.state[i, j + 1]):
                    self.state[i, j] *= 2
                    self.state[i, j + 1] = 0
                    self.score += self.state[i, j]

    def reverse(self) -> None:
        new_state = np.zeros((4, 4), dtype=np.int64)
        for i in range(4):
            for j in range(4):
                new_state[i, j] = self.state[i, 3 - j]
        self.state = new_state

    def transpose(self) -> None:
        """Take the transpose of the game board."""
        self.state = np.transpose(self.state)

    def add_new_tile(self) -> None:
        """Add a new 2 or 4 tile randomly to an empty cell."""
        if any(0 in row for row in self.state):
            row, col = self.np_random.randint(
                low=0, high=4, size=2, dtype=np.int64)
            while(self.state[row, col] != 0):
                row, col = self.np_random.randint(
                    low=0, high=4, size=2, dtype=np.int64)
            self.state[row, col] = self.np_random.choice([2, 4])

    def left_move_exists(self) -> bool:
        """Check if moving left is possible."""
        for i in range(4):
            for j in range(3):
                if self.state[i, j] != 0 and self.state[i, j + 1] != 0:
                    if self.state[i, j] == self.state[i, j + 1]:
                        return True
                elif self.state[i, j] == 0 and self.state[i, j + 1] != 0:
                    return True
        return False

    def right_move_exists(self) -> bool:
        """Check if moving right is possible."""
        for i in range(4):
            for j in range(3):
                if self.state[i, j] != 0 and self.state[i, j + 1] != 0:
                    if self.state[i, j] == self.state[i, j + 1]:
                        return True
                elif self.state[i, j] != 0 and self.state[i, j + 1] == 0:
                    return True
        return False

    def up_move_exists(self) -> bool:
        """Check if moving up is possible."""
        for j in range(4):
            for i in range(3):
                if self.state[i, j] != 0 and self.state[i + 1, j] != 0:
                    if self.state[i, j] == self.state[i + 1, j]:
                        return True
                elif self.state[i, j] == 0 and self.state[i + 1, j] != 0:
                    return True
        return False

    def down_move_exists(self) -> bool:
        """Check if moving down is possible."""
        for j in range(4):
            for i in range(3):
                if self.state[i, j] != 0 and self.state[i + 1, j] != 0:
                    if self.state[i, j] == self.state[i + 1, j]:
                        return True
                elif self.state[i, j] != 0 and self.state[i + 1, j] == 0:
                    return True
        return False

    def horizontal_move_exists(self) -> bool:
        """Check if moving horizontal is possible."""
        if self.left_move_exists() or self.right_move_exists():
            return True
        else:
            return False

    def vertical_move_exists(self) -> bool:
        """Check if moving vertical is possible."""
        if self.up_move_exists() or self.down_move_exists():
            return True
        else:
            return False

    def is_game_over(self) -> bool:
        """Check if game is over."""
        if any(2048 in row for row in self.state):
            reward, done = 1.0, True
            return reward, done
        elif not any(0 in row for row in self.state) and not self.horizontal_move_exists() and not self.vertical_move_exists():
            reward, done = 0.0, True
            return reward, done
        else:
            reward, done = 0.0, False
            return reward, done

    def step(self, action: int):
        """Take an action, i.e., move left, right, up or down."""
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        if action == 0:  # move left
            if self.left_move_exists():
                self.stack()
                self.combine()
                self.stack()
                self.add_new_tile()
        elif action == 1:  # move right
            if self.right_move_exists():
                self.reverse()
                self.stack()
                self.combine()
                self.stack()
                self.reverse()
                self.add_new_tile()
        elif action == 2:  # move up
            if self.up_move_exists():
                self.transpose()
                self.stack()
                self.combine()
                self.stack()
                self.transpose()
                self.add_new_tile()
        elif action == 3:  # move down
            if self.down_move_exists():
                self.transpose()
                self.reverse()
                self.stack()
                self.combine()
                self.stack()
                self.reverse()
                self.transpose()
                self.add_new_tile()

        reward, done = self.is_game_over()

        return self.state.reshape((16,)), reward, done, {}

    def reset(self):
        """Reset the (game) state."""
        self.start_game()
        return self.state.reshape((16,))


    class _render(tk.Frame):
        def __init__(self, GameEnv):
            tk.Frame.__init__(self)
            self.GameEnv = GameEnv

        def start(self):
            self.grid()
            self.master.title("2048")
            self.main_grid = tk.Frame(self, bg=c.GRID_COLOR, bd=3, width=600, height=600)
            self.main_grid.grid(pady=(100, 0))
            self.create_board()
            self.refresh()

        def create_board(self):
            # make grid
            self.cells = []
            for i in range(4):
                row = []
                for j in range(4):
                    cell_frame = tk.Frame(
                        self.main_grid,
                        bg=c.EMPTY_CELL_COLOR,
                        width=150,
                        height=150)
                    cell_frame.grid(row=i, column=j, padx=5, pady=5)
                    cell_number = tk.Label(self.main_grid, bg=c.EMPTY_CELL_COLOR)
                    cell_number.grid(row=i, column=j)
                    cell_data = {"frame": cell_frame, "number": cell_number}
                    row.append(cell_data)
                self.cells.append(row)

            # make score heading
            score_frame = tk.Frame(self)
            score_frame.place(relx=0.5, y=45, anchor="center")
            tk.Label(
                score_frame,
                text="Score",
                font=c.SCORE_LABEL_FONT).grid(
                row=0)
            self.score_label = tk.Label(score_frame, text="0", font=c.SCORE_FONT)
            self.score_label.grid(row=1)

        def refresh(self):
            for i in range(4):
                for j in range(4):
                    cell_value = self.GameEnv.state[i, j]
                    if cell_value == 0:
                        self.cells[i][j]["frame"].configure(bg=c.EMPTY_CELL_COLOR)
                        self.cells[i][j]["number"].configure(
                            bg=c.EMPTY_CELL_COLOR, text="")
                    else:
                        self.cells[i][j]["frame"].configure(
                            bg=c.CELL_COLORS[cell_value])
                        self.cells[i][j]["number"].configure(
                            bg=c.CELL_COLORS[cell_value],
                            fg=c.CELL_NUMBER_COLORS[cell_value],
                            font=c.CELL_NUMBER_FONTS[cell_value],
                            text=str(cell_value))
            self.score_label.configure(text=self.GameEnv.score)
            self.update_idletasks()
            self.update()

            if any(2048 in row for row in self.GameEnv.state):
                game_over_frame = tk.Frame(self.main_grid, borderwidth=2)
                game_over_frame.place(relx=0.5, rely=0.5, anchor="center")
                tk.Label(
                    game_over_frame,
                    text="You win!",
                    bg=c.WINNER_BG,
                    fg=c.GAME_OVER_FONT_COLOR,
                    font=c.GAME_OVER_FONT).pack()
            elif not any(0 in row for row in self.GameEnv.state) and not self.GameEnv.horizontal_move_exists() and not self.GameEnv.vertical_move_exists():
                game_over_frame = tk.Frame(self.main_grid, borderwidth=2)
                game_over_frame.place(relx=0.5, rely=0.5, anchor="center")
                tk.Label(
                    game_over_frame,
                    text="Game over!",
                    bg=c.LOSER_BG,
                    fg=c.GAME_OVER_FONT_COLOR,
                    font=c.GAME_OVER_FONT).pack()

        def close(self):
            self.mainloop()

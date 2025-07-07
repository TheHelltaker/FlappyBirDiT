# Core game code taken from https://codewithcurious.com/projects/flappy-bird-game-using-python/
# Importing the libraries
import pygame
import sys
import time
import os
import random

from pathlib import Path

curpath = Path(__file__).parent
os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0' #for simplicity in mss screen grab


# Frames per second
class FlappyBird():

    width = 350
    height = 622

    def __init__(self):
        pygame.init()
        # Game window
        self.width, self.height = FlappyBird.width, FlappyBird.height
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird")

        # setting background and base image
        self.back_img = pygame.image.load(curpath / "img_46.png")
        self.floor_img = pygame.image.load(curpath / "img_50.png")
        self.floor_x = 0

        # different stages of bird
        bird_up = pygame.image.load(curpath / "img_47.png")
        bird_down = pygame.image.load(curpath / "img_48.png")
        bird_mid = pygame.image.load(curpath / "img_49.png")
        self.birds = [bird_up, bird_mid, bird_down]
        self.bird_index = 0
        self.bird_flap = pygame.USEREVENT
        pygame.time.set_timer(self.bird_flap, 200)
        self.bird_img = self.birds[self.bird_index]
        self.bird_rect = self.bird_img.get_rect(center=(67, 622 // 2))
        self.bird_movement = 0
        self.gravity = 0.17

        # Loading pipe image
        self.pipe_img = pygame.image.load(curpath /"greenpipe.png")
        self.pipe_height = [400, 350, 533, 490]

        # for the pipes to appear
        self.pipes = []
        self.create_pipe = pygame.USEREVENT + 1
        pygame.time.set_timer(self.create_pipe, 1200)

        # Displaying game over image
        self.game_over = True
        self.over_img = pygame.image.load(curpath / "img_45.png").convert_alpha ()
        self.over_rect = self.over_img.get_rect(center=(self.width // 2, self.height // 2))

        # setting variables and font for score
        self.score = 0
        self.high_score = 0
        self.score_time = True
        self.score_font = pygame.font.Font("freesansbold.ttf", 27)


    # Function to draw
    def draw_floor(self):
        self.screen.blit(self.floor_img, (self.floor_x, 520))
        self.screen.blit(self.floor_img, (self.floor_x + 448, 520))


    # Function to create pipes
    def create_pipes(self):
        pipe_y = random.choice(self.pipe_height)
        top_pipe = self.pipe_img.get_rect(midbottom=(467, pipe_y - 300))
        bottom_pipe = self.pipe_img.get_rect(midtop=(467, pipe_y))
        return top_pipe, bottom_pipe


    # Function for animation
    def pipe_animation(self):
        for pipe in self.pipes:
            if pipe.top < 0:
                flipped_pipe = pygame.transform.flip(self.pipe_img, False, True)
                self.screen.blit(flipped_pipe, pipe)
            else:
                self.screen.blit(self.pipe_img, pipe)

            pipe.centerx -= 3
            if pipe.right < 0:
                self.pipes.remove(pipe)

            if self.bird_rect.colliderect(pipe):
                self.game_over = True


    # Function to draw score
    def draw_score(self, game_state):
        if game_state == "game_on":
            score_text = self.score_font.render(str(self.score), True, (255, 255, 255))
            score_rect = score_text.get_rect(center=(self.width // 2, 66))
            self.screen.blit(score_text, score_rect)
        elif game_state == "game_over":
            score_text = self.score_font.render(f" Score: {self.score}", True, (255, 255, 255))
            score_rect = score_text.get_rect(center=(self.width // 2, 66))
            self.screen.blit(score_text, score_rect)

            high_score_text = self.score_font.render(f"High Score: {self.high_score}", True, (255, 255, 255))
            high_score_rect = high_score_text.get_rect(center=(self.width // 2, 506))
            self.screen.blit(high_score_text, high_score_rect)


    # Function to update the score
    def score_update(self):
        if self.pipes:
            for pipe in self.pipes:
                if 65 < pipe.centerx < 69 and self.score_time:
                    self.score += 1
                    self.score_time = False
                if pipe.left <= 0:
                    self.score_time = True

        if self.score > self.high_score:
            self.high_score = self.score

    def run(self):
        # Game loop
        running = True
        while running:
            self.clock.tick(120)

            # for checking the events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # QUIT event
                    running = False
                    sys.exit()

                if event.type == pygame.KEYDOWN:  # Key pressed event
                    if event.key == pygame.K_SPACE and not self.game_over:  # If space key is pressed
                        self.bird_movement = 0
                        self.bird_movement = -7

                    if event.key == pygame.K_SPACE and self.game_over:
                        self.game_over = False
                        self.pipes = []
                        self.bird_movement = 0
                        self.bird_rect = self.bird_img.get_rect(center=(67, 622 // 2))
                        self.score_time = True
                        self.score = 0

                # To load different stages
                if event.type == self.bird_flap:
                    self.bird_index += 1

                    if self.bird_index > 2:
                        self.bird_index = 0

                    self.bird_img = self.birds[self.bird_index]
                    self.bird_rect = self.birds[0].get_rect(center=self.bird_rect.center)

                # To add pipes in the list
                if event.type == self.create_pipe:
                    self.pipes.extend(self.create_pipes())

            self.screen.blit(self.floor_img, (self.floor_x, 550))
            self.screen.blit(self.back_img, (0, 0))

            # Game over conditions
            if not self.game_over:
                self.bird_movement += self.gravity
                self.bird_rect.centery += self.bird_movement
                rotated_bird = pygame.transform.rotozoom(self.bird_img, self.bird_movement * -6, 1)

                if self.bird_rect.top < 5 or self.bird_rect.bottom >= 550:
                    self.game_over = True

                self.screen.blit(rotated_bird, self.bird_rect)
                self.pipe_animation()
                self.score_update()
                self.draw_score("game_on")
            elif self.game_over:
                self.screen.blit(self.over_img, self.over_rect)
                self.draw_score("game_over")

            # To move the base
            self.floor_x -= 1
            if self.floor_x < -448:
                self.floor_x = 0

            self.draw_floor()

            # Update the game window
            pygame.display.update()
        pygame.quit()
        sys.exit()

    @classmethod
    def game_window(cls) -> tuple[int,int,int,int]:
        return 0, 0, cls.width, cls.height
    @classmethod
    def game_inputs(cls):
        return ['space']

if __name__ == "__main__":
    game = FlappyBird()
    game.run()
    # quiting the pygame and sys

import pygame
import sys

WIDTH, HEIGHT = 800, 700
FPS = 60

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Self-Driving Car - DQN")
    clock = pygame.time.Clock()

    track = pygame.image.load("assets/new_map.png")

    running = True
    while running:
        screen.blit(track, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

import pygame
import time

pygame.init()
screen = pygame.display.set_mode((100, 100))

try:
    while True:
        event = pygame.event.wait()
        if event.type == pygame.KEYDOWN:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                break
            print("{} {}".format("left" if keys[pygame.K_LEFT] else "", "right" if keys[pygame.K_RIGHT] else ""))

finally:
    pygame.quit()
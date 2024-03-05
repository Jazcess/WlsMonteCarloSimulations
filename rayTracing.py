import pygame
import math
import random

# Define constants
WIDTH, HEIGHT = 800, 600
WLS_WIDTH, WLS_HEIGHT = 400, 300
PMT_RADIUS = 20

# Define colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Photon Ray Tracing")

# Define WLS plate
wls_plate = pygame.Rect((WIDTH - WLS_WIDTH) // 2, (HEIGHT - WLS_HEIGHT) // 2, WLS_WIDTH, WLS_HEIGHT)

# Define PMT position
pmt_position = (WIDTH // 2, HEIGHT // 2)

# Define photon class
class Photon:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

    def move(self):
        # Move photon along its angle
        self.x += 5 * math.cos(self.angle)
        self.y += 5 * math.sin(self.angle)

    def check_collision(self):
        # Check collision with WLS plate
        if not wls_plate.collidepoint(self.x, self.y):
            # Reflect photon if it hits the edges
            self.angle = random.uniform(0, 2 * math.pi)

        # Check collision with PMT
        if math.sqrt((self.x - pmt_position[0]) ** 2 + (self.y - pmt_position[1]) ** 2) <= PMT_RADIUS:
            return True

        return False

# Create initial photon
photon = Photon(WIDTH // 4, HEIGHT // 2, random.uniform(0, 2 * math.pi))

# Run simulation
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move and check collision of the photon
    photon.move()
    if photon.check_collision():
        running = False

    # Draw the scene
    screen.fill(WHITE)
    pygame.draw.rect(screen, BLUE, wls_plate)
    pygame.draw.circle(screen, YELLOW, pmt_position, PMT_RADIUS)
    pygame.draw.circle(screen, YELLOW, (int(photon.x), int(photon.y)), 2)

    pygame.display.flip()
    pygame.time.delay(50)  # Adjust delay to control simulation speed

pygame.quit()

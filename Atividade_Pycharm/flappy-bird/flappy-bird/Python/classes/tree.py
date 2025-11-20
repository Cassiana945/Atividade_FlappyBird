import pygame

tree_up_img = pygame.image.load("../images/trees_up.png")
tree_down_img = pygame.image.load("../images/trees_down.png")

class Tree:
    def __init__(self, x, height, velocity):
        self.x = x
        self.height = height
        self.velocity = velocity
        self.scored = False

    def update(self):
        self.x -= self.velocity

    def draw(self, screen):

        # Draw bottom tree
        screen.blit(tree_up_img, (self.x, self.height))
import pygame

WHITE = (255, 255, 255)


class Bird(pygame.sprite.Sprite):
    # This class represents a bird. It derives from the "Sprite" class in Pygame.

    def __init__(self, x, y):
        # Call the parent class (Sprite) constructor
        super().__init__()

        self.x = x
        self.y = y
        self.velocity = 0
        self.alive = True  # Adicione esta linha

        # Load all the images for our animation and store these in a list
        self.images = []
        self.images.append(pygame.image.load('../images/fat_bird_1.png'))
        self.images.append(pygame.image.load('../images/fat_bird_2.png'))
        self.images.append(pygame.image.load('../images/fat_bird_3.png'))
        self.images.append(pygame.image.load('../images/fat_bird_4.png'))
        self.images.append(pygame.image.load('../images/fat_bird_5.png'))
        self.images.append(pygame.image.load('../images/fat_bird_6.png'))


        # Use the first image for our sprite
        self.index = 0
        self.image = self.images[self.index]

        # Fetch the rectangle object that has the dimensions of the image.
        self.rect = self.image.get_rect()

        # Position the sprite on the screen at the given coordinates
        self.rect.x = x
        self.rect.y = y

    def update_image(self):
        # Increment the inex by 1 everytimne the update method is called
        self.index += 1

        # Check if the index is larger than the total number of images
        if self.index >= len(self.images):
            # Reset the index to 0
            self.index = 0
        import pygame

        WHITE = (255, 255, 255)

        class Bird(pygame.sprite.Sprite):
            # This class represents a bird. It derives from the "Sprite" class in Pygame.

            def __init__(self, x, y):
                # Call the parent class (Sprite) constructor
                super().__init__()

                self.x = x
                self.y = y
                self.velocity = 0

                # Load all the images for our animation and store these in a list
                self.images = []
                self.images.append(pygame.image.load('../images/fat_bird_1.png'))
                self.images.append(pygame.image.load('../images/fat_bird_2.png'))
                self.images.append(pygame.image.load('../images/fat_bird_3.png'))
                self.images.append(pygame.image.load('../images/fat_bird_4.png'))
                self.images.append(pygame.image.load('../images/fat_bird_5.png'))
                self.images.append(pygame.image.load('../images/fat_bird_6.png'))

                # Use the first image for our sprite
                self.index = 0
                self.image = self.images[self.index]

                # Fetch the rectangle object that has the dimensions of the image.
                self.rect = self.image.get_rect()

                # Position the sprite on the screen at the given coordinates
                self.rect.x = x
                self.rect.y = y

            def update_image(self):
                # Increment the inex by 1 everytimne the update method is called
                self.index += 1

                # Check if the index is larger than the total number of images
                if self.index >= len(self.images):
                    # Reset the index to 0
                    self.index = 0

                # Update the image that will be displayed
                self.image = self.images[self.index]

            def jump(self):
                self.velocity = -10

            def update(self):
                self.velocity += 0.75
                self.y += self.velocity
                self.rect.y = self.y

            def draw(self, screen):
                screen.blit(self.image, (self.x, self.y))

        # Update the image that will be displayed
        self.image = self.images[self.index]


    def jump(self):
        self.velocity = -10


    def update(self):
        self.velocity += 0.75
        self.y += self.velocity
        self.rect.y = self.y


    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))

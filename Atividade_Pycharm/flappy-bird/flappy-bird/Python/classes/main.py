import pygame, sys, random
from bird import Bird
from tree import Tree


# Initialize Game Atividade_Pycharm
pygame.init()

game_state = 1
score = 0
has_moved = False


# Window Setup
window_w = 288
window_h = 208

screen = pygame.display.set_mode((window_w, window_h))
pygame.display.set_caption("Flappython")
clock = pygame.time.Clock()
fps = 60

# Load Fonts
font = pygame.font.Font("../fonts/BaiJamjuree-Bold.ttf", 40)

# Load Sounds
slap_sfx = pygame.mixer.Sound("../sounds/slap.wav")
woosh_sfx = pygame.mixer.Sound("../sounds/woosh.wav")
score_sfx = pygame.mixer.Sound("../sounds/score.wav")

# Load Images
tree_up_img = pygame.image.load("../images/trees_up.png")
tree_down_img = pygame.image.load("../images/trees_down.png")
ground_img = pygame.image.load("../images/ground.png")


bg_img = pygame.image.load("../images/background.png")
bg_width = bg_img.get_width()

# Variable Setup
bg_scroll_spd = 1
ground_scroll_spd = 2



def scoreboard():
    show_score = font.render(str(score), True, (10, 40, 9))
    score_rect = show_score.get_rect(center=(window_w//2, 64))
    screen.blit(show_score, score_rect)


def game():
    global game_state
    global score
    global has_moved

    bg_x_pos = 0
    ground_x_pos = 0

    bird = Bird(80, 256)
    trees = [Tree(288, random.randint(80, 350), 130, 2.4)]
    animation_counter = 0

    while game_state != 0:
        # Gameplay
        while game_state == 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    has_moved = True
                    if event.key == pygame.K_SPACE:
                        pygame.mixer.Sound.play(woosh_sfx)
                        bird.jump()


            if has_moved == True:
                bird.update()

                animation_counter += 1
                if animation_counter >= 5:
                    bird.update_image()
                    animation_counter = 0


                bird_rect = bird.rect


                for tree in trees:
                    tree_width = tree_up_img.get_width()
                    tree_top_height = tree.height
                    tree_gap = tree.gap
                    tree_bottom_y = tree_top_height + tree_gap

                    tree_top_rect = pygame.Rect(tree.x, 0, tree_width, tree_top_height)
                    tree_bottom_rect = pygame.Rect(tree.x, tree_bottom_y, tree_width, window_h - tree_bottom_y)

                    if bird_rect.colliderect(tree_top_rect) or bird_rect.colliderect(tree_bottom_rect):
                        bird = Bird(168, 300)
                        trees = [Tree(600, random.randint(30, 250), 220, 2.4)]
                        score = 0
                        has_moved = False
                        animation_counter = 0
                        pygame.mixer.Sound.play(slap_sfx)

                if bird.y < -64 or bird.y > 536:
                    bird = Bird(168, 300)
                    trees = [Tree(600, random.randint(30, 250), 220, 2.4)]
                    score = 0
                    has_moved = False
                    animation_counter = 0
                    pygame.mixer.Sound.play(slap_sfx)

                for tree in trees:
                    tree.update()

                if trees[0].x < -tree_up_img.get_width():
                    trees.pop(0)
                    trees.append(Tree(400, random.randint(30, 280), 220, 2.4))


                for tree in trees:
                    if not tree.scored and tree.x + tree_up_img.get_width() < bird.x:
                        score += 1
                        pygame.mixer.Sound.play(score_sfx)
                        tree.scored = True


            bg_x_pos -= bg_scroll_spd
            ground_x_pos -= ground_scroll_spd

            if bg_x_pos <= -bg_width:
                bg_x_pos = 0

            if ground_x_pos <= -bg_width:
                ground_x_pos = 0

            screen.fill("blue")
            screen.blit(bg_img, (bg_x_pos, 0))
            screen.blit(bg_img, (bg_x_pos + bg_width, 0))
            screen.blit(ground_img, (ground_x_pos, 536))
            screen.blit(ground_img, (ground_x_pos + bg_width, 536))

            for tree in trees:
                tree.draw(screen)

            bird.draw(screen)
            scoreboard()

            pygame.display.flip()
            clock.tick(fps)

game()
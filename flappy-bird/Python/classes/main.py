import os.path
import pygame
import sys
import random
import neat
from bird import Bird
from tree import Tree

ai_playing = True
generation = 0

# Initialize Game
pygame.init()

game_state = 1
score = 0

# Window Setup
window_w = 288
window_h = 208

screen = pygame.display.set_mode((window_w, window_h))
pygame.display.set_caption("Flappython")
clock = pygame.time.Clock()
fps = 60

# Load Fonts
font = pygame.font.Font("../fonts/BaiJamjuree-Bold.ttf", 20)

# Load Sounds
slap_sfx = pygame.mixer.Sound("../sounds/slap.wav")
woosh_sfx = pygame.mixer.Sound("../sounds/woosh.wav")
score_sfx = pygame.mixer.Sound("../sounds/score.wav")

# Load Images
tree_up_img = pygame.image.load("../images/trees_up.png")
ground_img = pygame.image.load("../images/ground.png")
bg_img = pygame.image.load("../images/background.png")
bg_width = bg_img.get_width()

# Variable Setup
bg_scroll_spd = 1
ground_scroll_spd = 2


def scoreboard(birds):
    # Show current score
    show_score = font.render(f"Score: {score}", True, (255, 255, 255))
    score_rect = show_score.get_rect(topleft=(10, 10))
    screen.blit(show_score, score_rect)

    if ai_playing:
        # Show generation and alive birds
        show_generation = font.render(f"Gen: {generation}", True, (255, 255, 255))
        generation_rect = show_generation.get_rect(topright=(window_w - 10, 10))
        screen.blit(show_generation, generation_rect)

        alive_count = len([bird for bird in birds if hasattr(bird, 'alive') and bird.alive])
        show_alive = font.render(f"Alive: {alive_count}", True, (255, 255, 255))
        alive_rect = show_alive.get_rect(topright=(window_w - 10, 35))
        screen.blit(show_alive, alive_rect)


def eval_genomes(genomes, config):
    global generation, score
    generation += 1

    # Initialize neural networks
    nets = []
    birds = []
    ge = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(50, 104))
        ge.append(genome)
        genome.fitness = 0

    # Game variables
    bg_x_pos = 0
    ground_x_pos = 0
    trees = [Tree(288, random.randint(80, 160), 2.4)]
    score = 0
    running = True

    while running and len(birds) > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if not ai_playing:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    for bird in birds:
                        bird.jump()
                        pygame.mixer.Sound.play(woosh_sfx)

        # Find the nearest tree
        tree_ind = 0
        if len(trees) > 1 and birds[0].x > trees[0].x + tree_up_img.get_width():
            tree_ind = 1

        # Update birds using neural networks
        for i, bird in enumerate(birds):
            if hasattr(bird, 'alive') and bird.alive:
                # Update fitness for staying alive
                ge[i].fitness += 0.1

                # Get inputs for neural network
                if len(trees) > 0:
                    # Inputs: bird y position, distance to tree gap, bird velocity
                    current_tree = trees[tree_ind]
                    output = nets[i].activate((
                        bird.y / window_h,  # Normalized y position
                        abs(bird.y - current_tree.height) / window_h,  # Distance to gap
                        bird.velocity / 20  # Normalized velocity
                    ))

                    # Jump if output > 0.5
                    if output[0] > 0.5:
                        bird.jump()
                        pygame.mixer.Sound.play(woosh_sfx)

                # Update bird
                bird.update()

                # Check collisions with ground or ceiling
                if bird.y >= window_h - 50 or bird.y <= -50:
                    bird.alive = False
                    ge[i].fitness -= 1
                    birds.pop(i)
                    nets.pop(i)
                    ge.pop(i)
                    continue

                # Check collision with trees
                bird_rect = pygame.Rect(bird.x, bird.y, 34, 24)  # Adjust size as needed

                collision = False
                for tree in trees:
                    tree_width = tree_up_img.get_width()
                    tree_bottom_rect = pygame.Rect(tree.x, tree.height, tree_width, window_h - tree.height)

                    if bird_rect.colliderect(tree_bottom_rect):
                        collision = True
                        break

                if collision:
                    bird.alive = False
                    ge[i].fitness -= 5
                    birds.pop(i)
                    nets.pop(i)
                    ge.pop(i)
                    pygame.mixer.Sound.play(slap_sfx)

        # Update trees
        for tree in trees:
            tree.update()

            # Check if bird passed the tree
            for bird in birds:
                if hasattr(bird, 'alive') and bird.alive and not tree.scored:
                    if tree.x + tree_up_img.get_width() < bird.x:
                        tree.scored = True
                        score += 1
                        pygame.mixer.Sound.play(score_sfx)
                        # Reward all genomes for passing a tree
                        for g in ge:
                            g.fitness += 5

        # Remove off-screen trees and add new ones
        if trees[0].x < -tree_up_img.get_width():
            trees.pop(0)
            trees.append(Tree(288, random.randint(80, 160), 2.4))

        # Add new tree periodically
        if len(trees) < 2 and trees[-1].x < window_w - 150:
            trees.append(Tree(288, random.randint(80, 160), 2.4))

        # Scroll background
        bg_x_pos -= bg_scroll_spd
        ground_x_pos -= ground_scroll_spd

        if bg_x_pos <= -bg_width:
            bg_x_pos = 0
        if ground_x_pos <= -bg_width:
            ground_x_pos = 0

        # Draw everything
        screen.fill("blue")
        screen.blit(bg_img, (bg_x_pos, 0))
        screen.blit(bg_img, (bg_x_pos + bg_width, 0))
        screen.blit(ground_img, (ground_x_pos, window_h - ground_img.get_height()))
        screen.blit(ground_img, (ground_x_pos + bg_width, window_h - ground_img.get_height()))

        for tree in trees:
            tree.draw(screen)

        for bird in birds:
            bird.draw(screen)
            # Update animation for alive birds
            if hasattr(bird, 'alive') and bird.alive:
                bird.update_image()

        # Draw score and info
        scoreboard(birds)

        pygame.display.flip()
        clock.tick(fps)

        # Break if all birds are dead
        if len(birds) == 0:
            break


def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Create population
    population = neat.Population(config)

    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run for up to 50 generations
    winner = population.run(eval_genomes, 50)

    # Show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    # Keep window open after training
    pygame.quit()


def main():
    """Função para modo manual (sem IA)"""
    global score, generation

    bg_x_pos = 0
    ground_x_pos = 0

    bird = Bird(50, 104)
    trees = [Tree(288, random.randint(80, 160), 2.4)]

    animation_counter = 0
    score = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bird.jump()
                    pygame.mixer.Sound.play(woosh_sfx)

        # Update bird
        bird.update()

        # Update animation
        animation_counter += 1
        if animation_counter >= 5:
            bird.update_image()
            animation_counter = 0

        # Check collisions
        bird_rect = pygame.Rect(bird.x, bird.y, 34, 24)

        for tree in trees:
            tree_width = tree_up_img.get_width()
            tree_bottom_rect = pygame.Rect(tree.x, tree.height, tree_width, window_h - tree.height)

            if bird_rect.colliderect(tree_bottom_rect):
                # Reset game
                bird = Bird(50, 104)
                trees = [Tree(288, random.randint(80, 160), 2.4)]
                score = 0
                animation_counter = 0
                pygame.mixer.Sound.play(slap_sfx)

        # Check boundaries
        if bird.y < -64 or bird.y > 208:
            bird = Bird(50, 104)
            trees = [Tree(288, random.randint(80, 160), 2.4)]
            score = 0
            animation_counter = 0
            pygame.mixer.Sound.play(slap_sfx)

        # Update trees
        for tree in trees:
            tree.update()

        # Remove off-screen trees
        if trees[0].x < -tree_up_img.get_width():
            trees.pop(0)
            trees.append(Tree(288, random.randint(80, 160), 2.4))

        # Score points
        for tree in trees:
            if not tree.scored and tree.x + tree_up_img.get_width() < bird.x:
                score += 1
                pygame.mixer.Sound.play(score_sfx)
                tree.scored = True

        # Scroll background
        bg_x_pos -= bg_scroll_spd
        ground_x_pos -= ground_scroll_spd

        if bg_x_pos <= -bg_width:
            bg_x_pos = 0
        if ground_x_pos <= -bg_width:
            ground_x_pos = 0

        # Draw everything
        screen.fill("blue")
        screen.blit(bg_img, (bg_x_pos, 0))
        screen.blit(bg_img, (bg_x_pos + bg_width, 0))
        screen.blit(ground_img, (ground_x_pos, window_h - ground_img.get_height()))
        screen.blit(ground_img, (ground_x_pos + bg_width, window_h - ground_img.get_height()))

        for tree in trees:
            tree.draw(screen)

        bird.draw(screen)
        scoreboard([bird])  # Pass bird as list for compatibility

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == '__main__':
    path = os.path.dirname(__file__)
    path_config = os.path.join(path, 'config.txt')

    # Verifica se o arquivo config existe
    if not os.path.exists(path_config):
        print("Erro: arquivo config.txt não encontrado!")
        print(f"Procurando em: {path_config}")
        sys.exit(1)

    if ai_playing:
        # Modo IA com algoritmo genético
        run(path_config)
    else:
        # Modo manual
        main()
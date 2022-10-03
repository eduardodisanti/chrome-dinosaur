##########################################################
## ORIGINAL GAME IMPLEMENTED WITH PYGAME BY:            ##
##                                                      ##
## https://github.com/codewmax/chrome-dinosaur          ##
##                                                      ##
##########################################################

import pygame
import os
import random

import numpy as np
from ANN import ANN2
#from evolution import Evolution

pygame.init()

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

RUNNING = [pygame.image.load(os.path.join("Assets/snowboarder", "snowboarder_run1.png")),
           pygame.image.load(os.path.join("Assets/snowboarder", "snowboarder_run2.png"))]

JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
JUMPING = pygame.image.load(os.path.join("Assets/snowboarder", "snowboarder_jump1.png"))

DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

DUCKING = [pygame.image.load(os.path.join("Assets/snowboarder", "snowboarder_duck1.png")),
           pygame.image.load(os.path.join("Assets/snowboarder", "snowboarder_duck2.png"))]


SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/bush", "bush_1.png")),
                pygame.image.load(os.path.join("Assets/bush", "bush_2.png")),
                pygame.image.load(os.path.join("Assets/bush", "bush_3.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/skier", "post_1.png")),
                pygame.image.load(os.path.join("Assets/skier", "post_2.png")),
                pygame.image.load(os.path.join("Assets/skier", "post_3.png"))]

LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))]

LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/plown", "plown_1.png")),
                pygame.image.load(os.path.join("Assets/plown", "plown_2.png")),
                pygame.image.load(os.path.join("Assets/plown", "plown_3.png"))]


BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/helicopter", "helicopter_1.png")),
        pygame.image.load(os.path.join("Assets/helicopter", "helicopter_2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "cloud_snowing.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))
UBG = pygame.image.load(os.path.join("Assets/Other", "track_2.png"))


class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5
    
    def set_ANN(self, NN):
        self.ANN = NN

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.alive = True
        self.points = 0

    def update(self, userInput, action=99):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        if (action==2 or userInput[pygame.K_UP]) and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif (action==1 or userInput[pygame.K_DOWN]) and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif not (self.dino_jump or userInput[pygame.K_DOWN]):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def kill(self):
        self.alive = False
        
    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel < -self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.Y_POS = 310
            self.dino_rect.y = self.Y_POS

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325
        self.the_type = 3


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 330
        self.the_type = 2


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = np.random.choice([220, 240, 200])
        self.index = 0
        self.the_type = 1

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index//5], self.rect)
        self.index += 1

def sample_action(env, dyno, random_action):
    
    if not random_action:
        model = dyno.ANN
        
        state    = env.copy()
        state[0] = dyno.dino_duck
        state[1] = dyno.dino_run
        state[2] = dyno.dino_jump
        state[5] = dyno.dino_rect.y - state[5]
        
        action = np.argmax(model.forward(state))
    else:
        action = np.random.randint(3)
        
    return action

def main(players, training=True, random_actions=False):
    
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles, generations, action_map, state_dim, alive
    
    environment = np.zeros(state_dim, dtype=np.float16)
    run = True
    clock = pygame.time.Clock()
    
    cloud = Cloud()
    game_speed = 20
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0
    font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []
    death_count = 0

    def player_score(player):
       
        if player.alive:
            player.points+=1

    def score():
        global points, game_speed, generations, alive
        
        points += 1
        if points % 100 == 0:
            game_speed += 1

        text = font.render("Points: " + str(points) + " alive: " + str(alive) + " Generation: " + str(generations), True, (0, 0, 0))

        textRect = text.get_rect()
        textRect.center = (500, 440)
        
        SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        image_height = UBG.get_height()
        ub_image_width = UBG.get_width()
        SCREEN.blit(UBG, (x_pos_bg, y_pos_bg - image_height))
        SCREEN.blit(UBG, (ub_image_width + x_pos_bg, y_pos_bg - image_height))
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            SCREEN.blit(UBG, (ub_image_width + x_pos_bg, y_pos_bg - image_height))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    while run:
        alive = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if len(obstacles) == 0:
            key = int(random.randint(0,3))
            if key == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif key == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif key == 2:
                obstacles.append(Bird(BIRD))
    
        i = 0
        environment = np.zeros(state_dim, dtype=np.float16)
        SCREEN.fill((255, 255, 255))
        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()
            i+=1
            for player in players:
                if player.alive:
                    if player.dino_rect.colliderect(obstacle.rect):
                        death_count += 1
                        player.alive = False
                        player.kill()
                        if not training:
                            pygame.time.delay(2000)
                            run = False
                else:
                    alive+=1

        if len(obstacles)>0:
            environment[3] = obstacles[0].the_type
            environment[4] = obstacles[0].rect.x
            environment[5] = obstacles[0].rect.y
            #environment[6] = np.log(obstacles[0].rect.w+0.1)
            #environment[7] = obstacles[0].rect.y

        for player in players:
            if player.alive:
                userInput = pygame.key.get_pressed()
                if training:
                    action = sample_action(environment, player, random_action=random_actions)
                    action_map[action]+=1
                else:
                    action = 99
                player.update(userInput, action)

        background()
        alive = 0
        for player in players:
            if player.alive:
                player.draw(SCREEN)
                alive+=1
        if alive==0:
            if not training:
                pygame.time.delay(2000)
            run = False
        cloud.draw(SCREEN)
        cloud.update()

        for player in players:
            player_score(player)
        score()
        
        clock.tick(game_speed)
        
        pygame.display.update()

def menu(death_count, players, training=True, random_actions=False):
        SCREEN.fill((255, 255, 255))
        font = pygame.font.Font('freesansbold.ttf', 30)

        if training:
            main(players, training=True, random_actions=random_actions)
        else:
            if death_count == 0:
                text = font.render("Press any Key to Start", True, (0, 0, 0))
            elif death_count > 0:
                text = font.render("Press any Key to Restart", True, (0, 0, 0))
                score = font.render("Your Score: " + str(points), True, (0, 0, 0))
                scoreRect = score.get_rect()
                scoreRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
                SCREEN.blit(score, scoreRect)
            textRect = text.get_rect()
            textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            SCREEN.blit(text, textRect)
            SCREEN.blit(RUNNING[0], (SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2 - 140))
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYDOWN:
                    main()

def performCrossing(father, mother,  mutation_rate=0.1):
    
    size = father.shape[0]
    new_individual = np.zeros(size)
    
    inherit_prob = 0.55
    for i in range(size):
        if np.random.random()<inherit_prob:
            new_individual[i] = father[i]
        else:
            new_individual[i] = mother[i]
            
    if np.random.random() < mutation_rate:
        a = np.random.randint(size)
        b = np.random.randint(size)
        for i in range(min(a,b), max(a,b)):
            gene = np.random.randint(size)
            new_individual[gene]*=np.random.rand() * 0.10
            
    return new_individual

def perform_Crossing(father, mother,  mutation_rate=0.1):
        """ CROSS TWO INDIVIDUALS WITH CROSS OVER STRATEGY, PERFORM MUTATION WHILE CROSSING """

        size = father.shape[0]
        midPoint = np.random.randint(0, size)
        new_individual = father.copy()

        muts = 0
        for ix in range(size):
            mute = np.random.random()
            new_individual[ix] = father[ix] if ix < midPoint else mother[ix]
            if mute < mutation_rate:
                new_individual[ix] = new_individual[ix] * np.random.randint(1, size=(size))[0]
                muts+=1
        return new_individual
        
def evolve(players, evolution_pool, num_parents=4, mutation_rate=0.1):

    rewards = []
    
    for p in players:
        rewards.append(p.points)
        
    parents = np.argsort(rewards)[-num_parents:]
    
    father = players[parents[-1]].ANN.get_params()
    mother = players[parents[-2]].ANN.get_params()
        
    print(rewards[parents[-1]], rewards[parents[-2]])
    
    for p in players:
        """
        the_father, the_mother = random.sample(list(parents), k=2)
        father = players[the_father].ANN.get_params()
        mother = players[the_mother].ANN.get_params()
        """
        new_individual = performCrossing(father, mother, mutation_rate=mutation_rate)
        p.ANN.set_params(new_individual)
    
    return players

training = True
state_dim   = 8
generations = 0
num_players = 50

np.random.default_rng()
D           = state_dim
M1          = 132
M2          = 64
K           = 3
action_max  = 2

num_parents = 4
lr = 0.1

evolution_pool = [[0, 0], [0, 0]]

players = []
for _ in range(num_players):
    d = Dinosaur()
    nn = ANN2(D, M1, M2, K, action_max)
    nn.init()
    d.set_ANN(nn)
    players.append(d)
while training:
    action_map = {0:0, 1:0, 2:0}
    generations += 1
    menu(death_count=0, players=players, random_actions=False, training=True)
    players = evolve(players, evolution_pool, num_parents=num_parents, mutation_rate=lr)
    lr*=0.991
    if lr<0.0001:
        lr = 0.0001
    alive = 0
    model = np.zeros((num_players, nn.get_params().shape[0]))
    for p in players:
        model[alive] = p.ANN.get_params()
        p.alive = True
        p.points = 0
        alive+=1
    np.save("models/gen_"+str(generations), model)
    print(action_map, lr)

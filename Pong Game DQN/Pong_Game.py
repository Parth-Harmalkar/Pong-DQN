import pygame
import sys
import random

def ball_animation():    
    global ball_speed_x, ball_speed_y
    global player_score, opponent_score
    global score_time

    # ball movement
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # walls collision
    if ball.top <= 0 or ball.bottom >= screen_height:
        ball_speed_y *= -1

    if ball.left <= 0: 
        player_score += 1
        score_time = pygame.time.get_ticks()

    if ball.right >= screen_width:
        opponent_score += 1
        score_time = pygame.time.get_ticks()

    # Player and enemy collision
    if ball.colliderect(player) and ball_speed_x > 0:
        if abs(ball.right - player.left) < 10:
            ball_speed_x *= -1
        elif abs(ball.bottom - player.top) < 10 and ball_speed_y > 0:
            ball_speed_y *= -1
        elif abs(ball.top - player.bottom) < 10 and ball_speed_y < 0:
            ball_speed_y *= -1
        
    if ball.colliderect(opponent) and ball_speed_x < 0:
        if abs(ball.left - opponent.right) < 10:
            ball_speed_x *= -1
        elif abs(ball.bottom - opponent.top) < 10 and ball_speed_y > 0:
            ball_speed_y *= -1
        elif abs(ball.top - opponent.bottom) < 10 and ball_speed_y < 0:
            ball_speed_y *= -1




def player_animation():
    player.y += player_speed
    if player.top <= 0:
        player.top = 0
    if player.bottom >= screen_height:
        player.bottom = screen_height

def opponent_ai():
    if opponent.centery < ball.y:
        opponent.y += opponent_speed
    if opponent.centery > ball.y:
        opponent.y -= opponent_speed

    if opponent.top <= 0:
        opponent.top = 0
    if opponent.bottom >= screen_height:
        opponent.bottom = screen_height

def ball_reset():
    global ball_speed_x, ball_speed_y, score_time, last_speed_increase

    ball.center = (screen_width / 2, screen_height / 2)
    curr_time = pygame.time.get_ticks()

    if curr_time - score_time < 700:
        number_three = game_font.render("3", False, light_grey)
        screen.blit(number_three, (screen_width/2 - 10, screen_height/2 + 15))
    if 700 < curr_time - score_time < 1400:
        number_three = game_font.render("2", False, light_grey)
        screen.blit(number_three, (screen_width/2 - 10, screen_height/2 + 15))
    if 1400 < curr_time - score_time < 2100:
        number_three = game_font.render("1", False, light_grey)
        screen.blit(number_three, (screen_width/2 - 10, screen_height/2 + 15))
    
    
    if curr_time - score_time < 2100:
        ball_speed_x, ball_speed_y = 0,0
    else:
        ball_speed_y = 9 * random.choice((1,-1))
        ball_speed_x = 9 * random.choice((1,-1))
        score_time = None


# General Setup
pygame.init()
clock = pygame.time.Clock()

# Setting up the main window
screen_width = 1200
screen_height = 800

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("PONG")

# Colors
bg_color = pygame.Color('grey12')
light_grey = (200,200,200)

# Game Rectangles
# (X-coord, Y-coord, width, height)
ball = pygame.Rect(screen_width/2 - 15, screen_height/2 - 15 ,30,30)
player = pygame.Rect(screen_width - 20, screen_height/2 - 70, 10, 140)
opponent = pygame.Rect(10, screen_height/2 - 70, 10, 140)

# Game Variables
ball_speed_x = 9 * random.choice((1,-1))
ball_speed_y = 9 * random.choice((1,-1))
player_speed = 0
opponent_speed = 8

# Text Variables
player_score = 0
opponent_score = 0
game_font = pygame.font.Font("freesansbold.ttf", 28)

# Score timer
score_time = True


# Game Loop
while True:
    # Handling input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # When key is pressed down
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                player_speed +=8
            if event.key == pygame.K_UP:
                player_speed -= 8
        # Key press is over
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN:
                player_speed -= 8
            if event.key == pygame.K_UP:
                player_speed += 8

    ball_animation()
    player_animation()
    opponent_ai()
    
    # Visuals
    screen.fill(bg_color)

    pygame.draw.rect(screen, light_grey, player)
    pygame.draw.rect(screen, light_grey, opponent)
    pygame.draw.ellipse(screen, light_grey, ball)    
    pygame.draw.aaline(screen, light_grey, (screen_width/2,0), (screen_width/2, screen_height))

    # Timer
    if score_time:
        ball_reset()

    # Score text
    player_text = game_font.render(f"{player_score}", False, light_grey)
    screen.blit(player_text, (615,400))

    opponent_text = game_font.render(f"{opponent_score}", False, light_grey)
    screen.blit(opponent_text, (565,400))
    
    # Updating the window
    pygame.display.flip()
    clock.tick(60)
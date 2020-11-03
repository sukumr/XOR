import pygame
import random

from neu_net import Neural_Network

WIDTH = 400
HEIGHT = 400

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("XOR")

close = False

nn = Neural_Network(2, 4, 1, 0.01) # Neural_Network(Input_Nodes, Hidden_Nodes, Output_Node, Learning_Rate)

training_data = [{"inputs":[0, 1], 'targets':[1]}, {"inputs":[0, 0], 'targets':[0]}, {"inputs":[1, 0], 'targets':[1]}, {"inputs":[1, 1], 'targets':[0]}]

while not close:
    screen.fill('white')

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            close = True

    # traning using 1000 data points
    for i in range(1000):
        data = random.choice(training_data)
        nn.train(data["inputs"], data["targets"])
    
    resolution = 10
    cols = int(WIDTH / resolution)
    rows = int(HEIGHT / resolution)

    # predicting and plotting the result
    for i in range(cols):
        for j in range(rows):
            x1 = i / cols
            x2 = j / rows
            inputs = [x1, x2]
            y = nn.predict(inputs)
            pix = y[0]
            pix= int(pix * 255)
            colour = (pix, pix, pix)
            rect = pygame.Rect(i*resolution, j*resolution, resolution, resolution)
            pygame.draw.rect(screen, colour,rect)

    pygame.display.flip()

pygame.quit()
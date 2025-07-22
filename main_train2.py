import pygame
from pygame.math import Vector2
from renderer import Renderer
from solver import *
from pendulum import *
import random
import torch
from torch import nn
import csv
import os
from network import NeuralNetwork, clone_model, mutate_model, device

#Network initialization
print(f"Using {device} device")
agent = NeuralNetwork()

print(agent)

#Pygame stuff

WIDTH = 1280
HEIGHT = 720
FPS = 30

pygame.init()
pygame.mixer.init()  ## For sound
screen = pygame.display.set_mode((WIDTH, HEIGHT))
background = pygame.image.load("./assets/img/background.png").convert()
pygame.display.set_caption("Balance!")
clock = pygame.time.Clock()

renderer = Renderer(screen)
solver = Solver(d=0.995)
solver.replay = False


running = True
paused = True
numPendulums = 300
numEpisodes = 100
numSteps = FPS*30
currentStep = 0
speed = 1

# Improved genetic algorithm parameters
mutation_rates = [0.1, 0.15, 0.2, 0.3]  # Different rates for diversity
top_selection = int(numPendulums * 0.25)  # Select top 25%
elite_count = int(numPendulums * 0.15)    # Keep top 15% unchanged

max_score = 0
scoreTimer = FPS*6

for i in range(numPendulums):
  solver.addPendulum()

screen.blit(background,(0,0))
renderer.draw(solver)
pygame.display.flip()

while running:
  while (solver.run < numEpisodes) and running:
    while(solver.currentStep < numSteps) and running:
      #1 Process input/events
      #clock.tick(FPS)     ## will make the loop run at the same speed all the time
      while(True):
        for event in pygame.event.get():        # gets all the events which have occured till now and keeps tab of them.
          ## listening for the the X button at the top
          if event.type == pygame.QUIT:
            running = False
          if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
              paused = not paused
            if event.key == pygame.K_r:
              solver.reset()
              solver.run +=1
              for i in range(numPendulums):
                solver.addPendulum()
            if event.key == pygame.K_d:
              renderer.drawclones = not renderer.drawclones
        if not paused: break
      
      # model action
      time1 = pygame.time.get_ticks()
      time3 = 0
      solver.getAllNextActions()
      time3 = pygame.time.get_ticks()
      for p in solver.pendulums:
        p.performAction()
      #2 Update
      solver.solve(1/FPS)
      solver.currentStep+=1
      scoreTimer-=1

      if max_score < solver.getTopScore():
        max_score = solver.getTopScore()
        scoreTimer = FPS*6
      if scoreTimer<=0:
        scoreTimer = FPS*6
        break
      time2= pygame.time.get_ticks()
      #print(f"Time to solve: {time2-time1} ms")
      #print(f"  Time to get action: {time3-time1}")
      #print(f"  Time to solve physics: {time2-time3}")
      #3 Draw/render
      time1 = pygame.time.get_ticks()
      screen.blit(background,(0,0))
      renderer.draw(solver)
      time2= pygame.time.get_ticks()
      #print(f"Time to render: {time2-time1} ms")

      ## Done after drawing everything to the screen
      pygame.display.flip()
    solver.run+=1
    max_score = 0
    # sort pendulums by score:
    solver.pendulums.sort(key=lambda x: x.score,reverse=True)
    
    # Improved genetic algorithm
    print(f"Run {solver.run}")
    
    # After sorting by score:
    next_generation = []
    
    # Keep elite unchanged
    for i in range(elite_count):
        next_generation.append(clone_model(solver.pendulums[i].network))
        if i <= 10: print(f"  Rank {i}: {solver.pendulums[i].score}")
    
    # Create offspring from top performers
    for i in range(elite_count, top_selection):
        parent = solver.pendulums[i]
        if parent.score > 0:
            # Vary mutation rate based on performance
            mut_rate = mutation_rates[min(i // 10, len(mutation_rates) - 1)]
            child = mutate_model(clone_model(parent.network), 
                               mutation_rate=mut_rate, 
                               mutation_strength=0.5)
            next_generation.append(child)
        else:
            next_generation.append(NeuralNetwork())
    
    # Fill remaining with crossover + mutation
    remaining = numPendulums - len(next_generation)
    for i in range(remaining):
        parent1 = random.choice(solver.pendulums[:top_selection])
        parent2 = random.choice(solver.pendulums[:top_selection])
        
        # Simple crossover: average weights
        child = clone_model(parent1.network)
        with torch.no_grad():
            for c_param, p2_param in zip(child.parameters(), parent2.network.parameters()):
                if random.random() < 0.3:  # 30% crossover rate
                    c_param.copy_((c_param + p2_param) / 2)
        
        child = mutate_model(child, mutation_rate=0.5, mutation_strength=1)
        next_generation.append(child)
    
    print(len(solver.pendulums))
    
    solver.reset()
    for i in range(len(next_generation)):
      solver.addPendulum(new_network = next_generation[i])
      
pygame.quit()
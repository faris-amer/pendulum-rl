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
numPendulums = 400
numEpisodes = 100
numSteps = FPS*30
currentStep = 0
speed = 1
top_selection = int(numPendulums*0.3)
max_score = 0
scoreTimeout = FPS *10
scoreTimer = FPS*8

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
        scoreTimer = FPS*8
      if scoreTimer<=0:
        scoreTimer = FPS*8
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
      print(f"Time to render: {time2-time1} ms")

      ## Done after drawing everything to the screen
      pygame.display.flip()
    solver.run+=1
    max_score = 0
    # sort pendulums by score:
    solver.pendulums.sort(key=lambda x: x.score,reverse=True)
    # print leaderboards,
    next_generation = []
    print(f"Run {solver.run}")
    for i in range(top_selection):
      next_pendulum = solver.pendulums.pop(0)
      if i <= 10 : print(f"  Rank {i}: {next_pendulum.score}")
      if(next_pendulum.score>0):
        next_generation.append(clone_model(next_pendulum.network))
      else:
        next_generation.append(NeuralNetwork().to(device))
    print(len(solver.pendulums))
    # create x new pendulums by randomly selecting and slightly mutating the others:
    for i in range(len(solver.pendulums)):
      next_pendulum = solver.pendulums[random.randint(0,len(solver.pendulums)-1)]
      next_generation.append(mutate_model(clone_model(next_pendulum.network)))
    solver.reset()
    for i in range(len(next_generation)):
      solver.addPendulum(new_network = next_generation[i])
pygame.quit()
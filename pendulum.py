from pygame.math import Vector2
from pygame import Color
from network import NeuralNetwork, device
from typing import Optional
from math import atan2
import math
import torch

max_pivot_speed=100

class Particle:
  def __init__(self, position: Vector2, color : Color = Color(255, 204, 102),simulate = True):
    self.position = position
    self.lastPosition = position
    self.simulate = simulate
    self.velocity = Vector2(0,0)

class DistanceConstraint:
  def __init__(self, p1: Particle, p2 : Particle, length, stiffness = 1):
    self.p1 = p1
    self.p2 = p2
    self.length = length
    self.stiffness = stiffness
  def solve(self):
    delta = self.p1.position - self.p2.position
    distance = Vector2.length(delta)
    if(distance>0):
      difference = (self.length-distance)/distance
      correction = delta*difference*0.5*self.stiffness
      if(self.p1.simulate): self.p1.position += correction
      if(self.p2.simulate): self.p2.position -= correction

class PositionConstraint:
  def __init__(self,p: Particle,tl:Vector2,br:Vector2):
    self.p = p
    self.topLeft = tl
    self.botRight = br
  def solve(self):
    if(self.p.position.x<self.topLeft.x):  self.p.position.x = self.topLeft.x 
    if(self.p.position.x>self.botRight.x): self.p.position.x = self.botRight.x 
    if(self.p.position.y<self.topLeft.y):  self.p.position.y = self.topLeft.y 
    if(self.p.position.y>self.botRight.y): self.p.position.y = self.botRight.y

class Pendulum:
  def __init__(self, pivot,weight,length, arm : DistanceConstraint, network : Optional[NeuralNetwork] = None,color : Optional[Color] = None):
    self.pivot = pivot
    self.weight = weight
    self.length = length
    self.arm = arm
    self.velocity = 0
    self.nextAction = 0
    self.main = False
    self.state = torch.tensor([[0,0,0,0]], dtype=torch.float32, device=device)
    self.score = 0
    self.scoreTimer = 0
    self.scoreEffect = 0

    if color == None:
      color = Color(241, 157, 250,150)
      self.color = color
    if network == None:
      network = NeuralNetwork().to(device=device)
    self.network = network

  def getState(self, screen_center_x=640, screen_half_width=500, dt=(1/30)):
    # Vector from pivot to weight
    delta = self.weight.position - self.pivot.position
    last_delta = self.weight.lastPosition - self.pivot.lastPosition

    # Angle and angular velocity
    angle = atan2(delta.x, delta.y)
    prev_angle = atan2(last_delta.x, last_delta.y)
    angular_velocity = (angle - prev_angle) / dt

    # Normalize angle velocity to approx [-1, 1]
    angular_velocity /= 4.0

    # Normalize pivot position relative to screen center
    x_pos = (self.pivot.position.x - screen_center_x) / screen_half_width

    # Pivot velocity
    x_velocity = self.velocity
    x_velocity /= max_pivot_speed  # Normalize

    # Return the normalized state
    return [
        math.sin(angle),        # directional component
        math.cos(angle),        # uprightness
        angular_velocity,
        x_pos,
        x_velocity,
    ]

  def getNextAction(self):
    temp_state = self.getState()
    for i in range(len(temp_state)):
      self.state[0,i] = temp_state[i]
    self.nextAction = self.network(self.state)
  def performAction(self):
    self.velocity += self.nextAction
    if self.velocity > max_pivot_speed:
      self.velocity = max_pivot_speed
    if self.velocity < -max_pivot_speed:
      self.velocity = -max_pivot_speed
    self.pivot.position.x += self.velocity
  def setMain(self):
    self.color = Color(255, 204, 102)
    self.main = True
  def notMain(self):
    self.color.a = 150
    self.main = False
  def calcScore(self, dt):
    if self.pivot.position.x < 160 or self.pivot.position.x > 1120: 
      self.scoreTimer = 0
      return
    delta = self.pivot.position - self.weight.position
    angle = atan2(delta.x, delta.y)*180/3.1415
    if abs(angle)<15:
      self.scoreTimer = (self.scoreTimer+1)
    else:
      self.scoreTimer = 0
    if self.scoreTimer > 0:
      self.score += self.scoreTimer
      self.scoreEffect = 100
    self.scoreEffect -= 10
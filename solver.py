from pygame.math import Vector2
import pygame
from network import NeuralNetwork
from pendulum import *
from typing import Optional

class Solver:
  
  def __init__(self, g=Vector2(0,3000),d=0.98, replay = True):
    self.pendulums = []
    self.particles = []
    self.constraints = []
    self.gravity = g
    self.damping = d
    self.replay = replay
    self.replaying = False
    self.run = 0
    self.currentStep = 0
    self.mainPendulum = None

  def addParticle(self, p:Particle):
    self.particles.append(p)

  def addPendulum(self, pos: Optional[Vector2] = None, length = 200, new_network = None):
    if pos == None:
      pos = Vector2(640,330)
    pivot = Particle(pos,simulate=False)
    weight = Particle(pos+Vector2(0,length))
    self.particles.append(pivot)
    self.particles.append(weight)
    arm = DistanceConstraint(pivot,weight,length)
    self.constraints.append(PositionConstraint(pivot,Vector2(150,330),Vector2(1130,330)))
    self.constraints.append(arm)

    pendulum = Pendulum(pivot,weight,length, arm, network=new_network)
    if len(self.pendulums) == 0:
      self.setMain(pendulum)
    self.pendulums.append(pendulum)
    return pendulum
  
  def solve(self,dt):
    # simulate gravity
    for p in self.particles:
      if(p.simulate):
        temp = p.position.copy()
        p.velocity = p.position - p.lastPosition
        p.position = p.position + p.velocity*self.damping + self.gravity*dt*dt
        p.lastPosition = temp
    for c in self.constraints:
      for i in range(4):
        c.solve()
    for p in self.pendulums:
      p.calcScore(dt)
  
  def reset(self):
    self.pendulums.clear()
    self.particles.clear()
    self.constraints.clear()
    self.currentStep = 0
  
  def setMain(self,pendulum):
    if self.mainPendulum:
      self.mainPendulum.notMain()
    self.mainPendulum = pendulum
    pendulum.setMain()
  def getTopScore(self):
    max = 0
    for p in self.pendulums:
      if p.score>max: max = p.score
    return max
  def getAllNextActions0(self):
    time1 = pygame.time.get_ticks()
    states = torch.stack([torch.tensor(p.getState(), dtype=torch.float32) for p in self.pendulums])
    outputs = torch.stack([p.network(states[i:i+1]) for i, p in enumerate(self.pendulums)])
    for i, p in enumerate(self.pendulums):
      p.nextAction = outputs[i].item()
    #print(f"time to solve:{pygame.time.get_ticks()-time1}")
  def getAllNextActions(self):
    if not self.pendulums:
        return
    
    # Collect all states at once
    states = torch.tensor([p.getState() for p in self.pendulums], 
                         dtype=torch.float32, device=device)
    
    # Process in batches to avoid memory issues
    batch_size = 500
    with torch.no_grad():
        for i in range(0, len(self.pendulums), batch_size):
            end_idx = min(i + batch_size, len(self.pendulums))
            batch_states = states[i:end_idx]
            
            # Get actions for this batch
            for j, pendulum in enumerate(self.pendulums[i:end_idx]):
                action = pendulum.network(batch_states[j:j+1])
                pendulum.nextAction = action.item()
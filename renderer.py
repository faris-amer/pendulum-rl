import pygame
from pygame import Color
from solver import DistanceConstraint

def blend_colors(color1, color2, t):
    t = max(0, min(1, t))  # Clamp between 0 and 1
    r = int(color1.r + t * (color2.r - color1.r))
    g = int(color1.g + t * (color2.g - color1.g))
    b = int(color1.b + t * (color2.b - color1.b))
    a = color1.a
    return pygame.Color(r, g, b, a)

class Renderer:
  def __init__(self, surface):
    self.surface = surface
    self.font = pygame.font.Font("assets/CascadiaMono.ttf",24)
    self.drawclones = True

  def draw(self, solver):
    maxscore = 0
    radius = 20
    temp_surface = pygame.Surface(self.surface.get_size(), pygame.SRCALPHA)
    main_pendulum_surface = pygame.Surface(self.surface.get_size(), pygame.SRCALPHA)
    for p in solver.pendulums:
      if p.score > maxscore: maxscore = p.score
      
      if isinstance(p.arm,DistanceConstraint):
        pygame.draw.line(temp_surface,Color(255,255,255,p.color.a),p.arm.p1.position,p.arm.p2.position,5)
      
      color = Color(p.color)
      if p.scoreEffect > 0:
        color = blend_colors(color,Color(0,255,0,0),p.scoreEffect/100)
      if p.main == False:
        pygame.draw.circle(temp_surface,color,p.pivot.position,radius)
        pygame.draw.circle(temp_surface,color,p.weight.position,radius)
        pygame.draw.circle(temp_surface,Color(255,255,255,p.color.a),p.pivot.position,20,5)
        pygame.draw.circle(temp_surface,Color(255,255,255,p.color.a),p.weight.position,20,5)
      else:
        pygame.draw.line(main_pendulum_surface,Color(255,255,255,p.color.a),p.arm.p1.position,p.arm.p2.position,5)
        pygame.draw.circle(main_pendulum_surface,color,p.pivot.position,radius)
        pygame.draw.circle(main_pendulum_surface,color,p.weight.position,radius)
        pygame.draw.circle(main_pendulum_surface,Color(255,255,255,p.color.a),p.pivot.position,20,5)
        pygame.draw.circle(main_pendulum_surface,Color(255,255,255,p.color.a),p.weight.position,20,5)
    
    if self.drawclones == True: self.surface.blit(temp_surface,(0,0))
    temp_surface.fill((0,0,0,0))
    self.surface.blit(main_pendulum_surface,(0,0))
    
    text = self.font.render(f"Gravity: {solver.gravity.y}",True,(255,255,255))
    self.surface.blit(text,(75,610))
    text = self.font.render(f"Friction: {solver.damping}",True,(255,255,255))
    self.surface.blit(text,(75,635))

    if not (solver.replay):
      text = self.font.render("Saving...",True,(255,255,255))
      self.surface.blit(text,(75,25))
    elif(solver.replaying):
      text = self.font.render("REPLAYING...",True,(255,255,255))
      self.surface.blit(text,(75,25))
    else:
      text = self.font.render("Replay complete",True,(255,255,255))
      self.surface.blit(text,(75,25))
    text = self.font.render(f"Time: {round(solver.currentStep /30,1)}",True,(255,255,255))
    self.surface.blit(text,(275,25))
    text = self.font.render(f"Run: {solver.run}",True,(255,255,255))
    self.surface.blit(text,(475,25))
    text = self.font.render(f"Max Score: {maxscore}",True,(255,255,255))
    self.surface.blit(text,(675,25))
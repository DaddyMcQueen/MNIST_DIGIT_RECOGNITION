import pygame
import numpy as np
from nn import DeepNN
import math
from deepnn.constants import WIN_WIDTH
from deepnn.constants import WIN_HEIGHT
from deepnn.constants import WHITE
from deepnn.constants import BLACK
from deepnn.constants import GRAY

pygame.font.init()
FONT = pygame.font.SysFont("comicsans", 50)

class Drawing(object):

	def __init__(self):
		self.reset()

	def reset(self):
		self.grid = np.zeros((28,28))

	def update_grid(self, row, col):
		if row < 28 and col < 28:
			self.grid[row][col] = 1
			if row > 0 and col > 0:

				if self.grid[row - 1][col] >= 1:
					pass
				else: 
					self.grid[row - 1][col] += 0.1

				if self.grid[row][col - 1] >= 1:
					pass
				else: 
					self.grid[row][col - 1] += 0.1

				if self.grid[row - 1][col - 1] == 1:
					pass
				else: 
					self.grid[row - 1][col - 1] += 0.1

			if row < 27 and col < 27:

				if self.grid[row + 1][col] >= 1:
					pass
				else: 
					self.grid[row + 1][col] += 0.1

				if self.grid[row][col + 1] >= 1:
					pass
				else: 
					self.grid[row][col + 1] += 0.1

				if self.grid[row + 1][col + 1] == 1:
					pass
				else: 
					self.grid[row + 1][col + 1] += 0.1
		else:
			return True

	def draw_grid(self, win):
		pygame.draw.rect(win, BLACK, [0, 560, 560, 140])
		pygame.draw.rect(win, WHITE, [10, 570, 540, 120])
		text = FONT.render("PREDICT NUMBER ", 1, BLACK)
		win.blit(text, (122, 610))
		for row in range(28):
			for col in range(28):
				if self.grid[row][col] >= 1:
					self.grid[row][col] = 1
				if self.grid[row][col] == 1:
					pygame.draw.rect(win, BLACK, [col * 20, row * 20, 20, 20])
				if self.grid[row][col] == 0.5:
					pygame.draw.rect(win, GRAY, [col * 20, row * 20, 20, 20])
		
def click_to_grid(mx, my):
	row = math.floor(my / 20)
	col = math.floor(mx / 20)
	return row, col


def get_drawing():
	run = True
	testing = True
	win = pygame.display.set_mode((560, 700))
	pygame.draw.rect(win, WHITE, [ 0, 0, WIN_WIDTH, WIN_HEIGHT])
	img = Drawing()
	pygame.key.set_repeat(True)
	while run:
		
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				testing = False
				pygame.quit()
		
			if event.type == pygame.MOUSEBUTTONDOWN:
				clicked = True

				while clicked:	
					for event in pygame.event.get():
						if event.type == pygame.MOUSEBUTTONUP:
							clicked = False
					mx, my = pygame.mouse.get_pos()
					row, col = click_to_grid(mx, my)
					done = img.update_grid(row, col)

					if done:
						run = False
					else:
						img.draw_grid(win)
						pygame.display.update()
		out = img.grid
		img.reset
		img.draw_grid(win)
		pygame.display.update()

	return out, testing

						
# def main():
# 	output = get_drawing()

# main()
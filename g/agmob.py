import random
from random import randint
from random import randrange, choice
from copy import deepcopy

w = [1,1]

def fx(v):
	return int(v,2);
def gx(v):
	return int(v,2);
	
##individuo
class Individuo:
	def __init__(self, cro):
		self.cro = cro
		self.fitness = 0
		self.val_fx = 0
		self.val_gx = 0
	
	def calcular_fitness(self):
		self.val_fx = fx(self.cro)
		self.val_gx = gx(self.cro)
		self.fitness = w[0]*self.val_fx + w[1]*self.val_gx

	def __repr__(self):
		return "".join(["Individuo(",str(self.cro), ",", str(self.fitness),")"])



def get_poblacion_inicial(tam_poblacion):
	P = []
	cont = 0
	for i in range(tam_poblacion):
		I = Individuo([])
		I.cro = ""
		for j in range(9):
			I.cro += choice(["0", "1"])
		I.calcular_fitness()
		P.append(I)
	return P;

def imprimir_poblacion(P):
	i=0
	for I in P:
		print i,") ", "CRO: ",I.cro, " ",I.val_fx," ",I.val_gx,"	","	|",I.fitness
		i+=1


def test():
	poblacion = get_poblacion_inicial(10)
	imprimir_poblacion(poblacion)

test()
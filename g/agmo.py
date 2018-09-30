import random
from random import randint
from copy import deepcopy

##individuo
class Individuo:
	def __init__(self, cro, fitness):
		self.cro = cro
		self.fitness = fitness

	def __repr__(self):
		return "".join(["Individuo(",str(self.cro), ",", str(self.fitness),")"])

#poblacion -> vector<individuo>
class Poblacion:
	def __init__(self,P):
		self.P = P
	
	def insert(self,individuo):
		self.P.append(individuo)

	def __repr__(self):
		return str(self.P)
			
#pila
class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def top(self):
         return self.items[len(self.items)-1]

     def size(self):
         return len(self.items)

def getRandom(li,ls): #double
  	return random.uniform(li,ls)

def fx(I):
	return 4.0*pow(I.cro[0],2.0) + 4.0*pow(I.cro[1],2.0)

def gx(I):
	return pow((I.cro[0]-5),2.0) + pow((I.cro[1]-5),2.0)
'''
cro = [1,1]
ind = Individuo(cro,8)
print ind
ind.cro[0]=2
print ind
print fx(ind)
print gx(ind)
'''

w = [1,1]
#print w

# 0<= x <=5
# 0<= y <=3
mix_x = 0
max_x = 5
min_y = 0
max_y = 3
PROB_MUT = 5 # 0.5-1% [0-100]Probabilidad de Mutacion: 0.05

def get_fitness(I):
	return w[0]*fx(I) + w[1]*gx(I)


def es_valido(I):
	if I.cro[0] < mix_x:
		I.cro[0] = mix_x
	if I.cro[1] < min_y:
		I.cro[1] = min_y

	if I.cro[0] > max_x:
		I.cro[0] = max_x	
	if I.cro[1] > max_y:
		I.cro[1] = max_y
	I.fitness = w[0]*fx(I) + w[1]*gx(I)
	
	x=I.cro[0]
	y=I.cro[1]
	ff1 = pow((x-5),2.0) + pow(y,2.0) - 25
	ff2 = -1*pow((x-8),2.0) - pow((y+3),2.0) +7.7
	if 0.0>=ff1 and 0.0>=ff2:
		return True
	return False


def get_poblacion_inicial(tam_poblacion):
	P = Poblacion([])
	cont = 0
	while True:
		I = Individuo([],0)
		x = getRandom(mix_x, max_x);
		y = getRandom(min_y, max_y);
		I.cro = [x,y]
		I.fitness = w[0]*fx(I) + w[1]*gx(I);
		
		if es_valido(I):
			P.insert(I)
			cont += 1
		if cont >= tam_poblacion:
			break
	return P;

def imprimir_poblacion(P):
	i=0
	for I in P.P:
		print i,") ", "CRO[",I.cro[0],",",I.cro[1],"] ",fx(I)," ",gx(I),"	","	|",I.fitness
		i+=1

'''
P = get_poblacion_inicial(4)
imprimir_poblacion(P)
'''

def ruleta(P):
	total=0;
	for I in P.P:
		total += I.fitness
		
	cont = 0.0;
	v_pro = [];#ruleta, vector de probabilidades

	for I in P.P:
		cont += (I.fitness*100.0)/total;
		v_pro.append(cont); 
		
	#seleccion
	seleccionados = Poblacion([]);
	for i in range(len(P.P)):
		s = getRandom(0,100)
		for j in range(len(v_pro)):
			if s<=v_pro[j]:
				seleccionados.insert(P.P[j])
				break
	return seleccionados;


def torneo(P):
	seleccionados = Poblacion([])
	tam_torneo = 3;
	for I in P.P:
		P_torneo = Poblacion([])
		for i in range(tam_torneo):
			P_torneo.insert(P.P[randint(0,tam_torneo-1)])
		P_torneo.P.sort(key=takeFitness)
		seleccionados.insert(P_torneo.P[tam_torneo-1])
	return seleccionados


def takeFitness(elem):
    return elem.fitness


def seleccion(P):
	return ruleta(P);
	#return torneo(P);


def cruzamiento_blx(P):
	hijos = Poblacion([])
	tam_pob =len(P.P)
	cont = 0
	while True:
		h = randint(0,tam_pob-1)
		k = randint(0,tam_pob-1)
		B = getRandom(-5.0,1.5) 
		#cruzar
		I = Individuo([0,0],0) #--
		#p1 + B (P2 - P1)
		I.cro[0] = P.P[h].cro[0] + B*( P.P[k].cro[0]-P.P[h].cro[0] );
		I.cro[1] = P.P[h].cro[1] + B*( P.P[k].cro[1]-P.P[h].cro[1] );
		I.fitness = w[0]*fx(I) + w[1]*gx(I);
		if es_valido(I):
			hijos.insert(I)
			cont += 1
		if cont >= tam_pob:####
			break
	return hijos;



def mutar(P):
	tam_pob =len(P.P)
	for i in range(tam_pob):
		pro_mut = randint(0,100)
		if pro_mut <= PROB_MUT:
			print "Mutacion I: ", i
			I = Individuo([],0)
			while True:
				I =deepcopy(P.P[i])
				k = randint(0,1)
				if k==0: #en x
					I.cro[0] = getRandom(mix_x,max_x)
				if k==1:
					I.cro[1] = getRandom(min_y,max_y)
				if es_valido(I):
					break
			P.P[i].cro = I.cro
			P.P[i].fitness = get_fitness(P.P[i])



#retorna la primera frontera y el resto
def get_frontier(P):
	p_frontera = Poblacion([]) #fontera
	P_resto = Poblacion([]) #no estan en la frontera
	#sort(P.begin(), P.end(), decresiente);#ordenar soluciones en x o fx
	P.P.sort(key=takeFitness, reverse=True)

	front = Stack();
	front.push(P.P[0]);
	for i in range(1,len(P.P)):
		print ":::::::::::: ",i
		while not front.isEmpty():
			top_x = fx( front.top() )
			top_y = gx( front.top() )
			x = fx( P.P[i]);
			y = gx( P.P[i]);
			if top_x >= x  and  top_y >= y:
				print top_x,">=",x,"	",top_y,">=",y, front.size(),i;
				print "enter"	
				P_resto.insert(front.top())   	
				front.pop()
			else:
				break	
		front.push(P.P[i])
	
	while not front.isEmpty():
		p_frontera.insert(front.top())
		#cout<<fx(front.top(),front.top())<<" "<<gx(front.top(),front.top())<<endl;
		front.pop()
	
	fronteras =[] 
	#vector<poblacion> fronteras;
	fronteras.append( p_frontera );
	fronteras.append( P_resto );
	return fronteras;

def get_fronteras(P):
	contenedor=deepcopy(P);
	fronteras = [];#vector de fronteras
	while contenedor.size()>0:
		#r = [pobl,pobl,pobl] #ronteras
		r = get_frontier(contenedor);
		fronteras.append(r[0]);
		contenedor = r[1];#trabajar con e resto
	return fronteras;








def main():
	P = get_poblacion_inicial(10)
	imprimir_poblacion(P)
	
	print "======================="
	P_nueva = seleccion(P)
	imprimir_poblacion(P_nueva) 
	print "=============cruzar=========="
	hijos = cruzamiento_blx(P)
	imprimir_poblacion(hijos)
	print "=============mutar=========="
	mutar(hijos)
	imprimir_poblacion(hijos)
	print "=============fronteras=========="
	R = get_frontier(hijos);
	print "=============frontera 1=========="
	imprimir_poblacion(R[0])
	print "=============frontera 2=========="
	imprimir_poblacion(R[1])
	print "======================="
	

main()
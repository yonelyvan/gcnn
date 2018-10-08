import random
from random import randint
from copy import deepcopy

w = [1,1]

def fx(v):
	return 4.0*pow(v[0],2.0) + 4.0*pow(v[1],2.0)

def gx(v):
	return pow((v[0]-5),2.0) + pow((v[1]-5),2.0)
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

#poblacion -> vector<individuo>
class Poblacion:
	def __init__(self,P):
		self.P = P
	
	def insert(self,individuo):
		self.P.append(individuo)

	def size(self):
		return len(self.P)

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

# 0<= x <=5
# 0<= y <=3
mix_x = 0
max_x = 5
min_y = 0
max_y = 3
PROB_MUT = 5 # 0.5-1% [0-100]Probabilidad de Mutacion: 0.05


def es_valido(I):
	if I.cro[0] < mix_x:
		I.cro[0] = mix_x
	if I.cro[1] < min_y:
		I.cro[1] = min_y

	if I.cro[0] > max_x:
		I.cro[0] = max_x	
	if I.cro[1] > max_y:
		I.cro[1] = max_y
	I.calcular_fitness()
	
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
		I = Individuo([])
		x = getRandom(mix_x, max_x);
		y = getRandom(min_y, max_y);
		I.cro = [x,y]
		I.calcular_fitness()
		
		if es_valido(I):
			P.insert(I)
			cont += 1
		if cont >= tam_poblacion:
			break
	return P;

def imprimir_poblacion(P):
	i=0
	for I in P.P:
		print i,") ", "CRO[",I.cro[0],",",I.cro[1],"] ",I.val_fx," ",I.val_gx,"	","	|",I.fitness
		i+=1


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

def takeVal_fx(elem):
    return elem.val_fx


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
		I = Individuo([0,0]) #--
		#p1 + B (P2 - P1)
		I.cro[0] = P.P[h].cro[0] + B*( P.P[k].cro[0]-P.P[h].cro[0] );
		I.cro[1] = P.P[h].cro[1] + B*( P.P[k].cro[1]-P.P[h].cro[1] );
		I.calcular_fitness()
		if es_valido(I):
			hijos.insert(I)
			cont += 1
		if cont >= tam_pob:####
			break
	return hijos;



def mutar(P): # --
	tam_pob =len(P.P)
	for i in range(tam_pob):
		pro_mut = randint(0,100)
		if pro_mut <= PROB_MUT:
			print "Mutacion I: ", i
			I = Individuo([])
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
			P.P[i].calcular_fitness()
	return P


#retorna la primera frontera y el resto
def get_frontier(P):
	p_frontera = Poblacion([]) #fontera
	P_resto = Poblacion([]) #no estan en la frontera
	
	P.P.sort(key=takeVal_fx, reverse=True)

	front = Stack();
	front.push(P.P[0]);
	for i in range(1,len(P.P)):
		while not front.isEmpty():
			top_x = front.top().val_fx
			top_y = front.top().val_gx
			x = P.P[i].val_fx
			y = P.P[i].val_gx
			if top_x >= x  and  top_y >= y:
				#print top_x,">=",x,"	",top_y,">=",y, front.size(),i;
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
	contenedor = deepcopy(P);
	fronteras = [];#vector de fronteras
	while contenedor.size()>0:
		#r = [pobl,pobl,pobl] #ronteras
		r = get_frontier(contenedor);
		fronteras.append(r[0]);
		contenedor = r[1];#trabajar con e resto
	return fronteras;

def distancia(I1, I2):
	dx = abs(I1.val_fx - I2.val_fx);#x
	dy = abs(I1.val_gx - I1.val_gx);#y
	return 2*dx + 2*dy;

def run():
	iteraciones = 50
	tam_poblacion = 20
	D = 4.0
	P = get_poblacion_inicial(tam_poblacion)
	for i in range(iteraciones):
		print ">>>>>>> Iteracion: ",i
		print "Poblacion"
		imprimir_poblacion(P)

		print "Seleccion"
		seleccionados = seleccion(P)

		print "Hijos"
		hijos = cruzamiento_blx(seleccionados)
		imprimir_poblacion(hijos)

		for e in hijos.P:
			P.insert(e)
		P=mutar(P)

		print "Nueva Poblacion"
		new_P = Poblacion([])
		fronteras = get_fronteras(P)

		#nueva poblacion
		cont=0
		fl=1
		index_f=0
		while fl:
			for F in fronteras:#F es una poblacion
				index_f +=1
				for i in range(F.size()):
					if i==0 or i == F.size()-1:
						new_P.insert(F.P[i]);
						cont +=1
					else:
						if distancia(F.P[i-1],F.P[i+1]) >= D:
							new_P.insert(F.P[i])
							cont+=1
					if cont == tam_poblacion:
						fl=0
						break
				if cont == tam_poblacion:
						fl=0
						break
		P = new_P
		imprimir_poblacion(P)





run()




def test():
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
	'''
	FF = get_fronteras(hijos)
	for p in FF:
		print "frontera"
		imprimir_poblacion(p)
	'''	

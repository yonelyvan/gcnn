import random
from utils import *
from random import *
from copy import deepcopy

def getRandom(li,ls): #double
  	return random.uniform(li,ls)

def get_poblacion_inicial(tam_poblacion):
	P = []
	cont = 0
	for i in range(tam_poblacion):
		I = Individuo([])
		I.cro = ""
		for j in range(TAM_CROMOSOMA):
			I.cro += choice(["0", "1"])
		#I.calcular_fitness() #******
		P.append(I)
	return P;

def calcular_fitness(P):
	R =[]
	for I in P:
		I.calcular_fitness()
		R.append(I)
	return R;


def imprimir_poblacion(P):
	i=0
	for I in P:
		print i,") ", "CRO: ",I.cro, " ",I.val_fx," ",I.val_gx,"	","	|",I.fitness
		i+=1

def ruleta(P):
	total = 0
	for I in P:
		total += I.fitness
		
	cont = 0.0;
	v_pro = [];#ruleta, vector de probabilidades

	for I in P:
		if(I.fitness <=0.0):
			I.fitness = 0.000001 #evitando divicion por 0
		cont += (I.fitness*100.0)/total; #total = 0 error
		v_pro.append(cont); 
		
	#seleccion
	seleccionados = []
	for i in range(len(P)):
		s = getRandom(0,100)
		for j in range(len(v_pro)):
			if s<=v_pro[j]:
				seleccionados.append(P[j])
				break
	return seleccionados

def takeFitness(elem):
    return elem.fitness

def torneo(P):
	seleccionados = []
	tam_torneo = 3;
	for I in P:
		P_torneo = []
		for i in range(tam_torneo):
			P_torneo.append(P[randint(0,TAM_POBLACION-1)])
		P_torneo.sort(key=takeFitness)
		#seleccionados.append(P_torneo[tam_torneo-1]) ## fitnes mas alto
		seleccionados.append(P_torneo[0]) ## mas bajo es mejor fitnes
	return seleccionados

def seleccion(P):
	#return ruleta(P);
	return torneo(P);


def cruzamiento(P):
	hijos = []
	tam_pob =len(P)
	tam_cro = len(P[0].cro)
	mascara = ""
	for i in range(tam_cro):
		mascara += choice(["0", "1"]);

	cont = 0
	while cont<TAM_POBLACION:
		#elegir dos padres
		id_p1 = randint(0,tam_pob-1)
		id_p2 = randint(0,tam_pob-1)
		#verificar la probabilidad de cruze
		if randint(0,100) < PROB_CRUZAMIENTO:
			#print "_______________________"
			#print "Mascara: ",mascara
			#print "Padre-1: ",P[id_p1].cro
			#print "Padre-2: ",P[id_p2].cro

			#cruzar
			hijo1 = Individuo([])
			hijo2 = Individuo([])
			hijo1.cro = ""
			hijo2.cro = ""
			for i in range(tam_cro):
				if mascara[i]=="1":
					hijo1.cro += P[id_p1].cro[i]
					hijo2.cro += P[id_p2].cro[i]
				else:
					hijo1.cro += P[id_p2].cro[i]
					hijo2.cro += P[id_p1].cro[i]

			#hijo1.calcular_fitness()#******
			#hijo2.calcular_fitness()#******
			hijos.append(hijo1)
			hijos.append(hijo2)
			#print "Hijo-1:  ",hijo1.cro
			#print "Hijo-2:  ",hijo2.cro
			cont+=2
	return hijos


def mutar(P):
	for i in range(TAM_POBLACION):
		pro_mut = randint(0,100)
		if pro_mut <= PROB_MUTACION:
			id_bit = randint(0,TAM_CROMOSOMA-1);
			print "Mutacion I: ", i,") ",P[i].cro, " en bit: ", id_bit
			cro = list(P[i].cro)
			if  P[i].cro[id_bit] =='1':
				cro[id_bit] = '0'
			else:
				cro[id_bit] = '1'
			P[i].cro = "".join(cro)
			print "Mutacion I: ", i,") ",P[i].cro
			#P[i].calcular_fitness();#****** 
	return P


def takeVal_fx(elem):
    return elem.val_fx

#retorna la primera frontera y el resto
def get_frontier(P):
	p_frontera = [] #fontera
	P_resto = [] #no estan en la frontera
	
	P.sort(key=takeVal_fx, reverse=True)

	front = Stack();
	front.push(P[0]);
	for i in range(1,len(P)):
		while not front.isEmpty():
			top_x = front.top().val_fx
			top_y = front.top().val_gx
			x = P[i].val_fx
			y = P[i].val_gx
			if top_x >= x  and  top_y >= y:
				#print top_x,">=",x,"	",top_y,">=",y, front.size(),i;
				P_resto.append(front.top())   	
				front.pop()
			else:
				break	
		front.push(P[i])
	
	while not front.isEmpty():
		p_frontera.append(front.top())
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
	while len(contenedor)>0:
		#r = [pobl,pobl,pobl] #ronteras
		r = get_frontier(contenedor);
		fronteras.append(r[0]);
		contenedor = r[1];#trabajar con e resto
	return fronteras;


def distancia(I1, I2):#mejorar distancia
	dx = abs(I2.val_fx - I1.val_fx);#x
	dy = abs(I2.val_gx - I1.val_gx);#y
	return 2.0*dx + 2.0*dy;

'''
def run():
	iteraciones = 3
	tam_poblacion = TAM_POBLACION
	D = 1
	P = get_poblacion_inicial(tam_poblacion)
	for i in range(iteraciones):
		print ">>>>>>> Iteracion: ",i
		print "Poblacion"
		imprimir_poblacion(P)

		print "Seleccion"
		seleccionados = seleccion(P)

		print "Hijos:"
		hijos = cruzamiento(seleccionados)
		imprimir_poblacion(hijos)

		for e in hijos:
			P.append(e)
		P=mutar(P)

		print "Nueva Poblacion"
		new_P = []
		fronteras = get_fronteras(P)

		#nueva poblacion
		cont=0
		fl=1
		index_f=0
		while fl:
			for F in fronteras:#F es una poblacion
				index_f +=1
				for i in range(len(F)):
					if i==0 or i == len(F)-1:
						new_P.append(F[i]);
						cont +=1
					else:
						if distancia(F[i-1],F[i+1]) >= D:
							new_P.append(F[i])
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
'''

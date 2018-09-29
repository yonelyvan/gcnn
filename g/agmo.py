import random

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
			
'''
P = Poblacion([])
cro = [1,2,3]
print cro

ind = Individuo(cro,88)
P.insert(ind)
P.insert(ind)
print ind
print P
'''

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
PROB_MUT = 1 # 0.5-1% [0-100]Probabilidad de Mutacion: 0.05

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
	for i in range(tam_poblacion):
		I = Individuo([],0)
		x = getRandom(mix_x, max_x);
		y = getRandom(min_y, max_y);
		I.cro = [x,y]
		I.fitness = w[0]*fx(I) + w[1]*gx(I);
		
		if es_valido(I):
			P.insert(I)
		else:
			i-=1
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

def ruleta(P){
	total=0;
	for (int i = 0; i < P.size(); ++i){
		total+=P[i].fitness;
	}
	double cont=0;
	vd v_pro;//ruleta
	for (int i = 0; i < P.size(); ++i){
		cont += (P[i].fitness*100.0)/total;
		v_pro.push_back(cont); 
	}
	//seleccion
	poblacion seleccionados;

	for (int i = 0; i < P.size(); ++i){//P.size()
		int s= rand()%100;
		for (int j = 0; j < v_pro.size(); ++j){//verificando a q rango pertenece
			if( s <= v_pro[j] ){
				seleccionados.push_back(P[j]);
				break;
			}
		}
	}
	return seleccionados;
}





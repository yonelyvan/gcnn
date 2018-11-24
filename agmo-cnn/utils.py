
w = [1,1] #pesos para cada funcion objetivo [fx,gx]
INF = 1e9

TAM_POBLACION = 30 #%
TAM_CROMOSOMA = 9
PROB_CRUZAMIENTO = 90 #%
PROB_MUTACION = 2 #%

#numero bits por valor
BITS_VALUE = 3

def fx(cro):
	v_result = get_crom_values(cro)
	return 4.0*pow(v_result[0],2.0) + 4.0*pow(v_result[1],2.0)
	#paso a la arquitectura

def gx(cro):
	v_result = get_crom_values(cro)
	return pow((v_result[0]-5),2.0) + pow((v_result[1]-5),2.0)

def get_crom_values(crom):
	v_resul = []
	for i in range(0, TAM_CROMOSOMA, BITS_VALUE):
		str_val = crom[i:i+BITS_VALUE]
		v_resul.append( int(str_val,2) )
	return v_resul


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
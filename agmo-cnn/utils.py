from cnn import *
w = [2,1] #pesos para cada funcion objetivo [fx,gx] [error, tiempo]
INF = 1e9

TAM_POBLACION = 10 #%
#numero bits por valor
BITS_VALUE = 3
TAM_CROMOSOMA = 6*BITS_VALUE
PROB_CRUZAMIENTO = 90 #%
PROB_MUTACION = 5 #%

'''
def fx(cro):#cnn_error
	config = get_crom_values(cro)
	return 4.0*pow(config[0],2.0) + 4.0*pow(config[1],2.0)
	
	#paso a la arquitectura

def gx(cro):#cnn_time
	config = get_crom_values(cro)
	return pow((config[0]-5),2.0) + pow((config[1]-5),2.0)
'''

def get_crom_values(crom):
	v_resul = []
	for i in range(0, TAM_CROMOSOMA, BITS_VALUE):
		str_val = crom[i:i+BITS_VALUE]
		dim = int(str_val,2)+1 ##convolucion valida si dimension >=1
		v_resul.append( dim ) 
	return v_resul


##individuo
class Individuo:
	def __init__(self, cro):
		self.cro = cro
		self.fitness = 0
		self.val_fx = 0 #error
		self.val_gx = 0 #tiempo
	
	def calcular_fitness(self):
		print "Calculando fitnes de individuo:"
		config = get_crom_values(self.cro)
		print "CONFIG:",config
		result = run_cnn( config )
		self.val_fx = result[0] #fx(self.cro)
		self.val_gx = result[1] #gx(self.cro)
		self.fitness = w[0]*self.val_fx + w[1]*self.val_gx

		#self.val_fx = fx(self.cro)
		#self.val_gx = gx(self.cro)
		#self.fitness = w[0]*self.val_fx + w[1]*self.val_gx

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



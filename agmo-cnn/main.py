from agmo import *

def load_poblacion():
	P = [] #poblacion vacia

	infile = open(ULTIMA_POBLACION, "r")
	lines = infile.read().strip().split('\n')
	
	first_line = lines[0].split(' ')
	iteracion = int(first_line[0])
	
	lines = lines[1:] #leer lineas desde la segunda linea
	for line in lines:
		I = line.split(' ')
		ind = Individuo([])
		ind.cro = I[0] 
		ind.val_fx = float(I[1])
		ind.val_gx = float(I[2])
		ind.fitness = float(I[3]) 
		P.append(ind)
	infile.close()
	return [P,iteracion]

#corre una iteracion
def run_iteracion(P, it, data):

	#load data
	#[x,y,classes] = get_data() #train
	#[tx, ty] = cargar_imagenes_prueba()#test

	D = 0.5
	print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Iteracion: ",it
	save_log(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Iteracion: "+ str(it) )		
	seleccionados = seleccion(P)
	ver_poblacion(seleccionados, ">>> Seleccionados")
	#curzamiento y mutacion
	hijos = cruzamiento(seleccionados)
	hijos = mutar(hijos)
	#calculo de fittnes
	hijos = calcular_fitness(hijos,data)
	ver_poblacion(hijos, ">>> Hijos")
	for I in hijos:
		P.append(I)

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
				if cont == TAM_POBLACION:
					fl=0
					break
			if cont == TAM_POBLACION:
					fl=0
					break
	P = new_P
	it=it+1
	imprimir_poblacion(P,">>>Nueva poblacion",it)
	return [P,it]



def run_agmo(data):
	save_log("LOG")
	iteraciones = 15
	P = get_poblacion_inicial(TAM_POBLACION)
	P = calcular_fitness(P,data)
	imprimir_poblacion(P,"Poblacion Inicial",0)
	''' correr solo la primera poblacion
	for it in range(iteraciones):
		[P,it] = run_iteracion(P, it, data)
		break #una ejecucion cada vez
	'''



def load_and_run_iteration(data):	
	[P,it]=load_poblacion()
	run_iteracion(P, it, data)
	print "\nFIN"



def evolucionar():
	#load data
	data = DATA()
	data.get_data()
	data.cargar_imagenes_prueba()

	#run_agmo(data) #runing agmo
	
	load_and_run_iteration(data) #run an iteration

evolucionar()




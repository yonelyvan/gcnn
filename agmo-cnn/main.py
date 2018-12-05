from agmo import *

def run_agmo():
	save_log("LOG")
	iteraciones = 15
	tam_poblacion = TAM_POBLACION
	D = 1
	P = get_poblacion_inicial(tam_poblacion)
	P = calcular_fitness(P)
	for i in range(iteraciones):
		print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Iteracion: ",i
		save_log(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Iteracion: "+ str(i) )
		print "Poblacion"
		save_log("Poblacion")
		imprimir_poblacion(P)

		print "Seleccion"
		save_log("Seleccion")
		seleccionados = seleccion(P)
		#curzamiento y mutacion
		hijos = cruzamiento(seleccionados)
		hijos = mutar(hijos)
		#calculo de fittnes
		hijos = calcular_fitness(hijos)
		print "HIJOS"
		save_log("HIJOS")
		imprimir_poblacion(hijos)
		for I in hijos:
			P.append(I)

		print "Nueva Poblacion"
		save_log("Nueva Poblacion")
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


run_agmo()


'''
config = get_crom_values("011100011000110001")
fitness = run_cnn(config)
print "FITENESS: error, tiempo"
print fitness
'''
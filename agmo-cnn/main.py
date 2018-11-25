from agmo import *

def run_agmo():
	iteraciones = 5
	tam_poblacion = TAM_POBLACION
	D = 1
	P = get_poblacion_inicial(tam_poblacion)
	for i in range(iteraciones):
		print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Iteracion: ",i
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


run_agmo()


'''
config = get_crom_values("001101101")
fitness = run_cnn(config)
print "FITENESS: error, tiempo"
print fitness
'''
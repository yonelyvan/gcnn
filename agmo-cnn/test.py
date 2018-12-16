#from cnn import *
from utils import *


#print get_crom_values("010000110110100010")
#print get_crom_values("011010111010100010")
#print get_crom_values("011010110110100010")

config = get_crom_values("011010110110100000")


data = DATA()
data.get_data()
data.cargar_imagenes_prueba()

print run_cnn_arq(config ,data)


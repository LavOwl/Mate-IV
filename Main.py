import pandas as pd #DataRetrival
import numpy as np #IndexingAndListFunctions
import matplotlib.pyplot as plt
from RegressionClass import lineal_regression
from MultipleRegressionClass import multiple_regression_class as MRC
from GradientDescentOptimized import multiple_linear_regression as GDO

#/////////////////////////////////////////////////////////
#CARGA DE DATOS DEL ARCHIVO CSV
df = pd.read_csv('players_21.csv')
#Removimos "defending_marking" ya que era null en la totalidad de los jugadores.
#df = df.fillna(0)
#print(df['defending_marking'].sum()) #usamos estas dos lineas para demostrarlo. Reemplazamos todos los null por 0, y al sumarlos nos dio 0, indicando que todos eran previamente nulls, o al menos que la variable no variaba entre jugadores y era siempre 0, quedando inmediatamente descartada como determinante de cualquier otro valor.
vectorStats = ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']

#/////////////////////////////////////////////////////////
#CALCULO DE COEFICIENTES DE DETERMINACIÓN, Y DE LINEALIDAD. REGISTRO DE ESTIMADORES PUNTUALES DE B1 Y B0

Y = df['value_eur'] #Carga datos de value_eur
r2scores = []

for stat in vectorStats:
    X = df[[stat]] #Carga los datos de una caracteristica
    lr = lineal_regression(X.values, Y.values) #Prepara una regresión lineal entre ambos valores
    r2scores.append(lr.r2) #Guarda R2
    print(stat + " B1: " + str(lr.b1) + " B0: " + str(lr.b0))
    print("R2: " + str(lr.r2))
    print("R: " + str(lr.r))
    print('--------') #Así obtuvimos los datos del coeficiente de determinación y de correlación lineal de cada caracteristica.
print("//////////////////////")
#/////////////////////////////////////////////////////////
#DETECCIÓN DE LA ESTADÍSTICA MÁS RELEVANTE, Y CALCULO DE SU REGRESIÓN LINEAL

max_r2 = max(r2scores) #Obtenemos el mejor resultado de predicción
maxIndex = r2scores.index(max_r2) #Obtenemos su indice
statSeleccionada = vectorStats[maxIndex] #Obtenemos la stat con mejor ratio de predicción de variabilidad, usando el indice previo
X = df[[statSeleccionada]] #Cargamos los datos de esa stat
lr = lineal_regression(X.values, Y.values) #Calculamos la línea de regresión

print("V2: ", lr.var)
print("Sxx: ", lr.Sxx)
print("B1: ", lr.b1)
print("Mean x: ", lr.mean_x)
print("N: ", lr.N)

print("Est b1: ", lr.b1/np.sqrt(lr.var/lr.Sxx))
print("Int b0: [" + str(lr.b0 - 1.96*np.sqrt(lr.var*(1/lr.N + (lr.mean_x**2)/lr.Sxx))) + "; " + str(lr.b0 + 1.96*np.sqrt(lr.var*(1/lr.N + (lr.mean_x**2)/lr.Sxx))) + "]")
print("Int b1: [" + str(lr.b1 - 1.96*np.sqrt(lr.var/lr.Sxx)) + "; " + str(lr.b1 + 1.96*np.sqrt(lr.var/lr.Sxx)) + "]")

#/////////////////////////////////////////////////////////
#GRAFICO DE LA RECTA SELECCIONADA

plt.scatter(X, Y, color='blue')  # Puntos
plt.plot(X, lr.predict(), color='red')  # Linea de regresión
plt.xlabel(statSeleccionada)
plt.ylabel("value_eur")
plt.title("Precio según " + statSeleccionada)
plt.show()


#/////////////////////////////////////////////////////////
#OBTENER TOP 3 ESTADÍSTICAS MÁS RELEVANTES
stats = []
X = []
for i in range(3):
    max_r2 = max(r2scores)
    maxIndex = r2scores.index(max_r2)
    stats.append(vectorStats[maxIndex]) #Así seleccionamos las tres estadísitcas más relevantes
    X.append(df[vectorStats[maxIndex]].values)
    r2scores[maxIndex] = 0 

multi_line = MRC(X, Y.values)
coefficients = multi_line.coefficients
for c in coefficients:
    print(float(c[0]))

print("////")
print("R2: ", multi_line.r2)
print("Sce: ",multi_line.sce)
print("Stc: ",multi_line.stc)
print("r: ", np.sqrt(multi_line.r2))
print("r2a: ", multi_line.r2a)


#print(GDO(df[stats], df['value_eur'], 0.00001))
#Este es el descenso de gradiente. Salvo que quieran optimizar el funcionamiento de su CPU y el orden de prioridad de programas para correrlo por más de 10h, les recomiendo no ejecutarlo.

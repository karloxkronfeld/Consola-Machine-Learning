# Consola-Machine-Learning

La consola de machine learning, son 6 algoritmos de regresion en aprendizaje supervisado, usando una interfaz de videojuegos.

![](https://media.giphy.com/media/dWNXUH0y1X0Behnx6H/giphy.gif)


## Comenzando 

Estas instrucciones te permitirán obtener una copia de la consola funcionando en tu máquina local para propósitos de desarrollo y pruebas.
Mira **Deployment** para conocer como desplegar el proyecto.


### Requisitos 

* [Pygame](https://www.pygame.org/wiki/GettingStarted)
* [Scikit-Learn](https://scikit-learn.org/) 
* [Numpy](https://numpy.org/) 
* [Matplotlib](https://matplotlib.org/)


## Ejecutando las pruebas 

Los datos son generados automaticamente, tambien, es posible modificar el origen de los datos usando las variables x,y
![](https://user-images.githubusercontent.com/63472277/79079398-68b7ac00-7cd4-11ea-8d42-be699029968f.png)
```
from sklearn import datasets
boston= datasets.load_boston()
x=boston.data[:,np.newaxis,5]
y=boston.target
```

### Analiza las pruebas 

Cada interaccion da como resultado una nueva interpretacion de los datos basada en un modelo de machine learning, con sus respectivas metricas

```
Lineal,Polinomial,Arboles de decision, Bosques...
```

```
R2, Mean absolute error(MAE), Mean Square Error (MSA),...
```
Estas pruebas sirven para identificar la naturaleza de los datos y fortalercer el analisis de datos.

## Despliegue 
Teniendo los requisitos de software, simplemente correr el programa con uno de los editores de python.

---
[KarloxKronfeld](https://github.com/karloxkronfeld) 
---

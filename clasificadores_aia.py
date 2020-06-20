#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================================================================
# Ampliación de Inteligencia Artificial
# Implementación de clasificadores 
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autores del trabajo:
#
# APELLIDOS: Barragán Rodríguez
# NOMBRE: Emilio
#
# APELLIDOS: Sánchez Alanís
# NOMBRE: Jesús Manuel
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo
# que debe realizarse de manera individual. La discusión y el intercambio de
# información de carácter general con los compañeros se permite, pero NO AL
# NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED o cualquier otro medio, se considerará plagio. Si tienen
# dificultades para realizar el ejercicio, consulten con el profesor. En caso
# de detectarse plagio, supondrá una calificación de cero en la asignatura,
# para todos los alumnos involucrados. Sin perjuicio de las medidas
# disciplinarias que se pudieran tomar. 
# *****************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARCTICULAR: NO HACERLO EN UN NOTEBOOK.

# NOTAS: 
# * En este trabajo NO SE PERMITE usar Scikit Learn (excepto las funciones que
#   se usan en carga_datos.py). 
# * Se permite (y se recomienda) usar numpy.  

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aia.tgz y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn, por lo que para que la carga se haga sin
# problemas, deberá estar instalado el módulosklearn). Todos los datos se
# cargan en arrays de numpy.

# * Datos sobre día adecuado para jugar al tenis: un ejemplo clásico "de
#   juguete", que puede servir para probar la implementación de Naive
#   Bayes. Se carga en las variables X_tenis, y_tenis. 

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.

# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 


# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y su correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser aleatoria y
# estratificada respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# >>> Xe_votos,Xp_votos,ye_votos,yp_votos
#           =particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los tres conjuntos de
# datos: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#  (array([0, 1]), array([168, 267]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array([0, 1]), array([112, 178]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array([0, 1]), array([56, 89]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=
#               particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------
import numpy as np
import carga_datos
import math


def particion_entr_prueba(X, y, test=0.20):
    # Agrupamos los elementos de X e y en tuplas para no perder su correspondencia al mezclar.
    zipped = np.array([x for x in zip(X, y)])
    # Mezclamos aleatoriamente dicho array de tuplas.
    np.random.shuffle(zipped)
    # Valores unicos en y (clases).
    y_unicos = np.unique(y)
    # Creamos las listas vacias para nuestros datos.
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    # Lista para guardar listas con ejemplos. Cada lista solo tiene una clase.
    lista = []
    # Punto por el que dividir nuestros datos.
    punto = round(len(X) * test / len(y_unicos))

    # Por cada clase tenemos una lista en lista, Cada lista son ejemplos con la clase i.
    for i in y_unicos:
        lista.append([x for x in zipped if x[1] == i])
    # Agregamos proporcionalmente (segun nuestro punto) elementos de cada clase.
    for z in lista:
        x_train.append([x[0] for x in z[punto:]])
        y_train.append([x[1] for x in z[punto:]])
        x_test.append([x[0] for x in z[:punto]])
        y_test.append([x[1] for x in z[:punto]])
    # Aplanamos nuestras listas.
    return np.array([x[i] for x in x_train for i in range(len(x))]), \
           np.array([x[i] for x in x_test for i in range(len(x))]), \
           np.array([y[i] for y in y_train for i in range(len(y))]), \
           np.array([y[i] for y in y_test for i in range(len(y))])

# ========================================================
# EJERCICIO 2: IMPLEMENTACIÓN DEL CLASIFICADOR NAIVE BAYES
# ========================================================

# Se pide implementar el clasificador Naive Bayes, en su versión categórica
# con suavizado y log probabilidades. Esta versión categórica NO es la versión
# multinomial (específica de vectorización de textos) que se ha visto en el
# tema 3. Lo que se pide es la versión básica del algoritmo Naive Bayes, vista
# en la Sección 5 del tema 6 de la asignatura "Inteligencia Artificial" del
# primer cuatrimestre. 


# ----------------------------------
# 2.1) Implementación de Naive Bayes
# ----------------------------------

# Definir una clase NaiveBayes con la siguiente estructura:

# class NaiveBayes():

#     def __init__(self,k=1):
#                 
#          .....

#     def entrena(self,X,y):

#         ......

#     def clasifica_prob(self,ejemplo):

#         ......

#     def clasifica(self,ejemplo):

#         ......

class NaiveBayes:
    def __init__(self, k=1):
        self.pvj = dict()             # P(vj)
        self.pav = dict()             # P(ai|vj)
        self.clases = []              # Clases que tenemos en nuestros datos de entrenamiento
        self.num_atribs = 0           # Numero de atributos que tenemos por cada ejemplo
        self.num_ejem = 0             # Numero de ejemplos en nuestros datos
        self.atribs = []              # Valores que toma cada atributo. Una lista de listas, en las que
                                      # cada lista i corresponde a los valores posibles para el atributo i
        self.probabilidades = dict()  # Probabilidades de cada clase
        self.k = k                    # Constante de suavizado

    def entrena(self, X, y):
        self.clases = np.unique(y)
        # El numero de atributos es el mismo para todos los ejemplos, por lo que cogemos el tamaño del primero.
        self.num_atribs = len(X[0])
        self.num_ejem = len(X)
        # Los valores posibles de cada atributo son los valores unicos de cada ejemplo que hay en cada columna i (atributo i)
        self.atribs = [np.unique(X[:, i]) for i in range(self.num_atribs)]
        # Para cada clase calculamos su probabilidad.
        for n, v in enumerate(self.clases):
            # Cogemos todos los elementos pertenecientes a la clase.
            listapvj = [X[i] for i in range(len(X)) if y[i] == v]
            # P(vj) = log( n(V = vj) / N )
            self.pvj[v] = np.log(len(listapvj) / self.num_ejem)
            # Calculamos las probabilidades de los valores de atributos de la clase v
            for m, i in enumerate(self.atribs):
                # Recorremos los valores que toma cada atributo
                for j in i:
                    # Ejemplos que en el atributo m tienen el valor j
                    listapav = [x for x in listapvj if x[m] == j]
                    # P(ai|vj) = log( n(Ai = ai, V = vj) / n(V = vj) )
                    self.pav[(m, j, v)] = np.log((len(listapav) + self.k) / (len(listapvj) + self.k * len(i)))

    def clasifica_prob(self, ejemplo):
        # Si no tenemos probabilidades de clase no hemos entrenado. Fallo
        if self.pvj == dict():
            raise ClasificadorNoEntrenado

        for c in self.clases:
            # Prob de la clase c
            probv = self.pvj[c]
            # Lista con las probabilidades condicionadas de c
            lista = np.asarray([self.pav[(n, x, c)] for n, x in enumerate(ejemplo)])
            # log( P(vj) ) + sum( log( P(ai|vj) ) )
            prob = probv + np.sum(lista)
            # Deshacemos los logaritmos
            self.probabilidades[c] = math.exp(prob)
        # Normalizamos las probabilidades
        suma = np.sum(list(self.probabilidades.values()))
        for x in self.probabilidades:
            self.probabilidades[x] = self.probabilidades[x] / suma
        return self.probabilidades

    def clasifica(self, ejemplo):
        # Si no tenemos probabilidades de clase no hemos entrenado. Fallo
        if self.pvj == dict():
            raise ClasificadorNoEntrenado
        # Devolvemos la clasificacion con mayor probabilidad
        probs = self.clasifica_prob(ejemplo)
        return max(probs, key=probs.get)

# * El constructor recibe como argumento la constante k de suavizado (por
#   defecto 1) 
# * Método entrena, recibe como argumentos dos arrays de numpy, X e y, con los
#   datos y los valores de clasificación respectivamente. Tiene como efecto el
#   entrenamiento del modelo sobre los datos que se proporcionan.  
# * Método clasifica_prob: recibe un ejemplo (en forma de array de numpy) y
#   devuelve una distribución de probabilidades (en forma de diccionario) que
#   a cada clase le asigna la probabilidad que el modelo predice de que el
#   ejemplo pertenezca a esa clase. 
# * Método clasifica: recibe un ejemplo (en forma de array de numpy) y
#   devuelve la clase que el modelo predice para ese ejemplo.   

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception):
    def __str__(self):
        return "Modelo no entrenado"


# ------------------------------------------------------------------------------
# Ejemplo "jugar al tenis":
#X_tenis = carga_datos.X_tenis
#y_tenis = carga_datos.y_tenis
#nb_tenis = NaiveBayes(k=0.5)
#x_train_tenis, x_test_tenis, y_train_tenis, y_test_tenis = particion_entr_prueba(X_tenis, y_tenis)
#nb_tenis.entrena(X_tenis, y_tenis)


#print(nb_tenis.pav)

#ej_tenis = np.array(['Soleado', 'Baja', 'Alta', 'Fuerte'])
#print(nb_tenis.clasifica(ej_tenis))
# {'no': 0.7564841498559081, 'si': 0.24351585014409202}
#print(nb_tenis.clasifica(ej_tenis))
# 'no'
# ------------------------------------------------------------------------------


# ----------------------------------------------
# 2.2) Implementación del cálculo de rendimiento
# ----------------------------------------------

# Definir una función "rendimiento(clasificador,X,y)" que devuelve la
# proporción de ejemplos bien clasificados (accuracy) que obtiene el
# clasificador sobre un conjunto de ejemplos X con clasificación esperada y.

def rendimiento(clasificador, X, y):
    # Hacemos predicciones con nuestro clasificador y nuestros datos X e y
    predicciones = np.array([clasificador.clasifica(Xi) for Xi in X])
    num_predicciones = len(predicciones)
    # Predicciones que hemos acertado y fallado
    bien = predicciones == y
    # Numero de aciertos
    num_bien = len(bien[bien == True])
    return num_bien / num_predicciones


# ------------------------------------------------------------------------------
# Ejemplo:

#print(rendimiento(nb_tenis, X_tenis, y_tenis))


# 0.9285714285714286
# ------------------------------------------------------------------------------


# --------------------------
# 2.3) Aplicando Naive Bayes
# --------------------------

# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Concesión de prestamos
# - Críticas de películas en IMDB 

# En todos los casos, será necesario separar los datos en entrenamiento y
# prueba, para dar la valoración final de los clasificadores obtenidos (usar
# para ello la función particion_entr_prueba anterior). Ajustar también el
# valor del parámetro de suavizado k. Mostrar el proceso realizado en cada
# caso, y los rendimientos obtenidos.

def rendimiento_k(x_train, y_train, max_k=10, min_k=0.5, k_step=0.5, x_test=None, y_test=None, test=0.1):
    """

    :param x_train: ejemplos de entrenamiento
    :param y_train: targets de entrenamiento
    :param max_k: valor maximo de k para el que comprobaremos el rendimiento
    :param min_k: valor minimo de k para el que comprobaremos el rendimiento
    :param k_step: cuanto aumentamos k en cada iteracion para llegar del min al max
    :param x_test: opcional. ejemplos de test
    :param y_test: opcional. targets de test
    :param test: proporcion que queremos para test y validacion
    :return: rendimiento del modelo naive bayes sobre los datos. Tomando valores de k entre min_k y max_k.
    """
    # Si damos x_test tenemos que dar y_test, y viceversa. Si no, salta una excepcion.
    # Si no los damos, sacamos el test de nuestros datos de entrenamiento
    # Si los damos, usamos dichos datos para test y de nuestros datos solo sacamos validacion
    if(x_test is None):
        if(y_test is not None):
            raise Exception("Falta x_test")
        x_train, x_test, y_train, y_test = particion_entr_prueba(x_train, y_train, test)

    elif (y_test is None):
        if (x_test is not None):
            raise Exception("Falta y_test")
        x_train, x_test, y_train, y_test = particion_entr_prueba(x_train, y_train, test)

    x_train, x_val, y_train, y_val = particion_entr_prueba(x_train, y_train,test)
    # Mejor valor (k, rendimiento)
    mejor = (0, 0)
    # Por cada iteracion creamos un modelo con k = i. Entrenamos y comprobamos el rendimiento sobre validacion.
    # Nos quedamos con el mejor.
    for i in np.arange(min_k, max_k, k_step):
        #print("k={}".format(i))
        nb = NaiveBayes(k=i)
        nb.entrena(x_train, y_train)
        actual = rendimiento(nb, x_val, y_val)
        #print("Rendimiento: {}".format(actual))
        if actual > mejor[1]:
            mejor = (i, actual)
            mejor_modelo = nb
    # Mostramos cual ha sido el mejor valor de k sobre validacion y comprobamos el rendimiento sobre test
    print("El mejor valor de k: ", mejor[0])
    print("Rendimiento sobre validación: ", mejor[1])
    print("Rendimiento sobre test: ", rendimiento(mejor_modelo, x_test, y_test))

#print("RENDIMIENTO VOTOS")
#print("----------------------------")
#X_votos = carga_datos.X_votos
#y_votos = carga_datos.y_votos

#rendimiento_k(X_votos, y_votos)
#print()

#print("RENDIMIENTO PRÉSTAMOS")
#print("----------------------------")
#X_credito = carga_datos.X_credito
#y_credito = carga_datos.y_credito

#rendimiento_k(X_credito, y_credito)
#print()

#print("RENDIMIENTO CINE")
#print("----------------------------")
#x_train_cine = np.load('/home/theesmox/Documentos/AIA/Trabajo AIA/datos/imdb_sentiment/vect_train_text.npy')
#y_train_cine = np.load('/home/theesmox/Documentos/AIA/Trabajo AIA/datos/imdb_sentiment/y_train_text.npy')
#x_test_cine = np.load('/home/theesmox/Documentos/AIA/Trabajo AIA/datos/imdb_sentiment/vect_test_text.npy')
#y_test_cine = np.load('/home/theesmox/Documentos/AIA/Trabajo AIA/datos/imdb_sentiment/y_test_text.npy')

#rendimiento_k(x_train_cine, y_train_cine, x_test=x_test_cine, y_test=y_test_cine)

# =================================================
# EJERCICIO 3: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================

# Este ejercicio es OPCIONAL, y servirá para el ajuste de parámetros en los
# ejercicios posteriores. Si no se realiza, se podrían ajustar siguiendo el
# método "holdout" implementado en el ejercicio 1

# Definir una función 

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador. El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)  

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=4, en el conjunto de datos de los votos, para estimar cómo de
# bueno es el valor k=0.1 para suavizado en NaiveBayes. Usando la función que
# se pide sería (nótese que debido a la aleatoriedad, no tiene por qué
# coincidir exactamente el resultado):

# >>> rendimiento_validacion_cruzada(NaiveBayes,{"k":0.1},Xe_votos,ye_votos,n=4)
# 0.8963744588744589

# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. 

# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de k), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

# >>> nb01=NaiveBayes(k=0.1)
# >>> nb01.entrena(Xe_votos,ye_votos)

# Ydaríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# >>> rendimiento(nb01,Xp_votos,yp_votos)
#  0.88195402298850575

# ------------------------------------------------------------------------------


# ========================================================
# EJERCICIO 4: MODELOS LINEALES PARA CLASIFICACIÓN BINARIA
# ========================================================

# En este ejercicio se pide implementar en Python un clasificador binario
# lineal, basado en regresión logística.


# ---------------------------------------------
# 4.1) Implementación de un clasificador lineal
# ---------------------------------------------

# En esta sección se pide implementar un clasificador BINARIO basado en
# regresión logística, con algoritmo de entrenamiento de descenso por el
# gradiente mini-batch (para minimizar la entropía cruzada).

# En concreto se pide implementar una clase: 

class RegresionLogisticaMiniBatch():

    def __init__(self,clases=[0,1],normalizacion=False,
                 rate=0.1,rate_decay=False,batch_tam=64,n_epochs=200,
                 pesos_iniciales=None):

        self.clases = clases
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.rate_d = (rate)*(1/(1+n_epochs))
        self.batch_tam = batch_tam
        self.n_epochs = n_epochs
    
    def sigmoide(x):
        return expit(x)

    def normalization(self,X,y):
        
        X_normalizado = np.zeros(shape=X.shape)
        y_normalizado = np.zeros(shape=y.shape)

        for i in range(len(X)):
            suma_total = 0
            suma_dist = 0
            desv_tipica = 0
            for j in range(len(X[i])):
                suma_total += X[i][j]
            media = suma_total/len(X[i])
            for j in range(len(X[i])):
                suma_dist += abs(X[i][j] - media)**2
            desv_tipica = math.sqrt(suma_dist/len(X[i]))
            for j in range(len(X[i])):
                X_normalizado[i][j] = (X[i][j] - media)/desv_tipica
        
        suma_total = 0
        suma_dist = 0
        desv_tipica = 0
        media = 0
        
        for i in range(len(y)):
            suma_total += y[i]
        media = suma_total/len(y)

        for i in range(len(y)):          
            suma_dist += abs(y[i]- media)**2
        desv_tipica = math.sqrt(suma_dist/len(y))

        for i in range(len(y)):
            y_normalizado[i] = (y[i]- media)/desv_tipica
        return X_normalizado, y_normalizado
        
    def mini_batch(self,X,y):
        indices = list(range(len(X)))
        indices = random.shuffle(indices)
        list_X_batch = []
        list_y_batch = []
        for i in range(0, X.shape[0], self.batch_tam):
            # Get pair of (X, y) of the current minibatch/chunk
            list_X_batch.append(X[i:i + self.batch_tam])
            list_y_batch.append(y[i:i + self.batch_tam])
        return list_X_batch, list_y_batch

    def entrena(self,X,y):

        X_normalizado = []
        y_normalizado = []
        
        if self.normalizacion == True:
            X_normalizado,y_normalizado = self.normalization(X,y)
        else:
            X_normalizado,y_normalizado = X,y

        #n_epochs
        n_epochs = self.n_epochs
        clases = np.unique(y)
        n_atributos = len(X_normalizado[0])
        n_ejemplos = len(X_normalizado)
        w = np.zeros(shape=np.shape(X))
        for n in range(n_epochs):
            list_X_mini, list_y_mini = self.mini_batch(X, y)
            for i in range(len(list_X_mini)):
                X_mini = [x for x in list_X_mini[i]]
                y_mini = [y for y in list_y_mini[i]]
                for j in range(len(X_mini)):
                    for z in range(len(X_mini[j])):
                        print(X_mini[j][z])
                        if self.rate_decay == True:    
                            w[j] += self.rate_d*sum(y[z]-self.sigmoide(w*X_mini[j][z]))*X_mini[j]
                        else:
                            w[j] += self.rate*sum(y[z]-self.sigmoide(w*X_mini[j][z]))*X_mini[j]
                    #error_cuadratico = error_cuadratico_medio(x,y_train,w)
        return w

#     def clasifica_prob(self,ejemplo):

#         ......

#     def clasifica(self,ejemplo):

#          ......


# Explicamos a continuación cada uno de estos elementos:


# * El constructor tiene los siguientes argumentos de entrada:

#   + Una lista clases (de longitud 2) con los nombres de las clases del
#     problema de clasificación, tal y como aparecen en el conjunto de datos. 
#     Por ejemplo, en el caso de los datos de las votaciones, esta lista sería
#     ["republicano","democrata"]. La clase que aparezca en segundo lugar de
#     esta lista se toma como la clase positiva.  

#   + El parámetro normalizacion, que puede ser True o False (False por
#     defecto). Indica si los datos se tienen que normalizar, tanto para el
#     entrenamiento como para la clasificación de nuevas instancias. La
#     normalización es la estándar: a cada característica se le resta la media
#     de los valores de esa característica en el conjunto de entrenamiento, y
#     se divide por la desviación típica de dichos valores.

#  + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate es la
#    tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#  + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#    cada epoch. En concreto, si rate_decay es True, la tasa de
#    aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#    con la siguiente fórmula: 
#       rate_n= (rate_0)*(1/(1+n)) 
#    donde n es el número de epoch, y rate_0 es la cantidad introducida
#    en el parámetro rate anterior. Su valor por defecto es False. 

#  + batch_tam: indica el tamaño de los mini batches (por defecto 64) que se
#    usan para calcular cada actualización de pesos.

#  + n_epochs: número de veces que se itera sobre todo el conjunto de
#    entrenamiento.

#  + pesos_iniciales: si no es None, es un array con los pesos iniciales. Este
#    parámetro puede ser útil para empezar con unos pesos que se habían obtenido
#    y almacenado como consecuencia de un entrenamiento anterior.


# * El método entrena tiene como parámteros de entrada dos arrays X e y, con
#   los datos del conjunto de entrenamiento y su clasificación esperada,
#   respectivamente.


# * Los métodos clasifica y clasifica_prob se describen como en el caso del
#   clasificador NaiveBayes. Igualmente se debe devolver
#   ClasificadorNoEntrenado si llama a los métodos de clasificación antes de
#   entrenar. 

# Se recomienda definir la función sigmoide usando la función expit de
# scipy.special, para evitar "warnings" por "overflow":

# from scipy.special import expit    
#
# def sigmoide(x):
#    return expit(x)


# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama:

# >>> Xe_cancer,Xp_cancer,ye_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer)

# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True,n_epochs=1000)

# >>> lr_cancer.entrena(Xe_cancer,ye_cancer)

# >>> rendimiento(lr_cancer,Xe_cancer,ye_cancer)
# 0.9912280701754386

# >>> rendimiento(lr_cancer,Xp_cancer,yp_cancer)
# 0.9557522123893806

# -----------------------------------------------------------------


# -----------------------------------
# 4.2) Aplicando Regresión Logística 
# -----------------------------------

# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay. En alguno de los conjuntos de datos puede ser necesaria
# normalización. Si se ha hecho el ejercicio 3, usar validación cruzada para
# el ajuste (si no, usar el "holdout" del ejercicio 1). 

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos. 


# =====================================
# EJERCICIO 5: CLASIFICACIÓN MULTICLASE
# =====================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica
# de "One vs Rest" (OvR)

# ------------------------------------
# 5.1) Implementación de One vs Rest
# ------------------------------------


#  En concreto, se pide implementar una clase python RL_OvR con la siguiente
#  estructura, y que implemente un clasificador OvR usando como base el
#  clasificador binario del apartado anterior.


# class RL_OvR():

#     def __init__(self,clases,rate=0.1,rate_decay=False,batch_tam=64,n_epochs=200):

#        ......

#     def entrena(self,X,y):

#        .......

#     def clasifica(self,ejemplo):

#        ......


#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, excepto que ahora "clases" puede ser una lista con más de dos
#  elementos. 

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris=RL_OvR([0,1,2],rate=0.001,batch_tam=20,n_epochs=1000)

# >>> rl_iris.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris,Xp_iris,yp_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------


# ---------------------------------------------------------
# 5.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación del apartado anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test).

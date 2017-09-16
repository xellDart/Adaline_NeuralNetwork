# -*- coding: utf-8 -*-
import random
from pylab import rand, plot
from sklearn import svm
import numpy as np
from mlxtend.plotting import plot_decision_regions
import AdalineGD
import matplotlib.pyplot as plt

random.randrange(start=-1, stop=1)


def generateData(n):
    xb = (rand(n) * 2 - 1) / 2 - 0.5
    yb = (rand(n) * 2 - 1) / 2 + 0.5
    xr = (rand(n) * 2 - 1) / 2 + 0.5
    yr = (rand(n) * 2 - 1) / 2 - 0.5
    inputs = []
    for i in range(len(xb)):
        inputs.append([xb[i], yb[i], 1])  # 1 Clase 1
        inputs.append([xr[i], yr[i], -1])  # -1 Clase 2
    return inputs


class Adaline:
    def __init__(self):
        """
        Inicialización Adaline
        """
        self.pesos = []
        self.salidas_deseadas = []
        self.entradas = []
        self.theta = 0.7  # grado de libertad
        self.alpha = 0.1  # taza de aprendizaje
        self.debug = True
        self.epocas = 0
        self.errorGlobal = 1  # para introducir al entrenamiento

    def setDebug(self, debug=True):
        """
        Metodo simple para activar mensajes
        """
        self.debug = debug

    def setEntradas(self, entradas):
        """
        Aplica las entradas y reajusta el número de pesos
        """
        if entradas.__len__() >= 2:
            # crea nuevo pesos
            self.entradas = entradas
            self.pesos = []
            for i in range(0, entradas[0].__len__()):
                self.pesos.append(random.random())

    def setSalidasDeseadas(self, saidas):
        """
        Ajusta las salidas deseadas necesarias para entrenar la neurona
        """
        if saidas.__len__() >= 2:
            self.salidas_deseadas = saidas

    def criterio_parada(self):
        """
        Asiganción de valor máximo vueltas o épcocas de entrenamiento.
        """
        return self.epocas == 100  # 100 vueltas máximo valor

    def sumatorio(self, numEntrada):
        """
        Hace la suma del producto de cada entrada con su respectivo peso
        """
        sumatorio = 0
        for i in range(0, self.entradas[numEntrada].__len__()):
            sumatorio += self.pesos[i] * self.entradas[numEntrada][i]
        return sumatorio

    def ativacion(self, numEntrada):
        """
        Función de activación.
        """
        sumatorio = self.sumatorio(numEntrada)
        if sumatorio >= self.theta:
            return 1
        elif sumatorio < -self.theta:
            return -1
        return 0

    def aprender(self):
        """
        Realiza el entrenamiento por épocas.
        """
        if self.salidas_deseadas.__len__() != self.entradas.__len__():
            raise RuntimeError("Error en los tamaños del conjunto de entrada y/o Salidas deseadas.")
        while self.errorGlobal != 0 and not self.criterio_parada():
            self.epocas += 1
            if self.debug: print "Iniciando epoca: ", self.epocas, ". Error Global:", self.errorGlobal
            self.errorGlobal = 0
            self.entrena()

    def entrena(self):
        """
        Recorre las entradas, calcula las salidas y pasa el error a la función de ajuste
        """
        for i in range(0, self.entradas.__len__()):  # entrena la neurona para cada entrada
            activacion = self.ativacion(i)
            # print self.salidas_deseadas[i][2]
            error = self.salidas_deseadas[i][2] - activacion
            self.errorGlobal += error.__abs__()
            if self.debug:
                #  print " Entradas",self.entradas[i]
                print " Salida deseada: ", self.salidas_deseadas[i][2]
                print " Salida encontrada: ", activacion
                print " Error: ", error
                if activacion == 1:
                    plot(self.salidas_deseadas[i][0], self.salidas_deseadas[i][1], 'ob')   # Clase 1
                else:
                    plot(self.salidas_deseadas[i][0], self.salidas_deseadas[i][1], 'or')  # Clase 2
                    # print " Pesos:",self.pesos
            self.ajusta_pesos(error, i)

    def ajusta_pesos(self, error, numEntrada):
        """
        Función principal: realiza el ajuste de pesos (aprendizaje) cuando hay error.
        """
        if error != 0:
            errores_ = self.pesos.__len__()
            for j in range(0, errores_):
                self.pesos[j] = self.pesos[j] + self.alpha * error * self.entradas[numEntrada][j]
            if self.debug: print " Pesos actualizados:", self.pesos
            


def main():
    ##############################################################

    neurona = Adaline()
    info = generateData(20)
    neurona.setEntradas(info)
    neurona.setSalidasDeseadas(info)
    neurona.aprender()
    print "Épocas de la neurona:", neurona.epocas


if __name__ == "__main__":
    main()

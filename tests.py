"""
Calificación del laboratorio
-----------------------------------------------------------------------------------------
"""

import sys

import preguntas


def test_01():
	#print(preguntas.pregunta_01())
	assert round(preguntas.pregunta_01(),3)==0.938


def test_02():
	#print(preguntas.pregunta_02().tolist())
	assert preguntas.pregunta_02().tolist() == [[250, 17], [10, 158]]


test = {
    "01": test_01,
    "02": test_02,
}[sys.argv[1]]

test()

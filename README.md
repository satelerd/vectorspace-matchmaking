# Vectorspace Matchmaking

Este proyecto usa el framework de embeddings ```SentenceTransformer```, ```UMAP``` y varias bibliotecas de Python, podemos crear una representación visual que revela las conexiones semánticas en un conjunto de datos de texto. 

<img src="https://github.com/satelerd/vectorspace-matchmaking/assets/62968964/8e46a626-14d5-43a4-8fd4-fdc1cac1a9f4" width="750" height="500">

## ¿Qué hace este proyecto?

Este script toma un archivo CSV que contiene opiniones y comentarios sobre los gustos musicales, obtenidos de una pregunta que publique en mi Instagram, y los convierte en vectores multidimencionales con ```SentenceTransformer```. 

Luego, para lograr la visualizacion, hay que reducir los vectores a solo dos dimensiones utilizando ```UMAP``` (Uniform Manifold Approximation and Projection). 

El resultado es un grafico donde la proximidad de los puntos representa la similitud semántica.

## ¿Qué son los NPCs y por qué están incluidos?

En este proyecto, los puntos azules son NPCs, que representan personajes generados mediante GPT-4, cada uno con sus propias opiniones y gustos musicales. Estos fueron creados en base a las respuestas originales.

La inclusión de NPCs en este proyecto permite aumentar la cantidad de datos disponibles para el análisis, lo que ayuda a ampliar el grafico para revelar patrones y conexiones más profundas. Además, proporciona un punto de comparación interesante para las opiniones de mis amigos humanos.

## ¿Cómo utilizar este proyecto?

1. Asegúrate de tener todas las bibliotecas requeridas instaladas en tu entorno Python.
2. Puedes usar 'data.csv' que tiene la data de los NPCs. O puedes crear tu propio dataset, solo tienes que reemplazar las dos columnas: una para los nombres (ya sean humanos o NPCs) y otra para los párrafos de texto.
3. Ejecuta el script.

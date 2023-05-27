# Vectorspace Matchmaking

Este proyecto nace de una fascinación por el procesamiento del lenguaje natural, el aprendizaje automático y la visualización de datos. Usando el framework de embeddings SentenceTransformer y varias bibliotecas de Python, podemos crear una representación visual multidimensional que revela las conexiones semánticas en un conjunto de datos de texto. 
<img src="https://github.com/satelerd/vectorspace-matchmaking/assets/62968964/8e46a626-14d5-43a4-8fd4-fdc1cac1a9f4" width="750" height="500">

## ¿Qué hace este proyecto?

Este script toma un archivo CSV que contiene declaraciones textuales - en este caso, opiniones y comentarios sobre los gustos musicales obtenidos en una pregunta que puse en mi historia de Instagram - y los convierte en vectores de incrustación de oraciones. Estos vectores se reducen a dos dimensiones para su visualización utilizando UMAP (Uniform Manifold Approximation and Projection). El resultado es una visualización donde la proximidad de los puntos representa la similitud semántica.

Aunque aquí lo hice con datos de gustos musicales, la versatilidad del proyecto te permite cambiar el conjunto de datos a cualquier otro tipo de texto que desees visualizar. Desde opiniones políticas hasta reseñas de libros o descripciones de películas, este proyecto puede ser adaptado a una variedad de casos de uso.

## ¿Qué son los NPCs y por qué están incluidos?

En este proyecto, los puntos azules son NPCs. En este proyecto, los NPCs son personajes generados mediante el modelo GPT-4 de OpenAI, cada uno con sus propias opiniones y gustos musicales. Estos fueron generados en base a las respuestas originales.

La inclusión de NPCs en este proyecto aporta un valor único. Nos permite aumentar la cantidad de datos disponibles para el análisis, lo que puede ayudar a ampliar un poco mas el grafico para revelar patrones y conexiones más profundas. Además, proporciona un punto de comparación interesante para las opiniones de nuestros amigos humanos. ¿Son las opiniones de nuestros amigos notoriamente diferentes de las generadas por la IA, o hay muchas similitudes?

## ¿Cómo utilizar este proyecto?

1. Asegúrate de tener todas las bibliotecas requeridas instaladas en tu entorno Python.
2. Sustituye 'data.csv' con el nombre de tu archivo CSV. Este archivo debe contener dos columnas: una para los nombres (ya sean amigos humanos o NPCs) y otra para los párrafos de texto.
3. Ejecuta el script. Se generará un gráfico de dispersión que representa las similitudes y diferencias en el conjunto de datos de texto que proporcionaste.

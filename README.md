

Objetivo es mostrar si hay algun modelo que funcione igual de bien con 100 datos que con 10 000 (tamaño original)


---


### Orden de la presentación

1. Mostrar todas las gráficas usando `train_size=0.7` y comprobar que los modelos tienen muy buen rendimiento.
2. Calcular cuantas predicciones realiza mal cada modelo y ver si son las mismas entre modelos.
3. Mostrar gráfica de como varía el rendimiento de cada modelo mientras de reduce el `train_size`. Los datos no están balanceados pero se hace la media.
4. Mostrar todas las gráficas usando `train_size=0.01` y decir que estás gráficas son mejores para el estudio.
5. Explicar los 2 tipos de preprocesamiento de datos que se van a realizar para comprobar si mejora o empeora el rendimiento.
6. Explicar cada gráfica del punto 4 junto con sus gemelas de los otros tipos de preprocesamiento. Cada gráfica tiene que ser hecha a partir de la media de varias repeticiones.
7. Cuando se acaben de explicar todas las gráficas, mostrar si las malas predicciones de cada modelo son las mismas o no y en que se diferencian.
8. Probar con datos propios distintos y similares a los que fallan los modelos y ver que pasa.
9. Concluir que modelo es el que:
   1. Aprende mas rápido.
   2. Falla más.
   3. Acierta más.
   4. Mejor modelo en general.

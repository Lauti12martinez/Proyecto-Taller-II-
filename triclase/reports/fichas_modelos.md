# Modelos evaluados, optimización y métrica rectora (Triclase)

## 1. Fichas técnicas de los algoritmos

### Naive Bayes (MultinomialNB / ComplementNB)
Fue el primer modelo que probamos y lo usamos como punto de partida para comparar al resto. Trabaja con el teorema de Bayes: mira las palabras de cada reseña y calcula qué tan probable es que pertenezca a cada clase. Para simplificar la cuenta supone que las palabras no dependen unas de otras, y de ahí viene lo de "ingenuo". Gracias a eso entrena muy rápido, aunque pierde cosas importantes: frente a un *"not bad"* ve un "not" y un "bad" por separado, cuando juntos quieren decir algo positivo. En la versión base usamos MultinomialNB, y para optimizar pasamos a ComplementNB, una variante que se lleva mejor con clases de distinto tamaño, como pasa con nuestra clase neutral. Ninguna de las dos variantes se probó sobre GloVe porque exigen valores no negativos y esos embeddings incluyen coordenadas negativas.

### Regresión Logística (LogReg)
La Regresión Logística le asigna un peso a cada palabra del vocabulario según cuánto la acerca a cada sentimiento, suma esos pesos y los transforma en probabilidades para las tres clases. Con el parámetro C se regula cuánto se "frena" al modelo para que no se aprenda de memoria los datos de entrenamiento. Lo que más nos gustó es que se puede interpretar: revisando los pesos se ve qué palabras empujan hacia cada clase, que es justo el análisis de palabras relevantes que pide la consigna. Su límite es el de cualquier modelo lineal: no entiende cómo se combinan las palabras, salvo lo poco que le aportan los bigramas.

### SVM lineal (LinearSVC)
El SVM busca una frontera entre las clases que las deje lo más separadas posible, con el mayor margen entre ellas, para que las reseñas no queden pegadas a la zona de duda. Como tenemos tres clases, arma un separador para cada una contra las otras dos. Se usa mucho para texto porque se maneja bien con vocabularios enormes y matrices con casi todos sus valores en cero. Por sí solo no devuelve probabilidades, así que en la versión optimizada le sumamos un calibrador (`CalibratedClassifierCV`) para obtenerlas.

### SGD (SGDClassifier)
SGD no es tanto un modelo como una manera de entrenar modelos lineales. En vez de procesar todo el dataset en cada paso, va corrigiendo los pesos de a pocas reseñas por vez (descenso de gradiente estocástico), y eso lo hace muy eficiente con un corpus del tamaño del nuestro. Qué modelo termina siendo depende de la función de pérdida: con `hinge` funciona como un SVM, y con `log_loss`, que fue la que elegimos, como una regresión logística. Casi todo se puede ajustar: el tipo de regularización (L1, L2 o elasticnet, que combina las dos), cuánta se aplica (α) y cuánto pesa cada clase. Esa libertad es un arma de doble filo, porque mal configurado rinde poco, y fue justamente lo que vimos al optimizarlo (sección 2).

### Random Forest (RF)
Random Forest junta muchos árboles de decisión. Cada árbol se entrena con una muestra distinta de los datos y, en cada decisión, solo puede mirar algunas de las variables. Al final, los árboles votan y gana la clase más elegida. Como cada árbol se equivoca de una forma diferente, el conjunto termina siendo más estable que cualquiera de ellos solo, y además puede encontrar relaciones no lineales. El problema en texto es la cantidad de columnas: con TF-IDF tendría que recorrer decenas de miles de palabras y se volvía muy lento. Por eso lo entrenamos sobre los embeddings GloVe, que resumen cada reseña en 100 números.

### XGBoost
XGBoost también usa muchos árboles, pero no los arma todos a la vez como Random Forest: los construye uno después del otro, y cada árbol nuevo intenta corregir lo que los anteriores hicieron mal (*gradient boosting*). Trae regularización incorporada y varias optimizaciones, y por eso es tan usado con datos tabulares. La contra es que tiene muchos hiperparámetros delicados, como la tasa de aprendizaje, la profundidad de los árboles o cuántos usar, y si no se controlan bien termina sobreajustando. Lo entrenamos sobre GloVe, igual que Random Forest.

### Perceptrón Multicapa (MLP)
El MLP es una red neuronal simple. Tiene una capa de entrada, una o más capas ocultas que transforman los datos de manera no lineal y una capa de salida con una neurona por clase. Aprende a partir de sus errores: cada vez que se equivoca, ajusta los pesos hacia atrás (retropropagación). En teoría, esas transformaciones le permiten captar relaciones entre palabras que un modelo lineal no ve, aunque a cambio es más difícil de interpretar y depende bastante de la arquitectura que se elija. Lo usamos sobre GloVe, escalando antes los datos. Nos dimos cuenta de que su techo no lo pone la red, sino la representación: GloVe promedia los vectores de todas las palabras de la reseña, y en ese promedio se pierden el orden y buena parte del efecto de las negaciones.

### RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
RoBERTa es el único modelo del proyecto basado en Transformers. Es una versión mejorada de BERT, entrenada con más datos y durante más tiempo. A diferencia de los demás, no hace falta vectorizar nada: lee la reseña tal cual está escrita y, con el mecanismo de atención (*self-attention*), interpreta cada palabra según el contexto en el que aparece. En principio, eso lo deja mejor parado frente a negaciones, ironías o frases con vueltas. La versión que usamos ya venía entrenada para clasificar sentimiento en las mismas tres clases, pero con tweets, y la aplicamos sin reentrenarla. Pasar de tweets a reseñas de productos le juega en contra, y ajustarlo con nuestras propias reseñas (*fine-tuning*) es probablemente la mejora con más potencial que nos queda.

## 2. Por qué optimizamos más de un modelo

La cátedra nos pidió entrenar un grupo de modelos base y nos dejó libertad para sumar otros y optimizar el que mejor resultado diera. Siguiendo esa idea, la elección era bastante obvia: la Regresión Logística con TF-IDF bigrama había salido primera en la tanda base.

Igual preferimos optimizar varios modelos y ver hasta dónde podía llegar cada uno, en lugar de dar por sentado que el orden inicial se iba a mantener. Y ahí apareció lo más interesante del trabajo:

| Modelo | Mejor F1-macro base | Puesto base (de 12) | F1-macro optimizado | Mejora |
|---|---|---|---|---|
| SGD | 0,581 (BoW) | 8.º | 0,657 (TF-IDF bigrama) | +7,6 pts |
| Naive Bayes → ComplementNB | 0,613 (BoW) | 4.º | 0,641 (TF-IDF bigrama) | +2,8 pts |
| Regresión Logística | 0,640 (TF-IDF bigrama) | 1.º | 0,653 (BoW bigrama) | +1,3 pts |
| LinearSVC | 0,628 (TF-IDF bigrama) | 2.º | 0,617 (TF-IDF bigrama) | −1,1 pts |

SGD había quedado octavo de doce, así que con el criterio original no lo hubiéramos tocado. Una vez optimizado, fue el que más creció y terminó primero. La Regresión Logística, que arrancaba arriba, subió apenas un punto, y LinearSVC hasta empeoró.

La explicación está en que SGD es el que más margen de ajuste tiene. En la búsqueda probamos tres tipos de regularización (L1, L2 y elasticnet, que mezcla ambas en distintas proporciones), cinco valores de α y el balanceo de clases, además de las opciones del vectorizador. Revisando el historial de la búsqueda (`cv_search_history.csv`) se ve que su configuración por defecto estaba muy lejos de lo que necesitaba nuestro problema, por dos motivos:

- No compensaba la clase neutral. Sin `class_weight="balanced"` el modelo se inclinaba hacia las clases más grandes. Todas las mejores configuraciones usan balanceo; la mejor sin balanceo llega a 0,597, unos 6 puntos menos.
- Estaba demasiado regularizado. Con el α por defecto (1e-4) el modelo quedaba tan frenado que no llegaba a aprovechar el vocabulario. El mejor valor resultó diez veces más chico (1e-5).

Esperábamos que elasticnet fuera la clave, pero la configuración ganadora terminó usando L2, con elasticnet y L1 bastante cerca (0,637 y 0,636). En esa diferencia también pesó el muestreo aleatorio de la búsqueda (`RandomizedSearchCV` con 12 combinaciones): la corrida ganadora con L2 salió sorteada con bigramas, mientras que las mejores con elasticnet y L1 cayeron con unigramas.

Del resto también sacamos algo. Naive Bayes ganó casi 3 puntos solo con pasar a la variante Complement y ajustar el suavizado. La Regresión Logística ya partía bien y tenía menos para mejorar, aunque al ajustar en conjunto el pipeline terminó rindiendo mejor con BoW en bigramas que con TF-IDF. LinearSVC nos mostró que optimizar no siempre suma: aunque el clasificador interno usaba balanceo de clases, el calibrador isotónico (`CalibratedClassifierCV`) reescaló las probabilidades hacia la proporción real del dataset (40/20/40) y tiró abajo la detección de la clase neutral.

## 3. Por qué el F1-macro es la métrica rectora en triclase

En triclase usamos las tres clases, que quedan repartidas en 40 % negativas, 20 % neutrales y 40 % positivas. Con esa distribución, la accuracy no es confiable. Mide el porcentaje total de aciertos, así que cada clase cuenta según cuántas reseñas tiene, y las neutrales son apenas una de cada cinco. Un modelo que nunca dijera "neutral" y acertara todas las negativas y positivas llegaría al 80 % de accuracy. El número parece muy bueno, pero ese modelo falla siempre en la clase más difícil y también en la más útil para el cliente: la de las opiniones tibias o con sentimientos mezclados.

El F1-macro no cae en esa trampa. Calcula el F1 de cada clase por separado (un balance entre *precision* y *recall*) y después saca el promedio simple, así que las tres clases valen lo mismo aunque tengan cantidades distintas de reseñas. En el ejemplo anterior, el F1 de la clase neutral daría 0 y el F1-macro no podría pasar de 0,667, con lo que el problema queda a la vista. También castiga los dos errores posibles: predecir "neutral" casi nunca o predecirlo demasiado.

Lo comprobamos con nuestros propios modelos:

| Modelo | Accuracy | F1-macro |
|---|---|---|
| LinearSVC optimizado | 0,730 | 0,617 |
| SGD optimizado | 0,721 | 0,657 |
| SGD sin optimizar (TF-IDF bigrama) | 0,701 | 0,548 |

Si nos guiáramos por la accuracy, elegiríamos LinearSVC. Pero su F1-macro queda 4 puntos por debajo del de SGD, porque saca su buena accuracy acertando en las clases grandes y fallando en la neutral. Con el SGD sin optimizar se ve todavía más: 70 % de accuracy y un F1-macro de apenas 0,548.

Para ser coherentes, el F1-macro no fue solo la métrica de comparación final. También lo usamos como `scoring` en la validación cruzada de 5 *folds* y en todas las búsquedas de hiperparámetros, tanto de los modelos lineales (`optimize_cv.py`) como de los densos (`train_dense.py`). Así, cada modelo se ajustó persiguiendo lo mismo con lo que después lo evaluamos, y por eso el balanceo de clases aparece en casi todas las mejores configuraciones.

Además, la consigna pide cerrar la comparativa con una conclusión clara sobre qué modelo usar, y para eso necesitábamos un único criterio: el F1-macro, la única de las métricas pedidas que le da el mismo peso a las tres clases.

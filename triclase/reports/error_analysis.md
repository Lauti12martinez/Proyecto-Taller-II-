# Análisis Cualitativo de Errores — Flujo Triclase

En esta sección analizamos casos concretos del banco adversarial (`0: Negativo`, `1: Neutral`, `2: Positivo`) para entender qué fenómenos del lenguaje confunden a las representaciones basadas en palabras sueltas (BoW y TF-IDF Unigrama), en qué situaciones logran resolverlos TF-IDF Bigrama o RoBERTa, y dónde aparecen los límites actuales de nuestros modelos.

## 1. Casos donde BoW / Unigrama fallan y Bigrama o RoBERTa aciertan

| # | Oración | Real | Fenómeno | BoW | TF-IDF Unigrama | TF-IDF Bigrama | RoBERTa | Fallos |
| :---: | :--- | :---: | :--- | :--- | :--- | :--- | :---: | :---: |
| 1 | *"This product is not bad at all, I actually like it."* | **2** | Negación (`not bad`) | 0/4 (`0`) | 0/4 (`0`) | Aciertan `logreg`, `linearsvc` (`2`) | **Acierta** (`2`) | 12/17 |
| 2 | *"I have mixed feelings about this keyboard after two weeks."* | **1** | Expresión neutral (`mixed feelings`) | 0/4 (`0`) | 1/4 (`logreg` `1`, resto `0`) | Acierta `linearsvc` (`1`) | Falla (`0`) | 14/17 |
| 3 | *"Great, another charger that breaks in less than a week."* | **0** | Ironía con adjetivo positivo (`Great`) | 1/4 (`logreg`, `svc`, `sgd` `2`) | 4/4 (`0`) | Aciertan `logreg`, `linearsvc`, `nb` (`0`) | **Acierta** (`0`) | 5/17 |
| 4 | *"Not the best design, yet it gets the job done reliably."* | **2** | Concesión inicial (`Not the best... yet`) | 4/4 (`2`) | Falla `logreg` (`1`) | Aciertan `logreg`, `nb`, `sgd` (`2`) | **Acierta** (`2`) | 4/17 |
| 5 | *"I could not be happier with this purchase."* | **2** | Negación enfática (`not be happier`) | Falla `sgd` (`0`) | Falla `sgd` (`0`) | Aciertan `logreg`, `linearsvc`, `nb` (`2`) | **Acierta** (`2`) | 3/17 |
| 6 | *"I expected it to be terrible, but it works great."* | **2** | Contraste (`terrible, but... great`) | Falla `nb` (`0`) | Falla `nb` (`0`) | Aciertan los 4 (`2`) | Falla (`0`) | 3/17 |
| 7 | *"The packaging was damaged, however the device itself is amazing."* | **2** | Adversativa (`damaged, however... amazing`) | Falla `nb` (`0`) | Fallan `nb`, `sgd` (`0`) | `logreg`, `svc` predicen `1`; acierta `sgd` (`2`) | **Acierta** (`2`) | 9/17 |

### Detalle de los casos

1. **Negación directa (*"This product is not bad at all, I actually like it."* — Real: `2`)**  
   Todos los modelos base de BoW y TF-IDF Unigrama predijeron `0` (Negativo). Al separar las palabras, `"not"` y `"bad"` empujan juntas hacia la clase negativa y tapan a `"like"`. En cambio, `logreg` y `linearsvc` con TF-IDF Bigrama capturan `"not bad"` como unidad positiva y aciertan (`2`), al igual que RoBERTa.

2. **Expresión neutral idiomática (*"I have mixed feelings about this keyboard after two weeks."* — Real: `1`)**  
   Siete de los ocho modelos base de BoW y Unigrama la mandaron a `0` (Negativo), porque `"mixed"` y `"feelings"` por separado no alcanzan para activar la clase neutral. `linearsvc` con TF-IDF Bigrama sí reconoce la combinación `"mixed feelings"` y acierta (`1`). RoBERTa, en cambio, falla (`0`) e interpreta la duda como una queja.

3. **Ironía con palabra positiva al inicio (*"Great, another charger that breaks in less than a week."* — Real: `0`)**  
   En BoW, `logreg`, `linearsvc` y `sgd` se dejan engañar por `"Great"` y predicen `2` (Positivo). Con TF-IDF Bigrama (`logreg`, `linearsvc`, `nb`) y con RoBERTa el peso de `"breaks in"` y `"less than"` corrige el sesgo y la oración pasa a `0` (Negativo).

4. **Negación concesiva (*"Not the best design, yet it gets the job done reliably."* — Real: `2`)**  
   `logreg` con TF-IDF Unigrama la baja a `1` (Neutral) por la presencia de `"Not"` al comienzo. Tanto TF-IDF Bigrama (`logreg`, `nb`, `sgd`) como RoBERTa logran ponderar la segunda mitad de la frase y clasifican correctamente `2`.

5. **Negación idiomática positiva (*"I could not be happier with this purchase."* — Real: `2`)**  
   `sgd` con BoW y con Unigrama dispara `0` (Negativo) por la aparición de `"not"`. Los modelos con TF-IDF Bigrama (`logreg`, `linearsvc`, `nb`) y RoBERTa entienden que la negación refuerza a `"happier"` y predicen `2`.

6. **Contraste de expectativa (*"I expected it to be terrible, but it works great."* — Real: `2`)**  
   Naive Bayes falla tanto en BoW como en Unigrama (`0`) porque `"terrible"` tiene un peso negativo demasiado fuerte cuando se evalúa aislado. Al incorporar bigramas (`"works great"`), los cuatro modelos base de TF-IDF Bigrama aciertan (`2`). Acá RoBERTa también falla (`0`), arrastrado por `"terrible"`.

7. **Cláusula adversativa mixta (*"The packaging was damaged, however the device itself is amazing."* — Real: `2`)**  
   En BoW y Unigrama, `nb` y `sgd` caen en `0` por `"damaged"`, e incluso `logreg` y `linearsvc` con Bigrama quedan a mitad de camino en `1` (Neutral). RoBERTa es el que resuelve bien la jerarquía que introduce `"however"` y predice `2`.

## 2. Casos límite extremos (17/17 fallos) y dificultad de la clase Neutral

Al revisar el banco completo, encontramos **dos edge cases máximos donde fallaron los 17 modelos evaluados** (incluidos TF-IDF Bigrama, los modelos optimizados y RoBERTa):

* **Sorpresa positiva con léxico negativo — 17/17 fallos:** *"Never thought a cheap cable could last this long."* (**Real: `2` Positivo — Todos los modelos predijeron `0` Negativo**).  
  Es el caso más extremo del conjunto porque hubo unanimidad total en el error. Ningún enfoque logró captar que *"Never thought... could last this long"* es un elogio por durabilidad inesperada; la combinación de `"Never"` y `"cheap"` arrastró a los 17 modelos a clasificarla como negativa.

* **Reseña neutral puramente descriptiva — 17/17 fallos:** *"It arrived on time and does what it is supposed to do."* (**Real: `1` Neutral — Ningún modelo predijo `1`**).  
  Al no tener adjetivos explícitos de opinión, todos los modelos polarizaron la frase: BoW y Unigrama se inclinaron a `0` (Negativo), mientras que Bigrama (`logreg`, `linearsvc`), los optimizados y RoBERTa la llevaron a `2` (Positivo).

A esto se suman otros dos casos neutrales con negación doble o correlativa donde fallaron **16 de 17 modelos**:
* *"The item works as described, neither great nor terrible."* (**Real: `1`**) — Casi todos los modelos lineales predijeron `0` y RoBERTa predijo `2` (solo acertó `logregopt`).
* *"Not terrible, not amazing either, just a standard charger."* (**Real: `1`**) — 16 modelos cayeron en `0` por la doble aparición de `"Not"` (solo acertó `logreg` con Unigrama).

## 3. Conclusiones del análisis de errores

* **Bigramas resuelven la negación local y las colocaciones:** Pasar de unigramas a TF-IDF Bigrama corrige errores clave donde el significado depende de dos palabras juntas (`"not bad"`, `"mixed feelings"`, `"works great"`), sin necesidad de un modelo pesado.
* **RoBERTa destaca en conectores discursivos, pero no es infalible:** Entiende mejor giros como `"however"` o la ironía de `"Great, another charger..."`, aunque sin *fine-tuning* sobre nuestras reseñas sigue fallando en contrastes (`"expected it to be terrible..."`) y en expresiones de ambivalencia (`"mixed feelings"`).
* **El principal cuello de botella está en la clase Neutral y la pragmática implícita:** Tanto las frases factuales sin carga emocional como las negaciones dobles (`neither...nor`) y los elogios irónicos o indirectos (`"Never thought a cheap cable..."`) siguen confundiendo a todos los enfoques por igual.
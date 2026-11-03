# Resumen de vectorizacion

Generado por `src/vectorize.py`. Particion estratificada: 134,399 reseñas de train y 33,600 de test (test_size=0.2, random_state=42).

| Representacion | Tamaño de vocabulario | Dimensiones | Train | Test | Notas |
|---|---:|---:|---|---|---|
| bow | 12,472 | 12,472 | (134399, 12472) | (33600, 12472) | min_df=5, densidad=0.12049% |
| tfidf_unigram | 12,472 | 12,472 | (134399, 12472) | (33600, 12472) | min_df=5, densidad=0.12049% |
| tfidf_bigram | 71,213 | 71,213 | (134399, 71213) | (33600, 71213) | min_df=5, densidad=0.03143% |
| glove | 400,001 | 100 | (134399, 100) | (33600, 100) | cobertura del vocabulario del corpus: 27,388/43,214 (63.4%); reseñas sin palabras conocidas: 7 |
| hf_pipeline | 30,522 | 768 | (134399, 512) | - | distilbert-base-uncased-finetuned-sst-2-english; tokens por reseña: media 42, max 788; truncadas (>512): 0.02% |

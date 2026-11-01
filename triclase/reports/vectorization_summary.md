# Resumen de vectorizacion (Triclase)

Generado por `triclase/src/vectorize.py`. Particion estratificada en 3 clases: 167,997 reseñas de train y 42,000 de test (test_size=0.2, random_state=42).

| Representacion | Tamaño de vocabulario | Dimensiones | Train | Test | Notas |
|---|---:|---:|---|---|---|
| bow | 13,824 | 13,824 | (167997, 13824) | (42000, 13824) | min_df=5, densidad=0.11018% |
| tfidf_unigram | 13,824 | 13,824 | (167997, 13824) | (42000, 13824) | min_df=5, densidad=0.11018% |
| tfidf_bigram | 89,870 | 89,870 | (167997, 89870) | (42000, 89870) | min_df=5, densidad=0.02574% |
| glove | 400,001 | 100 | (167997, 100) | (42000, 100) | cobertura del vocabulario del corpus: 29,385/48,401 (60.7%); reseñas sin palabras conocidas: 8 |
| hf_pipeline | 50,265 | 768 | (167997, 512) | - | cardiffnlp/twitter-roberta-base-sentiment-latest; tokens por reseña: media 43, max 949; truncadas (>512): 0.02% |

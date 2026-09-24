import time
from pathlib import Path
import requests
import pandas as pd
from tqdm import tqdm

# Configuracion de la API de la Profe
# Preferentemente escribir codigo en ingles, pero si es necesario, se puede escribir en español.

BASE_URL = "https://amazon-reviews-api-g5ae.onrender.com"
REVIEWS_ENDPOINT = f"{BASE_URL}/reviews"
HEALTH_ENDPOINT = f"{BASE_URL}/health"

# Reglas que detallo la Profe:
# 1. La API tiene un limite de 1000 reviews por request. (Para no tirar el server)
# Total de reseñas aprox = 210.000

BATCH_SIZE = 1000

# Ruta de destino del CSV de reseñas extraidas. (No cambiar)
OUTPUT_FILE = Path("data/raw/amazon_reviews_raw.csv")

def check_server_status():
    """
    Verifica que el servidor esté despierto antes de iniciar la descarga.
    Hace ping al endpoint /health hasta recibir un código 200 OK.
    """
    print("Verificando disponibilidad del servidor en Render...")

    # Bucle infinito que no se detiene hasta que el servidor responda positivamente
    while True:
        try:
            # timeout=120 porque un cold-start en Render suele demorar 60-90 segundos
            response = requests.get(HEALTH_ENDPOINT, timeout=200)
            if response.status_code == 200:
                print(">> Servidor activo y listo para utilizar.\n")
                break
        except requests.exceptions.RequestException:
            print(">> El servidor está iniciando... esperando 10 segundos.")
            time.sleep(10)

def extract_all_reviews():
    # Aseguramos que la carpeta data/raw exista en el disco
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Verificamos si el servidor está activo.
    check_server_status()

    all_reviews = []
    offset = 0
    total_records = None
    progress_bar = None

    print("Iniciando proceso de extracción. Puede tardar unos minutos...")

    while True:
        params = {
            "limit": BATCH_SIZE,
            "offset": offset
        }
        try:
            # Petición HTTP GET con parametros de consulta.
            res = requests.get(REVIEWS_ENDPOINT, params=params, timeout=30)

            # Si el servidor responde con error temporal (ej. 500 o 503 por sobrecarga).
            if res.status_code != 200:
                print(f"\n[Aviso] HTTP {res.status_code} en offset {offset}. Reintentando en 5s...")
                time.sleep(5)
                continue
            payload = res.json()
            batch = payload.get("data", [])
            returned = payload.get("returned", len(batch))

            # En la primera iteración inicializamos la barra de progreso con el total real.
            if total_records is None:
                total_records = payload.get("total_matching", 210001)
                progress_bar = tqdm(total=total_records, desc="Descargando reseñas", unit="reviews")

            # Condición de salida 1: La API no devuelve más registros.
            if not batch:
                break

            # extend() agrega los elementos de la lista batch a all_reviews de forma plana.
            all_reviews.extend(batch)
            progress_bar.update(returned)

            # Actualizamos el offset sumando los registros devueltos.
            offset += returned

            # Condición de salida 2: Alcanzamos o superamos el total informado por la API.
            if offset >= total_records:
                break

            # Pausa breve de 5ms para no saturar el socket de redes y evitar errores de conexión. 
            time.sleep(0.05)

        # Captura cualquier error de conexión.
        except requests.exceptions.RequestException as err:
            # Muestra el fallo y espera 5 segundos antes de reintentar la misma petición.
            print(f"\n[Fallo de red temporal] {err}. Reintentando offset {offset} en 5s...")
            time.sleep(5)
    # Si la barra de progreso llegó al final, la cerramos para liberar la consola.      
    if progress_bar:
        progress_bar.close()

    # Devuelve la lista completa de todas las reseñas acumuladas en memoria.
    return all_reviews

def main():
    """
    Función de orquestación general del script:
    Coordina la extracción, la carga a Pandas, el guardado en CSV y el análisis inicial.
    """
    reviews_data = extract_all_reviews()

    print("\nProcesando datos en memoria...")
    df = pd.DataFrame(reviews_data)

    print(f"Guardando archivo en: {OUTPUT_FILE} ...")
    # index=False evita crear una columna extra de indice numerico.
    df.to_csv(OUTPUT_FILE, index=False)

    # Reporte en consola de la extraccion. (Decorado bonito ^_^ ).
    # No comento en detalle que hace cada cosa, deberian saberlo por lo visto en clase.
    print("\n" + "="*50)
    print(">> EXTRACCIÓN COMPLETADA CON ÉXITO")
    print("="*50)
    print(f"Total de registros guardados: {len(df):,}")
    print(f"Columnas disponibles: {list(df.columns)}")
    print("\nPrimeras 3 filas:")
    print(df.head(3))

    if "label" in df.columns:
        # Muestra el conteo de frecuencias de cada clase para evaluar el balanceo del dataset.
        print("\nDistribución de etiquetas (label):")
        print(df["label"].value_counts().sort_index())

# Condicional estándar de Python: asegura que main() solo se ejecute si corremos el script directamente.
if __name__ == "__main__":
    main()
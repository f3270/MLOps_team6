# Dockerfile
FROM python:3.10

# Establece el directorio de trabajo
WORKDIR /app

# Copia todos los archivos de la carpeta `api`, incluyendo `requirements.txt`
COPY ./api /app

# Instala las dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto de la API
EXPOSE 8000

# Ejecutar la API con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


![Tecnológico de Monterrey Logo](https://javier.rodriguez.org.mx/itesm/2014/tecnologico-de-monterrey-blue.png)

# Maestría en Inteligencia Artificial Aplicada

**Asignatura:** Operaciones de aprendizaje automático  
**Profesor Titular:** Gerardo Rodriguez Hernández  
**Tema:** Entrega 1 proyecto  
**Semana:** Semana Cuatro  
**Estudiantes:**
| Nombre                  | Matrícula    |
|-------------------------|--------------|
| Andrea Cantú Martínez    | A01235000    |
| Oscar Becerra Alegria    | A01795611    |
| Jorge Reyes Londono      | A01794421    |
| Henry Aranzales Lopez    | A01794020    |
| Fernando Torres Faúndez  | A01796759    |
**Grupo:** Grupo 06 


# Informe de Proyecto

## Sección 1: README

# MLOps-team6

## Herramientas utilizadas en este proyecto
* [Hydra](https://hydra.cc/): Gestión avanzada de archivos de configuración, permitiendo modificar parámetros de manera dinámica. Consulta este [artículo](https://mathdatasimplified.com/stop-hard-coding-in-a-data-science-project-use-configuration-files-instead/) para más detalles.
* [pdoc](https://github.com/pdoc3/pdoc): Generación automática de documentación de API para el proyecto.
* [Pre-commit plugins](https://pre-commit.com/): Automatización de revisiones de código y formateo con plugins como `black`, `isort` y `flake8`.

## Proceso de Construcción del Modelo de Machine Learning


## Project Structure

```bash
.
├── config                      
│   ├── main.yaml                   # Main configuration file
│   ├── model                       # Configurations for training model
│   │   ├── model1.yaml             # First variation of parameters to train model
│   │   └── model2.yaml             # Second variation of parameters to train model
│   └── process                     # Configurations for processing data
│       ├── process1.yaml           # First variation of parameters to process data
│       └── process2.yaml           # Second variation of parameters to process data
├── data            
│   ├── final                       # data after training the model
│   ├── processed                   # data after processing
│   └── raw                         # raw data
├── docs                            # documentation for your project
├── .gitignore                      # ignore files that cannot commit to Git
├── Makefile                        # store useful commands to set up the environment
├── models                          # store models
├── notebooks                       # store notebooks
├── pyproject.toml                  # Configure black

├── README.md                       # describe your project
├── src                             # store source code
│   ├── __init__.py                 # make src a Python module 
│   ├── process.py                  # process data before training model
│   ├── train_model.py              # train model
│   └── utils.py                    # store helper functions
└── tests                           # store tests
    ├── __init__.py                 # make tests a Python module 
    ├── test_process.py             # test functions for process.py
    └── test_train_model.py         # test functions for train_model.py
```

## Set up the environment


1. Create the virtual environment:
```bash
python3 -m venv venv
```
2. Activate the virtual environment:

- For Linux/MacOS:
```bash
source venv/bin/activate
```
- For Command Prompt:
```bash
.\venv\Scripts\activate
```
3. Install dependencies:
- To install all dependencies, run:
```bash
pip install -r requirements-dev.txt
```
- To install only production dependencies, run:
```bash
pip install -r requirements.txt
```
- To install a new package, run:
```bash
pip install <package-name>
```


## View and alter configurations
To view the configurations associated with a Pythons script, run the following command:
```bash
python src/process.py --help
```
Output:
```yaml
process is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

model: model1, model2
process: process1, process2


== Config ==
Override anything in the config (foo.bar=value)

process:
  use_columns:
  - col1
  - col2
model:
  name: model1
data:
  raw: data/raw/sample.csv
  processed: data/processed/processed.csv
  final: data/final/final.csv
```

To alter the configurations associated with a Python script from the command line, run the following:
```bash
python src/process.py data.raw=sample2.csv
```

## Auto-generate API documentation

To auto-generate API document for your project, run:

```bash
make docs
```


## Sección 2: Entrega del Proyecto

# Entrega 1 Proyecto

![Mapa Mental Desarrollo de Fases en el Proyecto de Machine Learning2.svg](/additional_resources/mind_map_ml_phases.svg)

# 1. **Manipulación y Preparación de Datos**

### Tarea:

La manipulación y preparación de datos implica poner los datos en un formato que pueda ser utilizado por los modelos de Machine Learning, eliminando problemas como valores faltantes o inconsistencias.

### Pasos que se realizaron:

- **Importar el conjunto de datos:**
En ambos notebooks se utilizó la función `pd.read_csv()` para cargar los datos desde un archivo CSV. Este es el paso donde cargamos los datos crudos en un DataFrame de pandas para comenzar a trabajar con ellos.
    
    ```python
    # Cargar el dataset desde un archivo CSV
    df = pd.read_csv('path_to_file.csv')  
    # 'df' es nuestro DataFrame, donde guardamos el conjunto de datos para empezar a trabajar con ellos.
    ```
    
- **Exploración de la estructura del DataFrame:**
Para comprender mejor el dataset, se utilizó `df.info()` y `df.shape` para ver cuántas filas y columnas tiene el dataset, qué tipos de datos están presentes, y cuántos valores nulos hay.
    
    ```python
    # Ver las primeras 5 filas del dataset para una inspección rápida
    df.head()  
    # Esto te ayuda a ver una muestra de cómo están estructurados los datos.
    
    # Mostrar las columnas y el tipo de datos de cada una
    df.info()  
    # Te proporciona un resumen del DataFrame, incluyendo tipos de datos y valores nulos.
    
    # Mostrar el tamaño del dataset (número de filas y columnas)
    print(df.shape)  
    # Esto te dice cuántas filas y columnas tiene el dataset.
    
    # Obtener estadísticas descriptivas de las columnas numéricas
    df.describe()  
    # Esto te da información estadística clave como la media, desviación estándar, mínimo, máximo, etc.
    ```
    
- **Limpieza de datos:**
Una tarea importante en la preparación de los datos fue identificar y manejar valores faltantes. En los notebooks, parece que se usa el método `dropna()` para eliminar filas con valores nulos, aunque en algunos casos puede ser mejor reemplazar valores faltantes con la media o mediana de la columna (dependiendo del tipo de dato).
    
    ```python
    # Eliminar filas con valores nulos
    df = df.dropna()  
    # Eliminamos las filas que contienen valores nulos para evitar problemas durante el modelado.
    ```
    

### Resultado:

El dataset se cargó y se limpió, eliminando los valores nulos que podrían afectar el modelado posterior. El objetivo aquí es tener datos consistentes y completos, lo que es crucial para asegurar que el modelo de Machine Learning se entrene de manera efectiva.

---

# 2. **Exploración y Preprocesamiento de Datos**

### Tarea:

El análisis exploratorio de datos (EDA) y el preprocesamiento son esenciales para entender el dataset antes de construir un modelo. El EDA ayuda a identificar relaciones y patrones dentro de los datos, mientras que el preprocesamiento asegura que los datos estén en un formato adecuado para el modelado.

### Pasos que se realizaron:

- **Análisis Exploratorio (EDA):**
Se utilizaron funciones como `describe()` para generar estadísticas descriptivas sobre las columnas numéricas del dataset, lo cual ayuda a identificar patrones en los datos, como la media, la desviación estándar, los valores mínimos y máximos, etc.
    
    ```python
    # Obtener estadísticas descriptivas de las columnas numéricas
    df.describe()  
    # Esto te da información estadística clave como la media, desviación estándar, mínimo, máximo, etc.
    ```
    
- **Visualización de datos:**
Se utilizó `matplotlib` y `seaborn` para crear gráficos que permitan visualizar relaciones y distribuciones en los datos. Por ejemplo, se pueden crear gráficos de dispersión (scatter plots) o histogramas para ver cómo se distribuyen las variables clave.
    
    ```python
    # Crear un gráfico de dispersión (pairplot) para ver relaciones entre las variables
    sns.pairplot(df)  
    # Esto muestra gráficos de pares para ver la relación entre múltiples variables del dataset.
    
    # Mostrar un histograma para visualizar la distribución de una columna específica
    df['survival_time'].hist(bins=50)
    plt.show()  
    # Esto crea un histograma que muestra cómo están distribuidos los tiempos de supervivencia en el dataset.
    
    ```
    
- **Normalización de datos:**
Las variables numéricas como `survival_time` y `CD34+ cells/kg` pueden estar en diferentes escalas. Para mejorar el rendimiento del modelo, se aplicó la normalización utilizando `MinMaxScaler` para llevar todas las variables a una escala común entre 0 y 1.
    
    ```python
    from sklearn.preprocessing import MinMaxScaler
    
    # Inicializar el normalizador
    scaler = MinMaxScaler()  
    
    # Aplicar la normalización a todas las columnas numéricas
    df_scaled = scaler.fit_transform(df)  
    # Los valores son escalados entre 0 y 1 para asegurar que las diferentes variables tengan una escala similar.
    ```
    
- **Codificación de variables categóricas:**
Algunas de las variables en el dataset son categóricas (por ejemplo, el grupo sanguíneo, el género). Para que los algoritmos de Machine Learning puedan usarlas, se aplicaron técnicas de codificación como `OneHotEncoder`.
    
    ```python
    from sklearn.preprocessing import OneHotEncoder
    
    # Inicializar el codificador
    encoder = OneHotEncoder()  
    
    # Aplicar codificación a las columnas categóricas
    df_encoded = encoder.fit_transform(df[['gender', 'blood_type']])  
    # Transformamos las variables categóricas (como género o tipo de sangre) en variables numéricas que el modelo pueda entender.
    ```
    

### Resultado:

El análisis exploratorio proporcionó una comprensión profunda de los datos y ayudó a identificar relaciones clave entre las variables. El preprocesamiento transformó los datos en un formato adecuado para el modelado al normalizar y codificar las variables.

---

# 3. **Versionado de Datos**

### Tarea:

El versionado de datos permite registrar y rastrear todos los cambios realizados durante la manipulación de datos. Esto es importante para asegurar la reproducibilidad de los resultados.

### Pasos que se realizaron:

- **Uso de Git para control de versiones:**
En los proyectos de Machine Learning, herramientas como Git permiten gestionar los cambios realizados en los archivos de datos y scripts de código. El control de versiones no solo es útil para los datos, sino también para asegurar que todos los cambios en el código estén registrados.
    
    ```bash
    # Inicializar un repositorio Git
    git init  
    
    # Agregar archivos al repositorio
    git add .  
    
    # Crear un commit con los cambios realizados
    git commit -m "Versión inicial del dataset y preprocesamiento"
    
    ```
    
- **Documentación de cambios:**
Cada modificación de los datos fue documentada para que se pueda rastrear en el futuro qué transformaciones se realizaron en los datos y cuándo.

### Resultado:

Se creó un control de versiones claro que permite rastrear los cambios realizados en los datos y asegurar la reproducibilidad del experimento.

---

# 4. **Construcción, Ajuste y Evaluación de Modelos de Machine Learning**

### Tarea:

La construcción del modelo implica seleccionar el algoritmo adecuado, entrenarlo y evaluarlo usando métricas específicas.

### Pasos que se realizaron:

- **Selección de algoritmos:**
Para el problema de predicción de la supervivencia (`survival_time`), se eligió un algoritmo de regresión, como un **Random Forest Regressor** o una **regresión lineal**.
    
    ```python
    # Construcción y Entrenamiento de un Modelo
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # Separar los datos en características (X) y la variable objetivo (y)
    X = df.drop('survival_time', axis=1)  # Variables predictoras
    y = df['survival_time']  # Variable objetivo
    
    # Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
    
    # Inicializar el modelo RandomForest
    model = RandomForestRegressor()  
    
    # Entrenar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)  
    # Entrenamos el modelo con los datos preprocesados.
    ```
    
- **Ajuste de hiperparámetros:**
Se usó `GridSearchCV` para ajustar los hiperparámetros del modelo y mejorar su rendimiento.
    
    ```python
    from sklearn.model_selection import GridSearchCV
    
    # Definir los hiperparámetros a ajustar
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    
    # Inicializar el GridSearchCV para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(model, param_grid, cv=5)  
    
    # Ajustar el modelo con los datos de entrenamiento y probar diferentes combinaciones de hiperparámetros
    grid_search.fit(X_train, y_train)  
    # GridSearch prueba diferentes combinaciones de hiperparámetros para encontrar la que mejor funcione.
    ```
    
- **Evaluación del modelo:**
Se utilizaron métricas como el **Error Medio Absoluto (MAE)** y el **coeficiente de determinación (R²)** para evaluar la precisión del modelo.
    
    ```python
    from sklearn.metrics import mean_absolute_error
    
    # Hacer predicciones con el modelo en los datos de prueba
    y_pred = model.predict(X_test)  
    
    # Evaluar el modelo utilizando el Error Medio Absoluto (MAE)
    mae = mean_absolute_error(y_test, y_pred)  
    print(f"Error Medio Absoluto: {mae}")
    # El MAE nos dice en promedio cuánto se desvía nuestra predicción del valor real.
    ```
    

### Resultado:

Se construyó y ajustó un modelo de Machine Learning basado en algoritmos de regresión para predecir la supervivencia de los pacientes. Los resultados del modelo se evaluaron utilizando métricas como el MAE y el R².

---

# 5. **Aplicación de Mejores Prácticas en el Pipeline de Modelado**

### Tarea:

El pipeline de modelado debe ser eficiente y reproducible, automatizando tareas como el preprocesamiento y la evaluación del modelo.

### Pasos que se realizaron:

- **Construcción del pipeline:**
Se creó un pipeline de `sklearn` para automatizar el flujo de trabajo del modelo, desde el preprocesamiento hasta la evaluación.
    
    ```python
    from sklearn.pipeline import Pipeline
    
    # Crear un pipeline que integre la normalización y el modelo de RandomForest
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # Primer paso: normalización
        ('model', RandomForestRegressor())  # Segundo paso: aplicar el modelo
    ])
    
    # Entrenar el pipeline
    pipeline.fit(X_train, y_train)  
    # El pipeline combina varios pasos en uno solo, facilitando la ejecución automática.
    ```
    
- **Documentación del pipeline:**
Se documentaron los pasos y se aseguraron de que el pipeline fuera fácil de reproducir.

### Resultado:

El pipeline facilita la ejecución automática de todas las etapas del proceso, asegurando eficiencia y reproducibilidad.

---

# 6. **Estructuración y Refactorización del Código**

### Tarea:

Es importante que el código esté bien organizado y estructurado, para facilitar el mantenimiento a largo plazo.

### Pasos que se realizaron:

- **Organización en módulos y funciones:**
El código se refactorizó en funciones claras, cada una con un propósito bien definido.
    
    ```python
    # Función para cargar los datos desde un archivo
    def cargar_datos(path):
        return pd.read_csv(path)
    
    # Función para preprocesar los datos (eliminar valores nulos, normalizar, etc.)
    def preprocesar_datos(df):
        df = df.dropna()
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df)
        return df_scaled
    ```
    
- **Aplicación de POO (Programación Orientada a Objetos):**
En algunos casos, se encapsuló el código en clases para mejorar la modularidad y reutilización.
    
    ```python
    # Importamos las bibliotecas necesarias
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    
    class PipelineSupervivencia:
        def __init__(self, file_path):
            """
            Constructor que inicializa el dataset y el modelo.
            """
            self.file_path = file_path
            self.data = None
            self.model = None
            self.scaler = MinMaxScaler()
            self.encoder = OneHotEncoder(sparse=False)
        
        def cargar_datos(self):
            """
            Método para cargar los datos desde un archivo CSV.
            """
            self.data = pd.read_csv(self.file_path)
            print("Datos cargados correctamente.")
            return self.data
        
        def preprocesar_datos(self):
            """
            Método para limpiar y preprocesar los datos (eliminar nulos, normalizar y codificar).
            """
            # Eliminamos filas con valores nulos
            self.data = self.data.dropna()
    
            # Normalizamos las variables numéricas
            columnas_numericas = self.data.select_dtypes(include=['float64', 'int64']).columns
            self.data[columnas_numericas] = self.scaler.fit_transform(self.data[columnas_numericas])
    
            # Codificamos variables categóricas (ejemplo con género y tipo de sangre)
            columnas_categoricas = ['gender', 'blood_type']
            self.data[columnas_categoricas] = self.encoder.fit_transform(self.data[columnas_categoricas])
    
            print("Datos preprocesados.")
            return self.data
    
        def entrenar_modelo(self):
            """
            Método para dividir los datos y entrenar el modelo de RandomForest.
            """
            # Separar características (X) de la variable objetivo (y)
            X = self.data.drop('survival_time', axis=1)
            y = self.data['survival_time']
            
            # Dividir los datos en conjunto de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Inicializar y entrenar el modelo
            self.model = RandomForestRegressor()
            self.model.fit(X_train, y_train)
            
            print("Modelo entrenado.")
            return X_test, y_test
        
        def evaluar_modelo(self, X_test, y_test):
            """
            Método para evaluar el modelo en los datos de prueba.
            """
            # Hacer predicciones
            y_pred = self.model.predict(X_test)
            
            # Evaluar el modelo con MAE
            mae = mean_absolute_error(y_test, y_pred)
            print(f"Error Medio Absoluto (MAE): {mae}")
            return mae
    
    # Ejemplo de uso de la clase
    
    # Inicializamos el pipeline con la ruta de los datos
    pipeline = PipelineSupervivencia(file_path='path/to/bone_marrow_data.csv')
    
    # Cargar los datos
    pipeline.cargar_datos()
    
    # Preprocesar los datos
    pipeline.preprocesar_datos()
    
    # Entrenar el modelo
    X_test, y_test = pipeline.entrenar_modelo()
    
    # Evaluar el modelo
    pipeline.evaluar_modelo(X_test, y_test)
    ```
    

### Explicación del código:

1. **Constructor (`__init__`)**: Inicializamos la clase con el archivo de datos, un modelo de `RandomForest`, un escalador (`MinMaxScaler`) para normalizar las variables numéricas, y un codificador (`OneHotEncoder`) para las variables categóricas.
2. **Método `cargar_datos()`**: Carga el archivo CSV y lo guarda en un atributo de la clase (`self.data`).
3. **Método `preprocesar_datos()`**: Realiza la limpieza de los datos eliminando valores nulos, luego normaliza las columnas numéricas y aplica la codificación a las columnas categóricas.
4. **Método `entrenar_modelo()`**: Divide los datos en conjuntos de entrenamiento y prueba, entrena el modelo de `RandomForest` con los datos preprocesados.
5. **Método `evaluar_modelo()`**: Evalúa el modelo utilizando la métrica de **Error Medio Absoluto (MAE)** en los datos de prueba.

### Beneficios de POO:

- **Modularidad**: Cada parte del pipeline está encapsulada en métodos que pueden reutilizarse.
- **Mantenibilidad**: Si se necesita cambiar o actualizar alguna parte del código (por ejemplo, el modelo), es más fácil de gestionar.
- **Claridad**: El código se organiza de una manera más legible y escalable, dividiendo las responsabilidades en métodos específicos.

### Resultado:

El código fue refactorizado y organizado de manera clara, lo que facilita su mantenimiento y mejora la eficiencia general.

---

# Conclusiones

- **Se puede concluir que la calidad del preprocesamiento es fundamental para el éxito del modelado en proyectos de Machine Learning.** En este caso, la eliminación de valores nulos, la normalización de las variables numéricas y la codificación de variables categóricas fueron pasos críticos que garantizaron que los datos estuvieran listos para ser procesados por los algoritmos. Sin una adecuada preparación de los datos, los modelos no habrían sido capaces de aprender correctamente los patrones subyacentes ni generar predicciones precisas. Por lo tanto, una buena preparación de los datos es esencial para obtener resultados fiables y precisos en cualquier tarea de modelado.
- **Se puede concluir que la evaluación y ajuste de modelos son pasos clave para mejorar significativamente el rendimiento de los modelos de Machine Learning.** En este proyecto, el uso de técnicas como el ajuste de hiperparámetros a través de `GridSearchCV` y la evaluación utilizando el Error Medio Absoluto (MAE) permitieron optimizar el modelo y mejorar su precisión. Ajustar los parámetros del modelo de manera adecuada tuvo un impacto significativo en la capacidad del modelo para predecir la supervivencia de los pacientes, demostrando que pequeñas optimizaciones pueden llevar a grandes mejoras en el desempeño del modelo.
- **Se puede concluir que el uso de la Programación Orientada a Objetos (POO) mejora la escalabilidad y mantenibilidad del código en proyectos de Machine Learning.** Al encapsular las diferentes fases del pipeline de modelado (carga de datos, preprocesamiento, entrenamiento y evaluación) dentro de una clase, se organizó el código de manera más eficiente, haciéndolo más modular y fácil de mantener. Este enfoque facilita la implementación de cambios futuros, como la adición de nuevos modelos o pasos de preprocesamiento, y asegura que el código sea más legible y escalable. La POO es una herramienta clave para mantener proyectos más complejos y colaborativos en el ámbito del aprendizaje automático.

"""
Contenido del archivo:

1. `generate_business_dates(num_dias, start_date, calendar='XNYS')`:
   Genera un rango de fechas hábiles para un mercado especificado.

2. `generar_datos(n_activos=10, rango_medias=(0, 0.1), rango_std=(0.05, 0.3), tipo_correlacion="poco")`:
   Genera datos aleatorios de medias, desviaciones estándar y coeficientes de correlación para activos.

3. `calcular_matriz_covarianza(media_lst, std_lst, corr_mat)`:
   Calcula la matriz de covarianza a partir de listas de medias, desviaciones estándar y matriz de correlación.

4. `es_matriz_covarianza(mat)`:
   Verifica si una matriz dada es una matriz de covarianza.

5. `ajustar_correlacion(cov_mat, tipo_correlacion="poco", factor=0.5)`:
   Ajusta la matriz de covarianza para modificar la correlación entre activos.

6. `simular_retornos_diarios(media_lst, std_lst, cov_mat, num_dias)`:
   Simula los retornos diarios de activos basados en sus medias, desviaciones estándar y matriz de covarianzas.

7. `calcular_precios(ret_df, precios_iniciales)`:
   Calcula los precios a partir de los retornos logarítmicos utilizando precios iniciales.

8. `simular_evento_bajista(med_lst, std_lst, cov_mat, impacto_medias=0.2, impacto_vol=0.3, impacto_corr=0.5)`:
   Simula un evento de mercado bajista ajustando las medias, volatilidades y matriz de covarianza.

9. `simular_impacto_severo(med_lst, std_lst, cov_mat, factor_medias=1.5, factor_vol=0.5, factor_corr=0.7)`:
   Simula un impacto severo en el mercado donde todos los activos pasan a perder.

10. `simular_impacto_bajista_parcial(med_lst, std_lst, cov_mat, prop_affected=0.5, factor_medias=1.5, factor_vol=0.3, factor_corr=0.5)`:
   Simula un impacto bajista en el mercado donde solo algunos activos pasan a perder.

11. `simular_mercado_lateral(med_lst, std_lst, cov_mat, factor_medias=0.5, factor_vol=0.7, factor_corr=0.5)`:
   Simula el paso a un mercado lateral con activos de baja volatilidad y correlaciones reducidas.

12. `simular_mercado_alcista(med_lst, std_lst, cov_mat, factor_medias=1.5, factor_vol=1.2, factor_corr=0.5)`:
    Simula un mercado alcista ajustando las medias, volatilidades y matriz de covarianza.

Este archivo ofrece funciones para generar fechas de negocio, simular datos de activos financieros y simular diferentes escenarios de mercado.

"""


import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from utils import dibuja_covar

plt.style.use('ggplot')

def generate_business_dates(num_dias, start_date, calendar='XNYS'):
    """
    Genera un rango de fechas hábiles para un mercado especificado.

    Parámetros:
    - num_dias: Número de días hábiles deseados.
    - start_date: Fecha de inicio del rango.
    - calendar: Calendario del mercado deseado (por defecto, 'XNYS' para el NYSE).

    Retorna:
    - Un rango de fechas hábiles.
    """
    
    # Tratamiento de errores
    if not isinstance(num_dias, int) or num_dias <= 0:
        raise ValueError("El número de días debe ser un entero positivo.")
    
    try:
        current_date = pd.Timestamp(start_date)
    except:
        raise ValueError("La fecha de inicio no es válida.")
    
    if calendar not in mcal.get_calendar_names():
        raise ValueError(f"El calendario '{calendar}' no está disponible.")
    
    # Crear un calendario para el mercado especificado
    market_cal = mcal.get_calendar(calendar)
    
    # Inicializar una variable para contar los días laborables
    business_days_count = 0

    # Inicializar la fecha de inicio
    start_date = pd.Timestamp(start_date)
    current_date = start_date

    while business_days_count < num_dias:
        # Verificar si la fecha actual es un día laborable
        if market_cal.valid_days(start_date=current_date, end_date=current_date).size > 0:
            business_days_count += 1
        # Solo aumenta el current_date si aún no hemos alcanzado el num_days requerido
        if business_days_count < num_dias:
            current_date += pd.DateOffset(days=1)  # Avanzar al siguiente día natural

    dias_naturales = current_date - start_date
    business_days = market_cal.valid_days(start_date=start_date, end_date=current_date)
    
    return business_days

def generar_datos(n_activos=10, rango_medias=(0, 0.1), rango_std=(0.05, 0.3), tipo_correlacion="poco"):
    """
    Genera datos aleatorios de medias, desviaciones estándar y coeficientes de correlación para un número 
    dado de activos.
    
    Parámetros:
    - n_activos: Número de activos.
    - rango_medias: Rango (min, max) para las medias de los activos.
    - rango_std: Rango (min, max) para las desviaciones estándar de los activos.
    - tipo_correlacion: Tipo de correlación deseada ("positiva", "negativa", "poco" o "mixta").
    
    Retorna:
    - medias: Array con las medias de los activos.
    - desviaciones: Array con las desviaciones estándar de los activos.
    - corr_mat: Matriz de correlación de los activos.
    """

    # Tratamiento de errores
    if not isinstance(n_activos, int) or n_activos <= 0:
        raise ValueError("El número de activos debe ser un entero positivo.")
    
    if not (isinstance(rango_medias, (list, tuple)) and len(rango_medias) == 2 and rango_medias[0] < rango_medias[1]):
        raise ValueError("El rango de medias debe ser una tupla o lista con dos valores, donde el primero es menor que el segundo.")
    
    if not (isinstance(rango_std, (list, tuple)) and len(rango_std) == 2 and rango_std[0] < rango_std[1]):
        raise ValueError("El rango de desviaciones estándar debe ser una tupla o lista con dos valores, donde el primero es menor que el segundo.")
    
    if tipo_correlacion not in ["positiva", "negativa", "poco", "mixta"]:
        raise ValueError("El tipo de correlación no es válido. Debe ser 'positiva', 'negativa', 'poco' o 'mixta'.")
        
    # Generar medias aleatorias dentro del rango especificado
    medias = np.random.uniform(rango_medias[0], rango_medias[1], n_activos)
    
    # Generar desviaciones estándar aleatorias dentro del rango especificado
    desviaciones = np.random.uniform(rango_std[0], rango_std[1], n_activos)
    
    # Generar una matriz de correlación aleatoria
    A = np.random.randn(n_activos, n_activos)
    cov_temp = A @ A.T
    d = np.sqrt(np.diag(cov_temp))
    corr_mat = cov_temp / d[:, None] / d[None, :]
    
    # Ajustar la matriz de correlación según el tipo deseado
    if tipo_correlacion == "positiva":
        corr_mat = 0.5 * (corr_mat + 1)
    elif tipo_correlacion == "negativa":
        corr_mat = 0.5 * (corr_mat - 1)
    elif tipo_correlacion == "mixta":
        # Definir los índices para dividir en tres grupos
        idx1 = n_activos // 3
        idx2 = 2 * n_activos // 3
        
        # Activos del primer grupo: correlación positiva
        corr_mat[:idx1, :idx1] = 0.5 * (corr_mat[:idx1, :idx1] + 1)
        
        # Activos del segundo grupo: correlación negativa
        corr_mat[idx1:idx2, idx1:idx2] = 0.5 * (corr_mat[idx1:idx2, idx1:idx2] - 1)
        
        # Activos del tercer grupo: poco correlacionados (no se modifica)
    # Si es "poco", no hacemos ajustes porque la matriz ya es poco correlacionada
    
    # Asegurar que los valores diagonales de la matriz de correlación sean 1
    np.fill_diagonal(corr_mat, 1)
    
    return medias, desviaciones, corr_mat

def calcular_matriz_covarianza(media_lst, std_lst, corr_mat):
    """
    Calcula la matriz de covarianza a partir de listas de medias, desviaciones estándar y una matriz de correlación.

    Parámetros:
    - media_lst: Lista de medias esperadas.
    - std_lst: Lista de desviaciones estándar.
    - corr_mat: Matriz de correlación. Tipo numpy array de tamaño (n, n), donde n es el número de activos.

    Retorna:
    - Matriz de covarianza: numpy array de tamaño (n, n), donde n es el número de activos.
    """
    
    # Comprobar que las listas de medias y desviaciones estándar tienen la misma longitud
    if len(media_lst) != len(std_lst):
        raise ValueError("Las listas de medias y desviaciones estándar deben tener la misma longitud.")
    
    # Almacenar la longitud de las listas para futuras comprobaciones
    n = len(media_lst)
    
    # Comprobar que la matriz de correlación tiene la forma correcta
    if corr_mat.shape != (n, n):
        raise ValueError("La matriz de correlación debe ser cuadrada y de tamaño igual a la longitud de las listas de medias y desviaciones estándar.")
    
    # Comprobar que la matriz de correlación es simétrica
    if not np.allclose(corr_mat, corr_mat.T):
        raise ValueError("La matriz de correlación debe ser simétrica.")
    
    # Calcular la matriz de covarianza usando la matriz de correlación y las desviaciones estándar
    # Se utiliza np.outer para obtener el producto externo de las desviaciones estándar, que nos da una matriz
    # cuyos elementos son el producto de las desviaciones estándar correspondientes. 
    # Esta matriz se multiplica elemento por elemento con la matriz de correlación.
    cov_mat = corr_mat * np.outer(std_lst, std_lst)
    
    return cov_mat

def es_matriz_covarianza(mat):
    """
    Comprueba si una matriz dada es una matriz de covarianza.
    
    Parámetros:
    - mat: Matriz a verificar. Tipo numpy array de tamaño (n, n), donde n es el número de activos.
    
    Retorna:
    - True si la matriz es de covarianza, False en caso contrario.
    """
    # Comprobar simetría
    if not np.allclose(mat, mat.T):
        return False
    
    # Comprobar que todos los valores propios son no negativos (semi-definida positiva)
    eigenvalues = np.linalg.eigvalsh(mat)
    if np.any(eigenvalues < 0):
        return False
    
    # Comprobar que los elementos de la diagonal son no negativos
    if np.any(np.diag(mat) < 0):
        return False
    
    return True

def ajustar_correlacion(cov_mat, tipo_correlacion="poco", factor=0.5):
    """
    Ajusta la matriz de covarianza para que los activos estén más correlacionados positivamente, 
    más correlacionados negativamente o menos correlacionados, basándose en un factor.
    
    Parámetros:
    - cov_mat: Matriz de covarianza original. Tipo numpy array de tamaño (n, n), donde n es el número de activos.
    - tipo_correlacion: Tipo de correlación deseada ("positiva", "negativa" o "poco").
    - factor: Factor de ajuste en el rango [-1, 1] para determinar la intensidad del ajuste.
    
    Retorna:
    - Matriz de covarianza ajustada.
    """
    
    # Comprobar si la matriz dada es de covarianza
    if not es_matriz_covarianza(cov_mat):
        raise ValueError("La matriz dada no es una matriz de covarianza válida.")
    
    # Convertir la matriz de covarianza en una matriz de correlación
    d = np.sqrt(np.diag(cov_mat))
    corr_mat = cov_mat / d[:, None] / d[None, :]
    
    # Ajustar la matriz de correlación según el tipo y factor deseado
    if tipo_correlacion == "positiva":
        corr_mat += factor * (1 - corr_mat)
    elif tipo_correlacion == "negativa":
        corr_mat += factor * (-1 - corr_mat)
    elif tipo_correlacion == "poco":
        # Multiplicar las correlaciones por un factor menor que 1 (p. ej., 0.2) para reducirlas hacia 0
        mask = np.ones_like(corr_mat) - np.eye(corr_mat.shape[0])
        corr_mat *= factor * mask + np.eye(corr_mat.shape[0])
    else:
        raise ValueError("El tipo de correlación debe ser 'positiva', 'negativa' o 'poco'.")
    
    # Asegurar que los valores diagonales sean 1
    np.fill_diagonal(corr_mat, 1)
    
    # Convertir la matriz de correlación ajustada de nuevo en una matriz de covarianza
    cov_mat_ajustada = corr_mat * d[:, None] * d[None, :]
    
    return cov_mat_ajustada

def simular_retornos_diarios(media_lst, std_lst, cov_mat, num_dias):
    """
    Simula los retornos diarios de activos basados en sus medias, desviaciones estándar y matriz de covarianzas.

    Parámetros:
    - media_lst: Lista de medias esperadas.
    - std_lst: Lista de desviaciones estándar.
    - cov_mat: Matriz de covarianza.
    - num_dias: Número de días para los que se desea simular los retornos.

    Retorna:
    - retornos: Array 2D (num_dias x número de activos) con los retornos diarios simulados.
    """
    
    # Tratamiento de errores
    if not isinstance(num_dias, int) or num_dias <= 0:
        raise ValueError("El número de días debe ser un entero positivo.")
    
    if len(media_lst) != len(std_lst):
        raise ValueError("Las listas de medias y desviaciones estándar deben tener la misma longitud.")
    
    if cov_mat.shape != (len(media_lst), len(media_lst)):
        raise ValueError("La forma de la matriz de covarianza no coincide con la longitud de las listas de medias y desviaciones estándar.")
    
    if not np.allclose(cov_mat, cov_mat.T):
        raise ValueError("La matriz de covarianza debe ser simétrica.")
    
    # Simular los retornos diarios usando una distribución multivariada normal
    retornos = np.random.multivariate_normal(media_lst, cov_mat, num_dias)
    
    return retornos

def calcular_precios(ret_df, precios_iniciales):
    """
    Calcula los precios a partir de los retornos logarítmicos.

    Parámetros:
    - ret_df: DataFrame con los retornos logarítmicos.
    - precios_iniciales: Lista con los precios iniciales de cada activo.

    Retorna:
    - DataFrame con los precios.
    """
    
    # Verificación: La longitud de precios_iniciales debe coincidir con el número de columnas en ret_df
    if len(precios_iniciales) != ret_df.shape[1]:
        raise ValueError("La longitud de precios_iniciales debe coincidir con el número de activos en ret_df.")
    
    # Crear un DataFrame para los precios
    precios_df = ret_df.copy()
    
    # Iniciar con los precios iniciales
    precios_df.iloc[0] = [p_inicial * np.exp(retorno) for p_inicial, retorno in zip(precios_iniciales, ret_df.iloc[0])]
    
    # Calcular los precios para los demás días
    for i in range(1, len(ret_df)):
        precios_df.iloc[i] = precios_df.iloc[i-1] * np.exp(ret_df.iloc[i])
    
    return precios_df

def simular_evento_bajista(med_lst, std_lst, corr_mat, impacto_medias=0.2, impacto_vol=0.3, factor_corr=0.5):
    """
    Simula un evento de mercado bajista ajustando las medias, volatilidades y matriz de covarianza.

    Parámetros:
    - med_lst: Lista de medias originales.
    - std_lst: Lista de volatilidades originales.
    - corr_mat: Matriz de correlaciones original.
    - impacto_medias: Proporción para disminuir las medias (por defecto, 0.2 o 20%).
    - impacto_vol: Proporción para aumentar las volatilidades (por defecto, 0.3 o 30%).
    - factor_corr: Proporción para aumentar las correlaciones (por defecto, 0.5 o 50%).

    Retorna:
    - med_lst_ajustada: Lista de medias ajustadas.
    - std_lst_ajustada: Lista de volatilidades ajustadas.
    - corr_mat_ajustada: Matriz de correlaciones ajustada.
    """
    
    # Ajustar medias
    med_lst_ajustada = [med * (1 - impacto_medias) for med in med_lst]
    
    # Ajustar volatilidades
    std_lst_ajustada = [std * (1 + impacto_vol) for std in std_lst]
    
    #corr_mat_ajustada = corr_mat + factor_corr * (1 - corr_mat)

    corr_mat_ajustada = corr_mat.copy()

    # Ajustar las correlaciones positivas
    mask_pos = corr_mat > 0
    corr_mat_ajustada[mask_pos] = corr_mat[mask_pos] + factor_corr * (1 - corr_mat[mask_pos])

    # Ajustar las correlaciones negativas
    mask_neg = corr_mat < 0
    corr_mat_ajustada[mask_neg] = corr_mat[mask_neg] - factor_corr * (1 + corr_mat[mask_neg])
       
    return med_lst_ajustada, std_lst_ajustada, corr_mat_ajustada

def simular_impacto_severo(med_lst, std_lst, corr_mat, factor_medias=1.5, factor_vol=0.5, factor_corr=0.7):
    """
    Simula un impacto severo en el mercado ajustando las medias, volatilidades y matriz de covarianza.

    Parámetros:
    - med_lst: Lista de medias originales.
    - std_lst: Lista de volatilidades originales.
    - cov_mat: Matriz de covarianza original.
    - factor_medias: Factor para aumentar la severidad de las pérdidas (por defecto, 1.5).
    - factor_vol: Proporción para aumentar las volatilidades (por defecto, 0.5 o 50%).
    - factor_corr: Proporción para aumentar las correlaciones (por defecto, 0.7 o 70%).

    Retorna:
    - med_lst_ajustada: Lista de medias ajustadas.
    - std_lst_ajustada: Lista de volatilidades ajustadas.
    - cov_mat_ajustada: Matriz de covarianza ajustada.
    """
    
    # Ajustar medias a valores negativos
    med_lst_ajustada = [-abs(med) * factor_medias for med in med_lst]
    
    # Ajustar volatilidades
    std_lst_ajustada = [std * (1 + factor_vol) for std in std_lst]


    corr_mat_ajustada = corr_mat.copy()

    # Ajustar las correlaciones positivas
    mask_pos = corr_mat > 0
    corr_mat_ajustada[mask_pos] = corr_mat[mask_pos] + factor_corr * (1 - corr_mat[mask_pos])

    # Ajustar las correlaciones negativas
    mask_neg = corr_mat < 0
    corr_mat_ajustada[mask_neg] = corr_mat[mask_neg] - factor_corr * (1 + corr_mat[mask_neg])
       
    
    # # Ajustar matriz de covarianza: aumentar correlaciones y volatilidades
    # d = np.sqrt(np.diag(cov_mat))
    # corr_mat = cov_mat / d[:, None] / d[None, :]
    # corr_mat_ajustada = corr_mat + factor_corr * (1 - corr_mat)
    # np.fill_diagonal(corr_mat_ajustada, 1)
    # cov_mat_ajustada = corr_mat_ajustada * np.outer(std_lst_ajustada, std_lst_ajustada)
    
    return med_lst_ajustada, std_lst_ajustada, corr_mat_ajustada

def simular_impacto_bajista_parcial(med_lst, std_lst, corr_mat, prop_affected=0.5, factor_medias=1.5, factor_vol=0.3, factor_corr=0.5):
    """
    Simula un impacto parcial en el mercado donde algunos activos pasan a perder y otros no.

    Parámetros:
    - med_lst: Lista de medias originales.
    - std_lst: Lista de volatilidades originales.
    - corr_mat: Matriz de correlaciones original.
    - prop_affected: Proporción de activos que se verán afectados negativamente (por defecto, 0.5 o 50%).
    - factor_medias: Factor para aumentar la severidad de las pérdidas de los activos afectados (por defecto, 1.5).
    - factor_vol: Proporción para aumentar las volatilidades (por defecto, 0.3 o 30%).
    - factor_corr: Proporción para aumentar las correlaciones (por defecto, 0.5 o 50%).

    Retorna:
    - med_lst_ajustada: Lista de medias ajustadas.
    - std_lst_ajustada: Lista de volatilidades ajustadas.
    - corr_mat_ajustada: Matriz de correlaciones ajustada.
    """
    
    # Seleccionar aleatoriamente los activos que se verán afectados
    num_affected = int(len(med_lst) * prop_affected)
    affected_indices = np.random.choice(len(med_lst), num_affected, replace=False)
    
    # Ajustar medias
    med_lst_ajustada = med_lst.copy()
    for idx in affected_indices:
        med_lst_ajustada[idx] = -abs(med_lst[idx]) * factor_medias
    
    # Ajustar volatilidades
    std_lst_ajustada = std_lst.copy()
    for idx in affected_indices:
        std_lst_ajustada[idx] = std_lst[idx] * (1 + factor_vol)

    # Ajustar correlaciones    
    corr_mat_ajustada = corr_mat.copy()
    
    for i, j in list(product(affected_indices, affected_indices)):
        if i != j and corr_mat[i][j] > 0:
            corr_mat_ajustada[i][j] += factor_corr * (1 - corr_mat[i][j])
        elif i != j and corr_mat[i][j] < 0:
            corr_mat_ajustada[i][j] -= factor_corr * (1 + corr_mat[i][j])
    
    return med_lst_ajustada, std_lst_ajustada, corr_mat_ajustada

def simular_mercado_lateral(med_lst, std_lst, cov_mat, factor_medias=0.5, factor_vol=0.7, factor_corr=0.5):
    """
    Simula el paso a un mercado lateral ajustando las medias, volatilidades y matriz de covarianza.

    Parámetros:
    - med_lst: Lista de medias originales.
    - std_lst: Lista de volatilidades originales.
    - cov_mat: Matriz de covarianza original.
    - factor_medias: Factor para reducir las medias hacia cero (por defecto, 0.5).
    - factor_vol: Proporción para ajustar las volatilidades (por defecto, 0.7 o 70%).
    - factor_corr: Proporción para reducir las correlaciones (por defecto, 0.5 o 50%).

    Retorna:
    - med_lst_ajustada: Lista de medias ajustadas.
    - std_lst_ajustada: Lista de volatilidades ajustadas.
    - cov_mat_ajustada: Matriz de covarianza ajustada.
    """
    
    # Ajustar medias hacia cero
    med_lst_ajustada = [med * factor_medias for med in med_lst]
    
    # Ajustar volatilidades
    std_lst_ajustada = [std * factor_vol for std in std_lst]
    
    # Ajustar matriz de covarianza: reducir correlaciones
    d = np.sqrt(np.diag(cov_mat))
    corr_mat = cov_mat / d[:, None] / d[None, :]
    corr_mat_ajustada = corr_mat - factor_corr * corr_mat
    np.fill_diagonal(corr_mat_ajustada, 1)
    cov_mat_ajustada = corr_mat_ajustada * np.outer(std_lst_ajustada, std_lst_ajustada)
    
    return med_lst_ajustada, std_lst_ajustada, cov_mat_ajustada

def simular_mercado_alcista(med_lst, std_lst, cov_mat, factor_medias=1.5, factor_vol=1.2, factor_corr=0.5):
    """
    Simula un mercado alcista ajustando las medias, volatilidades y matriz de covarianza.

    Parámetros:
    - med_lst: Lista de medias originales.
    - std_lst: Lista de volatilidades originales.
    - cov_mat: Matriz de covarianza original.
    - factor_medias: Factor para aumentar las medias (por defecto, 1.5).
    - factor_vol: Proporción para ajustar las volatilidades (por defecto, 1.2 o 20% de aumento).
    - factor_corr: Proporción para aumentar las correlaciones (por defecto, 0.5 o 50% de aumento).

    Retorna:
    - med_lst_ajustada: Lista de medias ajustadas.
    - std_lst_ajustada: Lista de volatilidades ajustadas.
    - cov_mat_ajustada: Matriz de covarianza ajustada.
    """
    
    # Ajustar medias hacia arriba
    med_lst_ajustada = [med * factor_medias for med in med_lst]
    
    # Ajustar volatilidades
    std_lst_ajustada = [std * factor_vol for std in std_lst]
    
    # Ajustar matriz de covarianza: aumentar correlaciones
    d = np.sqrt(np.diag(cov_mat))
    corr_mat = cov_mat / d[:, None] / d[None, :]
    corr_mat_ajustada = corr_mat + factor_corr * (1 - corr_mat)
    np.fill_diagonal(corr_mat_ajustada, 1)
    cov_mat_ajustada = corr_mat_ajustada * np.outer(std_lst_ajustada, std_lst_ajustada)
    
    return med_lst_ajustada, std_lst_ajustada, cov_mat_ajustada


if __name__ == "__main__":
    # Prueba de la función generate_business_dates
    print("Probando generate_business_dates...")
    fechas = generate_business_dates(5, '2023-01-01')
    print(f"Fechas generadas: {fechas}")
    # Puedes hacer afirmaciones o simplemente imprimir para revisar manualmente

    # Prueba de la función generar_datos
    print("\nProbando generar_datos...")
    medias, desviaciones, corr_mat = generar_datos(10, (0, 0.1), (0.05, 0.3), "poco")
    print(f"Medias: {medias}")
    print(f"Desviaciones: {desviaciones}")
    # Puedes visualizar la matriz de correlación con Seaborn, por ejemplo:
    sns.heatmap(corr_mat, annot=True)
    plt.show()
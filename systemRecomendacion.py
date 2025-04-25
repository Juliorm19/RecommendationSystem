import pandas as pd
import numpy as np

# 1. Crear el Dataset de 20 películas y ratings
def create_movie_ratings_dataset():
    data = {
        'UserA': [5, np.nan, 4, np.nan, 3, 5, np.nan, 4, 2, 5, 1, np.nan, 4, 3, np.nan, 5, 2, 4, 3, np.nan],
        'UserB': [4, 5, np.nan, 4, 3, np.nan, 5, 3, 2, 4, np.nan, 5, 3, np.nan, 4, 3, 5, np.nan, 2, 4],
        'UserC': [np.nan, 4, 5, 3, np.nan, 4, 5, np.nan, 1, 3, 2, 4, np.nan, 5, 4, np.nan, 3, 5, 4, 3],
        'UserD': [3, 3, 4, np.nan, 5, 2, np.nan, 5, 3, np.nan, 4, 3, 5, 4, np.nan, 2, np.nan, 3, 5, 4],
        'UserE': [5, np.nan, 3, 4, np.nan, 5, 4, 3, np.nan, 5, 1, 4, 2, np.nan, 5, 4, 3, 2, np.nan, 5],
        'UserF': [np.nan, 5, 4, 5, 2, np.nan, 3, 4, np.nan, 4, 3, np.nan, 5, 3, 4, 5, np.nan, 4, 3, np.nan],
        'UserG': [4, np.nan, 3, np.nan, 4, 5, np.nan, 5, 2, 4, 1, 5, np.nan, 4, 3, np.nan, 5, 2, np.nan, 4],
    }

    movies = [
        'The Shawshank Redemption', #0
        'The Godfather', #1
        'The Dark Knight',
        'Pulp Fiction',
        'Forrest Gump',
        'Inception',
        'The Matrix',
        'Parasite',
        'Spider-Man: Into the Spider-Verse',
        'Toy Story',
        'The Lion King (1994)',
        'Jurassic Park',
        'Star Wars: Episode IV - A New Hope',
        'The Lord of the Rings: The Fellowship of the Ring',
        'Titanic',
        'Avatar',
        'Frozen',
        'Get Out',
        'Mad Max: Fury Road',
        'La La Land'
    ]

    # Crear el DataFrame. Los usuarios serán las filas por conveniencia para la correlación
    # entre usuarios. Transpondremos si necesitamos películas como filas.
    ratings_df = pd.DataFrame(data, index=movies).T

    return ratings_df

# 2. Calcular la similitud entre usuarios (usando matriz)
def calculate_user_similarity(ratings_df):
    # Calcular la matriz de correlación de Pearson entre las filas (usuarios)
    # Pandas calcula la correlación solo para las columnas (películas) que tienen en común
    user_similarity = ratings_df.T.corr(method='pearson')
    return user_similarity

# 3. Generar recomendaciones para un usuario
def recommend_movies(user, ratings_df, user_similarity_matrix, k=3):

    
    user_ratings = ratings_df.loc[user].dropna() # ratings del usuario 

    all_movies = ratings_df.columns     # películas sin calificado
    movies_unrated_by_user = all_movies[~all_movies.isin(user_ratings.index)]

    if movies_unrated_by_user.empty:     # vacia si estan calificadas
        print(f"El usuario {user} ya ha calificado todas las películas disponibles.")
        return pd.Series([], dtype=float) # Retorna una serie vacía

    # similitud del usuario con los demas y se quita el user
    similar_users = user_similarity_matrix.loc[user].drop(user).dropna()

    # Si no hay usuarios similares no hay recomendaciones
    if similar_users.empty:
         print(f"No hay usuarios similares a {user} con ratings en común.")
         return pd.Series([], dtype=float)


    # Calcular un score para no calificadas
    predicted_ratings = {}

    for movie in movies_unrated_by_user:
        weighted_sum = 0
        similarity_sum = 0

        # Iterar sobre los usuarios similares
        for other_user, similarity in similar_users.items():
            # Verificar si las calificaron
            other_user_rating = ratings_df.loc[other_user, movie]

            if pd.notna(other_user_rating):
                # la similitud ponderada por el rating
                weighted_sum += similarity * other_user_rating
                # Sumar el valor absoluto de la similitud para normalizar
                similarity_sum += abs(similarity)

        # Calcular el rating promedio
        # Evitar división por cero si no se califico
        if similarity_sum > 0:
            predicted_ratings[movie] = weighted_sum / similarity_sum

    # Convertir a Serie de Pandas 
    predicted_ratings_series = pd.Series(predicted_ratings)

    # Ordenar y tomar el top
    recommendations = predicted_ratings_series.sort_values(ascending=False).head(k)

    return recommendations

if __name__ == "__main__":
    # 1. Crear el dataset
    ratings_df = create_movie_ratings_dataset()
    print("Dataset de Ratings (Usuarios x Películas):")
    print(ratings_df)

    # 2. Calcular la matriz de similitud entre usuarios
    user_similarity_matrix = calculate_user_similarity(ratings_df)
    print("\nMatriz de Similitud de Usuarios (Pearson Correlation):")
    print(user_similarity_matrix)

    # 3. Obtener recomendaciones para un usuario específico
    target_user = 'UserA'
    print(f"\n--- Recomendaciones para {target_user} ---")
    recommendations = recommend_movies(target_user, ratings_df, user_similarity_matrix, k=5)

    if recommendations.empty:
         print(f"No se pudieron generar recomendaciones para {target_user} en este momento.")
    else:
         print("Películas recomendadas con score predicho:")
         print(recommendations)

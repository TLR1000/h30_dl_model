import pandas as pd
import numpy as np
from scipy.stats import poisson
import sys
import os
from itertools import product
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Functie om wedstrijduitslagen te lezen en teamnamen te normaliseren
def read_match_results(file_path):
    try:
        df = pd.read_csv(file_path)
        required_columns = {'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Het bestand moet de kolommen {required_columns} bevatten.")
        
        # Leidende en volgende spaties verwijderen uit teamnamen
        df['HomeTeam'] = df['HomeTeam'].str.strip()
        df['AwayTeam'] = df['AwayTeam'].str.strip()
        return df
    except FileNotFoundError:
        print(f"Bestand {file_path} niet gevonden.")
        sys.exit(1)
    except Exception as e:
        print(f"Fout bij het lezen van {file_path}: {e}")
        sys.exit(1)

# Functie om thuisvoordeel te berekenen
def calculate_home_advantage(df):
    total_home_goals = df['HomeGoals'].sum()
    total_away_goals = df['AwayGoals'].sum()
    if total_away_goals == 0:
        return 1.0  # Vermijd deling door nul
    home_advantage = total_home_goals / total_away_goals
    return home_advantage

# Functie om data voor Deep Learning model voor te bereiden
def prepare_deep_learning_data(df):
    # Leidende en volgende spaties verwijderen uit teamnamen
    df['HomeTeam'] = df['HomeTeam'].str.strip()
    df['AwayTeam'] = df['AwayTeam'].str.strip()
    
    # Label Encoding voor teams
    le = LabelEncoder()
    df['HomeTeamEncoded'] = le.fit_transform(df['HomeTeam'])
    df['AwayTeamEncoded'] = le.transform(df['AwayTeam'])  # Zorg dat dezelfde encoder gebruikt wordt
    
    # One-Hot Encoding
    home_ohe = pd.get_dummies(df['HomeTeamEncoded'], prefix='HomeTeam')
    away_ohe = pd.get_dummies(df['AwayTeamEncoded'], prefix='AwayTeam')
    
    # Combineer features
    X = pd.concat([home_ohe, away_ohe], axis=1)
    
    # Targets
    y_home = df['HomeGoals']
    y_away = df['AwayGoals']
    
    # Splitsen in training en validatie sets
    X_train, X_val, y_home_train, y_home_val, y_away_train, y_away_val = train_test_split(
        X, y_home, y_away, test_size=0.2, random_state=42)
    
    # Schaal de features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    return X_train, X_val, y_home_train, y_home_val, y_away_train, y_away_val, le, scaler

# Functie om een Deep Learning model te definiëren en trainen
def train_deep_learning_model(X_train, X_val, y_home_train, y_home_val, y_away_train, y_away_val):
    input_dim = X_train.shape[1]
    
    # Input layer
    input_layer = Input(shape=(input_dim,))
    
    # Verborgen lagen
    dense1 = Dense(128, activation='relu')(input_layer)
    dropout1 = Dropout(0.3)(dense1)
    dense2 = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(dense2)
    
    # Output lagen voor HomeGoals en AwayGoals
    output_home = Dense(1, activation='linear', name='HomeGoals')(dropout2)
    output_away = Dense(1, activation='linear', name='AwayGoals')(dropout2)
    
    # Model definiëren
    model = Model(inputs=input_layer, outputs=[output_home, output_away])
    
    # Model compileren
    model.compile(optimizer='adam',
                  loss={'HomeGoals': 'mse', 'AwayGoals': 'mse'},
                  metrics={'HomeGoals': 'mae', 'AwayGoals': 'mae'})
    
    # Early stopping om overfitting te voorkomen
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Model trainen
    model.fit(X_train, [y_home_train, y_away_train],
              epochs=100,
              batch_size=32,
              validation_data=(X_val, [y_home_val, y_away_val]),
              callbacks=[early_stop],
              verbose=1)
    
    return model

# Functie om de nog niet gespeelde wedstrijden te genereren
def generate_remaining_fixtures(df, teams, le, scaler):
    # Strip spaties uit teamnamen
    teams = [team.strip() for team in teams]
    
    # Genereer alle mogelijke thuis-uit combinaties
    all_matches = pd.DataFrame(list(product(teams, teams)), columns=['HomeTeam', 'AwayTeam'])
    # Verwijder wedstrijden tegen zichzelf
    all_matches = all_matches[all_matches['HomeTeam'] != all_matches['AwayTeam']]
    
    # Voeg MatchID toe om reeds gespeelde wedstrijden uit te sluiten
    all_matches['MatchID'] = all_matches['HomeTeam'] + '_' + all_matches['AwayTeam']
    df['MatchID'] = df['HomeTeam'] + '_' + df['AwayTeam']
    
    # Exclude reeds gespeelde wedstrijden
    remaining_matches = all_matches[~all_matches['MatchID'].isin(df['MatchID'])]
    
    # Label Encoding
    remaining_matches['HomeTeamEncoded'] = le.transform(remaining_matches['HomeTeam'])
    remaining_matches['AwayTeamEncoded'] = le.transform(remaining_matches['AwayTeam'])
    
    # One-Hot Encoding
    home_ohe = pd.get_dummies(remaining_matches['HomeTeamEncoded'], prefix='HomeTeam')
    away_ohe = pd.get_dummies(remaining_matches['AwayTeamEncoded'], prefix='AwayTeam')
    
    # Combine home and away one-hot encodings
    X_fixtures = pd.concat([home_ohe, away_ohe], axis=1)
    
    # Reindex X_fixtures to match the scaler's feature names
    X_fixtures = X_fixtures.reindex(columns=scaler.feature_names_in_, fill_value=0)
    
    # Zorg ervoor dat de volgorde van de kolommen hetzelfde is
    X_fixtures = X_fixtures[scaler.feature_names_in_]
    
    # Schaal de features
    X_fixtures_scaled = scaler.transform(X_fixtures)
    
    # Voeg de gefilterde fixtures samen
    remaining_fixtures = remaining_matches[['HomeTeam', 'AwayTeam']].reset_index(drop=True)
    
    return remaining_fixtures, X_fixtures_scaled

# Functie om wedstrijdresultaat te voorspellen
def predict_match_outcome_poisson(lambda_home, lambda_away, max_goals=12):
    # Mogelijke doelpuntenaantallen
    home_goals = np.arange(0, max_goals+1)
    away_goals = np.arange(0, max_goals+1)

    # Kansenmatrix opstellen
    prob_matrix = np.outer(poisson.pmf(home_goals, lambda_home), poisson.pmf(away_goals, lambda_away))

    # Kansen op winst, gelijkspel, verlies
    home_win_prob = np.tril(prob_matrix, -1).sum()
    draw_prob = np.trace(prob_matrix)
    away_win_prob = np.triu(prob_matrix, 1).sum()

    # Meest waarschijnlijke uitslag
    max_prob_index = np.unravel_index(prob_matrix.argmax(), prob_matrix.shape)
    most_probable_score = (home_goals[max_prob_index[0]], away_goals[max_prob_index[1]])

    # Waarschijnlijkheid van de meest waarschijnlijke uitslag
    most_probable_score_prob = prob_matrix[max_prob_index]

    return home_win_prob, draw_prob, away_win_prob, most_probable_score, most_probable_score_prob

# Hoofdprogramma
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Gebruik: python deeplearning.py <invoerbestand> <uitvoerbestand>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Debugging: Print de ontvangen parameters
    print(f"Invoerbestand: {input_file}")
    print(f"Uitvoerbestand: {output_file}")

    # Wedstrijduitslagen lezen
    match_results = read_match_results(input_file)
    print("Wedstrijduitslagen gelezen.")

    # Thuisvoordeel berekenen
    home_advantage = calculate_home_advantage(match_results)
    print(f"Thuisvoordeel berekend: {home_advantage}")

    # Data voorbereiden voor Deep Learning
    X_train, X_val, y_home_train, y_home_val, y_away_train, y_away_val, le, scaler = prepare_deep_learning_data(match_results)
    print("Data voorbereid voor Deep Learning.")

    # Train het Deep Learning model
    print("Train het Deep Learning model...")
    model = train_deep_learning_model(X_train, X_val, y_home_train, y_home_val, y_away_train, y_away_val)
    print("Deep Learning model getraind.")

    # Lijst van teams verkrijgen
    teams = match_results['HomeTeam'].unique()
    print(f"Teams gevonden: {teams}")

    # Nog niet gespeelde wedstrijden genereren
    fixtures, X_fixtures_scaled = generate_remaining_fixtures(match_results, teams, le, scaler)
    print(f"Nog niet gespeelde wedstrijden gegenereerd: {len(fixtures)} wedstrijden.")

    # Controleren of er nog fixtures zijn om te voorspellen
    if fixtures.empty:
        print("Er zijn geen nog niet gespeelde wedstrijden om te voorspellen.")
        sys.exit(0)

    # Voorspellingen genereren met het Deep Learning model
    print("Voorspellingen genereren...")
    predictions = []
    predictions_home, predictions_away = model.predict(X_fixtures_scaled)

    # Ronde voorspellingen om naar integer doelpunten
    predictions_home = np.round(predictions_home.flatten()).astype(int)
    predictions_away = np.round(predictions_away.flatten()).astype(int)

    for i in range(len(fixtures)):
        home_team = fixtures.iloc[i]['HomeTeam']
        away_team = fixtures.iloc[i]['AwayTeam']
        lambda_home = predictions_home[i]
        lambda_away = predictions_away[i]

        home_win_prob, draw_prob, away_win_prob, most_probable_score, most_probable_score_prob = predict_match_outcome_poisson(
            lambda_home, lambda_away, max_goals=12)

        # Bepalen van de voorspelde winnaar
        if home_win_prob > away_win_prob and home_win_prob > draw_prob:
            voorspelling = f"{home_team} wint"
            voorspelling_prob = home_win_prob
        elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
            voorspelling = f"{away_team} wint"
            voorspelling_prob = away_win_prob
        else:
            voorspelling = "Gelijkspel"
            voorspelling_prob = draw_prob

        predictions.append({
            'Wedstrijd': f"{home_team} vs {away_team}",
            'Verwachte Doelpunten': f"{lambda_home} - {lambda_away}",
            'Meest Waarschijnlijke Uitslag': f"{most_probable_score[0]} - {most_probable_score[1]}",
            'Waarschijnlijkheid Uitslag': f"{most_probable_score_prob*100:.2f}%",
            'Voorspelling': voorspelling,
            'Waarschijnlijkheid Voorspelling': f"{voorspelling_prob*100:.2f}%",
            'Winstkansen': f"{home_win_prob*100:.1f}% - {draw_prob*100:.1f}% - {away_win_prob*100:.1f}%"
        })

    predictions_df = pd.DataFrame(predictions)
    print("Voorspellingen gegenereerd.")

    # Voorspellingen opslaan naar het opgegeven uitvoerbestand
    predictions_df.to_csv(output_file, index=False)
    print(f"Voorspellingen zijn opgeslagen in '{output_file}'")

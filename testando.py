#%%
import pandas as pd
import pyarrow
import numpy as np
from pathlib import Path

# 1 - Leitura dos dados:

df = pd.read_csv('yellow_tripdata_2016-03.csv')
print(f"Total de registros: {len(df):,}")
print(f"Total de colunas: {len(df.columns)}")
df.head(5)

df.info()
df.describe()

# 2 - Quality/Anomalias:

# convertendo colunas de data:
df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# Verificando valores nulos:
print("== VALORES NULOS ==")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "Nenhum valor nulo encontrado")

# Detectando anomalias:
total = len(df)

# Distância inválida
invalid_distance = (df['trip_distance'] <= 0) | (df['trip_distance'] > 100)

# Número de passageiros inválido
invalid_passengers = (df['passenger_count'] <= 0) | (df['passenger_count'] > 6)

# Tarifa negativa
invalid_fare = df['fare_amount'] < 0

# Duração inválida (pickup depois do dropoff)
invalid_duration = df['pickup_datetime'] > df['dropoff_datetime']

# Coordenadas fora de NYC (latitude ~40.4-41.0, longitude ~-74.3 a -73.7)
invalid_pickup_coords = (
    (df['pickup_latitude'] < 40.4) | (df['pickup_latitude'] > 41.0) |
    (df['pickup_longitude'] < -74.3) | (df['pickup_longitude'] > -73.7)
)
invalid_dropoff_coords = (
    (df['dropoff_latitude'] < 40.4) | (df['dropoff_latitude'] > 41.0) |
    (df['dropoff_longitude'] < -74.3) | (df['dropoff_longitude'] > -73.7)
)

print("=== ANOMALIAS DETECTADAS ===")
print(f"Distância inválida (<=0 ou >100mi): {invalid_distance.sum():,} ({invalid_distance.sum()/total*100:.2f}%)")
print(f"Passageiros inválidos (<=0 ou >6): {invalid_passengers.sum():,} ({invalid_passengers.sum()/total*100:.7f}%)")
print(f"Tarifa negativa: {invalid_fare.sum():,} ({invalid_fare.sum()/total*100:.4f}%)")
print(f"Duração inválida: {invalid_duration.sum():,} ({invalid_duration.sum()/total*100:.7f}%)")
print(f"Coordenadas pickup fora de NYC: {invalid_pickup_coords.sum():,} ({invalid_pickup_coords.sum()/total*100:.2f}%)")
print(f"Coordenadas dropoff fora de NYC: {invalid_dropoff_coords.sum():,} ({invalid_dropoff_coords.sum()/total*100:.2f}%)")

# Remover registros inválidos:
invalid_mask = (
    invalid_distance |
    invalid_passengers |
    invalid_fare |
    invalid_duration |
    invalid_pickup_coords |
    invalid_dropoff_coords
)

df_clean = df[~invalid_mask].copy()
removed = total - len(df_clean)
print(f"Registros antes:     {total:,}")
print(f"Registros removidos: {removed:,} ({removed/total*100:.2f}%)")
print(f"Registros após:      {len(df_clean):,}")


# 3 - Transformando colunas calculadas:
    # enriquecendo o dataset com métricas derivadas

# Duração da corrida (minutos):
df_clean['trip_duration_min'] = (
    (df_clean['dropoff_datetime'] - df_clean['pickup_datetime'])
    .dt.total_seconds() / 60
)

# Velocidade média (milhas/hora):
hours = df_clean['trip_duration_min'] / 60
df_clean['trip_speed_mph'] = (
    df_clean['trip_distance'] / hours.replace(0, np.nan)
).fillna(0)

# Receita por milha:
df_clean['revenue_per_mile'] = (
    df_clean['total_amount'] / df_clean['trip_distance'].replace(0, np.nan)
).fillna(0)

# Hora do dia:
df_clean['hour_of_day'] = df_clean['pickup_datetime'].dt.hour

# Data (sem hora):
df_clean['date'] = df_clean['pickup_datetime'].dt.date

# Dia da semana
df_clean['day_of_week'] = df_clean['pickup_datetime'].dt.day_name()

# Gorjeta percentual:
df_clean['tip_pct'] = (
    (df_clean['tip_amount'] / df_clean['fare_amount'].replace(0, np.nan)) * 100
).fillna(0)

print("=== NOVAS COLUNAS ===")
new_cols = ['trip_duration_min', 'trip_speed_mph', 'revenue_per_mile',
            'hour_of_day', 'date', 'day_of_week', 'tip_pct']
for col in new_cols:
    print(f"  {col}")

print(f"\nTotal de colunas: {len(df_clean.columns)}")
df_clean[new_cols].head(10)


# 4 - Agregações & Métricas:
# Sumário por hora do dia:
hourly = df_clean.groupby('hour_of_day').agg(
    total_trips=('hour_of_day', 'size'),
    avg_distance=('trip_distance', 'mean'),
    avg_fare=('fare_amount', 'mean'),
    avg_duration=('trip_duration_min', 'mean'),
    avg_speed=('trip_speed_mph', 'mean'),
    avg_tip_pct=('tip_pct', 'mean'),
).round(2)

print("=== SUMÁRIO POR HORA DO DIA ===")
hourly


# Sumário por vendor:
vendor = df_clean.groupby('VendorID').agg(
    total_trips=('VendorID', 'size'),
    avg_distance=('trip_distance', 'mean'),
    avg_fare=('fare_amount', 'mean'),
    total_revenue=('total_amount', 'sum'),
    avg_tip_pct=('tip_pct', 'mean'),
    avg_speed=('trip_speed_mph', 'mean'),
).round(2)

print("=== SUMÁRIO POR VENDOR ===")
vendor


# Sumário por tipo de pagamento:
payment_labels = {1: 'Credit Card', 2: 'Cash', 3: 'No Charge', 4: 'Dispute', 5: 'Unknown', 6: 'Voided'}
df_clean['payment_label'] = df_clean['payment_type'].map(payment_labels).fillna('Other')

payment = df_clean.groupby('payment_label').agg(
    total_trips=('payment_label', 'size'),
    avg_fare=('fare_amount', 'mean'),
    avg_tip=('tip_amount', 'mean'),
    avg_tip_pct=('tip_pct', 'mean'),
    total_revenue=('total_amount', 'sum'),
).round(2)

payment['pct_trips'] = ((payment['total_trips'] / payment['total_trips'].sum()) * 100).round(2)

print("=== SUMÁRIO POR TIPO DE PAGAMENTO ===")
payment


# Sumário por dia da semana:
weekday = df_clean.groupby('day_of_week').agg(
    total_trips=('day_of_week', 'size'),
    avg_distance=('trip_distance', 'mean'),
    avg_fare=('fare_amount', 'mean'),
    avg_duration=('trip_duration_min', 'mean'),
    avg_tip_pct=('tip_pct', 'mean'),
).round(2)

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday = weekday.reindex(day_order)

print("=== SUMÁRIO POR DIA DA SEMANA ===")
weekday

#%%

# 5 - MACHINE LEARNING: Regressão Linear para Predição de Tarifa:

# Objetivo: prever fare_amount a partir de features conhecidas
# ANTES da corrida terminar (distância, hora, dia, passageiros).

# Features usadas:
    #   - trip_distance     = principal driver da tarifa
    #   - trip_duration_min = tempo da corrida
    #   - hour_of_day       = hora do dia (trânsito afeta preço)
    #   - passenger_count   = número de passageiros
    #   - RatecodeID        = tipo de tarifa (padrão, JFK, etc.)

# Saída gerada no Parquet:
    #   - predicted_fare    = tarifa prevista pelo modelo
    #   - fare_difference   = fare_amount - predicted_fare (positivo = cobrou mais que o esperado)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

print("=== ML: REGRESSÃO LINEAR - PREDIÇÃO DE TARIFA ===\n")

# 5.1 Preparação das features:
FEATURES = [
    'trip_distance',
    'trip_duration_min',
    'hour_of_day',
    'passenger_count',
    'RatecodeID',
]
TARGET = 'fare_amount'

# Usar apenas corridas com tarifa padrão e JFK para treino
    # (remove outliers extremos que distorceriam o modelo)
df_ml = df_clean[
    df_clean['fare_amount'].between(2.5, 150) &
    df_clean['trip_duration_min'].between(1, 120) &
    df_clean['trip_speed_mph'].between(1, 60)
].copy()

print(f"Registros usados para treino/teste: {len(df_ml):,}")

X = df_ml[FEATURES]
y = df_ml[TARGET]

# 5.2 Divisão treino/teste (80/20):
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Treino: {len(X_train):,} registros | Teste: {len(X_test):,} registros")

# 5.3 Normalização:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 5.4 Treinamento:
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 5.5 Avaliação:
y_pred_test = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred_test)
r2  = r2_score(y_test, y_pred_test)

print(f"\n📊 Métricas no conjunto de TESTE:")
print(f"   MAE (Erro Médio Absoluto): US$ {mae:.2f}")
print(f"   R²  (Coef. Determinação):  {r2:.4f}  ({r2*100:.1f}% da variância explicada)")

print(f"\n📐 Coeficientes do modelo:")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"   {feat:<22}: {coef:+.4f}")
print(f"   {'intercept':<22}: {model.intercept_:+.4f}")

# 5.6 Aplicar o modelo em TODOS os registros limpos:
    # (inclusive os excluídos do treino, para análise completa no dashboard)
X_all = df_clean[FEATURES].copy()
X_all_scaled = scaler.transform(X_all)

df_clean['predicted_fare']    = model.predict(X_all_scaled).round(2)
df_clean['fare_difference']   = (df_clean['fare_amount'] - df_clean['predicted_fare']).round(2)

print(f"\n✅ Colunas adicionadas ao dataset:")
print(f"   predicted_fare   → tarifa prevista pelo modelo")
print(f"   fare_difference  → diferença (real - previsto)")
print(f"\n   Top 5 corridas com maior diferença positiva (cobrado bem acima do previsto):")
print(
    df_clean[['trip_distance', 'trip_duration_min', 'fare_amount', 'predicted_fare', 'fare_difference']]
    .sort_values('fare_difference', ascending=False)
    .head(5)
    .to_string(index=False)
)

# 6 - Load Parquet:

# Converter coluna date para string (pyarrow não serializa datetime.date):
df_clean['date'] = df_clean['date'].astype(str)

# Remover colunas originais de datetime (já temos pickup_datetime e dropoff_datetime):
df_clean = df_clean.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], errors='ignore')

output_path = Path('yellow_taxi_2016-03.parquet')
output_path.parent.mkdir(parents=True, exist_ok=True)

df_clean.to_parquet(output_path, engine='pyarrow', index=False)

size_mb = output_path.stat().st_size / (1024 * 1024)
print(f"=== PARQUET ===")
print(f"Arquivo: {output_path}")
print(f"Tamanho: {size_mb:.1f} MB")
print(f"Colunas ML incluídas: predicted_fare, fare_difference ✅")

# Validação: ler de volta o Parquet e conferir:
df_parquet = pd.read_parquet(output_path)
print(f"=== VALIDAÇÃO ===")
print(f"Registros no Parquet: {len(df_parquet):,}")
print(f"Registros esperados:  {len(df_clean):,}")
print(f"Match: {'OK' if len(df_parquet) == len(df_clean) else 'ERRO'}")
print(f"\nColunas: {list(df_parquet.columns)}")
df_parquet.head()
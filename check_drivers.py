import pandas as pd

df = pd.read_csv('F1 Races 2020-2024.csv', encoding='latin-1')

print("="*60)
print("DRIVER MAPPING CHECK")
print("="*60)

# CSV columns
print("\nColumns in CSV:")
print([col for col in df.columns])

# Unique drivers
unique_drivers = sorted(df['driverId'].unique())
print(f"\nUnique Driver IDs in CSV: {unique_drivers}")
print(f"Total: {len(unique_drivers)} drivers")

# Show sample with constructor
print("\n" + "="*60)
print("Sample data (Driver + Constructor):")
print("="*60)
sample = df[['driverId', 'constructorId', 'grid', 'driver_age']].drop_duplicates().sort_values('driverId').head(30)
print(sample)

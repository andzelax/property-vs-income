import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. WCZYTYWANIE DANYCH
# ==========================================
sale_files = sorted(glob.glob("apartment-prices-in-poland/apartments_pl_*.csv"))
sale_files = [f for f in sale_files if "rent" not in f]

dfs = []
for f in sale_files:
    df = pd.read_csv(f)
    basename = os.path.basename(f)
    parts = basename.replace(".csv","").split("_")
    df["year"]  = int(parts[2])
    df["month"] = int(parts[3])
    df["date"]  = pd.to_datetime(f"{parts[2]}-{parts[3]}-01")
    dfs.append(df)

df_sale = pd.concat(dfs, ignore_index=True)
print(f"Wczytano surowe dane sprzedaży: {len(df_sale):,} wierszy")

# ==========================================
# 2. CZYSZCZENIE DANYCH I USUWANIE DUPLIKATÓW
# ==========================================
# Usuwam duplikaty ofert z roznych miesiecy - zostawiam ostatni wpis
df_sale_dedup = (df_sale
    .sort_values("date")
    .drop_duplicates(subset="id", keep="last")
).copy()

# Liczenie ceny za metr
df_sale_dedup["price_per_sqm"] = df_sale_dedup["price"] / df_sale_dedup["squareMeters"]

# Filtrowanie wartosci skrajnych i bledow
df_clean = df_sale_dedup[
    (df_sale_dedup["price_per_sqm"] > 1000) &   
    (df_sale_dedup["price_per_sqm"] < 50000) &  
    (df_sale_dedup["squareMeters"] > 10) &
    (df_sale_dedup["squareMeters"] < 500)
].copy()

# Zamiana yes/no na 1/0
bool_cols = ["hasParkingSpace","hasBalcony","hasElevator","hasSecurity","hasStorageRoom"]
for col in bool_cols:
    df_clean[col] = (df_clean[col] == "yes").astype(int)

print(f"Dane po usunięciu duplikatów i outlierów: {len(df_clean):,} wierszy")

# ==========================================
# 3. POŁĄCZENIE Z DANYMI GUS O ZAROBKACH
# ==========================================
# Wczytanie pliku z zarobkami
df_earnings = pd.read_csv('WYNA_2497_CTAB_20260531120135.csv', delimiter=';')

# Zamiana formatu liczb (usuwanie spacji i zamiana przecinka na kropke)
df_earnings['salary_gross'] = (df_earnings['ogółem;2024;[zł]']
                               .str.replace(r'\s+', '', regex=True)
                               .str.replace(',', '.', regex=False)
                               .astype(float))

# Slownik do polaczenia nazw miast z GUS z nazwami z datasetu
gus_city_mapping = {
    'Powiat m. st. Warszawa': 'warszawa',
    'Powiat m. Kraków': 'krakow',
    'Powiat m. Gdańsk': 'gdansk',
    'Powiat m. Wrocław': 'wroclaw',
    'Powiat m. Poznań': 'poznan',
    'Powiat m. Łódź': 'lodz',
    'Powiat m. Szczecin': 'szczecin',
    'Powiat m. Gdynia': 'gdynia',
    'Powiat m. Katowice': 'katowice',
    'Powiat m. Lublin': 'lublin',
    'Powiat m. Białystok': 'bialystok',
    'Powiat m. Rzeszów': 'rzeszow',
    'Powiat m. Bydgoszcz': 'bydgoszcz',
    'Powiat m. Częstochowa': 'czestochowa',
    'Powiat m. Radom': 'radom'
}

df_earnings['city'] = df_earnings['Nazwa'].map(gus_city_mapping)

# Wybieram tylko glowne miasta i przeliczam na przyblizone zarobki netto (72% z brutto)
df_salaries_clean = df_earnings.dropna(subset=['city'])[['city', 'salary_gross']].copy()
df_salaries_clean['salary_net'] = (df_salaries_clean['salary_gross'] * 0.72).round(2)

# Laczenie tabel
df_final = pd.merge(df_clean, df_salaries_clean, on='city', how='inner')

# ==========================================
# 4. OBLICZANIE WSKAŹNIKA DOSTĘPNOŚCI
# ==========================================
# Licze ile pensji potrzeba na 1 metr kwadratowy
df_final['affordability_index'] = df_final['price_per_sqm'] / df_final['salary_net']

print("\n--- Realne wskaźniki dostępności z danych GUS 2024 (Mediany) ---")
ranking = df_final.groupby("city")[["price_per_sqm", "salary_net", "affordability_index"]].median().round(2)
print(ranking.sort_values('affordability_index', ascending=False))

# ==========================================
# 5. K-MEANS DLA MIAST
# ==========================================
# Agregacja danych do poziomu miast (wyciagam mediany i srednie wspolrzedne)
features_miast = df_final.groupby('city').agg({
    'price_per_sqm': 'median',
    'affordability_index': 'median',
    'latitude': 'mean',
    'longitude': 'mean'
}).reset_index()

# Zmienne do modelu
X = features_miast[['price_per_sqm', 'affordability_index', 'latitude', 'longitude']]

# Standaryzacja cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dopasowanie modelu K-means dla K=3
optimal_k = 3
model_km = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
features_miast['cluster'] = model_km.fit_predict(X_scaled)

# ==========================================
# 6. PODZIAŁ NA KLASTRY
# ==========================================
cluster_names = {
    2: "Metropolie Kryzysu (Niedostępne)",
    0: "Rynki Rozwojowe i Nadmorskie",
    1: "Oazy Dostępności i Rynki Lokalne"
}

features_miast['cluster_mapped'] = features_miast['cluster'].map(cluster_names)

print("\n--- PODZIAŁ MIAST PRZEZ K-MEANS ---")
for cluster_id in range(optimal_k):
    subset = features_miast[features_miast['cluster'] == cluster_id]
    miasta = subset['city'].tolist()
    srednia_cena = subset['price_per_sqm'].mean()
    srednia_niedostepnosc = subset['affordability_index'].mean()
    print(f"\nKlaster {cluster_id} ({cluster_names[cluster_id]}): {miasta}")
    print(f"   Średnia cena mkw: {srednia_cena:.0f} zł")
    print(f"   Ile pensji netto za 1 mkw? {srednia_niedostepnosc:.2f}")

# Przypisanie klastrow z powrotem do glównej tabeli
city_to_cluster_map = dict(zip(features_miast['city'], features_miast['cluster_mapped']))
df_final['cluster_mapped'] = df_final['city'].map(city_to_cluster_map)

# ==========================================
# 7. GENEROWANIE WYKRESU BOXPLOT
# ==========================================
print("\nGenerowanie wykresów...")
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df_final, 
    x='cluster_mapped', 
    y='price_per_sqm', 
    hue='cluster_mapped',  
    palette='Set2',
    legend=False,
    order=[
        "Metropolie Kryzysu (Niedostępne)",
        "Rynki Rozwojowe i Nadmorskie",
        "Oazy Dostępności i Rynki Lokalne"
    ]
)
plt.title('Rozkład cen za mkw w wyodrębnionych klastrach', fontsize=14)
plt.xlabel('Profil rynkowy (Klaster)')
plt.ylabel('Cena za metr kwadratowy (zł)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('profil_klastrow_boxplot.png', bbox_inches='tight')
plt.close()
print("-> Wygenerowano wykres: profil_klastrow_boxplot.png")

# ==========================================
# 8. MAPA POLSKI (GEOPANDAS)
# ==========================================
gdf_miasta = gpd.GeoDataFrame(
    features_miast, 
    geometry=gpd.points_from_xy(features_miast.longitude, features_miast.latitude),
    crs="EPSG:4326"
)

gdf_miasta['cluster_mapped'] = pd.Categorical(
    gdf_miasta['cluster_mapped'],
    categories=[
        "Metropolie Kryzysu (Niedostępne)",
        "Rynki Rozwojowe i Nadmorskie",
        "Oazy Dostępności i Rynki Lokalne"
    ]
)

# Pobranie konturu Polski
url_mapa = "https://nagix.github.io/geo-boundaries/database/POL/adm0.geojson"
try:
    polska_granice = gpd.read_file(url_mapa)
except Exception:
    polska_granice = gpd.read_file("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/poland.geojson")

fig, ax = plt.subplots(figsize=(10, 10))
polska_granice.plot(ax=ax, color='#f2f2f2', edgecolor='#9c9c9c', linewidth=1)

# Naniesienie miast na mape
gdf_miasta.plot(
    column='cluster_mapped', 
    categorical=True, 
    legend=True, 
    markersize=features_miast['price_per_sqm'] / 30, 
    cmap='Set1', 
    ax=ax,
    legend_kwds={'title': "Profile rynkowe (K-means)", 'loc': 'lower left'}
)

# Dodanie podpisow miast
for x, y, label in zip(gdf_miasta.geometry.x, gdf_miasta.geometry.y, gdf_miasta.city):
    ax.annotate(label.capitalize(), xy=(x, y), xytext=(5, 5), textcoords="offset points", fontsize=9, weight='bold')

plt.title('Geograficzny podział rynków mieszkaniowych w Polsce według K-means', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig('mapa_polska_geopandas.png', bbox_inches='tight')
plt.close()
print("-> Wygenerowano mapę: mapa_polska_geopandas.png")

# ==========================================
# 9. MAPA INTERAKTYWNA (FOLIUM)
# ==========================================
print("\nGenerowanie mapy w Folium...")

mapa_folium = folium.Map(location=[52.0, 19.0], zoom_start=6, tiles="cartodbpositron")

cluster_colors = {
    "Metropolie Kryzysu (Niedostępne)": "red",
    "Rynki Rozwojowe i Nadmorskie": "blue",
    "Oazy Dostępności i Rynki Lokalne": "green"
}

# Petla dodajaca punkty miast i popupy HTML
for _, row in features_miast.iterrows():
    kolor = cluster_colors[row['cluster_mapped']]
    
    html_popup = f"""
    <div style="font-family: Arial, sans-serif; width: 220px; font-size: 13px;">
        <h3 style="margin-bottom: 5px; color: {kolor}; text-transform: uppercase;">
            {row['city'].capitalize()}
        </h3>
        <strong style="color: #555;">Profil:</strong> {row['cluster_mapped']}<br>
        <hr style="border: 0; border-top: 1px solid #ccc; margin: 8px 0;">
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 3px 0; color: #666;">Mediana ceny:</td>
                <td style="text-align: right; font-weight: bold;">{row['price_per_sqm']:.0f} zł/m²</td>
            </tr>
            <tr>
                <td style="padding: 3px 0; color: #666;">Wskaźnik:</td>
                <td style="text-align: right; font-weight: bold; color: {kolor};">{row['affordability_index']:.2f} pensji</td>
            </tr>
        </table>
        <p style="font-size: 11px; color: #888; margin-top: 10px; font-style: italic;">
            *Wskaźnik określa liczbę lokalnych miesięcznych pensji netto potrzebnych na 1m²
        </p>
    </div>
    """
    
    promien = row['price_per_sqm'] / 500
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=promien,
        popup=folium.Popup(html_popup, max_width=250),
        color=kolor,
        fill=True,
        fill_color=kolor,
        fill_opacity=0.6,
        weight=2
    ).add_to(mapa_folium)

# Dodanie legendy do mapy
legenda_html = """
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 260px; height: 110px; 
     border:2px solid grey; z-index:9999; font-size:12px;
     background-color:white; opacity: 0.9; padding: 10px;
     font-family: Arial, sans-serif; border-radius: 5px;">
     <b>Profile rynków (K-means):</b><br>
     <i class="fa fa-circle fa-1x" style="color:red"></i> Metropolie Kryzysu (Niedostępne)<br>
     <i class="fa fa-circle fa-1x" style="color:blue"></i> Rynki Rozwojowe i Nadmorskie<br>
     <i class="fa fa-circle fa-1x" style="color:green"></i> Oazy Dostępności i Rynki Lokalne
     </div>
     """
mapa_folium.get_root().html.add_child(folium.Element(legenda_html))

mapa_folium.save("mapa_folium_projekt.html")
print("-> Wygenerowano interaktywną mapę Folium: mapa_folium_projekt.html")
print("Projekt ukończony.")
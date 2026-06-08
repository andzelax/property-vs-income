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
# 1. WCZYTYWANIE I CZYSZCZENIE
# ==========================================
print("Wczytywanie danych...")
sale_files = sorted(glob.glob("data/apartment-prices-in-poland/apartments_pl_*.csv"))
sale_files = [f for f in sale_files if "rent" not in f]

dfs = []
for f in sale_files:
    df = pd.read_csv(f)
    basename = os.path.basename(f)
    
    try:
        parts = basename.replace(".csv","").split("_")
        # Zakładamy format: np. apartments_pl_2024_01.csv -> parts[2]=rok, parts[3]=mies
        df["date"] = pd.to_datetime(f"{parts[2]}-{parts[3]}-01")
    except (IndexError, ValueError):
        df["date"] = pd.to_datetime("2024-01-01") 
        
    dfs.append(df)

if not dfs:
    print("BŁĄD: Nie znaleziono żadnych plików CSV w folderze!")
    exit()

df_sale = pd.concat(dfs, ignore_index=True)

# ==========================================
# 2. CZYSZCZENIE DANYCH I USUWANIE DUPLIKATÓW
# ==========================================
df_sale_dedup = (df_sale
    .sort_values("date")
    .drop_duplicates(subset="id", keep="last")
).copy()

df_sale_dedup["price_per_sqm"] = df_sale_dedup["price"] / df_sale_dedup["squareMeters"]

df_clean = df_sale_dedup[
    (df_sale_dedup["price_per_sqm"] > 1000) &   
    (df_sale_dedup["price_per_sqm"] < 50000) &  
    (df_sale_dedup["squareMeters"] > 10) &
    (df_sale_dedup["squareMeters"] < 500)
].copy()

print(f"Dane po usunięciu duplikatów i outlierów: {len(df_clean):,} wierszy")

# ==========================================
# 3. POŁĄCZENIE Z DANYMI GUS O ZAROBKACH
# ==========================================
df_earnings = pd.read_csv('data/gus/WYNA_2497_CTAB_20260531120135.csv', delimiter=';')

df_earnings['salary_gross'] = (df_earnings['ogółem;2024;[zł]']
                               .str.replace(r'\s+', '', regex=True)
                               .str.replace(',', '.', regex=False)
                               .astype(float))

gus_city_mapping = {
    'Powiat m. st. Warszawa': 'warszawa', 'Powiat m. Kraków': 'krakow',
    'Powiat m. Gdańsk': 'gdansk', 'Powiat m. Wrocław': 'wroclaw',
    'Powiat m. Poznań': 'poznan', 'Powiat m. Łódź': 'lodz',
    'Powiat m. Szczecin': 'szczecin', 'Powiat m. Gdynia': 'gdynia',
    'Powiat m. Katowice': 'katowice', 'Powiat m. Lublin': 'lublin',
    'Powiat m. Białystok': 'bialystok', 'Powiat m. Rzeszów': 'rzeszow',
    'Powiat m. Bydgoszcz': 'bydgoszcz', 'Powiat m. Częstochowa': 'czestochowa',
    'Powiat m. Radom': 'radom'
}

df_earnings['city'] = df_earnings['Nazwa'].map(gus_city_mapping)
df_salaries_clean = df_earnings.dropna(subset=['city'])[['city', 'salary_gross']].copy()
df_salaries_clean['salary_net'] = (df_salaries_clean['salary_gross'] * 0.72).round(2)

df_final = pd.merge(df_clean, df_salaries_clean, on='city', how='inner')

# ==========================================
# 4. OBLICZANIE WSKAŹNIKA DOSTĘPNOŚCI
# ==========================================
df_final['affordability_index'] = df_final['price_per_sqm'] / df_final['salary_net']

# ==========================================
# 5. K-MEANS DLA MIAST - AGREGACJA DO POZIOMU MIAST
# ==========================================
features_miast = df_final.groupby('city').agg({
    'price_per_sqm': 'median',
    'affordability_index': 'median',
    'latitude': 'mean',#przydatne do mapowania, ale nie bierzemy ich pod uwagę w klastrowaniu
    'longitude': 'mean'#przydatne do mapowania, ale nie bierzemy ich pod uwagę w klastrowaniu
}).reset_index()

X_ekonomiczne = features_miast[['price_per_sqm', 'affordability_index']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ekonomiczne)

optimal_k = 3
model_km = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
features_miast['cluster'] = model_km.fit_predict(X_scaled)

# ==========================================
# 6. DYNAMICZNE I STABILNE MAPOWANIE KLASTRÓW
# ==========================================
# Porządkujemy klastry od najtańszego do najdroższego średniego mkw, by zapobiec losowości etykiet
srednie_ceny = features_miast.groupby('cluster')['price_per_sqm'].mean().sort_values()

cluster_names = {
    srednie_ceny.index[0]: "Oazy Dostępności i Rynki Lokalne",
    srednie_ceny.index[1]: "Rynki Rozwojowe",
    srednie_ceny.index[2]: "Metropolie Kryzysu (Niedostępne)"
}
features_miast['cluster_mapped'] = features_miast['cluster'].map(cluster_names)

print("\n--- PODZIAŁ MIAST PRZEZ K-MEANS ---")
for cluster_id, nazwa in cluster_names.items():
    miasta = features_miast[features_miast['cluster'] == cluster_id]['city'].tolist()
    srednia_cena = features_miast[features_miast['cluster'] == cluster_id]['price_per_sqm'].mean()
    srednia_niedostepnosc = features_miast[features_miast['cluster'] == cluster_id]['affordability_index'].mean()
    print(f"\nKlaster {cluster_id} ({nazwa}): {miasta}")
    print(f"   Średnia cena mkw: {srednia_cena:.0f} zł")
    print(f"   Ile pensji netto za 1 mkw? {srednia_niedostepnosc:.2f}")

city_to_cluster_map = dict(zip(features_miast['city'], features_miast['cluster_mapped']))
df_final['cluster_mapped'] = df_final['city'].map(city_to_cluster_map)

# ==========================================
# 7. WYKRES ROZRZUTU K-MEANS Z CENTROIDAMI
# ==========================================
# Pobieramy współrzędne centroidów z modelu i odwracamy standaryzację
centroids_scaled = model_km.cluster_centers_
centroids_real = scaler.inverse_transform(centroids_scaled)

plt.figure(figsize=(10, 7))

kolory_wykresu = {
    "Metropolie Kryzysu (Niedostępne)": "#e41a1c",  # Czerwony
    "Rynki Rozwojowe": "#377eb8",    # Niebieski
    "Oazy Dostępności i Rynki Lokalne": "#4daf4a"  # Zielony
}

sns.scatterplot(
    data=features_miast,
    x='price_per_sqm',
    y='affordability_index',
    hue='cluster_mapped',
    palette=kolory_wykresu,
    s=150,
    edgecolor='black',
    alpha=0.8,
    zorder=3
)

plt.scatter(
    centroids_real[:, 0],
    centroids_real[:, 1],
    marker='*',
    s=400,
    color='black',
    label='Centroidy (Środki klastrów)',
    edgecolor='white',
    linewidth=2,
    zorder=4
)

for _, row in features_miast.iterrows():
    plt.text(
        x=row['price_per_sqm'] + 120,  
        y=row['affordability_index'] + 0.015,
        s=row['city'].capitalize(),
        fontsize=10,
        weight='bold',
        zorder=5
    )

plt.title('Podział K-means rynków mieszkaniowych w Polsce', fontsize=14, weight='bold', pad=15)
plt.xlabel('Mediana ceny za metr kwadratowy (zł)', fontsize=12)
plt.ylabel('Wskaźnik nieosiągalności (Liczba pensji netto za 1 mkw)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
plt.legend(title="Segmentacja miast", loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('kmeans_clusters_scatter.png', bbox_inches='tight', dpi=150)
plt.close()
print("-> Wygenerowano zoptymalizowany wykres rozrzutu: kmeans_clusters_scatter.png")

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
        "Rynki Rozwojowe",
        "Oazy Dostępności i Rynki Lokalne"
    ]
)

url_mapa = "https://nagix.github.io/geo-boundaries/database/POL/adm0.geojson"
try:
    polska_granice = gpd.read_file(url_mapa)
except Exception:
    polska_granice = gpd.read_file("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/poland.geojson")

fig, ax = plt.subplots(figsize=(10, 10))
polska_granice.plot(ax=ax, color='#f2f2f2', edgecolor='#9c9c9c', linewidth=1)

gdf_miasta.plot(
    column='cluster_mapped', 
    categorical=True, 
    legend=True, 
    markersize=features_miast['price_per_sqm'] / 30, 
    cmap='Set1', 
    ax=ax,
    legend_kwds={'title': "Profile rynkowe (K-means)", 'loc': 'lower left'}
)

for x, y, label in zip(gdf_miasta.geometry.x, gdf_miasta.geometry.y, gdf_miasta.city):
    ax.annotate(label.capitalize(), xy=(x, y), xytext=(5, 5), textcoords="offset points", fontsize=9, weight='bold')

plt.title('Geograficzny podział rynków mieszkaniowych w Polsce według K-means', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig('mapa_polska_geopandas.png', bbox_inches='tight')
plt.close()

# ==========================================
# 9. MAPA INTERAKTYWNA (FOLIUM)
# ==========================================
print("Generowanie mapy w Folium...")
mapa_folium = folium.Map(location=[52.0, 19.0], zoom_start=6, tiles="cartodbpositron")

cluster_colors = {
    "Metropolie Kryzysu (Niedostępne)": "red",
    "Rynki Rozwojowe": "blue",
    "Oazy Dostępności i Rynki Lokalne": "green"
}

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

legenda_html = """
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 260px; height: 110px; 
     border:2px solid grey; z-index:9999; font-size:12px;
     background-color:white; opacity: 0.9; padding: 10px;
     font-family: Arial, sans-serif; border-radius: 5px;">
     <b>Profile rynków (K-means):</b><br>
     <i class="fa fa-circle fa-1x" style="color:red"></i> Metropolie Kryzysu (Niedostępne)<br>
     <i class="fa fa-circle fa-1x" style="color:blue"></i> Rynki Rozwojowe<br>
     <i class="fa fa-circle fa-1x" style="color:green"></i> Oazy Dostępności i Rynki Lokalne
     </div>
     """
mapa_folium.get_root().html.add_child(folium.Element(legenda_html))
mapa_folium.save("mapa_folium_projekt.html")

print("-> Wygenerowano interaktywną mapę Folium: mapa_folium_projekt.html")
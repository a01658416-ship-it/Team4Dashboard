import os
#import re
import json
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, HeatMapWithTime, MarkerCluster
import numpy as np
import osmnx as ox
import branca.colormap as cm
import geopandas as gpd
from shapely.geometry import Point


#Carolina Torres Aguirre A01658416

#Leonardo Ramirez Cardoso A01657266

#Juan Alberto Lopez Govantes A01655112

#Santiago Ulloa Flores A01571492

#Emilio Adrián Gutiérrez Terrones A01654071

# Page configuration
st.set_page_config(
    page_title="Crime Data Visualization Dashboard",
    page_icon="🗺️",
    layout="wide"
)

# Title and description
st.title("🗺️ Crime Data Visualization Dashboard - Mexico City")
st.markdown("Interactive crime data analysis with multiple visualization layers")

# Sidebar controls
st.sidebar.header("Dashboard Controls")

# ============================================================================
# DATA CLEANING FUNCTIONS (Adapted from final_eda.py)
# ============================================================================

def _strip_accents_capitalize(text):
    """
    Normalize text by removing accents and capitalizing the first letter of each word.
    Example: "álvaro obregón norte" -> "Alvaro Obregon Norte"
    """
    if pd.isna(text):
        return text
    
    # Remove accents
    repl = (("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),
            ("Á","A"),("É","E"),("Í","I"),("Ó","O"),("Ú","U"),
            ("ñ","n"),("Ñ","N"))
    for a, b in repl:
        text = text.replace(a, b)
    
    # Capitalize the first letter of each word
    words = text.split()
    words = [w.capitalize() for w in words]
    
    return " ".join(words)

def _fill_missing_alcaldias(crimes_df, geojson_path, alcaldia_column="NOMGEO"):
    """
    Fill missing 'alcaldia_hecho' using coordinates (lat/lon) mapped to Mexico City's alcaldías polygons.
    """
    if not os.path.exists(geojson_path):
        st.warning(f"GeoJSON file not found at {geojson_path}. Skipping spatial filling.")
        return crimes_df
    
    try:
        # Load the GeoJSON polygons
        gdf_municipalities = gpd.read_file(geojson_path).to_crs(epsg=4326)
        
        # Work only with rows that have coordinates
        coords_df = crimes_df.dropna(subset=["latitud", "longitud"]).copy()
        
        if not coords_df.empty:
            # Build Point geometries
            coords_df["geometry"] = [Point(xy) for xy in zip(coords_df["longitud"], coords_df["latitud"])]
            
            # Convert to GeoDataFrame
            crimes_gdf = gpd.GeoDataFrame(coords_df, geometry="geometry", crs="EPSG:4326")
            
            # Spatial join with municipality polygons
            joined = gpd.sjoin(crimes_gdf, gdf_municipalities, how="left", predicate="within")
            
            # Fill missing values
            crimes_df = crimes_df.copy()
            subset = crimes_df.loc[coords_df.index, "alcaldia_hecho"].copy()
            subset = subset.fillna(joined[alcaldia_column].reset_index(drop=True))
            crimes_df.loc[coords_df.index, "alcaldia_hecho"] = subset
    
    except Exception as e:
        st.warning(f"Error during spatial filling: {e}")
    
    return crimes_df

def _fill_to_unknown(df, column="alcaldia_hecho"):
    """
    Fill missing values in the specified alcaldía column with "Unknown".
    Also converts any entry equal to "CDMX (indeterminada)" to "Unknown".
    """
    df = df.copy()
    # Replace "CDMX (indeterminada)" with "Unknown"
    df[column] = df[column].replace("CDMX (indeterminada)", "Unknown")
    # Fill remaining NaN values
    df[column] = df[column].fillna("Unknown")
    return df

def clean_crime_data(df, geojson_path=None):
    """
    Clean and prepare crime data following the methodology from final_eda.py
    
    Steps:
    1. Handle missing values in alcaldia_hecho using coordinates
    2. Fill remaining missing values with "Unknown"
    3. Normalize alcaldía names
    4. Filter valid coordinates for Mexico City
    5. Convert date columns to datetime
    6. Remove duplicates
    """
    st.info("🧹 Starting data cleaning process...")
    
    # Original row count
    original_count = len(df)
    
    # Keep categoria_delito column if it exists
    required_cols = ['delito', 'alcaldia_hecho', 'latitud', 'longitud']
    if 'categoria_delito' in df.columns:
        required_cols.append('categoria_delito')
    if 'fecha_inicio' in df.columns:
        required_cols.append('fecha_inicio')
    if 'fecha_hecho' in df.columns:
        required_cols.append('fecha_hecho')
    
    # Select only required columns
    df = df[required_cols].copy()
    
    # Step 1: Handle missing alcaldías using coordinates (if GeoJSON provided)
    if geojson_path and os.path.exists(geojson_path):
        st.info("📍 Filling missing alcaldías using coordinate mapping...")
        df = _fill_missing_alcaldias(df, geojson_path)
    
    # Step 2: Fill remaining missing values with "Unknown"
    st.info("🔍 Filling remaining missing alcaldías with 'Unknown'...")
    df = _fill_to_unknown(df)
    
    # Step 3: Normalize alcaldía names
    st.info("✨ Normalizing alcaldía names...")
    df["alcaldia_hecho"] = df["alcaldia_hecho"].apply(_strip_accents_capitalize)
    
    # Step 4: Clean and validate coordinates
    st.info("🗺️ Validating coordinates...")
    
    # Remove rows with missing coordinates
    before_coord_filter = len(df)
    df = df.dropna(subset=['latitud', 'longitud'])
    
    # Filter valid coordinates for Mexico City (19.0 to 19.6 N, -99.4 to -98.9 W)
    df = df[(df['latitud'] >= 19.0) & (df['latitud'] <= 19.6) & 
            (df['longitud'] >= -99.4) & (df['longitud'] <= -98.9)]
    
    after_coord_filter = len(df)
    coord_removed = before_coord_filter - after_coord_filter
    
    # Step 5: Convert date columns to datetime
    st.info("📅 Converting date columns...")
    for date_col in ['fecha_inicio', 'fecha_hecho']:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Step 6: Normalize crime names and categories
    st.info("🏷️ Normalizing crime type names...")
    if 'delito' in df.columns:
        df['delito'] = df['delito'].str.strip().str.upper()
    if 'categoria_delito' in df.columns:
        df['categoria_delito'] = df['categoria_delito'].str.strip().str.upper()
    
    # Step 7: Remove duplicates
    st.info("🔄 Removing duplicates...")
    before_dedup = len(df)
    df = df.drop_duplicates()
    after_dedup = len(df)
    duplicates_removed = before_dedup - after_dedup
    
    # Final row count
    final_count = len(df)
    total_removed = original_count - final_count
    
    # Display cleaning summary
    st.success("✅ Data cleaning completed!")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Original Records", f"{original_count:,}")
    with col2:
        st.metric("Invalid Coordinates", f"{coord_removed:,}")
    with col3:
        st.metric("Duplicates Removed", f"{duplicates_removed:,}")
    with col4:
        st.metric("Final Records", f"{final_count:,}", 
                 delta=f"-{total_removed:,}" if total_removed > 0 else "0")
    
    return df
    return df

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_crime_data():
    """Load and clean crime data"""
    try:
        # Try Downloads folder (English and Spanish)
        downloads_path = Path.home() / "Downloads" / "carpetasFGJ_acumulado_2025_01.csv"
        if not downloads_path.exists():
            downloads_path = Path.home() / "Descargas" / "carpetasFGJ_acumulado_2025_01.csv"
        
        if not downloads_path.exists():
            st.error(f"File not found. Please ensure 'carpetasFGJ_acumulado_2025_01.csv' is in your Downloads folder.")
            return None
        
        # Read CSV with proper encoding
        df = pd.read_csv(downloads_path, encoding='latin-1', low_memory=False)
        
        # Clean column names (remove spaces)
        df.columns = df.columns.str.strip()
        
        # Check if GeoJSON exists for spatial filling
        geojson_options = [
            Path.home() / "Downloads" / "limite-de-las-alcaldias.json",
            Path.home() / "Descargas" / "limite-de-las-alcaldias.json",
            "limite-de-las-alcaldias.json"
        ]
        
        geojson_path = None
        for path in geojson_options:
            if os.path.exists(path):
                geojson_path = str(path)
                break
        
        # Clean the data
        df = clean_crime_data(df, geojson_path)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
with st.spinner("Loading crime data..."):
    crime_df = load_crime_data()

if crime_df is None:
    st.error("Could not load crime data. Please check the file path.")
    st.stop()

# ============================================================================
# FILTERS AND CONTROLS
# ============================================================================

st.sidebar.success(f"✅ Loaded {len(crime_df):,} clean crime records")

# Visualization type selector
viz_type = st.sidebar.selectbox(
    "Select Visualization Type",
    ["Base Map with Markers", "Heatmap", "Grid Sectors (Probability)", 
     "Dynamic Hot Spots", "Animated Timeline", "All Layers Combined"]
)

# Filter controls
st.sidebar.subheader("Filters")

# Date range filter
if 'fecha_hecho' in crime_df.columns:
    min_date = crime_df['fecha_hecho'].min()
    max_date = crime_df['fecha_hecho'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        crime_df_filtered = crime_df[
            (crime_df['fecha_hecho'] >= pd.Timestamp(date_range[0])) &
            (crime_df['fecha_hecho'] <= pd.Timestamp(date_range[1]))
        ].copy()
    else:
        crime_df_filtered = crime_df.copy()
else:
    crime_df_filtered = crime_df.copy()

# Crime category filter
if 'categoria_delito' in crime_df_filtered.columns:
    st.sidebar.subheader("Crime Category Filter")
    crime_categories = sorted(crime_df_filtered['categoria_delito'].dropna().unique())
    
    col_cat1, col_cat2 = st.sidebar.columns(2)
    with col_cat1:
        if st.button("Select All Categories"):
            st.session_state.selected_categories = crime_categories
    with col_cat2:
        if st.button("Clear Categories"):
            st.session_state.selected_categories = []
    
    if 'selected_categories' not in st.session_state:
        st.session_state.selected_categories = crime_categories
    
    selected_categories = st.sidebar.multiselect(
        "Select Crime Categories",
        crime_categories,
        default=st.session_state.selected_categories
    )
    
    if selected_categories:
        crime_df_filtered = crime_df_filtered[crime_df_filtered['categoria_delito'].isin(selected_categories)]

# Crime type filter
st.sidebar.subheader("Crime Type Filter")
crime_types = sorted(crime_df_filtered['delito'].unique())

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Select All Types"):
        st.session_state.selected_crimes = crime_types
with col2:
    if st.button("Clear Types"):
        st.session_state.selected_crimes = []

if 'selected_crimes' not in st.session_state:
    st.session_state.selected_crimes = crime_types[:5] if len(crime_types) > 5 else crime_types

selected_crimes = st.sidebar.multiselect(
    "Select Crime Types",
    crime_types,
    default=st.session_state.selected_crimes
)

if selected_crimes:
    crime_df_filtered = crime_df_filtered[crime_df_filtered['delito'].isin(selected_crimes)]

# Alcaldia (borough) filter
alcaldias = sorted(crime_df_filtered['alcaldia_hecho'].dropna().unique())
selected_alcaldias = st.sidebar.multiselect(
    "Select Alcaldías",
    alcaldias,
    default=alcaldias
)

if selected_alcaldias:
    crime_df_filtered = crime_df_filtered[crime_df_filtered['alcaldia_hecho'].isin(selected_alcaldias)]

# Additional layers toggle
st.sidebar.subheader("Additional Layers")
show_alcaldias = st.sidebar.checkbox("Show Alcaldía Boundaries", value=True)
show_schools = st.sidebar.checkbox("Show Schools", value=False)
show_hospitals = st.sidebar.checkbox("Show Hospitals", value=False)
show_metro = st.sidebar.checkbox("Show Metro Stations", value=False)
show_parking = st.sidebar.checkbox("Show Parking Areas", value=False)

# Grid sector settings
if viz_type in ["Grid Sectors (Probability)", "Dynamic Hot Spots", "All Layers Combined"]:
    st.sidebar.subheader("Grid Settings")
    grid_size = st.sidebar.slider("Grid Cell Size (km)", 0.5, 5.0, 1.0, 0.5)
    probability_threshold = st.sidebar.slider("Probability Threshold (%)", 0, 100, 50, 5)

# Animation settings
if viz_type == "Animated Timeline":
    st.sidebar.subheader("Animation Settings")
    time_window = st.sidebar.slider("Time Window (hours)", 4, 48, 24, 4)

# ============================================================================
# MAP CREATION FUNCTIONS
# ============================================================================

def create_base_map():
    """Create base Folium map centered on Mexico City"""
    center_lat = crime_df_filtered['latitud'].mean()
    center_lon = crime_df_filtered['longitud'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Add alternative tile layers
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
    
    return m

@st.cache_data
def get_schools_data():
    """Get schools data using OSMnx"""
    try:
        tags = {'amenity': 'school'}
        gdf = ox.features_from_place('Ciudad de México, Mexico', tags)
        return gdf
    except:
        return None

@st.cache_data
def get_hospitals_data():
    """Get hospitals data using OSMnx"""
    try:
        tags = {'amenity': 'hospital'}
        gdf = ox.features_from_place('Ciudad de México, Mexico', tags)
        return gdf
    except:
        return None

@st.cache_data
def get_metro_data():
    """Get metro stations using OSMnx"""
    try:
        tags = {'railway': 'station', 'station': 'subway'}
        gdf = ox.features_from_place('Ciudad de México, Mexico', tags)
        return gdf
    except:
        return None

@st.cache_data
def get_parking_data():
    """Get parking areas using OSMnx"""
    try:
        tags = {'amenity': 'parking'}
        gdf = ox.features_from_place('Ciudad de México, Mexico', tags)
        return gdf
    except:
        return None

def create_grid_sectors(df, cell_size_km=1.0):
    """Create grid sectors with crime probability"""
    # Convert km to degrees (approximate)
    cell_size = cell_size_km / 111.0
    
    # Create grid
    lat_bins = np.arange(df['latitud'].min(), df['latitud'].max() + cell_size, cell_size)
    lon_bins = np.arange(df['longitud'].min(), df['longitud'].max() + cell_size, cell_size)
    
    # Count crimes in each cell
    grid_counts = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    
    for i in range(len(lat_bins)-1):
        for j in range(len(lon_bins)-1):
            mask = (
                (df['latitud'] >= lat_bins[i]) & 
                (df['latitud'] < lat_bins[i+1]) &
                (df['longitud'] >= lon_bins[j]) & 
                (df['longitud'] < lon_bins[j+1])
            )
            grid_counts[i, j] = mask.sum()
    
    # Calculate probabilities
    total_crimes = grid_counts.sum()
    grid_probs = (grid_counts / total_crimes * 100) if total_crimes > 0 else grid_counts
    
    return lat_bins, lon_bins, grid_counts, grid_probs

def add_grid_to_map(m, lat_bins, lon_bins, grid_probs, threshold=0):
    """Add colored grid sectors to map"""
    # Create colormap
    max_prob = grid_probs.max()
    colormap = cm.LinearColormap(
        colors=['green', 'yellow', 'orange', 'red'],
        vmin=0,
        vmax=max_prob,
        caption='Crime Probability (%)'
    )
    
    # Add rectangles for each cell
    for i in range(len(lat_bins)-1):
        for j in range(len(lon_bins)-1):
            if grid_probs[i, j] >= threshold:
                folium.Rectangle(
                    bounds=[
                        [lat_bins[i], lon_bins[j]],
                        [lat_bins[i+1], lon_bins[j+1]]
                    ],
                    color=colormap(grid_probs[i, j]),
                    fill=True,
                    fillColor=colormap(grid_probs[i, j]),
                    fillOpacity=0.5,
                    popup=f"Probability: {grid_probs[i, j]:.2f}%<br>Count: {int(grid_probs[i, j] * len(crime_df_filtered) / 100)}"
                ).add_to(m)
    
    colormap.add_to(m)

def add_alcaldias_to_map(m, geojson_path, crime_counts_df):
    """
    Add alcaldía boundaries to map with crime count information
    
    Parameters:
    - m: folium map object
    - geojson_path: path to the GeoJSON file with alcaldía boundaries
    - crime_counts_df: DataFrame with alcaldía names and crime counts
    """
    if not os.path.exists(geojson_path):
        st.warning(f"GeoJSON file not found at {geojson_path}")
        return
    
    try:
        # Load GeoJSON
        with open(geojson_path, "r", encoding="utf-8") as f:
            alcaldias_geo = json.load(f)
        
        # Create a dictionary of crime counts by alcaldía (normalized)
        crime_dict = {}
        for idx, row in crime_counts_df.iterrows():
            normalized_name = _strip_accents_capitalize(row['alcaldia_hecho'])
            crime_dict[normalized_name.upper()] = row['count']
        
        # Create a colormap for the choropleth
        max_crimes = max(crime_dict.values()) if crime_dict else 1
        colormap = cm.LinearColormap(
            colors=['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026'],
            vmin=0,
            vmax=max_crimes,
            caption='Number of Crimes'
        )
        
        def style_function(feature):
            """Style each alcaldía based on crime count"""
            alcaldia_name = feature['properties'].get('NOMGEO', '').upper()
            crime_count = crime_dict.get(alcaldia_name, 0)
            
            return {
                'fillColor': colormap(crime_count) if crime_count > 0 else '#cccccc',
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0.6
            }
        
        def highlight_function(feature):
            """Highlight on hover"""
            return {
                'fillColor': '#ffff00',
                'color': 'black',
                'weight': 3,
                'fillOpacity': 0.8
            }
        
        # Add GeoJSON layer with tooltips and popups
        folium.GeoJson(
            alcaldias_geo,
            name="Alcaldías",
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['NOMGEO'],
                aliases=['Alcaldía:'],
                localize=True
            ),
            popup=folium.GeoJsonPopup(
                fields=['NOMGEO'],
                aliases=['Alcaldía:'],
                localize=True
            )
        ).add_to(m)
        
        # Add custom popups with crime counts
        gdf = gpd.read_file(geojson_path)
        for idx, row in gdf.iterrows():
            alcaldia_name = row['NOMGEO'].upper()
            crime_count = crime_dict.get(alcaldia_name, 0)
            
            # Get centroid for label placement
            centroid = row.geometry.centroid
            
            # Add a marker at the centroid with crime count
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(html=f'''
                    <div style="
                        font-size: 10px;
                        font-weight: bold;
                        color: white;
                        text-align: center;
                        background-color: rgba(0, 0, 0, 0.7);
                        border-radius: 5px;
                        padding: 2px 5px;
                        white-space: nowrap;
                    ">
                        {row['NOMGEO']}<br>{crime_count:,} crimes
                    </div>
                ''')
            ).add_to(m)
        
        # Add colormap to map
        colormap.add_to(m)
        
    except Exception as e:
        st.warning(f"Error adding alcaldías to map: {e}")

@st.cache_data
def get_crime_counts_by_alcaldia(df):
    """Get crime counts grouped by alcaldía"""
    return df.groupby('alcaldia_hecho').size().reset_index(name='count')

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================

st.subheader(f"Visualization: {viz_type}")

# Get GeoJSON path for alcaldías
geojson_options = [
    Path.home() / "Downloads" / "limite-de-las-alcaldias.json",
    Path.home() / "Descargas" / "limite-de-las-alcaldias.json",
    "limite-de-las-alcaldias.json"
]

geojson_path = None
for path in geojson_options:
    if os.path.exists(path):
        geojson_path = str(path)
        break

# Calculate crime counts by alcaldía for the current filtered data
crime_counts = get_crime_counts_by_alcaldia(crime_df_filtered)

if viz_type == "Base Map with Markers":
    m = create_base_map()
    
    # Add alcaldías layer if enabled
    if show_alcaldias and geojson_path:
        add_alcaldias_to_map(m, geojson_path, crime_counts)
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Sample data if too many points
    sample_df = crime_df_filtered.sample(min(1000, len(crime_df_filtered)))
    
    for idx, row in sample_df.iterrows():
        folium.CircleMarker(
            location=[row['latitud'], row['longitud']],
            radius=3,
            popup=f"<b>Crime:</b> {row['delito']}<br><b>Date:</b> {row['fecha_hecho']}<br><b>Alcaldía:</b> {row['alcaldia_hecho']}",
            color='red',
            fill=True,
            fillColor='red'
        ).add_to(marker_cluster)

elif viz_type == "Heatmap":
    m = create_base_map()
    
    # Add alcaldías layer if enabled
    if show_alcaldias and geojson_path:
        add_alcaldias_to_map(m, geojson_path, crime_counts)
    
    # Prepare heatmap data
    heat_data = crime_df_filtered[['latitud', 'longitud']].values.tolist()
    
    # Add heatmap
    HeatMap(
        heat_data,
        radius=15,
        blur=25,
        max_zoom=13,
        name='Crime Heatmap'
    ).add_to(m)

elif viz_type == "Grid Sectors (Probability)":
    m = create_base_map()
    
    # Add alcaldías layer if enabled
    if show_alcaldias and geojson_path:
        add_alcaldias_to_map(m, geojson_path, crime_counts)
    
    # Create and add grid
    lat_bins, lon_bins, grid_counts, grid_probs = create_grid_sectors(crime_df_filtered, grid_size)
    add_grid_to_map(m, lat_bins, lon_bins, grid_probs, probability_threshold/100)

elif viz_type == "Dynamic Hot Spots":
    m = create_base_map()
    
    # Add alcaldías layer if enabled
    if show_alcaldias and geojson_path:
        add_alcaldias_to_map(m, geojson_path, crime_counts)
    
    # Create grid with dynamic threshold
    lat_bins, lon_bins, grid_counts, grid_probs = create_grid_sectors(crime_df_filtered, grid_size)
    
    # Only show hot spots above threshold
    add_grid_to_map(m, lat_bins, lon_bins, grid_probs, probability_threshold/100)
    
    # Add heatmap overlay
    heat_data = crime_df_filtered[['latitud', 'longitud']].values.tolist()
    HeatMap(
        heat_data,
        radius=20,
        blur=30,
        max_zoom=13,
        name='Hot Spots'
    ).add_to(m)

elif viz_type == "Animated Timeline":
    m = create_base_map()
    
    # Add alcaldías layer if enabled
    if show_alcaldias and geojson_path:
        add_alcaldias_to_map(m, geojson_path, crime_counts)
    
    # Prepare time-series data
    crime_df_sorted = crime_df_filtered.sort_values('fecha_hecho')
    
    # Group by time windows
    time_groups = []
    time_labels = []
    
    start_date = crime_df_sorted['fecha_hecho'].min()
    end_date = crime_df_sorted['fecha_hecho'].max()
    current_date = start_date
    
    while current_date <= end_date:
        next_date = current_date + timedelta(hours=time_window)
        mask = (crime_df_sorted['fecha_hecho'] >= current_date) & (crime_df_sorted['fecha_hecho'] < next_date)
        group_data = crime_df_sorted[mask][['latitud', 'longitud']].values.tolist()
        
        if group_data:
            time_groups.append(group_data)
            time_labels.append(current_date.strftime('%Y-%m-%d %H:%M'))
        
        current_date = next_date
    
    if time_groups:
        # Add animated heatmap
        HeatMapWithTime(
            time_groups,
            index=time_labels,
            auto_play=True,
            radius=15,
            max_opacity=0.8,
            name='Animated Crime Timeline'
        ).add_to(m)

elif viz_type == "All Layers Combined":
    m = create_base_map()
    
    # Add alcaldías layer if enabled
    if show_alcaldias and geojson_path:
        add_alcaldias_to_map(m, geojson_path, crime_counts)
    
    # Add grid sectors
    lat_bins, lon_bins, grid_counts, grid_probs = create_grid_sectors(crime_df_filtered, grid_size)
    add_grid_to_map(m, lat_bins, lon_bins, grid_probs, probability_threshold/100)
    
    # Add heatmap
    heat_data = crime_df_filtered[['latitud', 'longitud']].values.tolist()
    HeatMap(
        heat_data,
        radius=15,
        blur=25,
        max_zoom=13,
        name='Crime Heatmap'
    ).add_to(m)

# Add additional layers if selected
if show_schools:
    with st.spinner("Loading schools..."):
        schools = get_schools_data()
        if schools is not None:
            school_group = folium.FeatureGroup(name='Schools')
            for idx, row in schools.iterrows():
                if hasattr(row.geometry, 'centroid'):
                    centroid = row.geometry.centroid
                    folium.Marker(
                        location=[centroid.y, centroid.x],
                        popup='School',
                        icon=folium.Icon(color='blue', icon='graduation-cap', prefix='fa')
                    ).add_to(school_group)
            school_group.add_to(m)

if show_hospitals:
    with st.spinner("Loading hospitals..."):
        hospitals = get_hospitals_data()
        if hospitals is not None:
            hospital_group = folium.FeatureGroup(name='Hospitals')
            for idx, row in hospitals.iterrows():
                if hasattr(row.geometry, 'centroid'):
                    centroid = row.geometry.centroid
                    folium.Marker(
                        location=[centroid.y, centroid.x],
                        popup='Hospital',
                        icon=folium.Icon(color='red', icon='plus', prefix='fa')
                    ).add_to(hospital_group)
            hospital_group.add_to(m)

if show_metro:
    with st.spinner("Loading metro stations..."):
        metro = get_metro_data()
        if metro is not None:
            metro_group = folium.FeatureGroup(name='Metro Stations')
            for idx, row in metro.iterrows():
                if hasattr(row.geometry, 'centroid'):
                    centroid = row.geometry.centroid
                    folium.Marker(
                        location=[centroid.y, centroid.x],
                        popup='Metro Station',
                        icon=folium.Icon(color='orange', icon='subway', prefix='fa')
                    ).add_to(metro_group)
            metro_group.add_to(m)

if show_parking:
    with st.spinner("Loading parking areas..."):
        parking = get_parking_data()
        if parking is not None:
            parking_group = folium.FeatureGroup(name='Parking')
            for idx, row in parking.iterrows():
                if hasattr(row.geometry, 'centroid'):
                    centroid = row.geometry.centroid
                    folium.CircleMarker(
                        location=[centroid.y, centroid.x],
                        radius=5,
                        popup='Parking',
                        color='purple',
                        fill=True,
                        fillColor='purple'
                    ).add_to(parking_group)
            parking_group.add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Display map
st.components.v1.html(m._repr_html_(), height=600)

# Download button
st.sidebar.subheader("Export Map")
if st.sidebar.button("Download Map as HTML"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"crime_map_{viz_type.replace(' ', '_')}_{timestamp}.html"
    m.save(filename)
    st.sidebar.success(f"Map saved as {filename}")

# ============================================================================
# STATISTICS
# ============================================================================

st.subheader("Crime Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Crimes", f"{len(crime_df_filtered):,}")

with col2:
    st.metric("Crime Types", len(crime_df_filtered['delito'].unique()))

with col3:
    st.metric("Alcaldías", len(crime_df_filtered['alcaldia_hecho'].unique()))

with col4:
    if 'fecha_hecho' in crime_df_filtered.columns:
        date_range_days = (crime_df_filtered['fecha_hecho'].max() - crime_df_filtered['fecha_hecho'].min()).days
        st.metric("Date Range", f"{date_range_days} days")

# Top crimes chart
st.subheader("Top 10 Crime Types")
top_crimes = crime_df_filtered['delito'].value_counts().head(10)
st.bar_chart(top_crimes)

# Crimes by Alcaldía
st.subheader("Crimes by Alcaldía")
crimes_by_alcaldia = crime_df_filtered['alcaldia_hecho'].value_counts()
st.bar_chart(crimes_by_alcaldia)

# ============================================================================
# DATA QUALITY REPORT
# ============================================================================

with st.expander("📊 Data Quality Report"):
    st.markdown("### Data Cleaning Summary")
    st.markdown("""
    The data cleaning process follows best practices from the EDA methodology:
    
    **Steps Applied:**
    1. ✅ **Coordinate-based filling**: Missing alcaldías filled using lat/lon spatial joins
    2. ✅ **Unknown handling**: Remaining missing values marked as "Unknown"
    3. ✅ **Name normalization**: Removed accents and standardized capitalization
    4. ✅ **Coordinate validation**: Filtered to valid Mexico City boundaries (19.0-19.6°N, -99.4--98.9°W)
    5. ✅ **Date conversion**: Converted date strings to proper datetime objects
    6. ✅ **Crime type normalization**: Standardized crime names (uppercase, trimmed)
    7. ✅ **Duplicate removal**: Eliminated duplicate records
    
    **Quality Metrics:**
    - Missing values handled: ✓
    - Invalid coordinates removed: ✓
    - Data normalized: ✓
    - Duplicates eliminated: ✓
    """)
    
    st.markdown("### Current Dataset Info")
    st.write(f"**Total Records:** {len(crime_df_filtered):,}")
    st.write(f"**Unique Alcaldías:** {crime_df_filtered['alcaldia_hecho'].nunique()}")
    st.write(f"**Unique Crime Types:** {crime_df_filtered['delito'].nunique()}")
    st.write(f"**Date Range:** {crime_df_filtered['fecha_hecho'].min()} to {crime_df_filtered['fecha_hecho'].max()}")

# Footer
st.markdown("---")
st.markdown("""
### Documentation
**Features:**
- 🧹 **Professional Data Cleaning**: Follows EDA best practices with coordinate validation, name normalization, and missing value handling
- 🗺️ **Multiple Visualizations**: Heatmaps, Grid Sectors, Animated Timeline with time-based analysis
- 🔍 **Interactive Filtering**: Filter by date range, crime type, and location
- 📊 **Dynamic Hot Spots**: Probability-based detection with adjustable thresholds
- 🏫 **Additional Layers**: Schools, Hospitals, Metro Stations, Parking areas via OSMnx
- 📥 **Export Capability**: Download maps as HTML files
- 📈 **Statistics Dashboard**: Real-time metrics and visualizations

**Data Sources:**
- Crime data: Fiscalía General de Justicia CDMX (2025-01)
- Geographic data: OpenStreetMap via OSMnx
- Alcaldía boundaries: limite-de-las-alcaldias.json

**Data Quality:**
- Coordinates validated for Mexico City boundaries
- Missing alcaldías filled using spatial joins
- All text normalized (accents removed, proper capitalization)
- Duplicates removed
- Invalid records filtered

**Technologies:** Python, Folium, Streamlit, OSMnx, GeoPandas, Pandas
""")
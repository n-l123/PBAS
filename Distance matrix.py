import pandas as pd
import requests
import numpy as np

# 1. Load the store data
df_stores = pd.read_csv('data/store general.csv')

# 2. Hardcode DC Tilburg location 
dc_lon = 5.115950
dc_lat = 51.578055

# 3. Create a list of all locations (DC is placed first, followed by the stores)
locations = [(dc_lon, dc_lat, 'DC_Tilburg')]
for index, row in df_stores.iterrows():
    locations.append((row['Longitude'], row['Latitude'], f"Store_{row['Store nr']}"))

# 4. Format coordinates for the OSRM API
coords_string = ';'.join([f"{lon},{lat}" for lon, lat, name in locations])

# 5. Call the free OSRM Table API
url = f"http://router.project-osrm.org/table/v1/driving/{coords_string}?annotations=duration,distance"

print("Fetching data from OSRM API...")
response = requests.get(url)
data = response.json()

# 6. Parse the data, apply truck factor, round up times, and save it
if data.get('code') == 'Ok':
    # Create row/column labels
    labels = [loc[2] for loc in locations]
    
    # Extract Time Matrix (OSRM returns duration in seconds)
    df_time = pd.DataFrame(data['durations'], index=labels, columns=labels)
    
    # Convert seconds to minutes and apply a 1.15 truck penalty factor
    TRUCK_FACTOR = 1.15
    df_time_minutes_truck = (df_time / 60) * TRUCK_FACTOR
    
    # Round up to the nearest 5 minutes
    df_time_minutes_rounded = np.ceil(df_time_minutes_truck / 5) * 5
    
    # Extract Distance Matrix (OSRM returns distance in meters)
    df_distance = pd.DataFrame(data['distances'], index=labels, columns=labels)
    df_distance_km = df_distance / 1000  # Convert meters to kilometers
    
    # Save to CSV files
    df_time_minutes_rounded.to_csv('time_matrix_minutes_truck_rounded.csv')
    df_distance_km.to_csv('distance_matrix_km.csv')
    
    print("Success! Matrices saved to 'time_matrix_minutes_truck_rounded.csv' and 'distance_matrix_km.csv'.")
    
    # Print a small sample to the console
    print("\nSample Time Matrix (Truck factored and rounded up to nearest 5 mins):")
    print(df_time_minutes_rounded.iloc[:4, :4])
else:
    print("Error fetching data:", data.get('message'))
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# 1. Load the new clean CSV data
schedule_df = pd.read_csv('data/old.csv')
stores_df = pd.read_csv('data/store general.csv')
trucks_df = pd.read_csv('data/truck types.csv')
info_df = pd.read_csv('data/information dc.csv', header=None, index_col=0)
info_df = info_df.T

# Clean column names just in case there are trailing spaces
schedule_df.columns = schedule_df.columns.str.strip()
stores_df.columns = stores_df.columns.str.strip()
trucks_df.columns = trucks_df.columns.str.strip()
info_df.columns = info_df.columns.str.strip()

# 2. Filter out electric trucks for the baseline
traditional_trucks = trucks_df[~trucks_df['Trucktype'].str.contains('Electric', na=False, case=False)].copy()
truck_data = traditional_trucks.set_index('Trucktype').to_dict('index')

# Clean numeric data in truck_data
for t in truck_data:
    truck_data[t]['Capacity Ambient'] = int(truck_data[t]['Capacity Ambient'])
    truck_data[t]['Cost per km'] = float(truck_data[t]['Cost per km'])
    truck_data[t]['Cost per hour'] = float(truck_data[t]['Cost per hour'])
    truck_data[t]['kg CO2 emission per km'] = float(truck_data[t]['kg CO2 emission per km'])

# 3. Merge schedule with store distances and constraints
df = schedule_df.merge(
    stores_df[['Store nr', 'Distance to DC (km)', 'Driving time to DC', 'Max. allowed truck type']], 
    left_on='Store', 
    right_on='Store nr', 
    how='left'
)

# Convert driving time to decimal hours
def time_to_hours(time_str):
    if pd.isna(time_str): return 0
    parts = str(time_str).split(':')
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    s = int(parts[2]) if len(parts) > 2 else 0
    return h + m / 60.0 + s / 3600.0

df['Driving time hours'] = df['Driving time to DC'].apply(time_to_hours)

# 4. Logic to assign the cheapest valid truck combination per trip
truck_hierarchy = ['Euro', 'City', 'Rigid', 'Small']

def assign_baseline_trucks(row):
    volume = row['Total volume']
    max_truck = row['Max. allowed truck type']
    dist_km = row['Distance to DC (km)']
    time_h = row['Driving time hours']
    
    if pd.isna(volume) or volume == 0:
        return None, 0, 0, 0
        
    # Determine which trucks are allowed to enter the city
    allowed_trucks = []
    start_adding = False
    for t in truck_hierarchy:
        if t == max_truck:
            start_adding = True
        if start_adding:
            allowed_trucks.append(t)
            
    if not allowed_trucks:
        allowed_trucks = truck_hierarchy # Fallback if data is missing
        
    best_cost = float('inf')
    best_combo = []
    best_co2 = 0
    
    # Check combinations up to a reasonable max number of trucks
    max_trips = int(volume // 18) + 2 
    
    for k in range(1, max_trips + 1):
        for combo in itertools.combinations_with_replacement(allowed_trucks, k):
            total_cap = sum(truck_data[t]['Capacity Ambient'] for t in combo)
            
            if total_cap >= volume:
                cost = 0
                co2 = 0
                for t in combo:
                    # Point-to-point distance (roundtrip)
                    dist_cost = 2 * dist_km * truck_data[t]['Cost per km']
                    # Time: 2x driving + 30m load + 30m unload
                    time_cost = (2 * time_h + 1.0) * truck_data[t]['Cost per hour'] 
                    
                    cost += (dist_cost + time_cost)
                    co2 += 2 * dist_km * truck_data[t]['kg CO2 emission per km']
                    
                if cost < best_cost:
                    best_cost = cost
                    best_combo = list(combo)
                    best_co2 = co2
                    
    return best_combo, best_cost, best_co2, sum(2 * dist_km for _ in best_combo)

# 5. Apply calculation
df[['Assigned Trucks', 'Trip Cost', 'Trip CO2', 'Trip Distance (km)']] = df.apply(
    lambda row: pd.Series(assign_baseline_trucks(row)), axis=1
)

# 6. Aggregate Baseline KPIs
total_weekly_cost = df['Trip Cost'].sum()
total_weekly_co2 = df['Trip CO2'].sum()
total_weekly_distance = df['Trip Distance (km)'].sum()
total_trucks_dispatched = df['Assigned Trucks'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()

print("=== BASELINE CURRENT SITUATION ===")
print(f"Total Weekly Cost: €{total_weekly_cost:,.2f}")
print(f"Total Weekly CO2 Emissions: {total_weekly_co2:,.2f} kg")
print(f"Total Weekly Distance Driven: {total_weekly_distance:,.2f} km")
print(f"Total Trucks Dispatched: {total_trucks_dispatched}")

# 7. Calculate Dispatched Capacity for the chart
df['Dispatched Capacity'] = df['Assigned Trucks'].apply(
    lambda trucks: sum(truck_data[t]['Capacity Ambient'] for t in trucks) if isinstance(trucks, list) else 0
)

# 8. Group data by Day of Week
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
df['Day of Week'] = pd.Categorical(df['Day of Week'], categories=day_order, ordered=True)
daily_stats = df.groupby('Day of Week', observed=False)[['Total volume', 'Dispatched Capacity']].sum().reset_index()


# Function to calculate distance between two GPS coordinates
def store_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# 3. Calculate distance to the nearest neighboring store for every store
lats = stores_df['Latitude'].values
lons = stores_df['Longitude'].values
n = len(stores_df)
min_dists = []
nearest_neighbors_types = []

for i in range(n):
    dists = store_distance(lats[i], lons[i], lats, lons)
    dists[i] = np.inf # Ignore distance to itself
    min_idx = np.argmin(dists)
    min_dists.append(np.min(dists))
    nearest_neighbors_types.append(stores_df['Max. allowed truck type'].iloc[min_idx])
    

stores_df['Nearest Neighbor (km)'] = min_dists
stores_df['Nearest_Store_Type'] = nearest_neighbors_types
avg_dist = np.mean(min_dists)

# Check how many stores share the SAME truck type as their closest neighbor
stores_df['Same_Truck_Type_As_Neighbor'] = stores_df['Max. allowed truck type'] == stores_df['Nearest_Store_Type']

# print metrics about nearest neighbor distances
print(f"Average distance to nearest store: {avg_dist:.2f} km")
pct_within_10km = (np.array(min_dists) <= 10).mean() * 100
pct_within_5km = (np.array(min_dists) <= 5).mean() * 100
print(f"Stores within 10km of another store: {pct_within_10km:.1f}%")
print(f"Stores within 5km of another store: {pct_within_5km:.1f}%")
print("Store Types Breakdown:")
print(stores_df['Max. allowed truck type'].value_counts())
print("\nNeighbors with same truck type:")
same_type_pct = stores_df['Same_Truck_Type_As_Neighbor'].mean() * 100
print(f"{same_type_pct:.1f}% of stores have the SAME truck restriction as their nearest neighbor.")
# only at the stores that are close
print("\nFor stores that are within 5km of another store:")
close_stores = stores_df[stores_df['Nearest Neighbor (km)'] <= 5]
close_same_type_pct = close_stores['Same_Truck_Type_As_Neighbor'].mean() * 100
print(f"Of the stores within 5km of each other, {close_same_type_pct:.1f}% have the SAME truck restriction.")

#  Generate and save the bar chart of Total Volume vs Dispatched Capacity
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(daily_stats['Day of Week']))
width = 0.35

ax.bar(x - width/2, daily_stats['Total volume'], width, label='Actual Volume Delivered', color='#00ADE6')
ax.bar(x + width/2, daily_stats['Dispatched Capacity'], width, label='Dispatched Truck Capacity', color='#78B833')

ax.set_ylabel('Containers (Rollcages)', fontsize=12, fontweight='bold', color='#003B64')
ax.set_title('Weekly Inefficiency: Unused Truck Capacity per Day', fontsize=14, fontweight='bold', color='#003B64')
ax.set_xticks(x)
ax.set_xticklabels(daily_stats['Day of Week'], fontsize=11)
ax.legend(fontsize=11)

# Formatting grid and spines
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#003B64')
ax.spines['bottom'].set_color('#003B64')
ax.tick_params(colors='#003B64', which='both')

plt.tight_layout()
plt.savefig('plots/capacity_vs_volume.png', dpi=300)

# Map of Stores and DC
plt.figure(figsize=(10, 8))
sns.scatterplot(data=stores_df, x='Longitude', y='Latitude', size='Distance to DC (km)', sizes=(20, 200), color='blue', alpha=0.6)

dc_lon = info_df['Longitude'].iloc[0]
dc_lat = info_df['Latitude'].iloc[0]
plt.scatter(dc_lon, dc_lat, color='red', marker='*', s=400, label='DC (Distribution Center)')
plt.title('Geographical Network: Stores relative to Distribution Center')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/store_map.png')
plt.close()

# Bar chart of truck restrictions among neighboring stores
plt.figure(figsize=(8, 6))
sns.countplot(data=close_stores, x='Max. allowed truck type', hue='Same_Truck_Type_As_Neighbor', palette='Set2')
plt.title('Truck Restrictions among Clustered Stores (< 5km apart)')
plt.xlabel('Maximum Allowed Truck Type')
plt.ylabel('Number of Stores')
plt.legend(title='Shares Restriction\nwith Nearest Neighbor', labels=['Different', 'Same'])
plt.tight_layout()
plt.savefig('plots/neighbor_truck_types.png')
plt.close()

# Pie Chart of the Store Types
type_counts = stores_df['Max. allowed truck type'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140, 
        colors=['#99DDF4', '#00ADE6', '#0072CE', '#003B64'],
        explode=[0.05, 0, 0, 0], shadow=False)
plt.title('Network Flexibility: Store Access Restrictions', fontsize=16, fontweight='bold', pad=20)
plt.legend(title='Max Allowed Truck', loc="best")
plt.tight_layout()
plt.savefig('plots/store_types_pie.png')
plt.close()

# Donut chart of nearest neighbor truck type similarity
sizes = [
    stores_df['Same_Truck_Type_As_Neighbor'].mean() * 100,
    100 - stores_df['Same_Truck_Type_As_Neighbor'].mean() * 100
]
fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    sizes, 
    labels=['Same Truck Restriction\nas Nearest Neighbor', 'Different Restriction'], 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=['#0072CE', '#E0E0E0'], 
    wedgeprops=dict(width=0.4, edgecolor='white', linewidth=3), # 'width' creates the donut hole
    textprops=dict(fontsize=12, fontweight='bold', color='#003B64')
)

# Text on chart
autotexts[0].set_color('black')
autotexts[0].set_fontsize(16)
autotexts[1].set_color('black')
autotexts[1].set_fontsize(14)
plt.text(0, 0, 'High\nCompatibility', ha='center', va='center', fontsize=14, fontweight='bold', color='#003B64')
plt.title('Store Compatibility for Neighboring Stores', fontsize=16, fontweight='bold', color='#003B64', pad=20)
plt.tight_layout()
plt.savefig('plots/neighbor_compatibility_donut.png')
plt.close()

# Store to neighbor heatmap
matrix = pd.crosstab(
    stores_df['Max. allowed truck type'], 
    stores_df['Nearest_Store_Type'], 
    margins=True, 
    margins_name='Total'
)
ordered_cols = ['Euro', 'City', 'Small', 'Rigid', 'Total']
matrix = matrix.reindex(index=ordered_cols, columns=ordered_cols, fill_value=0)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(matrix, annot=True, cmap='Blues', fmt='d', linewidths=2, linecolor='white', cbar=False,
            annot_kws={'size': 16, 'weight': 'bold'}, ax=ax)
for i in range(4): # fpr every type
    # Draw a rectangle on cell (i, i)
    rect = patches.Rectangle((i, i), 1, 1, fill=False, edgecolor='#FF7F0E', lw=4, zorder=10)
    ax.add_patch(rect)
# extra text box
math_text = "Highlighted Diagonal:\n28 + 1 = 29 Compatible Stores\n\n(29 / 49 Total = 59.2%)"
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#FF7F0E', lw=2)
ax.text(1.05, 0.5, math_text, transform=ax.transAxes, fontsize=14,
        verticalalignment='center', bbox=props, fontweight='bold', color='#003B64')
plt.title('Store-to-Neighbor Access Matrix\nShowing the 59.2% Compatible Pairs', 
          fontsize=16, fontweight='bold', color='#003B64', pad=20)
plt.xlabel('Nearest Neighbor Maximum Truck Allowed', fontsize=12, fontweight='bold', color='#003B64')
plt.ylabel('Store Maximum Truck Allowed', fontsize=12, fontweight='bold', color='#003B64')
plt.yticks(rotation=0)
plt.subplots_adjust(right=0.75) 
plt.savefig('plots/heatmap_with_totals_highlighted.png', bbox_inches='tight', dpi=300)
plt.close()


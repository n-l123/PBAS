import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

# 1. Load the new clean CSV data
schedule_df = pd.read_csv('data/old.csv')
stores_df = pd.read_csv('data/store general.csv')
trucks_df = pd.read_csv('data/truck types.csv')

# Clean column names just in case there are trailing spaces
schedule_df.columns = schedule_df.columns.str.strip()
stores_df.columns = stores_df.columns.str.strip()
trucks_df.columns = trucks_df.columns.str.strip()

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

# 9. Generate and save the bar chart
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
print("Chart created successfully as capacity_vs_volume.png")

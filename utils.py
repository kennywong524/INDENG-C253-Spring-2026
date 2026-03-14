import numpy as np
import pandas as pd
import folium
import itertools
import time
import re
import subprocess
import os


# Here we provide a helper function populate_shared_ride_lengths() to calculate the driving cost and cost allocation of the shared ride involving any two riders i and j.
# The following notations denote choices of the pick-up and drop-off order for a shared ride between rider i and j. They are used in the calculation of the shortest path for a shared ride.
IIJJ = 0  # pick up rider i - drop off rider i - pick up rider j - drop off rider j
IJIJ = 1  # pick up rider i - pick up rider j - drop off rider i - drop off rider j
IJJI = 2  # pick up rider i - pick up rider j - drop off rider j - drop off rider i
JIJI = 3  # pick up rider j - pick up rider i - drop off rider j - drop off rider i
JIIJ = 4  # pick up rider j - drop off rider i - pick up rider i - drop off rider j


# Function to calculate the shortest trip length for a shared ride between two riders
def populate_shared_ride_lengths(origin_i, destination_i, origin_j, destination_j):
    """
    :param origin_i: (lat, lon) of rider i's origin
    :param destination_i: (lat, lon) of rider i's destination
    :param origin_j: (lat, lon) of rider j's origin
    :param destination_j: (lat, lon) of rider j's destination
    :return trip_length: the total length of the shared ride
    :return shared_length: the length of the shared part of the shared ride
    :return i_solo_length: the solo length of rider i in the shared ride
    :return j_solo_length: the solo length of rider j in the shared ride
    :return trip_order: the pick-up and drop-off order for the shared trip, denoted by 0 to 4 (IIJJ, IJIJ, IJJI, JIJI, JIIJ)
    """

    # calculate the matrix of pairwise haversine distances between the four points
    data = np.array([origin_i, destination_i, origin_j, destination_j])  # create a 4*2 shaped numpy array
    data = np.deg2rad(data)  # convert to radians
    lat = data[:, 0]  # extract the latitudes
    lon = data[:, 1]  # extract the longitudes
    # elementwise differentiations for lats & lons
    diff_lat = lat[:, None] - lat
    diff_lon = lon[:, None] - lon
    # calculate the distance matrix (in miles)
    d = np.sin(diff_lat/2)**2 + np.cos(lat[:, None])*np.cos(lat) * np.sin(diff_lon/2)**2
    distance_matrix = 2 * 6371 * np.arcsin(np.sqrt(d)) * 0.621371

    # four auxiliary distance submatrices
    # origin_i to destination_i, origin_j to destination_j
    O0D0 = np.repeat(
        np.array([distance_matrix[0, 1], distance_matrix[2, 3]])[
            :, np.newaxis, np.newaxis
        ],
        2,
        axis=1,
    )
    # (origin_i, origin_j) to (origin_i, origin_j)
    O0O1 = distance_matrix[[0, 2], :][:, [0, 2]][:, :, np.newaxis]
    # (destination_i, destination_j) to (destination_i, destination_j)
    D0D1 = distance_matrix[[1, 3], :][:, [1, 3]][:, :, np.newaxis]
    # (origin_i, origin_j) to (destination_i, destination_j)
    O0D1 = distance_matrix[[0, 2], :][:, [1, 3]][:, :, np.newaxis]

    # This method was adapted from code written by SeJIstien Martin for the paper “Detours in Shared Rides”.
    def match_efficiency_single(O0D0, O0O1, D0D1, O0D1):
        """Calculate the request length matrix, shared cost, solo cost,
        and the best pick-up and drop-off order for all rider type pairs."""

        n_riders = 2

        # Compute shortest ordering for each match
        IIJJ_triptime = O0D0 + O0D0.transpose(1, 0, 2)
        IJIJ_triptime = (
            O0O1 + O0D1.transpose(1, 0, 2) + D0D1
        )  # route IJIJ, we can transpose this matrix to get JIJI
        IJJI_triptime = (
            O0O1 + O0D0.transpose(1, 0, 2) + D0D1.transpose(1, 0, 2)
        )  # route IJJI, we can transpose this matrix to get JIIJ
        triptime_possibilities = np.stack(
            (
                IIJJ_triptime,  # 0
                IJIJ_triptime,  # 1
                IJJI_triptime,  # 2
                IJIJ_triptime.transpose(1, 0, 2),  # 3
                IJJI_triptime.transpose(1, 0, 2),  # 4
            ),
            axis=2,
        )
        best_triptime_choice = np.argmin(triptime_possibilities, axis=2)
        best_triptime = triptime_possibilities[
            np.arange(n_riders)[:, np.newaxis, np.newaxis],
            np.arange(n_riders)[np.newaxis, :, np.newaxis],
            best_triptime_choice,
            np.arange(1)[np.newaxis, np.newaxis, :],
        ]

        shared_length_matrix = np.zeros(
            (n_riders, n_riders)
        )  # matrix of the length of shared part of the trip
        i_solo_length_matrix = np.zeros(
            (n_riders, n_riders)
        )  # matrix of the length of type i's solo trip
        j_solo_length_matrix = np.zeros(
            (n_riders, n_riders)
        )  # matrix of the length of type j's solo trip
        for i, j in itertools.product(range(n_riders), range(n_riders)):
            if best_triptime_choice[i, j, 0] == IIJJ:
                shared_length_matrix[i, j] = 0  # II()JJ
                i_solo_length_matrix[i, j] = O0D1[i, i, 0]  # (II)JJ
                j_solo_length_matrix[i, j] = O0D1[j, j, 0]  # II(JJ)
            elif best_triptime_choice[i, j, 0] == IJIJ:
                shared_length_matrix[i, j] = O0D1[j, i, 0]  # I(JI)J
                i_solo_length_matrix[i, j] = O0O1[i, j, 0]  # (IJ)IJ
                j_solo_length_matrix[i, j] = D0D1[i, j, 0]  # IJ(IJ)
            elif best_triptime_choice[i, j, 0] == IJJI:
                shared_length_matrix[i, j] = O0D1[j, j, 0]  # I(JJ)I
                i_solo_length_matrix[i, j] = O0O1[i, j, 0] + D0D1[j, i, 0]  # (IJ)(JI)
                j_solo_length_matrix[i, j] = 0
            elif best_triptime_choice[i, j, 0] == JIJI:
                shared_length_matrix[i, j] = O0D1[i, j, 0]  # J(IJ)I
                i_solo_length_matrix[i, j] = D0D1[j, i, 0]  # JI(JI)
                j_solo_length_matrix[i, j] = O0O1[j, i, 0]  # (JI)JI
            elif best_triptime_choice[i, j, 0] == JIIJ:
                shared_length_matrix[i, j] = O0D1[i, i, 0]  # JIIJ
                i_solo_length_matrix[i, j] = 0
                j_solo_length_matrix[i, j] = O0O1[j, i, 0] + D0D1[i, j, 0]  # (JI)(IJ)

        # matrix of total trip length for each rider type pair
        trip_length_matrix = best_triptime[:, :, 0]
        # matrix of pick-up and drop-off order for each rider type pair,
        # denoted by 0 to 4 (IIJJ, IJIJ, IJJI, JIJI, JIIJ)
        trip_order_matrix = best_triptime_choice[:, :, 0]

        return (
            trip_length_matrix,
            shared_length_matrix,
            i_solo_length_matrix,
            j_solo_length_matrix,
            trip_order_matrix,
        )

    # calculate the optimal routing
    (
        trip_length_matrix,
        shared_length_matrix,
        i_solo_length_matrix,
        j_solo_length_matrix,
        trip_order_matrix,
    ) = match_efficiency_single(O0D0, O0O1, D0D1, O0D1)

    # extract the shortest trip length, shared length, solo length, and trip order
    trip_length = trip_length_matrix[0, 1]
    shared_length = shared_length_matrix[0, 1]
    i_solo_length = i_solo_length_matrix[0, 1]
    j_solo_length = j_solo_length_matrix[0, 1]
    trip_order = trip_order_matrix[0, 1]

    return trip_length, shared_length, i_solo_length, j_solo_length, trip_order


def create_route_map(origin_i, dest_i, origin_j, dest_j, trip_order):
    """
    Creates a folium map with labeled origins and destinations for two riders and displays the chosen route.

    Parameters:
    - origin_i: (lat, lon) tuple for Rider i's pickup location.
    - dest_i: (lat, lon) tuple for Rider i's dropoff location.
    - origin_j: (lat, lon) tuple for Rider j's pickup location.
    - dest_j: (lat, lon) tuple for Rider j's dropoff location.
    - trip_order: Integer (0-4) indicating the route order (0: IIJJ, 1: IJIJ, 2: IJJI, 3: JIJI, 4: JIIJ).

    Returns:
    - folium.Map: A folium map object with markers and route displayed.
    """

    # Center the map based on the average latitude and longitude of both riders
    center_lat = (origin_i[0] + origin_j[0]) / 2
    center_lon = (origin_i[1] + origin_j[1]) / 2
    map_folium = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )

    # Define colors and icons for rider_i and rider_j
    rider_i_color = 'blue'
    rider_j_color = 'red'
    pickup_icon = 'home'
    dropoff_icon = 'flag'

    # HTML style for the text box with proper sizing
    text_box_style = """
        display: inline-block;
        font-size: 13px;
        color: white;
        background-color: gray;
        border: 1px solid white;
        padding: 3px 8px;
        border-radius: 5px;
        max-width: 150x;
        white-space: nowrap;
        box-sizing: border-box;
        opacity: 0.8;
    """

    # Add markers with always-visible text labels for Rider i
    folium.Marker(location=origin_i, icon=folium.Icon(color=rider_i_color, icon=pickup_icon)).add_to(map_folium)
    folium.Marker(location=origin_i, icon=folium.DivIcon(html=f"<div style='{text_box_style}'><b>Rider 0 Pickup</b></div>")).add_to(map_folium)

    folium.Marker(location=dest_i, icon=folium.Icon(color=rider_i_color, icon=dropoff_icon)).add_to(map_folium)
    folium.Marker(location=dest_i, icon=folium.DivIcon(html=f"<div style='{text_box_style}'><b>Rider 0 Dropoff</b></div>")).add_to(map_folium)

    # Add markers with always-visible text labels for Rider j
    folium.Marker(location=origin_j, icon=folium.Icon(color=rider_j_color, icon=pickup_icon)).add_to(map_folium)
    folium.Marker(location=origin_j, icon=folium.DivIcon(html=f"<div style='{text_box_style}'><b>Rider 1 Pickup</b></div>")).add_to(map_folium)

    folium.Marker(location=dest_j, icon=folium.Icon(color=rider_j_color, icon=dropoff_icon)).add_to(map_folium)
    folium.Marker(location=dest_j, icon=folium.DivIcon(html=f"<div style='{text_box_style}'><b>Rider 1 Dropoff</b></div>")).add_to(map_folium)

    # Display the route based on the trip order
    route_color = 'black'
    route_weight = 3

    if trip_order == 0:  # IIJJ
        folium.PolyLine([origin_i, dest_i, origin_j, dest_j], color=route_color, weight=route_weight, tooltip='IIJJ Route').add_to(map_folium)
    elif trip_order == 1:  # IJIJ
        folium.PolyLine([origin_i, origin_j, dest_i, dest_j], color=route_color, weight=route_weight, tooltip='IJIJ Route').add_to(map_folium)
    elif trip_order == 2:  # IJJI
        folium.PolyLine([origin_i, origin_j, dest_j, dest_i], color=route_color, weight=route_weight, tooltip='IJJI Route').add_to(map_folium)
    elif trip_order == 3:  # JIJI
        folium.PolyLine([origin_j, origin_i, dest_j, dest_i], color=route_color, weight=route_weight, tooltip='JIJI Route').add_to(map_folium)
    elif trip_order == 4:  # JIIJ
        folium.PolyLine([origin_j, origin_i, dest_i, dest_j], color=route_color, weight=route_weight, tooltip='JIIJ Route').add_to(map_folium)

    return map_folium


def plot_riders(rider_types):
    """
    Plots the origins and destinations of rider types on a folium map.

    Parameters:
    - rider_types: List of tuples, where each tuple contains (origin_lat, origin_lon, dest_lat, dest_lon) for a rider type.

    Returns:
    - folium.Map: A folium map object with markers for all rider types.
    """

    # Center the map based on the average latitude and longitude of all riders
    all_lats = [rt[0] for rt in rider_types] + [rt[2] for rt in rider_types]
    all_lons = [rt[1] for rt in rider_types] + [rt[3] for rt in rider_types]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    map_folium = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=12,
        tiles='CartoDB positron'  # Simple grey/white style
    )

    # Generate colors for each rider type
    # Folium color names (for Icon colors, limited options)
    folium_colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 
                     'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 
                     'pink', 'lightblue', 'lightgreen', 'gray', 'black']

    text_box_style = """
        display: inline-block;
        font-size: 13px;
        color: white;
        background-color: gray;
        border: 1px solid white;
        padding: 3px 8px;
        border-radius: 5px;
        max-width: 150x;
        white-space: nowrap;
        box-sizing: border-box;
        opacity: 0.8;
    """

    # Add markers for each rider type
    for idx, (origin_lat, origin_lon, dest_lat, dest_lon) in enumerate(rider_types):
        color_name = folium_colors[idx % len(folium_colors)]
        
        folium.Marker(location=(origin_lat, origin_lon), icon=folium.Icon(color=color_name, icon='home'), tooltip=f'Rider {idx} Pickup').add_to(map_folium)
        folium.Marker(location=(dest_lat, dest_lon), icon=folium.Icon(color=color_name, icon='flag'), tooltip=f'Rider {idx} Dropoff').add_to(map_folium)

        # also draw lines between origins and destinations for each rider
        folium.PolyLine([(origin_lat, origin_lon), (dest_lat, dest_lon)], color=color_name, weight=3, tooltip=f'Rider {idx} Route').add_to(map_folium)
    
        # annotation for origin and destination
        folium.Marker(location=(origin_lat, origin_lon), icon=folium.DivIcon(html=f"<div style='{text_box_style}'><b>Rider {idx} Pickup</b></div>")).add_to(map_folium)
        folium.Marker(location=(dest_lat, dest_lon), icon=folium.DivIcon(html=f"<div style='{text_box_style}'><b>Rider {idx} Dropoff</b></div>")).add_to(map_folium)

    return map_folium


# Function to test the Pricing and Matching Policies
def test_policies(PricingPolicy, MatchingPolicy):
    """
    Test Policies Function: A function that tests policies based on a student's PricingPolicy and MatchingPolicy classes.

    Parameters:
        PricingPolicy (class): A student PricingPolicy class.
        MatchingPolicy (class): A student MatchingPolicy class.
    """

    instances = pd.read_pickle('data/test_examples.pickle')
    states = instances['states']
    rider = instances['rider']

    # Testing the Pricing Function
    PricingPolicyInstance = PricingPolicy()
    MatchingPolicyInstance = MatchingPolicy()

    for i, state in enumerate(states):
        # run the pricing function given state and incoming rider
        start = time.time()
        price = PricingPolicyInstance.pricing_function(state, rider)
        end = time.time()

        # assert the price is a float between 0 and 1
        assert(price <= 1 and price >= 0)

        # print the pricing decision and execution time
        print('\n=============== Pricing at State {} ({} waiting requests) ==============='.format(i, len(state)))
        print('Pricing decision: {:.5f}.'.format(price))
        print('Execution time of the pricing function is {:.5f} seconds.'.format(end - start))

        # run the matching function given state and incoming rider
        start = time.time()
        matched_request = MatchingPolicyInstance.matching_function(state, rider)
        end = time.time()

        # assert matched_request is either None or a rider instance in the state
        if matched_request is not None:
            # assert matched_request is a rider instance
            assert(matched_request.__class__.__name__ == 'rider')
            # assert matched_request is in state
            assert(matched_request in state)
        else:
            assert(matched_request is None)

        # print the matching decision and execution time
        print('\n=============== Matching at State {} ({} waiting requests) ==============='.format(i, len(state)))
        if matched_request is None:
            print('Matching decision: do not match.')
        else:
            print('Matching decision: match the incoming rider with a waiting request.')
        print('Execution time of the matching function is {:.5f} seconds.'.format(end - start))


def export_notebook(teamname, notebook_name="student_policies.ipynb"):
    """
    Exports the current notebook (.ipynb) to a Python script (.py), renaming it based on the team name,
    and removes test cells and export cells from the exported Python script.
    
    Parameters:
        teamname (str): The name to include in the exported .py file's name.
        notebook_name (str): Name of the notebook file to convert.
    """
    
    # Check if the notebook exists in the current directory
    if notebook_name not in os.listdir(os.getcwd()):
        raise FileNotFoundError(f"{notebook_name} not found in the current directory.")

    # Convert the notebook to a Python script
    try:
        # Run the nbconvert command
        subprocess.run(["jupyter", "nbconvert", "--to", "script", notebook_name], check=True)

        # Rename the output Python script based on the team name
        original_script_name = notebook_name.replace(".ipynb", ".py")
        renamed_script_name = f"{teamname}.py"
        
        # Check if the target file exists, and remove it if it does
        if os.path.exists(renamed_script_name):
            os.remove(renamed_script_name)

        os.rename(original_script_name, renamed_script_name)

        # Clean up the script
        with open(renamed_script_name, 'r', encoding='utf-8') as file:
            content = file.read()

        # Remove everything from the test_policies call onwards
        # This removes both the test cell and the export cells
        patterns_to_remove = [
            # Remove test_policies call and everything after
            r'# In\[\d+\]:\s*\n\nfrom utils import test_policies.*',
            # Alternative: remove from "# # Testing your Code" section onwards
            r'# # Testing your Code.*',
            # Remove any remaining cells that contain export_notebook
            r'# In\[\s*\]:\s*\n.*?export_notebook.*?(?=\n# In\[|\Z)',
        ]

        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.DOTALL)

        # Remove any trailing empty cells or whitespace
        content = re.sub(r'# In\[\s*\]:\s*\n\s*$', '', content)
        content = content.rstrip() + '\n'

        # Write back the cleaned script
        with open(renamed_script_name, 'w', encoding='utf-8') as file:
            file.write(content)

        print(f"Converted {notebook_name} to {renamed_script_name} successfully!")

    except FileNotFoundError:
        print("Error: jupyter nbconvert is not installed or not found.")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

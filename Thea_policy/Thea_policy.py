import pandas as pd
import numpy as np
from scipy.optimize import minimize
import joblib
import itertools
import os
# Change to project root directory
os.chdir(os.path.dirname(os.getcwd()))

from utils import populate_shared_ride_lengths


class TheaPricingPolicy:
    """
    The example pricing policy using the linear and logistic regression models from
    the previous notebooks. This is very simple and can be improved more.
    """

    def __init__(self, c=0.7):
        self.c = c

    @staticmethod
    def get_name():
        return "Thea"

    def pricing_function(self, state, rider):
        """
        Pricing function that calculates a price prediction for a rider.
        
        Parameters:
        - state: The current state of the system
        - rider: An object with rider features
        
        Returns:
        - price: A predicted price with added noise, clipped to be between 0.3 and 1
        """

        # load the models and encoded features
        cost_model = joblib.load('cost_model.joblib')
        conversion_model = joblib.load('conversion_model.joblib')
        encoded_feats = joblib.load('encoded_features.joblib')

        # Create a dataframe with the rider's features
        incoming_rider = pd.DataFrame({'solo_length': [rider.solo_length],
                                    'pickup_area': [rider.pickup_area],
                                    'dropoff_area': [rider.dropoff_area]})

        # One-hot encode the features
        incoming_rider = pd.get_dummies(incoming_rider, columns=['pickup_area', 'dropoff_area'], dtype=int)

        # Identify missing columns
        missing_cols = [col for col in encoded_feats if col not in incoming_rider]

        # Create a DataFrame for missing columns with zeros
        missing_data = pd.DataFrame(0, index=incoming_rider.index, columns=missing_cols)

        # Concatenate the incoming_rider DataFrame with missing_data
        incoming_rider = pd.concat([incoming_rider, missing_data], axis=1)

        # Ensure column order matches the training data
        incoming_rider = incoming_rider[encoded_feats]

        # Predict the cost
        cost = cost_model.predict(incoming_rider)[0]

        # Prepare features for the conversion model
        rider_features = incoming_rider.copy()
        rider_features = rider_features.drop('solo_length', axis=1)  # Drop solo_length as it's not in conversion_model

        # Helper function for optimization
        def objective(price):
            # Add quoted price to the rider features
            rider_features['quoted_price'] = price

            # Predict the conversion probability
            prob = conversion_model.predict_proba(rider_features)[:, 1][0]  # Probability of conversion

            # Objective: maximize (price - cost) * probability
            return -(price - cost) * prob

        # Optimize the price
        result = minimize(objective, x0=0.5, bounds=[(0.3, 1.0)])

        # Check if optimization was successful
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Return the optimized price
        optimized_price = result.x[0]

        return optimized_price


class TheaMatchingPolicy():
    """
    An example matching policy using Greedy Method.
    Selection is based on closest rider in the state.
    """

    def __init__(self, c=0.7):
        self.c = c

    @staticmethod
    def get_name():
        """
        :return: name of the team
        """
        return 'Thea'

    def matching_function(self, state, rider):
        """
        Match riders based on batching. If enough riders are in the system, match them.
        :param state: a list of rider instances waiting in the system
        :param rider: the arriving rider instance
        :return: a rider instance (the best match for the arriving rider); if no match, return None
        """
        # rider will be rider i
        origin_i, dest_i = (rider.pickup_lat, rider.pickup_lon), (rider.dropoff_lat, rider.dropoff_lon)
        dist_i = rider.solo_length

        best_match = None
        shortest_distance = float('inf')

        for rider_j in state:  #rider j is a potential match in the state
            # Get rider_j's origin and destination
            origin_j, dest_j = (rider_j.pickup_lat, rider_j.pickup_lon), (rider_j.dropoff_lat, rider_j.dropoff_lon)
            dist_j = rider_j.solo_length

            # Optimize the shared ride
            (trip_length, 
            shared_length, 
            i_solo_length, 
            j_solo_length, 
            trip_order) = populate_shared_ride_lengths(origin_i, dest_i, origin_j, dest_j)

            # Normalized trip lengths
            trip_length /= (dist_i + dist_j)

            # Update the shortest distance
            if trip_length < shortest_distance and shared_length > 0:
                shortest_distance = trip_length
                best_match = rider_j

        # If a match is found, return the best match
        if best_match:
            return best_match
        else:
            # If no match is found, return None
            return None
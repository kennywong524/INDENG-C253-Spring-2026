# Calyber: A Ridesharing Game

<p align="center">
  <img src="logo.png" width="280" />
</p>

Welcome to Calyber, a ridesharing game! Your mission is to develop intelligent pricing and matching policies that maximize platform profits using real historical shared rides data from Chicago.

### Installation
This project was developed and tested using Python 3.9.16. The following Python packages are required:
- numpy==1.24.3
- pandas==1.5.3
- matplotlib==3.7.1
- haversine==2.9.0
- folium==0.20.0
- joblib==1.5.1
- scipy==1.12.0
- scikit-learn==1.4.0
- statsmodels==0.14.5
- jupyter
- ipython

It is recommended to use [Conda](https://docs.conda.io/en/latest/) to manage the Python environment and install the dependencies. Run the following commands in your terminal to create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate calyber_env
```

## Repository Structure:

### Core Files:
- **`student_policies.ipynb`**: Skeleton notebook where you will implement your pricing and matching policies.
- **`rider.py`**: Rider class definition.
- **`utils.py`**: Helper functions including `populate_shared_ride_lengths()` for routing, `test_policies()` for validation, `create_route_map()` and `plot_riders()` for visualization, `export_notebook()` for submission preparation, and a few other helper functions used in the tutorial notebooks in `tutorials/`.
- **`helper_functions_demonstration.ipynb`**: Tutorial on how to use the key helper function `populate_shared_ride_lengths()` for shortest path calculation provided in `utils.py`, with additional functions for visualization.

### Data Directory (`data/`):
- **`training_data.csv`**: Historical shared rides data. Contains 11,788 historical rider records (including their conversion status and matching outcomes under the current policy -- i.e., Nick's engine as mentioned in the case). Features include:
    - `rider_id`: Rider’s unique ID (integer, 0–11787).
    - `arrival_week`: Week of arrival (integer, 1–6).
    - `arrival_time`: Arrival time within hour (in seconds, 0–3600).
    - `pickup_area`: Pickup community area (integer, 1–76).
    - `dropoff_area`: Dropoff community area (integer, 1–76).
    - `pickup_lat` and `pickup_lon`: Exact pickup coordinates (float).
    - `dropoff_lat` and `dropoff_lon`: Exact dropoff coordinates (float).
    - `solo_length`: OD distance (miles) (haversine distance).
    - `quoted_price`:  Quoted price ($/mile) under the current policy, i.e., Nick's engine (float, 0–1).
    - `convert_or_not`: Rider's conversion status (binary, 0/1).
    - `waiting_time`: Rider's waiting time, from arrival to dispatch (seconds; `nan` if not converted).
    - `matching_outcome`: Matched rider’s ID (integer; `nan` if not matched).
- **`community_areas.csv`**: Chicago's 76 community areas with their centroid coordinates.
    - `area`: Community area index (1–76).
    - `lat` and `lon`: Centroid coordinates.
- **`test_examples.pickle`**: Pre-generated test examples for validating your implementations.
    - Load using:
        ```python
        import pickle
        data = pickle.load(open('data/test_examples.pickle', 'rb'))
        # Access components
        incoming_rider = data['rider']  # Single rider object
        test_states = data['states']    # List of 4 states with 0/8/35/77 waiting riders
        ```
    - Contents:
        - `data['rider']`: A rider object as the incoming rider.
        - `data['states']`: Four example states, each a list of waiting rider objects (0, 8, 35, or 77 requests) accumulated over random time windows (0s, 15s, 1min, 2min).

### Thea's engine (`Thea_policy/`)
Contains the implementation of Thea's policy that serves as both a learning reference and evaluation benchmark:

- **`models/`** - Pre-trained machine learning models
  - `conversion_model.joblib` - Model for predicting rider conversion probability.
  - `cost_model.joblib` - Model for estimating per-rider cost.
  - `encoded_features.joblib` - Encoded feature set used by the conversion model.
- **`Thea_policy_training.ipynb`** - Training notebook demonstrating how the pre-trained models were built, trained, and evaluated.
- **`Thea_policy.py`** - Full policy implementation that loads and integrates the pre-trained models into a working solution.

**Note**: This policy will be used as a performance benchmark during evaluation. Study its approach to understand how to design strategies, but develop your own unique solution.

## Getting Started:
- Activate your environment: `conda activate calyber`
- Launch Jupyter Notebook: `jupyter notebook`
- Open the main notebook `student_policies.ipynb`, and follow the instructions to implement your pricing and matching policies in the provided cells.
- Test your implementation using the built-in testing functions in the notebook to verify that your policies work as expected.
- Prepare your submission: 
    - Create a folder `TEAM_NAME/` in the root directory of the repository.
    - Inside it, include `TEAM_NAME.py` (exported from the notebook `student_policies.ipynb`) and any other additional files (if needed) for your implementation.

**Note**:
- You may include additional files in your submission folder, such as pre-trained models. If you do, ensure you use the correct file paths and loading functions in your code to load these files. For example, the following snippet shows how to load a pre-trained model:
```python
import os
import joblib

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load your model (assuming it is a .joblib file)
model_path = os.path.join(current_dir, 'models', 'your_model.joblib')
your_model = joblib.load(model_path)
```
- Also, ensure that your folder `TEAM_NAME/` contains no other `.py` files except for `TEAM_NAME.py` to avoid potential conflicts during evaluation.
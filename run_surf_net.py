#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:28:15 2025

@author: sjgauva
"""

import numpy as np
import pandas as pd
from obspy import read
from geopy.distance import geodesic
import glob
import os
from surf_net import SurfNet  # Assuming the provided code is saved as surf_net.py
#%%
def load_station_data(file_path):
    """
    Load station data from text file into DataFrame
    
    Parameters:
    file_path (str): Path to the station data text file
    
    Returns:
    pandas.DataFrame: DataFrame containing station information
    """
    # Read CSV file
    stations = pd.read_csv(file_path, header=0)
    # Clean column names by stripping whitespace
    stations.columns = stations.columns.str.strip()
    # Clean station names by stripping whitespace
    stations['station name'] = stations['station name'].str.strip()
    return stations

def calculate_interstation_distances(stations_df):
    """
    Calculate distances between all station pairs
    Returns dictionary with station pairs as keys and distances in km as values
    """
    distances = {}
    stations = stations_df.copy()
    
    for i, row1 in stations.iterrows():
        station1 = row1['station name']
        coord1 = (row1['latitude'], row1['longitude'])
        
        for j, row2 in stations.iterrows():
            if i < j:  # Only calculate each pair once
                station2 = row2['station name']
                coord2 = (row2['latitude'], row2['longitude'])
                
                # Calculate distance in kilometers
                distance = geodesic(coord1, coord2).kilometers
                
                # Store with consistent ordering (alphabetical)
                pair = tuple(sorted([station1, station2]))
                distances[pair] = distance
    
    return distances

def read_ccf_data(mseed_folder):
    """
    Read all miniseed files from folder
    Returns dictionary with station pairs as keys and CCF traces as values
    """
    ccf_data = {}
    
    # Get all miniseed files in folder
    mseed_files = glob.glob(os.path.join(mseed_folder, "*.MSEED"))
    print(f"Found {len(mseed_files)} mseed files")
    
    for file in mseed_files:
        try:
            # Extract station names from filename
            basename = os.path.basename(file)
            stations = basename.split('.')[0]  # Remove .MSEED extension
            station_parts = stations.split('_')
            
            # Combine network and station codes
            station1 = f"{station_parts[0]}.{station_parts[1]}"
            station2 = f"{station_parts[2]}.{station_parts[3]}"
            
            print(f"\nProcessing file: {basename}")
            print(f"Extracted station pair: {station1} - {station2}")
            
            # Read mseed file
            st = read(file)
            
            # Store with consistent ordering (alphabetical)
            pair = tuple(sorted([station1, station2]))
            ccf_data[pair] = st[0].data
            
            print(f"Successfully added pair: {pair}")
            
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    return ccf_data

def prepare_surfnet_input(ccf_data, distances, input_length=3072):
    """
    Prepare input data for SurfNet model
    Returns X array with shape (num_samples, 3072, 2)
    
    Parameters:
    - ccf_data: Dictionary with station pairs as keys and CCF traces as values
    - distances: Dictionary with station pairs as keys and distances (km) as values
    - input_length: Length of input time series (default 3072)
    
    Channel 0: Normalized cross-correlation waveform
    Channel 1: Distance converted to time domain with boxcar function
    """
    num_samples = len(ccf_data)
    X = np.zeros((num_samples, input_length, 2))
    
    # Reference velocities for boxcar function (km/s)
    v0 = 1.5  # Lower velocity bound
    v1 = 5.0  # Upper velocity bound
    
    # Create time vector for input length
    dt = 0.02  # Assuming 50 Hz sampling rate from mseed stats
    t = np.arange(input_length) * dt
    
    for i, (pair, ccf) in enumerate(ccf_data.items()):
        # Channel 0: Normalized CCF
        ccf_norm = ccf / np.max(np.abs(ccf))
        
        # Ensure CCF length matches input_length
        if len(ccf_norm) > input_length:
            # Take center portion
            start = (len(ccf_norm) - input_length) // 2
            ccf_norm = ccf_norm[start:start+input_length]
        elif len(ccf_norm) < input_length:
            # Zero pad
            pad_width = (input_length - len(ccf_norm)) // 2
            ccf_norm = np.pad(ccf_norm, (pad_width, input_length-len(ccf_norm)-pad_width))
        
        X[i, :, 0] = ccf_norm
        
        # Channel 1: Distance boxcar function
        D = distances[pair]  # Distance in km
        boxcar = np.zeros(input_length)
        # Time window where velocity would be between v0 and v1
        t_min = D/v1  # Time for fastest velocity
        t_max = D/v0  # Time for slowest velocity
        # Set boxcar to 1 within this window
        boxcar[(t >= t_min) & (t <= t_max)] = 1
        X[i, :, 1] = boxcar
    
    return X

def prepare_target_data(ccf_data, distances, input_length=3072):
    """
    Prepare target data for SurfNet model
    Returns y array with shape (num_samples, 3072, 50)
    
    For each station pair and each frequency:
    - Calculate expected arrival time based on distance and group velocity
    - Create Gaussian distribution centered at arrival time
    - SD varies with frequency according to -0.5 log(f) - 0.4
    """
    num_samples = len(ccf_data)
    num_frequencies = 50
    y = np.zeros((num_samples, input_length, num_frequencies))
    
    # Create frequency array (assuming 0.07-10 Hz)
    frequencies = np.logspace(np.log10(0.07), np.log10(10), num_frequencies)
    
    # Create time vector
    dt = 0.02  # 50 Hz sampling rate
    t = np.arange(input_length) * dt
    
    # Approximate group velocity for each frequency
    group_velocities = np.ones_like(frequencies) * 3.0  # km/s
    
    for i, (pair, _) in enumerate(ccf_data.items()):
        distance = distances[pair]  # km
        
        for j, (f, v) in enumerate(zip(frequencies, group_velocities)):
            # Calculate expected arrival time
            arrival_time = distance / v
            
            # Calculate standard deviation for this frequency
            sd = -0.5 * np.log10(f) - 0.4
            
            # Create Gaussian distribution
            gaussian = np.exp(-0.5 * ((t - arrival_time) / sd) ** 2)
            
            # Normalize to make it a probability distribution
            gaussian = gaussian / np.max(gaussian)
            
            # Store in target array
            y[i, :, j] = gaussian
            
    return y, frequencies

def main():
    # Load station data from file
    station_file = "station_locs_localregional.txt"
    stations_df = load_station_data(station_file)
    print("Number of stations loaded:", len(stations_df))
    
    # Calculate interstation distances
    distances = calculate_interstation_distances(stations_df)
    print("Number of station pairs with distances:", len(distances))
    
    # Read CCF data
    mseed_folder = "/gpfs/sjgauva/ambient/"
    ccf_data = read_ccf_data(mseed_folder)
    print("Number of CCF traces loaded:", len(ccf_data))
    
    # Prepare input data
    X = prepare_surfnet_input(ccf_data, distances)
    print("Shape of prepared input data X:", X.shape)
    
    # Prepare target data
    y, frequencies = prepare_target_data(ccf_data, distances)
    print("Shape of target data y:", y.shape)
    print("Frequency range:", frequencies[0], "to", frequencies[-1], "Hz")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print("\nFinal shapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    
    # Initialize and train model
    model = SurfNet()
    model.compile()
    history = model.train(X_train, y_train, X_val, y_val, epochs=100)
    
    return distances, model, X_train, X_val, y_train, y_val, y, frequencies, history
#%%
if __name__ == "__main__":
    main()
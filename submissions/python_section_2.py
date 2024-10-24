import pandas as pd
import numpy as np


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here

    distance_matrix = pd.DataFrame(index=df.index, columns=df.index)
    
    
    for i in df.index:
        for j in df.index:
            distance = np.sqrt((df.at[i, 'x'] - df.at[j, 'x'])**2 + (df.at[i, 'y'] - df.at[j, 'y'])**2)
            distance_matrix.at[i, j] = distance
    
    return distance_matrix


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here

    unrolled_data = []
    
    
    for id_start in df.index:
        for id_end in df.columns:
            distance = df.at[id_start, id_end]
            unrolled_data.append([id_start, id_end, distance])
    
   
    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
    
    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    reference_avg = df.loc[reference_id].mean()
    
    
    threshold_min = reference_avg * 0.9
    threshold_max = reference_avg * 1.1
    
    
    avg_distances = df.mean(axis=1)
    
    
    ids_within_threshold = avg_distances[(avg_distances >= threshold_min) & (avg_distances <= threshold_max)]
    
    
    return pd.DataFrame(ids_within_threshold, columns=['average_distance'])


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    
    toll_rates = {
        'car': 0.05,  
        'truck': 0.10,
        'bus': 0.08,  
    }
    
    
    df['toll_rate'] = df.apply(lambda row: row['distance'] * toll_rates.get(row['vehicle_type'], 0), axis=1)
    
    return df[['id_start', 'id_end', 'vehicle_type', 'toll_rate']]


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    time_based_rates = {
        'peak': 1.5,  
        'off-peak': 1.0,  
        'night': 0.8,
    }
    
 
    df['time_multiplier'] = df['time_of_day'].map({
        'morning_peak': time_based_rates['peak'],
        'evening_peak': time_based_rates['peak'],
        'off_peak': time_based_rates['off-peak'],
        'night': time_based_rates['night'],
    })
    

    df['final_toll_rate'] = df['toll_rate'] * df['time_multiplier']
    
    return df[['id_start', 'id_end', 'vehicle_type', 'time_of_day', 'final_toll_rate']]

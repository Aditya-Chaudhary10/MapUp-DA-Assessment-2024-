import pandas as pd
import numpy as np

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame): The DataFrame containing distance information.

    Returns:
        pandas.DataFrame: The distance matrix with cumulative distances along known routes.
    """

    distance_matrix = df.copy()

    for i in range(len(df)):
        for j in range(len(df)):
            if i == j or not pd.isna(distance_matrix.iloc[i, j]):
                continue

            for k in range(len(df)):
                if not pd.isna(distance_matrix.iloc[i, k]) and not pd.isna(df.iloc[k, j]):
                    distance_matrix.iloc[i, j] = distance_matrix.iloc[i, k] + df.iloc[k, j]
                    break

    distance_matrix = distance_matrix + distance_matrix.T

    np.fill_diagonal(distance_matrix.values, 0)

    return distance_matrix



def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unrolls a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame): The distance matrix DataFrame.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """

    ids = df.columns

    combinations = [(start, end) for start in ids for end in ids if start != end]

    unrolled_data = [(start, end, df.loc[start, end]) for start, end in combinations]

    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])

    return unrolled_df


def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Finds IDs within a 10% threshold of the average distance of a reference ID.

    Args:
        df (pd.DataFrame): The unrolled DataFrame.
        reference_id (int): The reference ID.

    Returns:
        pd.DataFrame: A DataFrame containing the IDs within the 10% threshold.
    """

    # Filter the DataFrame for the reference ID
    reference_df = df[df['id_start'] == reference_id]

    # Calculate the average distance for the reference ID
    average_distance = reference_df['distance'].mean()

    # Calculate the 10% threshold
    lower_threshold = average_distance * 0.9
    upper_threshold = average_distance * 1.1

    # Filter the DataFrame for IDs within the threshold
    filtered_df = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]

    # Extract unique IDs within the threshold and return a DataFrame
    ids_within_threshold = filtered_df[['id_start']].drop_duplicates().sort_values('id_start')

    return ids_within_threshold    






def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame():
    """
    Calculates toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): The unrolled DataFrame containing 'id_start', 'id_end', and 'distance' columns.

    Returns:
        pandas.DataFrame: The DataFrame with additional columns for toll rates of different vehicle types.
    """

    vehicle_types = {
        'car': 0.1,
        'truck': 0.2,
        'bus': 0.15,
    }

    for vehicle_type, rate_per_km in vehicle_types.items():
        df[f'{vehicle_type}_toll'] = df['distance'] * rate_per_km

    return df



def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time-based toll rates for different time intervals within a day.

    Args:
        df (pd.DataFrame): The DataFrame with distance and toll rates.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for time-based toll rates.
    """

    time_intervals = [
        ((time(0, 0, 0), time(10, 0, 0)), 0.8),  
        ((time(10, 0, 0), time(18, 0, 0)), 1.2),  
        ((time(18, 0, 0), time(23, 59, 59)), 0.8),  
        ((time(0, 0, 0), time(23, 59, 59)), 0.7),  
    ]

    def apply_discount(row, interval, discount_factor):
        start_time, end_time = interval
        if (row['start_time'] >= start_time and row['start_time'] < end_time) or \
           (row['end_time'] > start_time and row['end_time'] <= end_time):
            for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                row[vehicle_type] *= discount_factor
        return row

    result_df = pd.DataFrame()

    for _, group in df.groupby(['id_start', 'id_end']):
        for start_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            for interval, discount_factor in time_intervals:
                if start_day in ['Saturday', 'Sunday']:
                    end_day = 'Sunday'
                elif start_day == 'Friday':
                    end_day = 'Friday'
                else:
                    end_day = 'Saturday'

                new_row = {
                    'id_start': group['id_start'].iloc[0],
                    'id_end': group['id_end'].iloc[0],
                    'distance': group['distance'].iloc[0],
                    'start_day': start_day,
                    'start_time': interval[0],
                    'end_day': end_day,
                    'end_time': interval[1],
                    'moto': group['moto'].iloc[0],
                    'car': group['car'].iloc[0],
                    'rv': group['rv'].iloc[0],
                    'bus': group['bus'].iloc[0],
                    'truck': group['truck'].iloc[0],
                }
                new_row = apply_discount(new_row, interval, discount_factor)

                result_df = result_df.append(new_row, ignore_index=True)

    return result_df

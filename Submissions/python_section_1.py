from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    
    length = len(lst)
    result = []
    for i in range(0, length, n):
        temp = []
       
        for j in range(i, min(i + n, length)):
            temp.insert(0, lst[j])
        
       
        result.extend(temp)
    lst = result
    return lst
    




def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    for string in lst:
        length = len(string)
        
        
        if length in result:
            result[length].append(string)
        else:
            
            result[length] = [string]
    
    
    return dict(result)




def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    
    def _flatten(obj, parent_key=''):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(_flatten(v, new_key).items())
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{parent_key}[{i}]"
                items.extend(_flatten(v, new_key).items())
        else:
            items.append((parent_key, obj))
        return dict(items)
    
    return _flatten(nested_dict)


from typing import List

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    from itertools import permutations

    
    unique_perms = set(permutations(nums))
    
   
    return [list(p) for p in unique_perms]



from typing import List
import re

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    
    date_pattern = r'(\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b)'
    
    
    dates = re.findall(date_pattern, text)
    
    return dates


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points 
    on the Earth's surface using the Haversine formula.

    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        float: Distance between the two points in meters.
    """
    R = 6371000  
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, 
    and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    
    coordinates = polyline.decode(polyline_str)
    
    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    
    distances = [0] 
    for i in range(1, len(coordinates)):
        dist = haversine(df.latitude[i - 1], df.longitude[i - 1], df.latitude[i], df.longitude[i])
        distances.append(dist)
    
    df['distance'] = distances
    
    return pd.DataFrame(df)


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then replace each element 
    with the sum of all elements in the same row and column, excluding itself.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])  
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  
    
    return final_matrix


def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Use shared dataset-1 to verify the completeness of the data by checking whether the timestamps for each unique 
    (`id`, `id_2`) pair cover a full 24-hour and 7 days period.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the dataset.

    Returns:
        pd.Series: A boolean series indicating if each (id, id_2) pair has incorrect timestamps.
    """
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    grouped = df.groupby(['id', 'id_2'])
    
    results = pd.Series(index=grouped.groups.keys(), dtype=bool)
    
    for (id_value, id_2_value), group in grouped:
        full_days = set(pd.date_range(start='2023-01-01', end='2023-01-07', freq='D').day_name())
        recorded_days = set(group['start_timestamp'].dt.day_name()).union(set(group['end_timestamp'].dt.day_name()))
        
        days_check = recorded_days == full_days
        
        min_time = group['start_timestamp'].min()
        max_time = group['end_timestamp'].max()
        
        hours_check = (max_time - min_time) >= pd.Timedelta(hours=24)
        
        results[(id_value, id_2_value)] = not (days_check and hours_check)
    
    return results

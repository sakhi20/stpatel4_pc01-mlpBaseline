import h5py
import numpy as np

def h5_to_dict(file_path):
    ''' Recursively converts an HDF5 file to a nested dictionary.
    Args:
        file_path (str): Path to the .h5 file.
    Returns:
        data (dict): A nested dictionary containing the data from the HDF5 file.
    '''
    data = {}
    with h5py.File(file_path, "r") as f:
        for key, item in f.items():
            if isinstance(item, h5py.Dataset):
                data[key] = item[()]  # NumPy array/scalar
            elif isinstance(item, h5py.Group):
                data[key] = h5_to_dict(item)  # recurse into group
    return data

def write_h5(file_path, data_dict):
    ''' Writes a dictionary to an HDF5 file.
    Args:
        file_path (str): Path to the output .h5 file.
        data_dict (dict): A dictionary containing the data to be written to the HDF5 file.
    '''
    with h5py.File(file_path, 'w') as f:
        for key, value in data_dict.items():
            arr = np.asarray(value)
            f.create_dataset(key, data=value)
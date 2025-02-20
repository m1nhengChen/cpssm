
import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
from scipy.signal import resample
import random
from scipy.stats import fisher_exact,pearsonr
from joblib import Parallel, delayed
from scipy.stats import zscore
from nilearn import plotting, datasets,image
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
def craddock2012_atlas():
    """
    Create a custom atlas dataset that mimics the output format of fetch_atlas_craddock_2012.
    
    Returns:
    - Bunch: A dictionary-like object containing paths to different parcellation versions.
    """
    # Define dummy file paths (update these with actual paths if needed)
    atlas_data = {
        'scorr_mean': '/home/minheng/brain library/atlas/craddock_2011_parcellations/scorr05_mean_all.nii.gz',
        'tcorr_mean': '/home/minheng/brain library/atlas/craddock_2011_parcellations/tcorr05_mean_all.nii.gz',
        'scorr_2level': '/home/minheng/brain library/atlas/craddock_2011_parcellations/scorr05_2level_all.nii.gz',
        'tcorr_2level': '//home/minheng/brain library/atlas/craddock_2011_parcellations/tcorr05_2level_all.nii.gz',
        'random': '/home/minheng/brain library/atlas/craddock_2011_parcellations/random_all.nii.gz',
        'description': 'This is a custom-generated atlas dataset mimicking the Craddock 2012 atlas.'
    }
    
    # Return a Bunch object
    return Bunch(**atlas_data)
def split_fmri_4d_to_3d(input_file, output_dir):
    """
    Splits a 4D fMRI image into multiple 3D MRI images along the time dimension.
    
    Args:
        input_file (str): Path to the 4D fMRI NIfTI file.
        output_dir (str): Directory where the 3D images will be saved.
    """
    # Load the 4D fMRI image
    img = sitk.ReadImage(input_file)
    
    # Get the number of time points
    num_volumes = img.GetSize()[-1]
    print(num_volumes)
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over time dimension and save each 3D volume
    for i in range(num_volumes):
        volume = img[:, :, :, i]  # Extract 3D volume
        volume_img = sitk.Image(volume)
        output_path = os.path.join(output_dir, f"volume_{i:03d}.nii.gz")
        sitk.WriteImage(volume_img, output_path)
        print(f"Saved: {output_path}")
    
    print(f"Successfully split {num_volumes} volumes into {output_dir}")
def signal_correlation(signal1, signal2):
    """
    Compute correlation between two signals, return 0 if any signal is all zeros.
    """
    if np.all(signal1 == 0) or np.all(signal2 == 0):
        return 0
    else:
        pcc, _ = pearsonr(signal1, signal2)
        return pcc

def fisher_z_transform(correlation_matrix):
    """
    Perform Fisher Z transformation on a correlation matrix.

    Parameters:
        correlation_matrix (numpy.ndarray): A symmetric matrix of Pearson correlation coefficients.

    Returns:
        numpy.ndarray: A matrix of Fisher Z-transformed values.
    """
    # Ensure correlation coefficients are within the valid range (-1, 1)
    correlation_matrix = np.clip(correlation_matrix, -0.999999, 0.999999)

    # Apply Fisher Z transformation
    z_matrix = 0.5 * np.log((1 + correlation_matrix) / (1 - correlation_matrix))

    return z_matrix

def compute_pcc_matrix(timeseries, n_jobs=-1):
    """
    Compute the PCC matrix for a given time series using signal_correlation.
    """
    n_rois = timeseries.shape[0]
    
    # Parallelize computation for the upper triangle of the PCC matrix
    def compute_row(i):
        row = np.zeros(n_rois)
        for j in range(i, n_rois):
            row[j] = signal_correlation(timeseries[i], timeseries[j])
        if len(row) != n_rois:
            raise ValueError(f"Row {i} has unexpected length {len(row)} (expected {n_rois}).")
        return row
    
    upper_triangle = Parallel(n_jobs=n_jobs)(
        delayed(compute_row)(i) for i in range(n_rois)
    )

    # Validate upper_triangle
    for i, row in enumerate(upper_triangle):
        if len(row) != n_rois:
            raise ValueError(f"Row {i} in upper_triangle has unexpected length {len(row)} (expected {n_rois}).")

    # Assemble the symmetric PCC matrix
    pcc_matrix = np.ones((n_rois, n_rois))
    for i in range(n_rois):
        pcc_matrix[i, i:] = upper_triangle[i][:n_rois - i]
        pcc_matrix[i:, i] = upper_triangle[i][:n_rois - i]

    return pcc_matrix


def normalize_mri_3d(input_file, output_file):
    """
    Normalizes a 3D MRI image to the range [0,1] and saves the output.
    
    Args:
        input_file (str): Path to the input 3D NIfTI file.
        output_file (str): Path to save the normalized 3D NIfTI file.
    """
    # Load the 3D MRI image
    img = sitk.ReadImage(input_file)
    img_array = sitk.GetArrayFromImage(img)
    
    # Normalize to [0,1]
    min_val = img_array.min()
    max_val = img_array.max()
    normalized_array = (img_array - min_val) / (max_val - min_val)
    
    # Convert back to SimpleITK image
    normalized_img = sitk.GetImageFromArray(normalized_array)
    normalized_img.CopyInformation(img)
    
    # Save the normalized image
    sitk.WriteImage(normalized_img, output_file)
    print(f"Normalized image saved to: {output_file}")
def mip_minip_projection(input_file, output_mip, output_minip):
    """
    Computes the Maximum Intensity Projection (MIP) and Minimum Intensity Projection (MinIP) 
    of a 4D fMRI image along the time dimension and saves the results as 3D NIfTI files.
    
    Args:
        input_file (str): Path to the input 4D NIfTI file.
        output_mip (str): Path to save the MIP result as a 3D NIfTI file.
        output_minip (str): Path to save the MinIP result as a 3D NIfTI file.
    """
    # Load the 4D fMRI image
    img = sitk.ReadImage(input_file)
    img_array = sitk.GetArrayFromImage(img)  # Convert to numpy array (T, Z, Y, X)
    
    # Compute Maximum Intensity Projection (MIP) along the time dimension
    mip_array = np.max(img_array, axis=0)  # Shape becomes (Z, Y, X)
    mip_img = sitk.GetImageFromArray(mip_array)
    
    # Compute Minimum Intensity Projection (MinIP) along the time dimension
    minip_array = np.min(img_array, axis=0)  # Shape becomes (Z, Y, X)
    minip_img = sitk.GetImageFromArray(minip_array)
    
    # Extract a reference 3D slice from the 4D image for metadata
    reference_3d = sitk.Extract(img, [img.GetSize()[0], img.GetSize()[1], img.GetSize()[2], 0], [0, 0, 0, 0])
    # reference_3d = sitk.Extract(img, extract_size, [0, 0, 0, 0])
    mip_img.CopyInformation(reference_3d)
    minip_img.CopyInformation(reference_3d)
    
    # Save the MIP and MinIP results as 3D NIfTI files
    sitk.WriteImage(mip_img, output_mip)
    print(f"MIP image saved to: {output_mip}")
    sitk.WriteImage(minip_img, output_minip)
    print(f"MinIP image saved to: {output_minip}")

import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, datasets, image

# Generate a 200x200 matrix with only 10 nonzero edges, with an option for symmetry

def generate_sparse_matrix(size=200, num_edges=10, symmetric=True):
    """
    Generate a matrix of given size with only a specified number of nonzero edges.
    Values of the edges are randomly sampled between -1 and 1.
    
    Parameters:
    - size (int): The size of the square matrix.
    - num_edges (int): The number of nonzero edges in the matrix.
    - symmetric (bool): Whether the matrix should be symmetric.
    
    Returns:
    - np.ndarray: A matrix of shape (size, size), either symmetric or asymmetric.
    """
    # Initialize a zero matrix
    matrix = np.zeros((size, size))
    
    # Seed for reproducibility
    # np.random.seed(42)
    
    # Track the number of added edges
    edges_added = 0
    
    while edges_added < num_edges:
        # Randomly select two distinct indices
        i, j = np.random.randint(0, size, size=2)
        
        if i != j and matrix[i, j] == 0:
            # Assign a random value between -1 and 1
            value = np.random.uniform(0, 1)
            
            matrix[i, j] = value
            
            if symmetric:
                # Ensure symmetry
                matrix[j, i] = value
            
            edges_added += 1
    
    return matrix


def filter_connectome(connectivity_matrix, coordinates):
    """
    Remove isolated nodes (nodes with no connections) from the connectivity matrix and coordinates.
    
    Parameters:
    - connectivity_matrix (np.ndarray): The full adjacency matrix.
    - coordinates (list of tuples): The coordinates of all nodes.
    
    Returns:
    - filtered_matrix (np.ndarray): The filtered adjacency matrix with non-isolated nodes.
    - filtered_coords (list of tuples): The coordinates of the remaining nodes.
    """
    # Identify nodes with at least one connection
    node_mask = np.any(connectivity_matrix != 0, axis=0)  # True if node has at least one connection
    
    # Apply mask to filter connectivity matrix and coordinates
    filtered_matrix = connectivity_matrix[node_mask][:, node_mask]  # Keep only connected nodes
    filtered_coords = np.array(coordinates)[node_mask]  # Keep corresponding coordinates
    
    return filtered_matrix, filtered_coords
# Example usage
# split_fmri_4d_to_3d("/home/minheng/Image/fmri_sample/rsfProcess/fmriReadyRetrend.nii.gz", "fmri_slices")
# normalize_mri_3d("/home/minheng/Image/fmri_sample/rsfProcess/fmri_slices/volume_006.nii.gz", "/home/minheng/Image/fmri_sample/rsfProcess/normalized_output.nii.gz")
# mip_minip_projection("/home/minheng/Image/fmri_sample/rsfProcess/fmriReadyRetrend.nii.gz", "/home/minheng/Image/fmri_sample/rsfProcess/mip_output.nii.gz", "/home/minheng/Image/fmri_sample/rsfProcess/minip_output.nii.gz")

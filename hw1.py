import numpy as np
import pandas as pd

# Define the normalized frequencies of amino acids based on the provided table
# Order: A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V
amino_acid_frequencies = {
    'A': 0.087, 'R': 0.041, 'N': 0.040, 'D': 0.047, 'C': 0.033,
    'Q': 0.038, 'E': 0.050, 'G': 0.089, 'H': 0.034, 'I': 0.037,
    'L': 0.085, 'K': 0.081, 'M': 0.015, 'F': 0.040, 'P': 0.051,
    'S': 0.070, 'T': 0.058, 'W': 0.010, 'Y': 0.030, 'V': 0.065
}

def generate_pam(x, input_path, output_path):

    # Read the input mutation matrix, skipping the comment line and using space as separator
    mut_matrix = pd.read_csv(input_path, sep='\s+', skiprows=1, index_col=0)
    print(f"Shape of the input matrix after reading: {mut_matrix.shape}")  # Debugging print statement
    print(mut_matrix.head())  # Display the first few rows for verification

    # Extract the index and columns before converting to NumPy array
    row_index = mut_matrix.index
    col_index = mut_matrix.columns    

    # Convert mutation matrix to a numpy array and divide by 10,000 as per instruction
    mut_matrix = mut_matrix.values / 10000

    # Normalize the mutation matrix
    # row_sums = mut_matrix.sum(axis=1)
    # norm_matrix = mut_matrix / row_sums[:, np.newaxis]

    # Multiply the normalized matrix 'x' times to get PAMx
    pam_matrix = np.linalg.matrix_power(mut_matrix, x)

    # Calculate the log-odds matrix
    # Reorder the frequencies based on the row and column indices of the matrix
    frequencies = np.array([amino_acid_frequencies[aa] for aa in row_index])
    
    # Calculate R_ij = M_ij / f_i (observed change divided by frequency of the amino acid)
    R_matrix = pam_matrix / frequencies[:, np.newaxis]

    # Log-odds calculation: 10 * log10(R_matrix)
    log_odds_matrix = 10 * np.log10(R_matrix)
    log_odds_matrix[np.isneginf(log_odds_matrix)] = 0  # Handle -inf values
    log_odds_matrix = np.nan_to_num(log_odds_matrix, nan=0)  # Replace NaNs with 0
    log_odds_matrix = np.round(log_odds_matrix).astype(int)  # Round to nearest integer

    # Create a DataFrame to store the output matrix using original indices and column names
    pam_df = pd.DataFrame(log_odds_matrix, index=row_index, columns=col_index)

    # Save to output file with the correct format
    try:
        pam_df.to_csv(output_path, sep='\t', index=True, header=True)
        print(f"Output written successfully to {output_path}")
    except Exception as e:
        raise IOError(f"An error occurred while writing to the output file: {e}")

# Example usage:
generate_pam(5, 'mut.txt', 'pam5.txt')
generate_pam(10, 'mut.txt', 'pam10.txt')
generate_pam(100, 'mut.txt', 'pam100.txt')
generate_pam(150, 'mut.txt', 'pam150.txt')
generate_pam(250, 'mut.txt', 'pam250.txt')

# ChatGPT, respond to my prompt on Obtober 6, 2024
# "please read this file and provide me the code to get the hw done"
# "how to make sure the compiler can read the provided file" 
# "the code can be run successfully but the resulting matrix is not correct according to the hints"
# unfortunately conversation link unable to provide 
# as the current version is not supporting the conversation with uploaded image 
from src.features import PhoneticFeatureEncoder
from src.distances import binary_vector_distance
import numpy as np
from src.features import FEATURES
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import get_project_root

class PhonemeSubstitutionMatrix(object):
    """Class for creating, saving and loading phoneme substitution matrices."""
    
    def __init__(self,
                 save_dir: Path = get_project_root() / 'data',
                 encoder: PhoneticFeatureEncoder = None, 
                 weighted: bool = True):
        """Initialize the substitution matrix.
        
        Args:
            save_dir: Folder to save the matrix and mapping to
            encoder: PhoneticFeatureEncoder instance to use for creating matrix.
                    If None, matrix must be loaded from file.
            weighted: Whether to use weighted features when creating matrix
        """
        self.save_dir = save_dir
        self.encoder = encoder
        self.weighted = weighted
        self.matrix = None
        self.phoneme_to_index = None
        self.matrix_file = save_dir / 'substitution_matrix.csv'
        self.mapping_file = save_dir / 'substitution_matrix_mapping.csv'
        self.load()
        
    def create_matrix(self):
        """Create and sort the substitution matrix using the encoder."""
        # Get all phonemes from our FEATURES dictionary
        phonemes = list(FEATURES.keys())
        
        # Sort phonemes by their features
        features_list = [(p, self.encoder.phoneme_to_categorical(p)) for p in phonemes]
        
        def sort_key(item):
            p, feat = item
            [voicing, place, manner, height, backness, roundedness, is_consonant] = feat
            
            if is_consonant:
                # Consonants: sort by manner, place, voicing
                return (1, manner, place, voicing)
            else:
                # Vowels: sort by height, backness, roundedness
                return (0, height, backness, roundedness)
        
        # Sort phonemes and create mapping
        sorted_phonemes = [p for p, _ in sorted(features_list, key=sort_key)]
        self.phoneme_to_index = {p: i for i, p in enumerate(sorted_phonemes)}
        n_phonemes = len(sorted_phonemes)
        
        # Initialize substitution matrix
        self.matrix = np.zeros((n_phonemes, n_phonemes))
        
        # Fill matrix with distances between binary feature vectors
        for i, p1 in enumerate(sorted_phonemes):
            v1 = self.encoder.phoneme_to_binary(p1, weighted=self.weighted)
            for j, p2 in enumerate(sorted_phonemes):
                v2 = self.encoder.phoneme_to_binary(p2, weighted=self.weighted)
                self.matrix[i,j] = binary_vector_distance(v1, v2)
                
        # Round to 2 decimals and save
        self.matrix = np.round(self.matrix, decimals=2)
        self.save()

    def save(self):
        """Save the substitution matrix and phoneme mapping to CSV files.
        """
        if self.matrix is None or self.phoneme_to_index is None:
            raise ValueError("Matrix not initialized")
        
        # Create DataFrame with phoneme labels
        phonemes = list(self.phoneme_to_index.keys())
        df = pd.DataFrame(
            self.matrix,
            index=phonemes,
            columns=phonemes
        )
        
        # Save matrix with phoneme labels
        df.to_csv(self.matrix_file)
        
        # Save phoneme mapping separately
        pd.DataFrame.from_dict(self.phoneme_to_index, orient='index', 
                             columns=['index']).to_csv(self.mapping_file)
    
    def load(self):
        """Load a substitution matrix and phoneme mapping from CSV files.
        """
        # Create matrix if not already created
        if not self.matrix_file.exists():
            self.create_matrix()

        # Load matrix with phoneme labels
        df = pd.read_csv(self.matrix_file, index_col=0)
        self.matrix = df.values
        
        # Load phoneme mapping
        mapping_df = pd.read_csv(self.mapping_file, index_col=0)
        self.phoneme_to_index = mapping_df['index'].to_dict()
    
    def get_matrix(self):
        """Get the substitution matrix."""
        if self.matrix is None:
            raise ValueError("Matrix not initialized")
        return self.matrix

    def visualize(self):
        """Create a heatmap visualization of the substitution matrix."""
        if self.matrix is None or self.phoneme_to_index is None:
            raise ValueError("Matrix not initialized")
                    
        phonemes = list(self.phoneme_to_index.keys())
        plt.figure(figsize=(15, 15))
        sns.heatmap(self.matrix, 
                   xticklabels=phonemes,
                   yticklabels=phonemes,
                   cmap='viridis_r')  # _r makes smaller distances darker
        plt.title('Phoneme Substitution Costs')
        plt.show()

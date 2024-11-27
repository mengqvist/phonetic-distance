from src.features import PhoneticFeatureEncoder, WordEncoder, FEATURES
from src.utils import get_project_root, setup_logger
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


logger = setup_logger(__name__)


def feature_vector_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate Euclidean distance between two feature vectors.
    
    Args:
        v1, v2: Feature vectors of phonemes (can be binary or weighted)
        
    Returns:
        float: Euclidean distance between the vectors
    """
    return np.sqrt(np.sum((v1 - v2) ** 2))


class PhonemeSubstitutionMatrix(object):
    """Class for creating, saving and loading phoneme substitution matrices."""
    
    def __init__(self, weighted: bool = True) -> None:
        """Initialize the substitution matrix.
        
        Args:
            weighted: Whether to use weighted features when creating matrix.
                     Defaults to True.
        """
        self.save_dir = get_project_root() / 'data'
        self.encoder = PhoneticFeatureEncoder()
        self.weighted = weighted
        self.matrix = None
        self.phoneme_to_index = None
        
        if weighted is True:    
            self.matrix_file = self.save_dir / 'substitution_matrix_weighted.csv'
            self.mapping_file = self.save_dir / 'substitution_matrix_mapping_weighted.csv'
        else:
            self.matrix_file = self.save_dir / 'substitution_matrix.csv'
            self.mapping_file = self.save_dir / 'substitution_matrix_mapping.csv'
        self._load()
        
    def _create_matrix(self) -> None:
        """Create and sort the substitution matrix using the encoder.
        
        Creates a matrix of phoneme distances by:
        1. Getting all phonemes from FEATURES dictionary
        2. Sorting phonemes by their features
        3. Computing distances between binary feature vectors
        4. Saving the matrix and phoneme mapping to files
        """
        # Get all phonemes from our FEATURES dictionary
        phonemes = list(FEATURES.keys())
        
        # Sort phonemes by their features
        features_list = [(p, self.encoder.phoneme_to_categorical(p)) for p in phonemes]
        
        def sort_key(item):
            p, feat = item
            [voicing, place, manner, height, backness, roundedness, is_consonant, tense] = feat
            
            if is_consonant:
                # Consonants: sort by manner, place, voicing
                return (1, manner, place, voicing)
            else:
                # Vowels: sort by height, backness, roundedness, tense
                return (0, height, backness, roundedness, tense)
        
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
                self.matrix[i,j] = feature_vector_distance(v1, v2)
                
        # Round to 2 decimals and save
        self.matrix = np.round(self.matrix, decimals=2)
        self._save()

    def _save(self) -> None:
        """Save the substitution matrix and phoneme mapping to CSV files.
        
        Raises:
            ValueError: If matrix is not initialized.
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
    
    def _load(self) -> None:
        """Load a substitution matrix and phoneme mapping from CSV files.
        
        Creates a new matrix if one does not exist at the specified path.
        Also adds aliases to handle 'y'/'j' and 'r'/'ɹ' as the same phonemes.
        """
        # Create matrix if not already created
        if not self.matrix_file.exists():
            self._create_matrix()

        # Load matrix with phoneme labels
        df = pd.read_csv(self.matrix_file, index_col=0)
        self.matrix = df.values
        
        # Load phoneme mapping
        mapping_df = pd.read_csv(self.mapping_file, index_col=0)
        self.phoneme_to_index = mapping_df['index'].to_dict()

        # Add aliases to mapping, to handle them as the same phoneme
        self.phoneme_to_index['y'] = self.phoneme_to_index['j']
        self.phoneme_to_index['r'] = self.phoneme_to_index['ɹ']
    
    def get_matrix(self) -> np.ndarray:
        """Get the substitution matrix.
        
        Returns:
            np.ndarray: The phoneme substitution matrix.
            
        Raises:
            ValueError: If matrix is not initialized.
        """
        if self.matrix is None:
            raise ValueError("Matrix not initialized")
        return self.matrix

    def visualize(self) -> None:
        """Create a heatmap visualization of the substitution matrix.
        
        Displays a heatmap using seaborn where smaller distances are shown
        in darker colors.
        
        Raises:
            ValueError: If matrix is not initialized.
        """
        if self.matrix is None or self.phoneme_to_index is None:
            raise ValueError("Matrix not initialized")
                    
        phonemes = list(self.phoneme_to_index.keys())
        # Ignore aliases
        phonemes = [p for p in phonemes if p not in ['y', 'r']]

        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(self.matrix, 
                   xticklabels=phonemes,
                   yticklabels=phonemes,
                   cmap='viridis_r'
                   )  # _r makes smaller distances darker
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title('Phoneme Substitution Costs')
        colorbar = heatmap.collections[0].colorbar
        colorbar.set_label('Phonetic distance')
        plt.show()


class WordAligner(object):
    """Class for computing alignments between words using phoneme substitution costs."""
    
    def __init__(self, weighted: bool = True, gap_penalty: float = 1.0):
        """Initialize the aligner.
        
        Args:
            weighted: Whether to use weighted substitution matrix
            gap_penalty: Cost for insertions/deletions (gaps in alignment)
        """
        self.gap_penalty = gap_penalty
        self.substitution_matrix = PhonemeSubstitutionMatrix(weighted=weighted)
        self.encoder = WordEncoder()
        
    def _align_words(self, word1: str, word2: str) -> tuple[float, list[tuple]]:
        """
        Calculate phonetic distance between two words and return the optimal alignment.
        
        Args:
            word1, word2: Words to compare
            
        Returns:
            tuple[float, list[tuple]]: (distance, alignment)
            where alignment is a list of (phoneme1, phoneme2) pairs,
            with None representing gaps in the alignment
        """
        # Convert words to phonemes
        phonemes1 = self.encoder._word_to_phonemes(word1)
        phonemes2 = self.encoder._word_to_phonemes(word2)
        
        # Get matrix and mapping
        matrix = self.substitution_matrix.get_matrix()
        phoneme_to_index = self.substitution_matrix.phoneme_to_index
        
        # Initialize dynamic programming matrices
        m, n = len(phonemes1), len(phonemes2)
        dp = np.zeros((m + 1, n + 1))
        moves = np.zeros((m + 1, n + 1), dtype=int)  # 0: diag, 1: up, 2: left
        
        # Initialize first row and column with gap penalties
        for i in range(m + 1):
            dp[i, 0] = i * self.gap_penalty
        for j in range(n + 1):
            dp[0, j] = j * self.gap_penalty
        
        # Fill dp matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Look up substitution cost from matrix
                p1, p2 = phonemes1[i-1], phonemes2[j-1]
                # Handle aliases
                if p1 == 'y': p1 = 'j'
                elif p1 == 'r': p1 = 'ɹ'
                if p2 == 'y': p2 = 'j'
                elif p2 == 'r': p2 = 'ɹ'
                idx1, idx2 = phoneme_to_index[p1], phoneme_to_index[p2]
                substitution_cost = matrix[idx1, idx2]
                
                # Find minimum cost move
                costs = [
                    dp[i-1, j-1] + substitution_cost,  # diagonal
                    dp[i-1, j] + self.gap_penalty,     # up (deletion)
                    dp[i, j-1] + self.gap_penalty,     # left (insertion)
                ]
                dp[i, j] = min(costs)
                moves[i, j] = np.argmin(costs)
        
        # Backtrack to find alignment
        alignment = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and moves[i, j] == 0:  # diagonal
                alignment.append((phonemes1[i-1], phonemes2[j-1]))
                i -= 1
                j -= 1
            elif i > 0 and moves[i, j] == 1:  # up
                alignment.append((phonemes1[i-1], None))
                i -= 1
            else:  # left
                alignment.append((None, phonemes2[j-1]))
                j -= 1
                
        # Reverse alignment since we backtracked
        alignment.reverse()
        
        return dp[m, n], alignment

    def compare_words(self, word1: str, word2: str) -> tuple[list, list, float]:
        """
        Compare two words and show their alignment.
        
        Args:
            word1, word2: Words to compare
            
        Returns:
            tuple[list, list, float]: (line1, line2, distance)
            where line1 and line2 are aligned phoneme sequences with '-' for gaps
        """
        try:
            distance, alignment = self._align_words(word1, word2)
            
            # Format alignment
            line1 = []
            line2 = []
            for p1, p2 in alignment:
                # Use '-' to represent gaps
                line1.append(p1 if p1 is not None else '-')
                line2.append(p2 if p2 is not None else '-')
            
            return ''.join(line1), ''.join(line2), float(distance)
            
        except Exception as e:
            logger.error(f"Error comparing words: {e}")
            raise
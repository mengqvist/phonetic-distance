import numpy as np
from .features import WordEncoder
from .utils import setup_logger

logger = setup_logger(__name__)

def binary_vector_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate distance between two binary feature vectors.
    
    Args:
        v1, v2: Binary feature vectors of phonemes
        
    Returns:
        float: Euclidean distance between the vectors
    """
    return np.sqrt(np.sum((v1 - v2) ** 2))

def calculate_word_distance_with_alignment(word1: str, word2: str, weighted: bool = False) -> tuple[float, list[tuple]]:
    """
    Calculate phonetic distance between two words and return the optimal alignment.
    
    Args:
        word1, word2: Words to compare
        weighted: Whether to use weighted feature vectors
        
    Returns:
        tuple[float, list[tuple]]: (distance, alignment)
        where alignment is a list of (phoneme1, phoneme2) pairs,
        with None representing gaps in the alignment
    """
    # Initialize word encoder
    encoder = WordEncoder()
    
    # Convert words to sequences of binary feature vectors and keep original phonemes
    phonemes1 = encoder._word_to_phonemes(word1)
    phonemes2 = encoder._word_to_phonemes(word2)
    vec1 = encoder.encode_word(word1, weighted=weighted)
    vec2 = encoder.encode_word(word2, weighted=weighted)
    
    # Initialize dynamic programming matrices
    m, n = len(vec1), len(vec2)
    dp = np.zeros((m + 1, n + 1))
    # Keep track of moves for backtracking
    moves = np.zeros((m + 1, n + 1), dtype=int)  # 0: diag, 1: up, 2: left
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    
    # Fill dp matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate substitution cost using binary vector distance
            substitution_cost = binary_vector_distance(vec1[i-1], vec2[j-1])
            
            # Find minimum cost move
            costs = [
                dp[i-1, j-1] + substitution_cost,  # diagonal
                dp[i-1, j] + 1,  # up (deletion)
                dp[i, j-1] + 1,  # left (insertion)
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

def compare_words(word1: str, word2: str, weighted: bool = False):
    """
    Compare two words and show their alignment.
    """
    try:
        distance, alignment = calculate_word_distance_with_alignment(word1, word2, weighted=weighted)
        
        # Display alignment
        print(f"Distance between '{word1}' and '{word2}': {distance:.2f}")
        print("\nAlignment:")
        line1 = []
        line2 = []
        for p1, p2 in alignment:
            # Use '-' to represent gaps
            line1.append(p1 if p1 is not None else '-')
            line2.append(p2 if p2 is not None else '-')
        print(f"Word1: {''.join(line1)}")
        print(f"Word2: {''.join(line2)}")
        
    except Exception as e:
        logger.error(f"Error comparing words: {e}")
        raise

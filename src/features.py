from .utils import setup_logger, FEATURE_WEIGHTS, FEATURES
import numpy as np
import eng_to_ipa
from sklearn.preprocessing import OneHotEncoder
import re

logger = setup_logger(__name__)



class PhoneticFeatureEncoder:
   def __init__(self):
       # Extract unique values for each feature position
       features_array = np.array(list(FEATURES.values()))
       self.feature_categories = [
           sorted(set(features_array[:, i])) 
           for i in range(features_array.shape[1])
       ]
       
       # Create and fit one-hot encoders for each feature
       self.encoders = [
           OneHotEncoder(categories=[cats], sparse_output=False).fit(np.array(cats).reshape(-1, 1))
           for cats in self.feature_categories
       ]

       # Initialize weights - expand to match binary feature vector length
       expanded_weights = []
       for i, cats in enumerate(self.feature_categories):
           # Repeat each weight by number of categories for that feature
           expanded_weights.extend([FEATURE_WEIGHTS[i]] * len(cats))
       self.weights = np.array(expanded_weights)

   def phoneme_to_categorical(self, phoneme: str) -> list:
       """Get the categorical feature vector for a phoneme."""
       if not isinstance(phoneme, str):
           raise TypeError(f"Phoneme must be a string, got {type(phoneme)}")
           
       if not phoneme or len(phoneme) > 1:
           raise ValueError(f"Invalid phoneme: {phoneme}")
           
       if phoneme not in FEATURES:
           raise ValueError(f"Unknown phoneme: {phoneme}")
       
       return FEATURES[phoneme]

   def phoneme_to_binary(self, phoneme: str, weighted: bool = False) -> np.ndarray:
       """Get the binary feature vector for a phoneme."""
       features = self.phoneme_to_categorical(phoneme)

       # Create binary vectors and then concatenate
       binary_vectors = []
       for encoder, feature in zip(self.encoders, features):
            binary = encoder.transform([[feature]])
            binary_vectors.append(binary.flatten())  # flatten each binary vector
       binary = np.concatenate(binary_vectors)

       if weighted:
           binary = binary * self.weights
       
       return binary
   
   def binary_to_categorical(self, binary_vector: np.ndarray) -> list:
       """Convert a binary feature vector back to categorical features."""
       if len(binary_vector) != sum(len(cats) for cats in self.feature_categories):
           raise ValueError("Invalid binary vector length")
       
       # Split binary vector into segments for each feature
       start = 0
       categorical = []
       for encoder, cats in zip(self.encoders, self.feature_categories):
           length = len(cats)
           segment = binary_vector[start:start + length]
           categorical.append(int(encoder.inverse_transform([segment])[0][0]))
           start += length
           
       return categorical



class WordEncoder(object):
    """Encodes words into their phonetic binary feature representations."""
    
    def __init__(self):
        """Initialize the word encoder."""
        self.feature_encoder = PhoneticFeatureEncoder()
        
    def _word_to_phonemes(self, word: str, strip_stress_marks: bool = True) -> list[str]:
        """Convert a word to its IPA phonetic representation.
        
        Args:
            word: Word to convert to phonemes
            strip_stress_marks: Whether to remove stress marks like 'ˈ'
            
        Returns:
            list[str]: List of IPA phoneme characters
            
        Raises:
            ValueError: If word cannot be converted to phonemes
        """
        try:
            phonemes = eng_to_ipa.convert(word)
            if strip_stress_marks:
                # Remove stress marks like 'ˈ', 'ˌ', and 'ˆ'
                phonemes = re.sub(r'[ˈˌˆ]', '', phonemes)

            # Check if conversion failed (indicated by '*' at end)
            if phonemes.endswith('*'):
                logger.error(f"Could not convert '{word}' to phonemes")
                raise ValueError(f"Could not convert '{word}' to phonemes")

            return list(phonemes)
                
        except Exception as e:
            logger.error(f"Error converting word to phonemes: {e}")
            raise ValueError(f"Error converting word to phonemes: {e}")
        
    def encode_word(self, word: str, weighted: bool = False) -> list[np.ndarray]:
        """Convert a word to binary phonetic feature vectors.
        
        Args:
            word: Word to encode
            weighted: Whether to weight the binary features
            
        Returns:
            list[np.ndarray]: List of binary feature vectors, one per phoneme
            
        Raises:
            TypeError: If word is not a string
        """
        if not isinstance(word, str):
            logger.error(f"Word must be a string, got {type(word)}")
            raise TypeError(f"Word must be a string, got {type(word)}")
            
        phonemes = self._word_to_phonemes(word)
        return [
            self.feature_encoder.phoneme_to_binary(phoneme, weighted=weighted)
            for phoneme in phonemes
        ]
        
    def get_categorical_features(self, word: str) -> list[list[int]]:
        """Get categorical feature vectors for each phoneme in a word.
        
        Args:
            word: Word to analyze
            
        Returns:
            list[list[int]]: List of categorical feature vectors, one per phoneme.
                Each vector contains 7 features:
                [voicing, place, manner, height, backness, roundedness, consonant]
                
        Raises:
            TypeError: If word is not a string
        """
        if not isinstance(word, str):
            logger.error(f"Word must be a string, got {type(word)}")
            raise TypeError(f"Word must be a string, got {type(word)}")
            
        phonemes = self._word_to_phonemes(word)
        return [
            self.feature_encoder.phoneme_to_categorical(phoneme)
            for phoneme in phonemes
        ]



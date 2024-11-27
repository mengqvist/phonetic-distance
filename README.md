# Phonetic Distance Calculator

A playful exploration of phonetic distances between words, with a focus on the NATO phonetic alphabet (alpha, bravo, charlie...). This project implements a feature-based approach to compare words based on their sounds rather than their spelling.

**IMPORTANT:** I am not a linguist. This is a fun modelling project and the results should be taken with a grain of salt. It is not an authoritative source of how to correctly calculate phonetic distances.

## Overview

The main components:
- Feature vectors for phonemes based on linguistic properties (voicing, place of articulation, etc.)
- Custom distance metric for comparing phonemes
- Sequence alignment for comparing full words
- Analysis tools for the NATO phonetic alphabet

## Installation

```bash
# Create conda environment
conda env create -f requirements.yml

# Activate environment
conda activate phonetic-distance
```

## Project Structure

```
phonetic-distance/
├── requirements.yml
├── README.md
├── data/
│   └── nato_alphabet.json
├── notebooks/
│   └── examples.ipynb
└── src/
   ├── __init__.py
   ├── features.py      # PhoneticFeatures class
   ├── distances.py     # Distance calculation functions  
   └── utils.py         # Helper functions
```


## Usage

### Basic phonetic feature visualization

Work with phonetic features using the `PhoneticFeatureEncoder` class:

```python
# Initialize encoder
from src.features import PhoneticFeatureEncoder
encoder = PhoneticFeatureEncoder()

# Obtain the feature vector for a phoneme
vector = encoder.phoneme_to_binary("b")
print(vector)

# Convert a feature vector back to a phoneme
categorical = encoder.binary_to_categorical(vector)
print(categorical)

# Convert a phoneme to its categorical representation
categorical = encoder.phoneme_to_categorical("b")
print(categorical)

# Visualize phonemes
encoder.visualize()
```

### Basic Word Comparison

Compare two words using the `WordAligner` class:

```python
from src.distances import WordAligner

# Initialize aligner (defaults: weighted=True, gap_penalty=1.0)
aligner = WordAligner()

# Compare two words
phonemes1, phonemes2, distance = aligner.compare_words("hello", "halo")
print(f"Aligned phonemes:")
print(f"Word 1: {phonemes1}")
print(f"Word 2: {phonemes2}")
print(f"Distance: {distance:.2f}")
```

### Substitution Matrix

Visualize or work with the phoneme substitution costs:

```python
from src.distances import PhonemeSubstitutionMatrix

# Create matrix with weighted features (recommended)
matrix = PhonemeSubstitutionMatrix(weighted=True)

# Visualize the substitution costs as a heatmap
matrix.visualize()

# Access the raw matrix and phoneme mapping
costs = matrix.get_matrix()
phoneme_mapping = matrix.phoneme_to_index
```

### Feature Encoding

Work directly with phonetic features:

```python
from src.features import WordEncoder

# Initialize encoder
encoder = WordEncoder()

# Get phonemes for a word
phonemes = encoder._word_to_phonemes("hello")
print(f"Phonemes: {phonemes}")

# Get categorical features for each phoneme
features = encoder.get_categorical_features("hello")
print("Features for each phoneme:")
for p, f in zip(phonemes, features):
    print(f"{p}: {f}")
```

### Working with IPA and Aliases

The system handles common alternative representations:
- 'r' is treated as 'ɹ' (English 'r' sound)
- 'y' is treated as 'j' (as in "yes")

Features for each phoneme include:
1. Voicing (0-1): voiceless -> voiced
2. Place (0-6): bilabial -> labiodental -> dental -> alveolar -> postalveolar -> velar -> glottal
3. Manner (0-6): plosive -> fricative -> nasal -> lateral -> approximant -> tap/trill -> affricate
4. Height (0-3): low -> mid -> high
5. Backness (0-2): front -> central -> back
6. Roundedness (0-1): unrounded -> rounded
7. Consonant/vowel (0-1): vowel -> consonant
8. Tense (0-1): lax -> tense

### Advanced Usage

Customize the alignment parameters:

```python
# Use unweighted features and custom gap penalty
aligner = WordAligner(weighted=False, gap_penalty=0.5)

# Compare a batch of words
word_pairs = [
    ("hello", "halo"),
    ("world", "word"),
    ("phone", "foam")
]

for w1, w2 in word_pairs:
    p1, p2, dist = aligner.compare_words(w1, w2)
    print(f"\nComparing {w1} vs {w2}")
    print(f"Alignment: {p1} / {p2}")
    print(f"Distance: {dist:.2f}")
```

### Note on Weighted vs Unweighted Features

- Weighted features (default) apply linguistic importance weights to different features
- For example, manner of articulation is weighted more heavily than voicing
- Use weighted=True for more linguistically motivated distances
- Use weighted=False for pure feature-based distances

The substitution matrices are cached in the data/ directory:
- weighted=True: 'substitution_matrix_weighted.csv'
- weighted=False: 'substitution_matrix.csv'


## Validation

Stop consonants are grouped (p/b/t/d/k/g)
Fricatives cluster together (f/v/s/z)
Vowels form their own region
Similar-sounding phonemes are generally closer together
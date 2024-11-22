# Phonetic Distance Calculator

A playful exploration of phonetic distances between words, with a focus on the NATO phonetic alphabet (alpha, bravo, charlie...). This project implements a feature-based approach to compare words based on their sounds rather than their spelling.

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


## Validation

Stop consonants are grouped (p/b/t/d/k/g)
Fricatives cluster together (f/v/s/z)
Vowels form their own region
Similar-sounding phonemes are generally closer together
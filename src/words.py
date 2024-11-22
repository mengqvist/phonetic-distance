import sys
sys.path.append('..')

import requests
import json
import os
from pathlib import Path
import time
from src.utils import setup_logger, get_project_root

logger = setup_logger(__name__)

def load_data(file_path: str) -> dict:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        logger.info(f"Loading data from {file_path}")
        return json.load(f)

def fetch_word_data(word: str, save_dir: Path = get_project_root() / 'data' / 'words') -> dict:
    """Fetch word data from API and save to file.
    
    Makes a request to the Dictionary API to get pronunciation and definition data for a word.
    Caches the response JSON in a local file to avoid repeated API calls.
    
    Args:
        word: The word to look up
        save_dir: Directory path to save cached word data. Defaults to project_root/data/words
        
    Returns:
        dict: The word data loaded from either the API response or cached file
        
    Raises:
        ValueError: If the API request fails
    """
    # Create directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define json file path
    file_path = save_dir / f"{word}.json"
    
    # Check if we already have the data
    if not file_path.exists():
        logger.info(f"Fetching data for word '{word}'")

        # If not, fetch from API
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url)
        
        if response.status_code != 200:
            raise ValueError(f"Could not fetch data for word '{word}'. Status code: {response.status_code}")
        
        data = response.json()
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        # Add small delay to be nice to the API
        time.sleep(0.1)

    return load_data(file_path)

def word_to_phonemes(word: str, strip_slashes: bool = True, strip_stress_marks: bool = True) -> str:
    """Extract phonetic notation from word data.
    
    Gets the IPA phonetic transcription for a word from the Dictionary API data.
    Uses the first available pronunciation if multiple exist.
    
    Args:
        word: The word to get phonemes for
        strip_slashes: Whether to remove surrounding '/' characters. Defaults to True
        strip_stress_marks: Whether to remove stress marks like 'ˈ'. Defaults to True
        
    Returns:
        str: The IPA phonetic transcription
        
    Raises:
        ValueError: If no phonetic notation is found for the word
    """
    # get json data for word
    word_data = fetch_word_data(word)

    # Get all phonetic notations
    phonetics = [p.get('text', '') for p in word_data[0].get('phonetics', []) if p.get('text')]
    if phonetics:
        # Get first pronunciation
        pronunciation = phonetics[0]

        # Strip slashes
        if strip_slashes:
            pronunciation = pronunciation.strip('/')

        # Remove stress marks
        if strip_stress_marks:
            pronunciation = pronunciation.replace('ˈ', '')

        return pronunciation
    
    else:
        raise ValueError(f"No phonetic notation found for word '{word}'")

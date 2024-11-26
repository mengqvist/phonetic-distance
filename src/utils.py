import logging
import sys
from pathlib import Path
import numpy as np

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Set up and return a logger with consistent formatting.
    
    Args:
        name: Name of the logger, typically __name__ from the calling module
        log_file: Optional path to log file. If None, logs only to console
        level: Logging level (default: logging.INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optionally add file handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_project_root() -> Path:
    """Get project root directory from any script location"""
    # Start with current working directory
    current_dir = Path.cwd()
    
    # Look for key project files/directories that indicate the root
    # For example, look for 'data' dir, requirements.yml, or .git
    while current_dir != current_dir.parent:  # stop at root
        if (current_dir / 'environment.yml').exists() or \
           (current_dir / '.git').exists() or \
           (current_dir / 'data').exists():
            return current_dir
        current_dir = current_dir.parent
    
    # If we couldn't find project root, use current directory
    return Path.cwd()


# Feature weights for distance calculation, may be tuned for better performance
FEATURE_WEIGHTS = np.array([
    0.5,      # Voicing: Different voicing is noticeable but not major
    1.0,      # Place: Place of articulation is important
    1.5,      # Manner: Manner of articulation is very important
    1.0,      # Height: Vowel height is important
    1.0,      # Backness: Vowel backness is important
    0.5,      # Roundedness: Roundedness is less crucial
    2.0,      # Consonant: Consonant vs vowel is very important
    0.8       # Tense: Tenseness is moderately important (new)
])
    
FEATURES = {
    # [voicing, place, manner, height, backness, roundedness, consonant/vowel, tense]
    # Voicing (0-1): voiceless -> voiced
    # Place (0-6): bilabial -> labiodental -> dental -> alveolar -> postalveolar -> velar -> glottal
    # Manner (0-6): plosive -> fricative -> nasal -> lateral -> approximant -> trill -> affricate
    # Height (0-3): low -> mid -> high
    # Backness (0-2): front -> central -> back
    # Roundedness (0-1): unrounded -> rounded
    # Consonant/vowel (0-1): vowel -> consonant
    # Tense (0-1): lax -> tense

    # Vowels
    'i': [1, 0, 0, 3, 0, 0, 0, 1],  # high front unrounded tense (see)
    'ɪ': [1, 0, 0, 3, 0, 0, 0, 0],  # high front unrounded lax (sit)
    'e': [1, 0, 0, 2, 0, 0, 0, 1],  # upper-mid front unrounded tense (say)
    'ɛ': [1, 0, 0, 2, 0, 0, 0, 0],  # mid front unrounded lax (bed)
    'æ': [1, 0, 0, 1, 0, 0, 0, 0],  # near-low front unrounded lax (cat)
    'ə': [1, 0, 0, 2, 1, 0, 0, 0],  # mid central unrounded lax (about)
    'ʌ': [1, 0, 0, 1, 2, 0, 0, 0],  # low-mid back unrounded lax (but)
    'ɑ': [1, 0, 0, 0, 2, 0, 0, 0],  # low back unrounded lax (hot)
    'ɔ': [1, 0, 0, 1, 2, 1, 0, 0],  # low-mid back rounded lax (thought)
    'o': [1, 0, 0, 2, 2, 1, 0, 1],  # upper-mid back rounded tense (go)
    'ʊ': [1, 0, 0, 3, 2, 1, 0, 0],  # high back rounded lax (put)
    'u': [1, 0, 0, 3, 2, 1, 0, 1],  # high back rounded tense (boot)

    # Stops
    'p': [0, 0, 0, 0, 0, 0, 1, 0],  # voiceless bilabial plosive (pat)
    'b': [1, 0, 0, 0, 0, 0, 1, 0],  # voiced bilabial plosive (bat)
    't': [0, 3, 0, 0, 0, 0, 1, 0],  # voiceless alveolar plosive (tap)
    'd': [1, 3, 0, 0, 0, 0, 1, 0],  # voiced alveolar plosive (dap)
    'k': [0, 5, 0, 0, 0, 0, 1, 0],  # voiceless velar plosive (cat)
    'g': [1, 5, 0, 0, 0, 0, 1, 0],  # voiced velar plosive (gap)

    # Fricatives
    'f': [0, 1, 1, 0, 0, 0, 1, 0],  # voiceless labiodental fricative (fat)
    'v': [1, 1, 1, 0, 0, 0, 1, 0],  # voiced labiodental fricative (vat)
    'θ': [0, 2, 1, 0, 0, 0, 1, 0],  # voiceless dental fricative (thin)
    'ð': [1, 2, 1, 0, 0, 0, 1, 0],  # voiced dental fricative (this)
    's': [0, 3, 1, 0, 0, 0, 1, 0],  # voiceless alveolar fricative (sat)
    'z': [1, 3, 1, 0, 0, 0, 1, 0],  # voiced alveolar fricative (zip)
    'ʃ': [0, 4, 1, 0, 0, 0, 1, 0],  # voiceless postalveolar fricative (ship)
    'ʒ': [1, 4, 1, 0, 0, 0, 1, 0],  # voiced postalveolar fricative (measure)
    'h': [0, 6, 1, 0, 0, 0, 1, 0],  # voiceless glottal fricative (hat)

    # Nasals
    'm': [1, 0, 2, 0, 0, 0, 1, 0],  # bilabial nasal (mat)
    'n': [1, 3, 2, 0, 0, 0, 1, 0],  # alveolar nasal (nat)
    'ŋ': [1, 5, 2, 0, 0, 0, 1, 0],  # velar nasal (sing)

    # Approximants and Liquids
    'l': [1, 3, 3, 0, 0, 0, 1, 0],  # alveolar lateral (lip)
    'ɹ': [1, 3, 4, 0, 0, 0, 1, 0],  # alveolar approximant (rat), alias for r
    'j': [1, 4, 4, 0, 0, 0, 1, 0],  # palatal approximant (yes), alias for y
    'w': [1, 5, 4, 0, 0, 1, 1, 0],  # labial-velar approximant (wet)

    # Affricates
    'ʧ': [0, 4, 6, 0, 0, 0, 1, 0],  # voiceless postalveolar affricate (church)
    'ʤ': [1, 4, 6, 0, 0, 0, 1, 0],  # voiced postalveolar affricate (judge)
}

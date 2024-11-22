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
    2.0])     # Consonant: Consonant vs vowel is very important
    
FEATURES = {
    # [voicing, place, manner, height, backness, roundedness, consonant/vowel]
    # Voicing (0-1): voiceless -> voiced
    # Place of articulation (0-6): bilabial -> labiodental -> dental -> alveolar -> postalveolar -> velar -> glottal
    # Manner of articulation (0-6): plosive -> fricative -> nasal -> lateral -> approximant -> tap/trill -> affricate
    # Height (0-3): low -> mid -> high
    # Backness (0-2): front -> central -> back
    # Roundedness (0-1): unrounded -> rounded
    # Consonant/vowel (0-1): vowel -> consonant

    # Vowels (height, backness, roundedness are primary features)
    'i': [1, 0, 0, 3, 0, 0, 0],  # high front unrounded (as in "see")
    'ɪ': [1, 0, 0, 2, 0, 0, 0],  # near-high front unrounded (as in "sit") 
    'e': [1, 0, 0, 2, 0, 0, 0],  # upper-mid front unrounded (as in "say")
    'ɛ': [1, 0, 0, 1, 0, 0, 0],  # mid front unrounded (as in "bed")
    'æ': [1, 0, 0, 0, 0, 0, 0],  # near-low front unrounded (as in "cat")
    'a': [1, 0, 0, 0, 0, 0, 0],  # low front unrounded (as in "father")
    'ɑ': [1, 0, 0, 0, 2, 0, 0],  # low back unrounded (as in "hot")
    'ɒ': [1, 0, 0, 0, 2, 1, 0],  # low back rounded (as in British "lot")
    'ɔ': [1, 0, 0, 1, 2, 1, 0],  # mid back rounded (as in "thought")
    'o': [1, 0, 0, 2, 2, 1, 0],  # upper-mid back rounded (as in "go")
    'ʊ': [1, 0, 0, 2, 2, 1, 0],  # near-high back rounded (as in "put")
    'u': [1, 0, 0, 3, 2, 1, 0],  # high back rounded (as in "boot")
    'y': [1, 0, 0, 3, 0, 1, 0],  # high front rounded (as in French "tu")
    'ʏ': [1, 0, 0, 2, 0, 1, 0],  # near-high front rounded (as in German "hübsch")
    'ø': [1, 0, 0, 2, 0, 1, 0],  # upper-mid front rounded (as in French "deux")
    'œ': [1, 0, 0, 1, 0, 1, 0],  # mid front rounded (as in French "sœur")
    'ə': [1, 0, 0, 1, 1, 0, 0],  # mid central unrounded (as in "about")
    'ɜ': [1, 0, 0, 1, 1, 0, 0],  # mid central unrounded (as in British "bird")
    'ɞ': [1, 0, 0, 1, 1, 1, 0],  # mid central rounded (as in Swedish "öga")
    'ʌ': [1, 0, 0, 1, 2, 0, 0],  # low-mid back unrounded (as in "but")
    
    # Consonants (place and manner are primary features)
    'p': [0, 0, 0, 0, 0, 0, 1],  # voiceless bilabial plosive (as in "pat")
    'b': [1, 0, 0, 0, 0, 0, 1],  # voiced bilabial plosive (as in "bat")
    't': [0, 2, 0, 0, 0, 0, 1],  # voiceless alveolar plosive (as in "tap")
    'd': [1, 2, 0, 0, 0, 0, 1],  # voiced alveolar plosive (as in "dap")
    'k': [0, 4, 0, 0, 0, 0, 1],  # voiceless velar plosive (as in "cat")
    'g': [1, 4, 0, 0, 0, 0, 1],  # voiced velar plosive (as in "gap")
    'q': [0, 5, 0, 0, 0, 0, 1],  # voiceless uvular plosive (as in Arabic "qalb")
    'ɢ': [1, 5, 0, 0, 0, 0, 1],  # voiced uvular plosive (as in Arabic "Quran")
    'ʔ': [0, 6, 0, 0, 0, 0, 1],  # glottal stop (as in "uh-oh")
    'f': [0, 1, 1, 0, 0, 0, 1],  # voiceless labiodental fricative (as in "fat")
    'v': [1, 1, 1, 0, 0, 0, 1],  # voiced labiodental fricative (as in "vat")
    'θ': [0, 2, 1, 0, 0, 0, 1],  # voiceless dental fricative (as in "thin")
    'ð': [1, 2, 1, 0, 0, 0, 1],  # voiced dental fricative (as in "this")
    's': [0, 2, 1, 0, 0, 0, 1],  # voiceless alveolar fricative (as in "sat")
    'z': [1, 2, 1, 0, 0, 0, 1],  # voiced alveolar fricative (as in "zip")
    'ʃ': [0, 3, 1, 0, 0, 0, 1],  # voiceless postalveolar fricative (as in "ship")
    'ʒ': [1, 3, 1, 0, 0, 0, 1],  # voiced postalveolar fricative (as in "measure")
    'h': [0, 6, 1, 0, 0, 0, 1],  # voiceless glottal fricative (as in "hat")
    'm': [1, 0, 2, 0, 0, 0, 1],  # bilabial nasal (as in "mat")
    'n': [1, 2, 2, 0, 0, 0, 1],  # alveolar nasal (as in "nat")
    'ŋ': [1, 4, 2, 0, 0, 0, 1],  # velar nasal (as in "sing")
    'l': [1, 2, 3, 0, 0, 0, 1],  # alveolar lateral (as in "lip")
    'ɹ': [1, 2, 4, 0, 0, 0, 1],  # alveolar approximant (as in "rat")
    'j': [1, 3, 4, 0, 0, 0, 1],  # palatal approximant (as in "yes")
    'w': [1, 4, 4, 0, 0, 1, 1],  # labial-velar approximant (as in "wet")
    'ɾ': [1, 2, 5, 0, 0, 0, 1],  # alveolar tap (as in Spanish "pero")
    'ʙ': [1, 0, 5, 0, 0, 0, 1],  # bilabial trill (as in Czech "brr")
    'r': [1, 2, 5, 0, 0, 0, 1],  # alveolar trill (as in Spanish "perro")
    'χ': [0, 5, 1, 0, 0, 0, 1],  # voiceless uvular fricative (as in German "Bach")
    'ʁ': [1, 5, 1, 0, 0, 0, 1],  # voiced uvular fricative (as in French "rouge")

    # Affricates
    # Note: Affricates combine features of plosives and fricatives
    'ʧ': [0, 3, 6, 0, 0, 0, 1],  # voiceless postalveolar affricate (as in "church")
    'ʤ': [1, 3, 6, 0, 0, 0, 1],  # voiced postalveolar affricate (as in "judge")
}

from src.features import WordEncoder
from src.distances import WordAligner


# Example usage:
"""


# Initialize

optimizer = GapPenaltyOptimizer(word_encoder)

# Find optimal penalty
penalties = 
best_penalty, results = optimizer.find_optimal_penalty(penalties)

# Analyze results
analysis = optimizer.analyze_results(results)

# Print summary
print(f"Best gap penalty: {best_penalty}")
print(f"Best score: {analysis['best_score']}")

# Example of accessing detailed results
for penalty in penalties:
    print(f"\nPenalty {penalty}:")
    score = results[penalty]['score']
    print(f"Overall score: {score:.2f}")
    
    # Print some example alignments
    for detail in results[penalty]['details'][:3]:  # First 3 cases
        print(f"{detail['word1']} - {detail['word2']}: {detail['alignment']}")
"""


class GapPenaltyOptimizer(object):
    """Class for finding optimal gap penalty using test cases."""
    
    def __init__(self):
        """Initialize with a WordEncoder instance."""
        # self.word_encoder = WordEncoder()
        self.penalties_to_try = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
        self.best_penalty = None
        self.best_score = -1
        self.all_results = {}
        self.analysis = None
        
        # Test cases where no gaps should appear
        self.no_gap_cases = [
            ('phone', 'foam', 0),
            ('light', 'right', 0),
            ('cat', 'kit', 0),
            ('bat', 'bet', 0),
            ('sin', 'sun', 0),
            ('pig', 'peg', 0),
            ('peak', 'pick', 0),
            ('pool', 'pull', 0)
        ]
        
        # Test cases where gaps should appear
        self.gap_cases = [
            ('strip', 'tip', 2),    # 2 gaps at start
            ('lamp', 'clamp', 1),   # 1 gap at start
            ('smile', 'mile', 1),   # 1 gap at start
            ('spark', 'park', 1),   # 1 gap at start
            ('stamp', 'tap', 2),    # 2 gaps (start and end)
            ('spring', 'ping', 2)   # 2 gaps at start
        ]

        self.find_optimal_penalty()
        
    def evaluate_single_case(self, word1, word2, expected_gaps, gap_penalty):
        """Evaluate alignment for a single test case.
        
        Args:
            word1: First word to align
            word2: Second word to align
            expected_gaps: Number of gaps expected in correct alignment
            gap_penalty: Gap penalty to test
            
        Returns:
            dict containing alignment results and analysis
        """
        aligner = WordAligner(gap_penalty=gap_penalty)
        line1, line2, distance = aligner.compare_words(word1, word2)
        
        # Count actual gaps
        actual_gaps = sum(1 for c in line1 if c == '-') + sum(1 for c in line2 if c == '-')
        
        return {
            'word1': word1,
            'word2': word2,
            'expected_gaps': expected_gaps,
            'actual_gaps': actual_gaps,
            'correct': actual_gaps == expected_gaps,
            'alignment': (line1, line2),
            'distance': distance
        }
    
    def evaluate_gap_penalty(self, gap_penalty, cases=None):
        """Evaluate how well a gap penalty works on test cases.
        
        Args:
            gap_penalty: float, the gap penalty to test
            cases: list of test cases to use, or None to use all cases
            
        Returns:
            tuple of (score, results dictionary)
        """
        if cases is None:
            cases = self.no_gap_cases + self.gap_cases
            
        results = []
        for word1, word2, expected_gaps in cases:
            result = self.evaluate_single_case(word1, word2, expected_gaps, gap_penalty)
            results.append(result)
        
        # Calculate overall score
        score = sum(r['correct'] for r in results) / len(results)
        
        return score, results
    
    def find_optimal_penalty(self):
        """Find the optimal gap penalty by testing a range of values.
        """
        best_score = -1
        best_penalty = None
        all_results = {}
        
        for penalty in self.penalties_to_try:
            score, details = self.evaluate_gap_penalty(penalty)
            all_results[penalty] = {
                'score': score,
                'details': details
            }
            
            if score > best_score:
                best_score = score
                best_penalty = penalty
        
        self.best_penalty = best_penalty
        self.best_score = best_score
        self.all_results = all_results
    
    def analyze_results(self):
        """Analyze optimization results in detail.

                Returns:
            dict containing analysis metrics
        """
        analysis = {
            'penalties_tested': list(self.all_results.keys()),
            'scores': [self.all_results[p]['score'] for p in self.all_results.keys()],
            'best_penalty': max(self.all_results.keys(), key=lambda p: self.all_results[p]['score']),
            'best_score': max(r['score'] for r in self.all_results.values()),
            'per_case_analysis': {}
        }
        
        # Analyze performance on each test case across penalties
        all_words = set((r['word1'], r['word2']) 
                       for p in self.all_results 
                       for r in self.all_results[p]['details'])
        
        for word1, word2 in all_words:
            case_results = {}
            for penalty in self.all_results:
                case = next((r for r in self.all_results[penalty]['details'] 
                           if r['word1'] == word1 and r['word2'] == word2), None)
                if case:
                    case_results[penalty] = {
                        'correct': case['correct'],
                        'actual_gaps': case['actual_gaps'],
                        'alignment': case['alignment']
                    }
            analysis['per_case_analysis'][(word1, word2)] = case_results
        
        self.analysis = analysis



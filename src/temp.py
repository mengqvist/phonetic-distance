
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """Store optimization results"""
    weights: np.ndarray
    threshold: float
    steepness: float
    loss_history: List[float]
    final_loss: float

class SymmetricFeatureOptimizer:
    """Optimizes phonetic feature weights using symmetrized confusion matrices.
    
    This class implements optimization of feature weights by minimizing the difference
    between predicted and actual confusion matrices. It uses symmetrized confusion
    matrices to focus on inherent phonetic similarities rather than directional effects.
    
    Attributes:
        weights: Feature weight vector for computing phonetic distances
        threshold: Sigmoid threshold parameter for converting distances to similarities
        steepness: Sigmoid steepness parameter for converting distances to similarities
        consonant_confusion: Symmetrized consonant confusion matrix
        vowel_confusion: Symmetrized vowel confusion matrix
        consonant_mapping: Mapping from consonant symbols to matrix indices
        vowel_mapping: Mapping from vowel symbols to matrix indices
        original_consonant_confusion: Original asymmetric consonant confusion matrix
        original_vowel_confusion: Original asymmetric vowel confusion matrix
    """
    
    def __init__(
        self,
        initial_weights: Optional[np.ndarray] = None,
        initial_threshold: float = 2.0,
        initial_steepness: float = 5.0
    ):
        """Initialize the optimizer.
        
        Args:
            initial_weights: Initial feature weights. If None, uses default FEATURE_WEIGHTS
            initial_threshold: Initial sigmoid threshold parameter
            initial_steepness: Initial sigmoid steepness parameter
        """
        self.weights = initial_weights if initial_weights is not None else FEATURE_WEIGHTS
        self.threshold = initial_threshold
        self.steepness = initial_steepness
        
        # Storage for matrices
        self.consonant_confusion = None
        self.vowel_confusion = None
        self.consonant_mapping = None
        self.vowel_mapping = None
        
        # Store original asymmetric matrices for analysis
        self.original_consonant_confusion = None
        self.original_vowel_confusion = None
    
    def load_confusion_matrices(
        self,
        consonant_matrix: np.ndarray,
        vowel_matrix: np.ndarray,
        consonant_mapping: Dict[str, int],
        vowel_mapping: Dict[str, int]
    ) -> None:
        """Load and symmetrize confusion matrices.
        
        Args:
            consonant_matrix: Raw consonant confusion matrix
            vowel_matrix: Raw vowel confusion matrix
            consonant_mapping: Mapping from consonant symbols to matrix indices
            vowel_mapping: Mapping from vowel symbols to matrix indices
        """
        # Store original matrices
        self.original_consonant_confusion = consonant_matrix.copy()
        self.original_vowel_confusion = vowel_matrix.copy()
        
        # Symmetrize matrices
        self.consonant_confusion = self._symmetrize_matrix(consonant_matrix)
        self.vowel_confusion = self._symmetrize_matrix(vowel_matrix)
        
        # Store mappings
        self.consonant_mapping = consonant_mapping
        self.vowel_mapping = vowel_mapping
    
    def _symmetrize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Symmetrize confusion matrix by averaging with its transpose.
        
        Args:
            matrix: Input confusion matrix to symmetrize
            
        Returns:
            Symmetrized and normalized confusion matrix
        """
        # Average matrix with its transpose
        symmetric = (matrix + matrix.T) / 2
        
        # Add small constant to avoid division by zero
        epsilon = 1e-10
        symmetric = symmetric + epsilon
        
        # Normalize rows to sum to 1
        row_sums = symmetric.sum(axis=1, keepdims=True)
        return symmetric / row_sums
    
    def _compute_feature_distance(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> float:
        """Compute weighted Euclidean distance between feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Weighted Euclidean distance between the vectors
        """
        return np.sqrt(np.sum(self.weights * (features1 - features2) ** 2))
    
    def _apply_threshold(self, distance: float) -> float:
        """Apply sigmoid threshold to distance.
        
        Args:
            distance: Input distance value
            
        Returns:
            Sigmoid-transformed similarity score
        """
        return 1.0 / (1.0 + np.exp(self.steepness * (distance - self.threshold)))
    
    def compute_predicted_confusion(
        self,
        features: Dict[str, List[int]],
        phonemes: List[str]
    ) -> np.ndarray:
        """Compute predicted confusion matrix with threshold.
        
        Args:
            features: Dictionary mapping phonemes to feature vectors
            phonemes: List of phonemes to include in matrix
            
        Returns:
            Predicted confusion matrix
        """
        n = len(phonemes)
        predicted = np.zeros((n, n))
        
        for i, p1 in enumerate(phonemes):
            for j, p2 in enumerate(phonemes):
                if p1 in features and p2 in features:
                    # Get feature vectors
                    feat1 = np.array(features[p1])
                    feat2 = np.array(features[p2])
                    
                    # Compute symmetric distance and apply threshold
                    distance = self._compute_feature_distance(feat1, feat2)
                    similarity = self._apply_threshold(distance)
                    
                    # Matrix is inherently symmetric
                    predicted[i, j] = similarity
                    predicted[j, i] = similarity
        
        # Normalize rows
        row_sums = predicted.sum(axis=1, keepdims=True)
        predicted = predicted / row_sums
        
        return predicted
    
    def compute_loss(self, features: Dict[str, List[int]]) -> float:
        """Compute total loss across consonant and vowel confusions.
        
        Args:
            features: Dictionary mapping phonemes to feature vectors
            
        Returns:
            Combined KL divergence loss for consonants and vowels
        """
        pred_consonants = self.compute_predicted_confusion(
            features, list(self.consonant_mapping.keys()))
        pred_vowels = self.compute_predicted_confusion(
            features, list(self.vowel_mapping.keys()))
        
        consonant_loss = self._kl_divergence(self.consonant_confusion, pred_consonants)
        vowel_loss = self._kl_divergence(self.vowel_confusion, pred_vowels)
        
        return consonant_loss + vowel_loss
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence between two distributions.
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            KL divergence from q to p
        """
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        return np.sum(p * np.log(p / q))
    
    def optimize(
        self,
        features: Dict[str, List[int]],
        max_iter: int = 1000,
        learning_rate: float = 0.01,
        momentum: float = 0.9
    ) -> OptimizationResult:
        """Optimize weights and threshold using gradient descent with momentum.
        
        Args:
            features: Dictionary mapping phonemes to feature vectors
            max_iter: Maximum number of optimization iterations
            learning_rate: Learning rate for gradient descent
            momentum: Momentum coefficient for gradient descent
            
        Returns:
            OptimizationResult containing best parameters and optimization history
        """
        weight_velocity = np.zeros_like(self.weights)
        threshold_velocity = 0.0
        steepness_velocity = 0.0
        
        best_weights = self.weights.copy()
        best_threshold = self.threshold
        best_steepness = self.steepness
        best_loss = float('inf')
        loss_history = []
        
        for _ in range(max_iter):
            current_loss = self.compute_loss(features)
            loss_history.append(current_loss)
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_weights = self.weights.copy()
                best_threshold = self.threshold
                best_steepness = self.steepness
            
            # Compute gradients numerically
            epsilon = 1e-6
            
            # Gradients for weights
            weight_gradients = np.zeros_like(self.weights)
            for i in range(len(self.weights)):
                self.weights[i] += epsilon
                loss_plus = self.compute_loss(features)
                self.weights[i] -= 2*epsilon
                loss_minus = self.compute_loss(features)
                weight_gradients[i] = (loss_plus - loss_minus) / (2*epsilon)
                self.weights[i] += epsilon
            
            # Gradient for threshold
            self.threshold += epsilon
            loss_plus = self.compute_loss(features)
            self.threshold -= 2*epsilon
            loss_minus = self.compute_loss(features)
            threshold_gradient = (loss_plus - loss_minus) / (2*epsilon)
            self.threshold += epsilon
            
            # Gradient for steepness
            self.steepness += epsilon
            loss_plus = self.compute_loss(features)
            self.steepness -= 2*epsilon
            loss_minus = self.compute_loss(features)
            steepness_gradient = (loss_plus - loss_minus) / (2*epsilon)
            self.steepness += epsilon
            
            # Update with momentum
            weight_velocity = momentum * weight_velocity - learning_rate * weight_gradients
            threshold_velocity = momentum * threshold_velocity - learning_rate * threshold_gradient
            steepness_velocity = momentum * steepness_velocity - learning_rate * steepness_gradient
            
            self.weights += weight_velocity
            self.threshold += threshold_velocity
            self.steepness += steepness_velocity
            
            # Ensure constraints
            self.weights = np.maximum(0.1, self.weights)
            self.threshold = max(0.1, self.threshold)
            self.steepness = max(0.1, self.steepness)
        
        return OptimizationResult(
            weights=best_weights,
            threshold=best_threshold,
            steepness=best_steepness,
            loss_history=loss_history,
            final_loss=best_loss
        )
    
    def analyze_asymmetry(self, features: Dict[str, List[int]], is_consonants: bool = True) -> np.ndarray:
        """Analyze asymmetry in confusion patterns not captured by the model.
        
        Args:
            features: Dictionary mapping phonemes to feature vectors
            is_consonants: If True, analyze consonants, otherwise analyze vowels
            
        Returns:
            Matrix of asymmetry values (original - original.T)
        """
        if is_consonants:
            original = self.original_consonant_confusion
            phonemes = list(self.consonant_mapping.keys())
        else:
            original = self.original_vowel_confusion
            phonemes = list(self.vowel_mapping.keys())
            
        predicted = self.compute_predicted_confusion(features, phonemes)
        
        # Compute asymmetry in original data
        original_asymmetry = original - original.T
        
        # Our predicted matrix is symmetric, so any asymmetry in the original
        # data represents effects we can't capture with features alone
        return original_asymmetry
    
    def plot_optimization_progress(self, result: OptimizationResult) -> None:
        """Plot loss history from optimization.
        
        Args:
            result: OptimizationResult containing loss history to plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(result.loss_history)
        plt.title('Optimization Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.show()
    
    def plot_confusion_comparison(
        self,
        features: Dict[str, List[int]],
        is_consonants: bool = True
    ) -> None:
        """Plot actual vs predicted confusion matrices.
        
        Args:
            features: Dictionary mapping phonemes to feature vectors
            is_consonants: If True, plot consonants, otherwise plot vowels
        """
        if is_consonants:
            actual = self.consonant_confusion
            phonemes = list(self.consonant_mapping.keys())
        else:
            actual = self.vowel_confusion
            phonemes = list(self.vowel_mapping.keys())
            
        predicted = self.compute_predicted_confusion(features, phonemes)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        im1 = ax1.imshow(actual, cmap='viridis')
        ax1.set_title('Actual (Symmetrized) Confusion Matrix')
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(predicted, cmap='viridis')
        ax2.set_title('Predicted Confusion Matrix')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.show()
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch

class ExpertOracle:
    def __init__(self, expert_obs, expert_actions, normalize_features=True, distance_threshold=None):
        """
        Initialize expert oracle with preprocessing and efficient nearest neighbor search
        
        Args:
            expert_obs: Expert observations (N, obs_dim)
            expert_actions: Expert actions (N, action_dim)
            normalize_features: Whether to normalize observation features
            distance_threshold: Maximum distance threshold for valid neighbors
        """
        self.expert_obs = expert_obs
        self.expert_actions = expert_actions
        self.distance_threshold = distance_threshold
        self.scaler = None
        
        # Normalize features if requested
        if normalize_features:
            self.scaler = StandardScaler()
            self.expert_obs_normalized = self.scaler.fit_transform(expert_obs)
        else:
            self.expert_obs_normalized = expert_obs
            
        # Build efficient kNN index
        self.nn_model = NearestNeighbors(
            n_neighbors=min(20, len(expert_obs)),  # Limit max neighbors
            algorithm='ball_tree',  # Often faster for high-dimensional data
            metric='euclidean'
        )
        self.nn_model.fit(self.expert_obs_normalized)
    
    def get_expert_action(self, obs, k=5, temporal_consistency=True, fallback_strategy='weighted_avg'):
        """
        Get expert action using improved k-nearest neighbors
        
        Args:
            obs: Current observation
            k: Number of nearest neighbors
            temporal_consistency: Whether to consider temporal smoothness
            fallback_strategy: Strategy when no good neighbors found
        """
        flatten_obs = self.flatten_observation(obs)
        
        # Normalize query observation
        if self.scaler is not None:
            flatten_obs_norm = self.scaler.transform(flatten_obs.reshape(1, -1))[0]
        else:
            flatten_obs_norm = flatten_obs
        
        # Find k nearest neighbors with distances
        distances, indices = self.nn_model.kneighbors(
            flatten_obs_norm.reshape(1, -1), 
            n_neighbors=min(k, len(self.expert_obs))
        )
        distances = distances[0]
        indices = indices[0]
        
        # Filter by distance threshold if specified
        if self.distance_threshold is not None:
            valid_mask = distances <= self.distance_threshold
            if not np.any(valid_mask):
                return self._fallback_action(flatten_obs, fallback_strategy)
            distances = distances[valid_mask]
            indices = indices[valid_mask]
        
        # Improved weighting scheme
        weights = self._compute_weights(distances, method='exponential')
        
        # Get candidate actions
        candidate_actions = self.expert_actions[indices]
        if isinstance(candidate_actions, torch.Tensor):
            candidate_actions = candidate_actions.cpu().numpy()
        
        # Apply temporal consistency if requested
        if temporal_consistency and len(candidate_actions) > 1:
            weights = self._apply_temporal_consistency(candidate_actions, weights)
        
        # Compute weighted action
        weighted_action = np.sum(candidate_actions * weights[:, None], axis=0)
        
        # Optional: Add confidence score
        confidence = self._compute_confidence(distances, weights)
        
        return weighted_action, confidence
    
    def _compute_weights(self, distances, method='exponential'):
        """Compute weights from distances using various methods"""
        if method == 'inverse':
            weights = 1 / (distances + 1e-8)
        elif method == 'exponential':
            # Exponential decay - more aggressive downweighting of far neighbors
            weights = np.exp(-distances / np.std(distances + 1e-8))
        elif method == 'gaussian':
            # Gaussian kernel
            sigma = np.std(distances) + 1e-8
            weights = np.exp(-(distances**2) / (2 * sigma**2))
        else:
            # Uniform weights
            weights = np.ones_like(distances)
        
        # Normalize weights
        return weights / (np.sum(weights) + 1e-8)
    
    def _apply_temporal_consistency(self, actions, weights):
        """Apply temporal consistency by penalizing large action changes"""
        if len(actions) <= 1:
            return weights
        
        # Compute action variations
        action_diffs = np.std(actions, axis=0)
        consistency_penalty = np.mean(action_diffs)
        
        # Adjust weights based on consistency
        # Actions that are more consistent with others get higher weights
        action_similarities = []
        for i, action in enumerate(actions):
            similarity = np.mean([np.exp(-np.linalg.norm(action - other_action)) 
                                for j, other_action in enumerate(actions) if i != j])
            action_similarities.append(similarity)
        
        action_similarities = np.array(action_similarities)
        action_similarities = action_similarities / (np.sum(action_similarities) + 1e-8)
        
        # Combine distance weights with temporal consistency
        combined_weights = 0.7 * weights + 0.3 * action_similarities
        return combined_weights / (np.sum(combined_weights) + 1e-8)
    
    def _compute_confidence(self, distances, weights):
        """Compute confidence score for the expert action"""
        # Lower average distance and higher weight concentration = higher confidence
        avg_distance = np.mean(distances)
        weight_entropy = -np.sum(weights * np.log(weights + 1e-8))
        max_weight_entropy = np.log(len(weights))
        
        # Normalize confidence to [0, 1]
        distance_conf = np.exp(-avg_distance)
        entropy_conf = 1 - (weight_entropy / max_weight_entropy) if max_weight_entropy > 0 else 1
        
        return 0.6 * distance_conf + 0.4 * entropy_conf
    
    def _fallback_action(self, obs, strategy='weighted_avg'):
        """Fallback strategy when no good neighbors are found"""
        if strategy == 'weighted_avg':
            # Use all expert actions with distance-based weighting
            distances = np.linalg.norm(self.expert_obs_normalized - obs, axis=1)
            weights = self._compute_weights(distances, method='exponential')
            return np.sum(self.expert_actions * weights[:, None], axis=0), 0.1
        elif strategy == 'nearest_single':
            # Just use the single nearest neighbor
            idx = np.argmin(np.linalg.norm(self.expert_obs_normalized - obs, axis=1))
            return self.expert_actions[idx], 0.2
        else:
            # Random expert action
            idx = np.random.randint(len(self.expert_actions))
            return self.expert_actions[idx], 0.05
    
    @staticmethod
    def flatten_observation(obs):
        """Flatten observation - customize this based on your observation structure"""
        if isinstance(obs, dict):
            # Handle dictionary observations
            flattened = []
            for key in sorted(obs.keys()):  # Sort for consistency
                val = obs[key]
                if key == 'robot0_joint_acc':
                    continue
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                flattened.append(val.flatten())
            return np.concatenate(flattened)
        elif isinstance(obs, torch.Tensor):
            return obs.cpu().numpy().flatten()
        else:
            return np.array(obs).flatten()


# Usage example:
def setup_expert_oracle(expert_observations, expert_actions):
    """Setup the expert oracle"""
    return ExpertOracle(
        expert_observations, 
        expert_actions,
        normalize_features=True,
        distance_threshold=None  # Set to filter out very distant neighbors
    )

# Modified function that's compatible with your current interface
def get_expert_action(obs, expert_obs, expert_actions, k=5):
    """Drop-in replacement for your current function"""
    # Create temporary oracle (not efficient for repeated calls)
    oracle = ExpertOracle(expert_obs, expert_actions, normalize_features=True)
    action, confidence = oracle.get_expert_action(obs, k=k)
    return action
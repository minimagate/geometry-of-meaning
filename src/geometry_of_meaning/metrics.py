"""
Mathematical metrics for the Geometry of Meaning project.

Centralized implementations of semantic similarity and distance metrics
used across experiments. All metric definitions live here so that different
experiments do not calculate the same concept in inconsistent ways.
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector (shape (d,)).
        b: Second vector (shape (d,)).

    Returns:
        Cosine similarity in range [-1, 1]. Returns 0.0 if either vector is zero.

    Raises:
        ValueError: If vectors have different shapes.
    """
    if a.shape != b.shape:
        raise ValueError(f"Vector shape mismatch: {a.shape} vs {b.shape}")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities for a matrix of embeddings.

    Args:
        embeddings: Array of shape (n, d) where each row is an embedding.

    Returns:
        Symmetric (n, n) matrix of cosine similarities.
    """
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = embeddings / norms
    return normalized @ normalized.T


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean (L2) distance between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Euclidean distance.
    """
    return float(np.linalg.norm(a - b))


def step_displacements(embeddings: list[np.ndarray]) -> list[float]:
    """
    Compute step-to-step Euclidean displacement along a sequence.

    The first element has displacement 0.0 (no previous step).

    Args:
        embeddings: Ordered list of embeddings representing a trajectory.

    Returns:
        List of displacements: displacement[i] = ||embedding[i] - embedding[i-1]||,
        with displacement[0] = 0.0.
    """
    if len(embeddings) < 2:
        return [0.0] * len(embeddings)

    displacements = [0.0]
    for i in range(1, len(embeddings)):
        d = euclidean_distance(embeddings[i], embeddings[i - 1])
        displacements.append(d)

    return displacements


def cumulative_trajectory_length(embeddings: list[np.ndarray]) -> list[float]:
    """
    Compute the cumulative trajectory length along a sequence of embeddings.

    cumulative[i] = sum of step displacements from position 0 to i.

    Args:
        embeddings: Ordered list of embeddings representing a trajectory.

    Returns:
        List where cumulative[i] is the total distance traveled from index 0 to i.
    """
    disp = step_displacements(embeddings)
    return list(np.cumsum(disp))


def direction_alignment(a: np.ndarray, b: np.ndarray, reference: np.ndarray) -> float:
    """
    Measure whether two displacement vectors point in the same direction
    relative to a reference.

    Computes: cos( (a - reference), (b - reference) )

    Useful for comparing whether two languages diverge from the canonical
    text in the same direction.

    Args:
        a: First embedded text.
        b: Second embedded text.
        reference: Reference embedding (e.g., the canonical original).

    Returns:
        Cosine similarity of the two displacement vectors from reference.
    """
    d_a = a - reference
    d_b = b - reference
    return cosine_similarity(d_a, d_b)


def cluster_density(
    embeddings: np.ndarray,
    k: int = 5,
) -> float:
    """
    Estimate local cluster density using average distance to k nearest neighbors.

    Args:
        embeddings: Array of shape (n, d).
        k: Number of nearest neighbors to consider.

    Returns:
        Mean distance to k nearest neighbors (excluding self), averaged over all points.
        Lower values indicate denser clustering.
    """
    sim_matrix = cosine_similarity_matrix(embeddings)
    # Convert similarities to distances for nearest-neighbor computation
    # Cosine distance = 1 - cosine_similarity
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, np.inf)

    # For each point, get distance to k nearest neighbors
    n = len(embeddings)
    k_effective = min(k, n - 1)
    nearest_dists = np.sort(dist_matrix, axis=1)[:, :k_effective]

    return float(np.mean(nearest_dists))


def neighborhood_overlap(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    k: int = 5,
) -> float:
    """
    Measure overlap between neighborhoods of two sets of embeddings.

    For each point in set A, compute the fraction of its k nearest neighbors
    in the combined set (A ∪ B) that belong to set A. Average this over all
    points in A, then symmetrically for B, and average the two.

    Args:
        embeddings_a: First set of embeddings (n_a, d).
        embeddings_b: Second set of embeddings (n_b, d).
        k: Number of neighbors to consider.

    Returns:
        Overlap score in [0, 1]. 1.0 = identical neighborhoods, 0.0 = disjoint.
    """
    combined = np.vstack([embeddings_a, embeddings_b])
    n_a = len(embeddings_a)
    n_b = len(embeddings_b)
    n_total = n_a + n_b
    k_effective = min(k, n_total - 1)

    sim_matrix = cosine_similarity_matrix(combined)
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, np.inf)

    # Indices of k nearest neighbors for each point
    neighbor_indices = np.argsort(dist_matrix, axis=1)[:, :k_effective]

    # For points in A, what fraction of neighbors are also in A?
    overlap_a = 0.0
    for i in range(n_a):
        n_in_a = np.sum(neighbor_indices[i] < n_a)
        overlap_a += n_in_a / k_effective
    overlap_a /= n_a if n_a > 0 else 1.0

    # For points in B, what fraction of neighbors are in B?
    overlap_b = 0.0
    for i in range(n_a, n_total):
        n_in_b = np.sum(neighbor_indices[i] >= n_a)
        overlap_b += n_in_b / k_effective
    overlap_b /= n_b if n_b > 0 else 1.0

    return float((overlap_a + overlap_b) / 2.0)


def phase_transition_score(
    embeddings: list[np.ndarray],
) -> float:
    """
    Estimate the sharpness of a trajectory's phase transition.

    Computes the ratio of the maximum step displacement to the mean step
    displacement. A high ratio suggests a sudden large jump (phase transition).

    Args:
        embeddings: Ordered list of embeddings representing a trajectory.

    Returns:
        Ratio of max step displacement to mean step displacement.
        Higher values indicate a sharper phase transition.
    """
    disp = np.array(step_displacements(embeddings))
    nonzero = disp[disp > 0]

    if len(nonzero) == 0:
        return 0.0

    return float(np.max(nonzero) / np.mean(nonzero))


def centroid_distance(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """
    Compute the cosine distance between centroids of two groups of embeddings.

    Args:
        group_a: First group of embeddings (n_a, d).
        group_b: Second group of embeddings (n_b, d).

    Returns:
        Cosine distance (1 - cosine_similarity) between the centroids.
    """
    centroid_a = np.mean(group_a, axis=0)
    centroid_b = np.mean(group_b, axis=0)
    return 1.0 - cosine_similarity(centroid_a, centroid_b)

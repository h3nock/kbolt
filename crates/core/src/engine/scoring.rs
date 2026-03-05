pub(super) fn dense_distance_to_score(distance: f32) -> f32 {
    1.0 / (1.0 + distance.max(0.0))
}

pub(super) fn max_option(left: Option<f32>, right: Option<f32>) -> Option<f32> {
    match (left, right) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

pub(super) fn normalize_scores(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if (max - min).abs() <= f32::EPSILON {
        return vec![0.5; scores.len()];
    }

    scores
        .iter()
        .map(|score| ((score - min) / (max - min)).clamp(0.0, 1.0))
        .collect::<Vec<_>>()
}

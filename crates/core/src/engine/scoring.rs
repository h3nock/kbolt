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

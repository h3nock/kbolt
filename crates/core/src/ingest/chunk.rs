use crate::config::{ChunkPolicy, ChunkingConfig};

/// Resolves the effective chunk policy for a file profile.
/// Precedence: CLI override > profile > defaults.
pub fn resolve_policy(
    config: &ChunkingConfig,
    profile: Option<&str>,
    cli_override: Option<&ChunkPolicy>,
) -> ChunkPolicy {
    if let Some(override_policy) = cli_override {
        return override_policy.clone();
    }

    if let Some(profile_name) = profile {
        let key = normalize_profile_key(profile_name);
        if let Some(policy) = config.profiles.get(&key) {
            return policy.clone();
        }
    }

    config.defaults.clone()
}

fn normalize_profile_key(raw: &str) -> String {
    raw.trim().trim_start_matches('.').to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::config::{ChunkPolicy, ChunkingConfig};
    use crate::ingest::chunk::resolve_policy;

    fn baseline_config() -> ChunkingConfig {
        ChunkingConfig {
            defaults: ChunkPolicy {
                target_tokens: 450,
                soft_max_tokens: 550,
                hard_max_tokens: 750,
                boundary_overlap_tokens: 48,
                neighbor_window: 1,
                contextual_prefix: true,
            },
            profiles: HashMap::from([(
                "md".to_string(),
                ChunkPolicy {
                    target_tokens: 300,
                    soft_max_tokens: 360,
                    hard_max_tokens: 480,
                    boundary_overlap_tokens: 24,
                    neighbor_window: 2,
                    contextual_prefix: false,
                },
            )]),
        }
    }

    #[test]
    fn resolve_policy_prefers_cli_override() {
        let config = baseline_config();
        let override_policy = ChunkPolicy {
            target_tokens: 128,
            soft_max_tokens: 160,
            hard_max_tokens: 196,
            boundary_overlap_tokens: 16,
            neighbor_window: 3,
            contextual_prefix: false,
        };

        let resolved = resolve_policy(&config, Some("md"), Some(&override_policy));
        assert_eq!(resolved, override_policy);
    }

    #[test]
    fn resolve_policy_uses_normalized_profile_key() {
        let config = baseline_config();

        let resolved = resolve_policy(&config, Some(".MD"), None);
        assert_eq!(resolved.target_tokens, 300);
        assert_eq!(resolved.soft_max_tokens, 360);
        assert_eq!(resolved.hard_max_tokens, 480);
        assert_eq!(resolved.boundary_overlap_tokens, 24);
        assert_eq!(resolved.neighbor_window, 2);
        assert!(!resolved.contextual_prefix);
    }

    #[test]
    fn resolve_policy_falls_back_to_defaults() {
        let config = baseline_config();

        let resolved = resolve_policy(&config, Some("txt"), None);
        assert_eq!(resolved, config.defaults);
    }
}

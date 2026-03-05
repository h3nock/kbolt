use std::path::Path;

use crate::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ModelFileRequirement {
    ExactPath {
        path: String,
        config_field: &'static str,
    },
    SingleExtension {
        extension: &'static str,
        config_field: &'static str,
    },
    SingleTokenizerJson {
        config_field: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelDownloadRequest {
    pub model_id: String,
    pub requirements: Vec<ModelFileRequirement>,
}

pub(crate) trait ModelArtifactProvider {
    fn download_model(&self, request: &ModelDownloadRequest, target_dir: &Path) -> Result<u64>;
}

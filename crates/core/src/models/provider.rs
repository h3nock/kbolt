use std::path::Path;

use crate::Result;

pub(crate) trait ModelArtifactProvider {
    fn download_model(&self, model_id: &str, target_dir: &Path) -> Result<u64>;
}

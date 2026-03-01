pub mod document;
pub mod error;
pub mod indexing;
pub mod model;
pub mod search;
pub mod space;
pub mod status;

pub use document::{
    DocumentResponse, FileEntry, GetRequest, Locator, MultiGetRequest, MultiGetResponse,
    OmitReason, OmittedFile,
};
pub use error::{KboltError, Result};
pub use indexing::{FileError, UpdateOptions, UpdateReport};
pub use model::PullReport;
pub use search::{SearchMode, SearchRequest, SearchResponse, SearchResult, SearchSignals};
pub use space::{ActiveSpace, ActiveSpaceSource, AddCollectionRequest, CollectionInfo, SpaceInfo};
pub use status::{
    CollectionStatus, DiskUsage, ModelInfo, ModelStatus, SpaceStatus, StatusResponse,
};

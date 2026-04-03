pub mod doctor;
pub mod document;
pub mod error;
pub mod eval;
pub mod indexing;
pub mod schedule;
pub mod search;
pub mod space;
pub mod status;

pub use doctor::{DoctorCheck, DoctorCheckStatus, DoctorReport, DoctorSetupStatus};
pub use document::{
    DocumentResponse, FileEntry, GetRequest, Locator, MultiGetRequest, MultiGetResponse,
    OmitReason, OmittedFile,
};
pub use error::{KboltError, Result};
pub use eval::{
    EvalCase, EvalDataset, EvalImportReport, EvalJudgment, EvalModeFailure, EvalModeReport,
    EvalQueryReport, EvalRunReport,
};
pub use indexing::{FileError, UpdateDecision, UpdateDecisionKind, UpdateOptions, UpdateReport};
pub use schedule::{
    AddScheduleRequest, RemoveScheduleRequest, RemoveScheduleSelector, ScheduleAddResponse,
    ScheduleBackend, ScheduleDefinition, ScheduleInterval, ScheduleIntervalUnit, ScheduleOrphan,
    ScheduleRemoveResponse, ScheduleRunResult, ScheduleRunState, ScheduleScope, ScheduleState,
    ScheduleStatusEntry, ScheduleStatusResponse, ScheduleTrigger, ScheduleWeekday,
};
pub use search::{
    SearchMode, SearchPipeline, SearchPipelineNotice, SearchPipelineStep,
    SearchPipelineUnavailableReason, SearchRequest, SearchResponse, SearchResult, SearchSignals,
};
pub use space::{
    ActiveSpace, ActiveSpaceSource, AddCollectionRequest, AddCollectionResult, CollectionInfo,
    InitialIndexingBlock, InitialIndexingOutcome, SpaceInfo,
};
pub use status::{
    CollectionStatus, DiskUsage, ModelInfo, ModelStatus, SpaceStatus, StatusResponse,
};

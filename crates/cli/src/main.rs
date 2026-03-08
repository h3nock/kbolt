use std::ffi::OsString;

use clap::Parser;
use kbolt_cli::args::{
    Cli, CollectionCommand, Command, EvalCommand, IgnoreCommand, ModelsCommand, OutputFormat,
    ScheduleAddArgs, ScheduleCommand, ScheduleDayArg, ScheduleRemoveArgs, SearchArgs, SpaceCommand,
};
use kbolt_cli::{resolve_no_rerank_for_mode, CliAdapter, CliSearchOptions};
use kbolt_core::config;
use kbolt_core::engine::Engine;
use kbolt_core::error::CoreError;
use kbolt_mcp::stdio;
use kbolt_mcp::McpAdapter;
use kbolt_types::{
    ActiveSpace, AddScheduleRequest, CollectionInfo, FileEntry, GetRequest, KboltError, Locator,
    ModelStatus, MultiGetRequest, RemoveScheduleRequest, RemoveScheduleSelector, ScheduleInterval,
    ScheduleIntervalUnit, ScheduleScope, ScheduleTrigger, ScheduleWeekday, SearchMode,
    SearchRequest, SpaceInfo, UpdateOptions,
};
use serde::Serialize;
use serde_json::json;

fn main() {
    let argv = std::env::args_os().collect::<Vec<_>>();
    let output_format = requested_output_format_from_args(&argv);

    if let Err(err) = run(argv) {
        emit_error(output_format, &err);
        std::process::exit(err.exit_code());
    }
}

fn run(argv: Vec<OsString>) -> std::result::Result<(), RunError> {
    if let Some(schedule_id) = parse_internal_schedule_run(&argv)? {
        let engine = Engine::new(None)?;
        engine.run_schedule(&schedule_id)?;
        return Ok(());
    }

    let cli = Cli::try_parse_from(argv)?;
    ensure_supported_output_format(cli.format, &cli.command)?;
    maybe_print_first_run_models_hint(&cli.command, cli.format);

    if matches!(cli.command, Command::Mcp) {
        let engine = Engine::new(None)?;
        let adapter = McpAdapter::new(engine);
        stdio::run_stdio(&adapter)?;
        return Ok(());
    }

    let output_format = cli.format;
    let wants_json = output_format == OutputFormat::Json;
    let interactive_output = supports_interactive_output(output_format, stdin_stdout_are_tty());
    let engine = Engine::new(None)?;
    let mut adapter = CliAdapter::new(engine);
    let print_text = |line: &str| emit_text_output(output_format, line);
    let print_message = |line: &str| emit_message_output(output_format, line);

    match cli.command {
        Command::Space(space) => match space.command {
            SpaceCommand::Add {
                name,
                description,
                strict,
                dirs,
            } => {
                let line = adapter.space_add(&name, description.as_deref(), strict, &dirs)?;
                print_message(&line);
            }
            SpaceCommand::Describe { name, text } => {
                let line = adapter.space_describe(&name, &text)?;
                print_message(&line);
            }
            SpaceCommand::Rename { old, new } => {
                let line = adapter.space_rename(&old, &new)?;
                print_message(&line);
            }
            SpaceCommand::Remove { name } => {
                let line = adapter.space_remove(&name)?;
                print_message(&line);
            }
            SpaceCommand::Current => {
                if wants_json {
                    let active_space = adapter.engine.current_space(cli.space.as_deref())?;
                    emit_structured_output(&ActiveSpaceJsonResponse { active_space })?;
                } else {
                    let line = adapter.space_current(cli.space.as_deref())?;
                    print_text(&line);
                }
            }
            SpaceCommand::Default { name } => {
                if let Some(space_name) = name.as_deref() {
                    let line = adapter.space_default(Some(space_name))?;
                    print_message(&line);
                } else if wants_json {
                    emit_structured_output(&DefaultSpaceJsonResponse {
                        default_space: adapter.engine.config().default_space.clone(),
                    })?;
                } else {
                    let line = adapter.space_default(None)?;
                    print_text(&line);
                }
            }
            SpaceCommand::List => {
                if wants_json {
                    let spaces = adapter.engine.list_spaces()?;
                    emit_structured_output(&SpacesJsonResponse { spaces })?;
                } else {
                    let line = adapter.space_list()?;
                    print_text(&line);
                }
            }
            SpaceCommand::Info { name } => {
                if wants_json {
                    let space = adapter.engine.space_info(&name)?;
                    emit_structured_output(&space)?;
                } else {
                    let line = adapter.space_info(&name)?;
                    print_text(&line);
                }
            }
        },
        Command::Collection(collection) => match collection.command {
            CollectionCommand::Add {
                path,
                name,
                description,
                extensions,
                no_index,
            } => {
                let run_collection_add = || {
                    adapter.collection_add(
                        cli.space.as_deref(),
                        &path,
                        name.as_deref(),
                        description.as_deref(),
                        extensions.as_deref(),
                        no_index,
                    )
                };
                match run_collection_add() {
                    Ok(line) => {
                        print_message(&line);
                    }
                    Err(err)
                        if should_offer_model_pull_for_collection_add(
                            no_index,
                            interactive_output,
                            &err,
                        ) =>
                    {
                        if prompt_pull_models()? {
                            let pull_report = adapter.models_pull()?;
                            print_message(&pull_report);
                            let retried = run_collection_add()?;
                            print_message(&retried);
                        } else {
                            return Err(with_collection_add_model_missing_guidance(err).into());
                        }
                    }
                    Err(err) => return Err(with_collection_add_model_missing_guidance(err).into()),
                }
            }
            CollectionCommand::List => {
                if wants_json {
                    let collections = adapter.engine.list_collections(cli.space.as_deref())?;
                    emit_structured_output(&CollectionsJsonResponse { collections })?;
                } else {
                    let line = adapter.collection_list(cli.space.as_deref())?;
                    print_text(&line);
                }
            }
            CollectionCommand::Info { name } => {
                if wants_json {
                    let collection = adapter
                        .engine
                        .collection_info(cli.space.as_deref(), &name)?;
                    emit_structured_output(&collection)?;
                } else {
                    let line = adapter.collection_info(cli.space.as_deref(), &name)?;
                    print_text(&line);
                }
            }
            CollectionCommand::Describe { name, text } => {
                let line = adapter.collection_describe(cli.space.as_deref(), &name, &text)?;
                print_message(&line);
            }
            CollectionCommand::Rename { old, new } => {
                let line = adapter.collection_rename(cli.space.as_deref(), &old, &new)?;
                print_message(&line);
            }
            CollectionCommand::Remove { name } => {
                let line = adapter.collection_remove(cli.space.as_deref(), &name)?;
                print_message(&line);
            }
        },
        Command::Ignore(ignore) => match ignore.command {
            IgnoreCommand::Show { collection } => {
                if wants_json {
                    let (space, content) = adapter
                        .engine
                        .read_collection_ignore(cli.space.as_deref(), &collection)?;
                    emit_structured_output(&IgnoreShowJsonResponse {
                        space,
                        collection,
                        patterns: ignore_patterns_to_lines(content),
                    })?;
                } else {
                    let line = adapter.ignore_show(cli.space.as_deref(), &collection)?;
                    print_text(&line);
                }
            }
            IgnoreCommand::Add {
                collection,
                pattern,
            } => {
                let line = adapter.ignore_add(cli.space.as_deref(), &collection, &pattern)?;
                print_message(&line);
            }
            IgnoreCommand::Remove {
                collection,
                pattern,
            } => {
                let line = adapter.ignore_remove(cli.space.as_deref(), &collection, &pattern)?;
                print_message(&line);
            }
            IgnoreCommand::List => {
                if wants_json {
                    let entries = adapter
                        .engine
                        .list_collection_ignores(cli.space.as_deref())?;
                    let ignores = entries
                        .into_iter()
                        .map(|entry| IgnoreListEntryJson {
                            space: entry.space,
                            collection: entry.collection,
                            pattern_count: entry.pattern_count,
                        })
                        .collect();
                    emit_structured_output(&IgnoreListJsonResponse { ignores })?;
                } else {
                    let line = adapter.ignore_list(cli.space.as_deref())?;
                    print_text(&line);
                }
            }
            IgnoreCommand::Edit { collection } => {
                let line = adapter.ignore_edit(cli.space.as_deref(), &collection)?;
                print_message(&line);
            }
        },
        Command::Models(models) => match models.command {
            ModelsCommand::List => {
                if wants_json {
                    let models = adapter.engine.model_status()?;
                    emit_structured_output(&ModelsJsonResponse { models })?;
                } else {
                    let line = adapter.models_list()?;
                    print_text(&line);
                }
            }
            ModelsCommand::Pull => {
                let line = adapter.models_pull()?;
                print_message(&line);
            }
        },
        Command::Eval(eval) => {
            ensure_eval_uses_local_scope(cli.space.as_deref())?;
            match eval.command {
                EvalCommand::Run => {
                    if wants_json {
                        let report = adapter.engine.run_eval()?;
                        emit_structured_output(&report)?;
                    } else {
                        let line = adapter.eval_run()?;
                        print_text(&line);
                    }
                }
            }
        }
        Command::Schedule(schedule) => {
            ensure_schedule_uses_local_scope(cli.space.as_deref())?;
            match schedule.command {
                ScheduleCommand::Add(add) => {
                    let request = schedule_add_request(add)?;
                    if wants_json {
                        let response = adapter.engine.add_schedule(request)?;
                        emit_structured_output(&response)?;
                    } else {
                        let line = adapter.schedule_add(request)?;
                        print_message(&line);
                    }
                }
                ScheduleCommand::Status => {
                    if wants_json {
                        let response = adapter.engine.schedule_status()?;
                        emit_structured_output(&response)?;
                    } else {
                        let line = adapter.schedule_status()?;
                        print_text(&line);
                    }
                }
                ScheduleCommand::Remove(remove) => {
                    let request = schedule_remove_request(remove)?;
                    if wants_json {
                        let response = adapter.engine.remove_schedule(request)?;
                        emit_structured_output(&response)?;
                    } else {
                        let line = adapter.schedule_remove(request)?;
                        print_message(&line);
                    }
                }
            }
        }
        Command::Mcp => unreachable!("mcp command handled before adapter setup"),
        Command::Search(search) => {
            let requested_mode = requested_search_mode(&search);
            let mode = if search.deep {
                SearchMode::Deep
            } else if search.keyword {
                SearchMode::Keyword
            } else if search.semantic {
                SearchMode::Semantic
            } else {
                SearchMode::Auto
            };
            let effective_no_rerank =
                resolve_no_rerank_for_mode(mode.clone(), search.rerank, search.no_rerank);

            if wants_json {
                let response = adapter.engine.search(SearchRequest {
                    query: search.query.clone(),
                    mode,
                    space: cli.space.clone(),
                    collections: search.collections.clone(),
                    limit: search.limit,
                    min_score: search.min_score,
                    no_rerank: effective_no_rerank,
                    debug: search.debug,
                })?;
                emit_structured_output(&response)?;
            } else {
                let run_search = |deep: bool, keyword: bool, semantic: bool| {
                    adapter.search(CliSearchOptions {
                        space: cli.space.as_deref(),
                        query: &search.query,
                        collections: &search.collections,
                        limit: search.limit,
                        min_score: search.min_score,
                        deep,
                        keyword,
                        semantic,
                        rerank: search.rerank,
                        no_rerank: search.no_rerank,
                        debug: search.debug,
                    })
                };

                match run_search(search.deep, search.keyword, search.semantic) {
                    Ok(line) => {
                        print_text(&line);
                    }
                    Err(err) if is_model_not_available_error(&err) && interactive_output => {
                        if prompt_pull_models()? {
                            let pull_report = adapter.models_pull()?;
                            print_message(&pull_report);
                            let retried = run_search(search.deep, search.keyword, search.semantic)?;
                            print_text(&retried);
                        } else if requested_mode == RequestedSearchMode::Auto {
                            print_text("models unavailable; falling back to keyword mode");
                            let fallback = run_search(false, true, false)?;
                            print_text(&fallback);
                        } else {
                            return Err(err.into());
                        }
                    }
                    Err(err) => return Err(err.into()),
                }
            }
        }
        Command::Update(update) => {
            let update_options = |no_embed: bool| UpdateOptions {
                space: cli.space.clone(),
                collections: update.collections.clone(),
                no_embed,
                dry_run: update.dry_run,
                verbose: update.verbose,
            };

            if wants_json {
                match adapter.engine.update(update_options(update.no_embed)) {
                    Ok(report) => emit_structured_output(&report)?,
                    Err(err) => return Err(with_update_model_missing_guidance(err).into()),
                }
            } else {
                let run_update = |no_embed: bool| {
                    adapter.update(
                        cli.space.as_deref(),
                        &update.collections,
                        no_embed,
                        update.dry_run,
                        update.verbose,
                    )
                };

                match run_update(update.no_embed) {
                    Ok(line) => {
                        print_message(&line);
                    }
                    Err(err)
                        if should_offer_model_pull_for_update(
                            update.no_embed,
                            interactive_output,
                            &err,
                        ) =>
                    {
                        if prompt_pull_models()? {
                            let pull_report = adapter.models_pull()?;
                            print_message(&pull_report);
                            let retried = run_update(false)?;
                            print_message(&retried);
                        } else {
                            return Err(with_update_model_missing_guidance(err).into());
                        }
                    }
                    Err(err) => return Err(with_update_model_missing_guidance(err).into()),
                }
            }
        }
        Command::Status => {
            if wants_json {
                let status = adapter.engine.status(cli.space.as_deref())?;
                emit_structured_output(&status)?;
            } else {
                let line = adapter.status(cli.space.as_deref())?;
                print_text(&line);
            }
        }
        Command::Ls(ls) => {
            if wants_json {
                let mut files = adapter.engine.list_files(
                    cli.space.as_deref(),
                    &ls.collection,
                    ls.prefix.as_deref(),
                )?;
                if !ls.all {
                    files.retain(|file| file.active);
                }
                emit_structured_output(&FilesJsonResponse { files })?;
            } else {
                let line = adapter.ls(
                    cli.space.as_deref(),
                    &ls.collection,
                    ls.prefix.as_deref(),
                    ls.all,
                )?;
                print_text(&line);
            }
        }
        Command::Get(get) => {
            if wants_json {
                let locator = Locator::parse(&get.identifier);
                let document = adapter.engine.get_document(GetRequest {
                    locator,
                    space: cli.space.clone(),
                    offset: get.offset,
                    limit: get.limit,
                })?;
                emit_structured_output(&document)?;
            } else {
                let line =
                    adapter.get(cli.space.as_deref(), &get.identifier, get.offset, get.limit)?;
                print_text(&line);
            }
        }
        Command::MultiGet(get) => {
            if wants_json {
                let locators = get
                    .locators
                    .iter()
                    .map(|item| Locator::parse(item))
                    .collect();
                let response = adapter.engine.multi_get(MultiGetRequest {
                    locators,
                    space: cli.space.clone(),
                    max_files: get.max_files,
                    max_bytes: get.max_bytes,
                })?;
                emit_structured_output(&response)?;
            } else {
                let line = adapter.multi_get(
                    cli.space.as_deref(),
                    &get.locators,
                    get.max_files,
                    get.max_bytes,
                )?;
                print_text(&line);
            }
        }
    }

    Ok(())
}

#[derive(Debug)]
enum RunError {
    Clap(clap::Error),
    Core(CoreError),
}

impl From<clap::Error> for RunError {
    fn from(value: clap::Error) -> Self {
        Self::Clap(value)
    }
}

impl From<CoreError> for RunError {
    fn from(value: CoreError) -> Self {
        Self::Core(value)
    }
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Clap(err) => write!(f, "{err}"),
            Self::Core(err) => write!(f, "{err}"),
        }
    }
}

impl RunError {
    fn exit_code(&self) -> i32 {
        match self {
            Self::Clap(err) => err.exit_code(),
            Self::Core(_) => 1,
        }
    }
}

#[derive(Debug, Serialize, PartialEq)]
struct SpacesJsonResponse {
    spaces: Vec<SpaceInfo>,
}

#[derive(Debug, Serialize, PartialEq)]
struct ActiveSpaceJsonResponse {
    active_space: Option<ActiveSpace>,
}

#[derive(Debug, Serialize, PartialEq)]
struct DefaultSpaceJsonResponse {
    default_space: Option<String>,
}

#[derive(Debug, Serialize, PartialEq)]
struct CollectionsJsonResponse {
    collections: Vec<CollectionInfo>,
}

#[derive(Debug, Serialize, PartialEq)]
struct FilesJsonResponse {
    files: Vec<FileEntry>,
}

#[derive(Debug, Serialize, PartialEq)]
struct IgnoreShowJsonResponse {
    space: String,
    collection: String,
    patterns: Vec<String>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct IgnoreListEntryJson {
    space: String,
    collection: String,
    pattern_count: usize,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct IgnoreListJsonResponse {
    ignores: Vec<IgnoreListEntryJson>,
}

#[derive(Debug, Serialize, PartialEq)]
struct ModelsJsonResponse {
    models: ModelStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestedSearchMode {
    Auto,
    Deep,
    Keyword,
    Semantic,
}

const INTERNAL_SCHEDULE_RUN_COMMAND: &str = "__schedule-run";

fn requested_search_mode(search: &SearchArgs) -> RequestedSearchMode {
    if search.deep {
        RequestedSearchMode::Deep
    } else if search.keyword {
        RequestedSearchMode::Keyword
    } else if search.semantic {
        RequestedSearchMode::Semantic
    } else {
        RequestedSearchMode::Auto
    }
}

fn ensure_schedule_uses_local_scope(space: Option<&str>) -> std::result::Result<(), RunError> {
    if space.is_none() {
        return Ok(());
    }

    Err(CoreError::Domain(KboltError::InvalidInput(
        "schedule commands do not use the top-level --space flag; use schedule --space instead"
            .to_string(),
    ))
    .into())
}

fn ensure_eval_uses_local_scope(space: Option<&str>) -> std::result::Result<(), RunError> {
    if space.is_none() {
        return Ok(());
    }

    Err(CoreError::Domain(KboltError::InvalidInput(
        "eval commands do not use the top-level --space flag; set scope inside eval.toml"
            .to_string(),
    ))
    .into())
}

fn schedule_add_request(
    args: ScheduleAddArgs,
) -> std::result::Result<AddScheduleRequest, RunError> {
    Ok(AddScheduleRequest {
        trigger: schedule_trigger(&args)?,
        scope: schedule_scope(args.space, args.collections),
    })
}

fn schedule_remove_request(
    args: ScheduleRemoveArgs,
) -> std::result::Result<RemoveScheduleRequest, RunError> {
    let selector = if args.all {
        RemoveScheduleSelector::All
    } else if let Some(id) = args.id {
        RemoveScheduleSelector::Id { id }
    } else {
        RemoveScheduleSelector::Scope {
            scope: schedule_scope(args.space, args.collections),
        }
    };

    Ok(RemoveScheduleRequest { selector })
}

fn schedule_trigger(args: &ScheduleAddArgs) -> std::result::Result<ScheduleTrigger, RunError> {
    if let Some(interval) = args.every.as_deref() {
        return Ok(ScheduleTrigger::Every {
            interval: parse_schedule_interval(interval)?,
        });
    }

    let time = args.at.clone().ok_or_else(|| {
        RunError::from(CoreError::Domain(KboltError::InvalidInput(
            "schedule trigger requires --every or --at".to_string(),
        )))
    })?;
    if args.on.is_empty() {
        return Ok(ScheduleTrigger::Daily { time });
    }

    Ok(ScheduleTrigger::Weekly {
        weekdays: args.on.iter().copied().map(schedule_weekday).collect(),
        time,
    })
}

fn schedule_scope(space: Option<String>, collections: Vec<String>) -> ScheduleScope {
    match space {
        Some(space) if collections.is_empty() => ScheduleScope::Space { space },
        Some(space) => ScheduleScope::Collections { space, collections },
        None => ScheduleScope::All,
    }
}

fn parse_schedule_interval(raw: &str) -> std::result::Result<ScheduleInterval, RunError> {
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.len() < 2 {
        return Err(invalid_schedule_interval(raw));
    }

    let (value, unit) = normalized.split_at(normalized.len() - 1);
    let value = value
        .parse::<u32>()
        .map_err(|_| invalid_schedule_interval(raw))?;
    let unit = match unit {
        "m" => ScheduleIntervalUnit::Minutes,
        "h" => ScheduleIntervalUnit::Hours,
        _ => return Err(invalid_schedule_interval(raw)),
    };

    Ok(ScheduleInterval { value, unit })
}

fn invalid_schedule_interval(raw: &str) -> RunError {
    CoreError::Domain(KboltError::InvalidInput(format!(
        "invalid schedule interval '{raw}': use <minutes>m or <hours>h"
    )))
    .into()
}

fn schedule_weekday(day: ScheduleDayArg) -> ScheduleWeekday {
    match day {
        ScheduleDayArg::Mon => ScheduleWeekday::Mon,
        ScheduleDayArg::Tue => ScheduleWeekday::Tue,
        ScheduleDayArg::Wed => ScheduleWeekday::Wed,
        ScheduleDayArg::Thu => ScheduleWeekday::Thu,
        ScheduleDayArg::Fri => ScheduleWeekday::Fri,
        ScheduleDayArg::Sat => ScheduleWeekday::Sat,
        ScheduleDayArg::Sun => ScheduleWeekday::Sun,
    }
}

fn is_model_not_available_error(err: &CoreError) -> bool {
    matches!(err, CoreError::Domain(KboltError::ModelNotAvailable { .. }))
}

fn parse_internal_schedule_run(args: &[OsString]) -> std::result::Result<Option<String>, RunError> {
    let Some(command) = args.get(1).and_then(|arg| arg.to_str()) else {
        return Ok(None);
    };

    if command != INTERNAL_SCHEDULE_RUN_COMMAND {
        return Ok(None);
    }

    if args.len() != 3 {
        return Err(CoreError::Domain(KboltError::InvalidInput(format!(
            "internal schedule runner usage: kbolt {INTERNAL_SCHEDULE_RUN_COMMAND} <id>"
        )))
        .into());
    }

    let Some(raw_id) = args.get(2).and_then(|arg| arg.to_str()) else {
        return Err(CoreError::Domain(KboltError::InvalidInput(
            "schedule id must be valid utf-8".to_string(),
        ))
        .into());
    };

    let schedule_id = raw_id.trim();
    if schedule_id.is_empty() {
        return Err(CoreError::Domain(KboltError::InvalidInput(
            "schedule id must not be empty".to_string(),
        ))
        .into());
    }

    Ok(Some(schedule_id.to_string()))
}

fn requested_output_format_from_args(args: &[OsString]) -> OutputFormat {
    let mut args_iter = args.iter().skip(1);
    while let Some(arg) = args_iter.next() {
        let Some(raw) = arg.to_str() else {
            continue;
        };

        if let Some(value) = raw.strip_prefix("--format=") {
            if value.eq_ignore_ascii_case("json") {
                return OutputFormat::Json;
            }
        }

        if raw == "--format" || raw == "-f" {
            if args_iter
                .next()
                .and_then(|value| value.to_str())
                .is_some_and(|value| value.eq_ignore_ascii_case("json"))
            {
                return OutputFormat::Json;
            }
        }
    }

    OutputFormat::Cli
}

fn ensure_supported_output_format(
    format: OutputFormat,
    command: &Command,
) -> std::result::Result<(), RunError> {
    if format == OutputFormat::Json && matches!(command, Command::Mcp) {
        return Err(CoreError::Domain(KboltError::InvalidInput(
            "--format json is not supported for the mcp command".to_string(),
        ))
        .into());
    }

    Ok(())
}

fn supports_interactive_output(format: OutputFormat, is_tty: bool) -> bool {
    format == OutputFormat::Cli && is_tty
}

fn emit_text_output(format: OutputFormat, line: &str) {
    println!("{}", render_text_output(format, line));
}

fn emit_message_output(format: OutputFormat, line: &str) {
    println!("{}", render_message_output(format, line));
}

fn emit_structured_output<T: Serialize>(value: &T) -> std::result::Result<(), RunError> {
    println!("{}", render_structured_output(value)?);
    Ok(())
}

fn emit_error(format: OutputFormat, err: &RunError) {
    let rendered = render_error_output(format, err);
    if err.exit_code() == 0 {
        println!("{rendered}");
    } else {
        eprintln!("{rendered}");
    }
}

fn render_text_output(format: OutputFormat, line: &str) -> String {
    match format {
        OutputFormat::Cli | OutputFormat::Json => line.to_string(),
    }
}

fn render_message_output(format: OutputFormat, line: &str) -> String {
    match format {
        OutputFormat::Cli => line.to_string(),
        OutputFormat::Json => json!({
            "ok": true,
            "message": line,
        })
        .to_string(),
    }
}

fn render_structured_output<T: Serialize>(value: &T) -> std::result::Result<String, RunError> {
    serde_json::to_string(value)
        .map_err(CoreError::from)
        .map_err(RunError::from)
}

fn render_error_output(format: OutputFormat, err: &RunError) -> String {
    if let RunError::Clap(clap_err) = err {
        if matches!(
            clap_err.kind(),
            clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
        ) {
            return clap_err.to_string();
        }
    }

    match format {
        OutputFormat::Cli => err.to_string(),
        OutputFormat::Json => json!({
            "ok": false,
            "error": {
                "kind": json_error_kind(err),
                "message": err.to_string(),
            }
        })
        .to_string(),
    }
}

fn ignore_patterns_to_lines(content: Option<String>) -> Vec<String> {
    content
        .into_iter()
        .flat_map(|value| value.lines().map(ToString::to_string).collect::<Vec<_>>())
        .collect()
}

fn json_error_kind(err: &RunError) -> &'static str {
    match err {
        RunError::Clap(_) => "invalid_input",
        RunError::Core(CoreError::Domain(KboltError::SpaceNotFound { .. }))
        | RunError::Core(CoreError::Domain(KboltError::CollectionNotFound { .. }))
        | RunError::Core(CoreError::Domain(KboltError::DocumentNotFound { .. }))
        | RunError::Core(CoreError::Domain(KboltError::FileNotFound(_))) => "not_found",
        RunError::Core(CoreError::Domain(KboltError::SpaceAlreadyExists { .. }))
        | RunError::Core(CoreError::Domain(KboltError::CollectionAlreadyExists { .. })) => {
            "already_exists"
        }
        RunError::Core(CoreError::Domain(KboltError::AmbiguousSpace { .. })) => "ambiguous_space",
        RunError::Core(CoreError::Domain(KboltError::NoActiveSpace))
        | RunError::Core(CoreError::Domain(KboltError::InvalidInput(_)))
        | RunError::Core(CoreError::Domain(KboltError::InvalidPath(_))) => "invalid_input",
        RunError::Core(CoreError::Domain(KboltError::ModelNotAvailable { .. })) => {
            "model_not_available"
        }
        RunError::Core(CoreError::Domain(KboltError::ModelDownload(_))) => "model_download",
        RunError::Core(CoreError::Domain(KboltError::Inference(_))) => "inference",
        RunError::Core(CoreError::Domain(KboltError::Config(_))) => "config",
        RunError::Core(CoreError::Domain(KboltError::Database(_)))
        | RunError::Core(CoreError::Domain(KboltError::Tantivy(_)))
        | RunError::Core(CoreError::Domain(KboltError::USearch(_)))
        | RunError::Core(CoreError::Domain(KboltError::FileDeleted(_)))
        | RunError::Core(CoreError::Domain(KboltError::Internal(_)))
        | RunError::Core(CoreError::Domain(KboltError::Io(_)))
        | RunError::Core(CoreError::Sqlite(_))
        | RunError::Core(CoreError::TomlDe(_))
        | RunError::Core(CoreError::TomlSer(_))
        | RunError::Core(CoreError::Json(_))
        | RunError::Core(CoreError::Tantivy(_))
        | RunError::Core(CoreError::Io(_))
        | RunError::Core(CoreError::Internal(_)) => "internal",
    }
}

fn maybe_print_first_run_models_hint(command: &Command, format: OutputFormat) {
    let config_file = match config::default_config_file_path() {
        Ok(path) => path,
        Err(_) => return,
    };
    if should_show_first_run_models_hint(
        command,
        format,
        stdin_stdout_are_tty(),
        config_file.exists(),
    ) {
        eprintln!(
            "hint: run `kbolt models pull` to download semantic/rerank models. keyword search works without models."
        );
    }
}

fn should_show_first_run_models_hint(
    command: &Command,
    format: OutputFormat,
    is_tty: bool,
    config_exists: bool,
) -> bool {
    !matches!(command, Command::Mcp) && format == OutputFormat::Cli && is_tty && !config_exists
}

fn stdin_stdout_are_tty() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

fn prompt_pull_models() -> std::result::Result<bool, CoreError> {
    use std::io::Write;

    print!("Models not downloaded. Download now and continue? (Y/n) ");
    std::io::stdout().flush()?;

    let mut response = String::new();
    std::io::stdin().read_line(&mut response)?;
    Ok(parse_pull_confirmation(&response))
}

fn parse_pull_confirmation(input: &str) -> bool {
    let normalized = input.trim().to_ascii_lowercase();
    normalized.is_empty() || normalized == "y" || normalized == "yes"
}

fn should_offer_model_pull_for_update(no_embed: bool, is_tty: bool, err: &CoreError) -> bool {
    !no_embed && is_tty && is_model_not_available_error(err)
}

fn should_offer_model_pull_for_collection_add(
    no_index: bool,
    is_tty: bool,
    err: &CoreError,
) -> bool {
    !no_index && is_tty && is_model_not_available_error(err)
}

fn with_update_model_missing_guidance(err: CoreError) -> CoreError {
    if is_model_not_available_error(&err) {
        return CoreError::Domain(KboltError::InvalidInput(format!(
            "{err}. for update, run `kbolt models pull` or re-run with `--no-embed`"
        )));
    }
    err
}

fn with_collection_add_model_missing_guidance(err: CoreError) -> CoreError {
    if is_model_not_available_error(&err) {
        return CoreError::Domain(KboltError::InvalidInput(format!(
            "{err}. for collection add, run `kbolt models pull` or re-run with `--no-index`"
        )));
    }
    err
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use clap::CommandFactory;
    use serde_json::json;

    use super::{
        ensure_eval_uses_local_scope, ensure_schedule_uses_local_scope,
        ensure_supported_output_format, is_model_not_available_error, parse_internal_schedule_run,
        parse_pull_confirmation, parse_schedule_interval, render_error_output,
        render_message_output, render_structured_output, requested_output_format_from_args,
        requested_search_mode, schedule_add_request, schedule_remove_request,
        should_offer_model_pull_for_collection_add, should_offer_model_pull_for_update,
        should_show_first_run_models_hint, supports_interactive_output,
        with_collection_add_model_missing_guidance, with_update_model_missing_guidance,
        DefaultSpaceJsonResponse, IgnoreShowJsonResponse, RequestedSearchMode, RunError,
        INTERNAL_SCHEDULE_RUN_COMMAND,
    };
    use kbolt_cli::args::{
        Cli, Command, OutputFormat, ScheduleAddArgs, ScheduleDayArg, ScheduleRemoveArgs, SearchArgs,
    };
    use kbolt_core::error::CoreError;
    use kbolt_types::{
        FileError, KboltError, RemoveScheduleSelector, ScheduleInterval, ScheduleIntervalUnit,
        ScheduleScope, ScheduleTrigger, ScheduleWeekday, UpdateDecision, UpdateDecisionKind,
        UpdateReport,
    };

    fn search_args() -> SearchArgs {
        SearchArgs {
            query: "query".to_string(),
            collections: Vec::new(),
            limit: 10,
            min_score: 0.0,
            deep: false,
            keyword: false,
            semantic: false,
            rerank: false,
            no_rerank: false,
            debug: false,
        }
    }

    #[test]
    fn requested_search_mode_defaults_to_auto() {
        let args = search_args();
        assert_eq!(requested_search_mode(&args), RequestedSearchMode::Auto);
    }

    #[test]
    fn requested_search_mode_respects_explicit_flags() {
        let mut args = search_args();
        args.deep = true;
        assert_eq!(requested_search_mode(&args), RequestedSearchMode::Deep);

        args.deep = false;
        args.keyword = true;
        assert_eq!(requested_search_mode(&args), RequestedSearchMode::Keyword);

        args.keyword = false;
        args.semantic = true;
        assert_eq!(requested_search_mode(&args), RequestedSearchMode::Semantic);
    }

    #[test]
    fn eval_rejects_top_level_space_flag() {
        let err = ensure_eval_uses_local_scope(Some("work")).expect_err("space should fail");
        assert!(
            err.to_string()
                .contains("eval commands do not use the top-level --space flag"),
            "unexpected error: {err}"
        );
        ensure_eval_uses_local_scope(None).expect("no top-level scope");
    }

    #[test]
    fn parse_pull_confirmation_accepts_default_yes_and_explicit_yes() {
        assert!(parse_pull_confirmation(""));
        assert!(parse_pull_confirmation("   "));
        assert!(parse_pull_confirmation("y"));
        assert!(parse_pull_confirmation("YES"));
    }

    #[test]
    fn parse_pull_confirmation_rejects_non_yes_answers() {
        assert!(!parse_pull_confirmation("n"));
        assert!(!parse_pull_confirmation("no"));
        assert!(!parse_pull_confirmation("anything else"));
    }

    #[test]
    fn parse_internal_schedule_run_recognizes_hidden_runner_command() {
        let parsed = parse_internal_schedule_run(&[
            OsString::from("kbolt"),
            OsString::from(INTERNAL_SCHEDULE_RUN_COMMAND),
            OsString::from("s2"),
        ])
        .expect("parse internal runner");

        assert_eq!(parsed.as_deref(), Some("s2"));
    }

    #[test]
    fn parse_internal_schedule_run_ignores_normal_cli_invocations() {
        let parsed =
            parse_internal_schedule_run(&[OsString::from("kbolt"), OsString::from("status")])
                .expect("parse normal cli");

        assert_eq!(parsed, None);
    }

    #[test]
    fn parse_internal_schedule_run_rejects_missing_or_empty_ids() {
        let missing = parse_internal_schedule_run(&[
            OsString::from("kbolt"),
            OsString::from(INTERNAL_SCHEDULE_RUN_COMMAND),
        ])
        .expect_err("missing id should fail");
        assert!(missing
            .to_string()
            .contains("internal schedule runner usage"));

        let empty = parse_internal_schedule_run(&[
            OsString::from("kbolt"),
            OsString::from(INTERNAL_SCHEDULE_RUN_COMMAND),
            OsString::from("   "),
        ])
        .expect_err("empty id should fail");
        assert!(empty.to_string().contains("schedule id must not be empty"));
    }

    #[test]
    fn schedule_add_request_builds_weekly_collection_scope() {
        let request = schedule_add_request(ScheduleAddArgs {
            every: None,
            at: Some("3pm".to_string()),
            on: vec![ScheduleDayArg::Mon, ScheduleDayArg::Fri],
            space: Some("work".to_string()),
            collections: vec!["api".to_string(), "docs".to_string()],
        })
        .expect("build schedule request");

        assert_eq!(
            request.trigger,
            ScheduleTrigger::Weekly {
                weekdays: vec![ScheduleWeekday::Mon, ScheduleWeekday::Fri],
                time: "3pm".to_string(),
            }
        );
        assert_eq!(
            request.scope,
            ScheduleScope::Collections {
                space: "work".to_string(),
                collections: vec!["api".to_string(), "docs".to_string()],
            }
        );
    }

    #[test]
    fn schedule_remove_request_prefers_explicit_selectors() {
        let by_id = schedule_remove_request(ScheduleRemoveArgs {
            id: Some("s2".to_string()),
            all: false,
            space: None,
            collections: vec![],
        })
        .expect("build id removal");
        assert_eq!(
            by_id.selector,
            RemoveScheduleSelector::Id {
                id: "s2".to_string()
            }
        );

        let by_scope = schedule_remove_request(ScheduleRemoveArgs {
            id: None,
            all: false,
            space: Some("work".to_string()),
            collections: vec!["api".to_string()],
        })
        .expect("build scoped removal");
        assert_eq!(
            by_scope.selector,
            RemoveScheduleSelector::Scope {
                scope: ScheduleScope::Collections {
                    space: "work".to_string(),
                    collections: vec!["api".to_string()],
                }
            }
        );
    }

    #[test]
    fn parse_schedule_interval_accepts_minutes_and_hours() {
        assert_eq!(
            parse_schedule_interval("30m").expect("parse minutes"),
            ScheduleInterval {
                value: 30,
                unit: ScheduleIntervalUnit::Minutes,
            }
        );
        assert_eq!(
            parse_schedule_interval("2h").expect("parse hours"),
            ScheduleInterval {
                value: 2,
                unit: ScheduleIntervalUnit::Hours,
            }
        );

        let err = parse_schedule_interval("7d").expect_err("reject invalid interval unit");
        assert!(
            err.to_string().contains("invalid schedule interval"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ensure_schedule_uses_local_scope_rejects_top_level_space() {
        let err = ensure_schedule_uses_local_scope(Some("work"))
            .expect_err("top-level space should fail");
        assert!(
            err.to_string()
                .contains("schedule commands do not use the top-level --space flag"),
            "unexpected error: {err}"
        );

        ensure_schedule_uses_local_scope(None).expect("no top-level scope");
    }

    #[test]
    fn requested_output_format_parses_json_flags() {
        let separate = requested_output_format_from_args(&[
            OsString::from("kbolt"),
            OsString::from("--format"),
            OsString::from("json"),
            OsString::from("status"),
        ]);
        assert_eq!(separate, OutputFormat::Json);

        let inline = requested_output_format_from_args(&[
            OsString::from("kbolt"),
            OsString::from("--format=json"),
            OsString::from("status"),
        ]);
        assert_eq!(inline, OutputFormat::Json);

        let cli =
            requested_output_format_from_args(&[OsString::from("kbolt"), OsString::from("status")]);
        assert_eq!(cli, OutputFormat::Cli);
    }

    #[test]
    fn interactive_output_requires_cli_format_and_tty() {
        assert!(supports_interactive_output(OutputFormat::Cli, true));
        assert!(!supports_interactive_output(OutputFormat::Cli, false));
        assert!(!supports_interactive_output(OutputFormat::Json, true));
    }

    #[test]
    fn render_message_output_wraps_json_success_envelope() {
        let rendered = render_message_output(OutputFormat::Json, "space added: work");
        let value: serde_json::Value =
            serde_json::from_str(&rendered).expect("rendered output should be valid json");
        assert_eq!(
            value,
            json!({
                "ok": true,
                "message": "space added: work",
            })
        );
    }

    #[test]
    fn render_error_output_wraps_json_error_envelope() {
        let err = RunError::Core(CoreError::Domain(KboltError::InvalidInput(
            "bad input".to_string(),
        )));
        let rendered = render_error_output(OutputFormat::Json, &err);
        let value: serde_json::Value =
            serde_json::from_str(&rendered).expect("rendered output should be valid json");
        assert_eq!(
            value,
            json!({
                "ok": false,
                "error": {
                    "kind": "invalid_input",
                    "message": "invalid input: bad input",
                }
            })
        );
    }

    #[test]
    fn render_error_output_preserves_clap_help_text() {
        let err =
            RunError::Clap(Cli::command().error(clap::error::ErrorKind::DisplayHelp, "help text"));
        let rendered = render_error_output(OutputFormat::Json, &err);
        assert!(
            rendered.contains("help text"),
            "unexpected output: {rendered}"
        );
        assert!(
            !rendered.trim_start().starts_with('{'),
            "help output should not be wrapped as json: {rendered}"
        );
    }

    #[test]
    fn render_structured_output_serializes_object_payloads() {
        let rendered = render_structured_output(&DefaultSpaceJsonResponse {
            default_space: Some("work".to_string()),
        })
        .expect("structured output should serialize");
        let value: serde_json::Value =
            serde_json::from_str(&rendered).expect("rendered output should be valid json");
        assert_eq!(
            value,
            json!({
                "default_space": "work",
            })
        );
    }

    #[test]
    fn ignore_show_json_response_preserves_pattern_lines() {
        let rendered = render_structured_output(&IgnoreShowJsonResponse {
            space: "work".to_string(),
            collection: "api".to_string(),
            patterns: vec!["dist/".to_string(), "# comment".to_string()],
        })
        .expect("structured output should serialize");
        let value: serde_json::Value =
            serde_json::from_str(&rendered).expect("rendered output should be valid json");
        assert_eq!(
            value,
            json!({
                "space": "work",
                "collection": "api",
                "patterns": ["dist/", "# comment"],
            })
        );
    }

    #[test]
    fn render_structured_output_serializes_update_reports_with_decisions() {
        let rendered = render_structured_output(&UpdateReport {
            scanned: 2,
            skipped_mtime: 0,
            skipped_hash: 0,
            added: 1,
            updated: 0,
            deactivated: 0,
            reactivated: 0,
            reaped: 0,
            embedded: 0,
            decisions: vec![UpdateDecision {
                space: "work".to_string(),
                collection: "api".to_string(),
                path: "src/lib.rs".to_string(),
                kind: UpdateDecisionKind::New,
                detail: None,
            }],
            errors: vec![FileError {
                path: "/tmp/work-api/src/lib.rs".to_string(),
                error: "read failed".to_string(),
            }],
            elapsed_ms: 12,
        })
        .expect("structured output should serialize");
        let value: serde_json::Value =
            serde_json::from_str(&rendered).expect("rendered output should be valid json");
        assert_eq!(
            value,
            json!({
                "scanned": 2,
                "skipped_mtime": 0,
                "skipped_hash": 0,
                "added": 1,
                "updated": 0,
                "deactivated": 0,
                "reactivated": 0,
                "reaped": 0,
                "embedded": 0,
                "decisions": [
                    {
                        "space": "work",
                        "collection": "api",
                        "path": "src/lib.rs",
                        "kind": "New",
                        "detail": null
                    }
                ],
                "errors": [
                    {
                        "path": "/tmp/work-api/src/lib.rs",
                        "error": "read failed"
                    }
                ],
                "elapsed_ms": 12
            })
        );
    }

    #[test]
    fn model_not_available_error_detection_is_specific() {
        let missing = CoreError::Domain(KboltError::ModelNotAvailable {
            name: "test-model".to_string(),
        });
        assert!(is_model_not_available_error(&missing));

        let other = CoreError::Domain(KboltError::InvalidInput("bad input".to_string()));
        assert!(!is_model_not_available_error(&other));
    }

    #[test]
    fn update_model_pull_prompt_offer_requires_embed_and_tty_and_model_error() {
        let missing = CoreError::Domain(KboltError::ModelNotAvailable {
            name: "test-model".to_string(),
        });
        assert!(should_offer_model_pull_for_update(false, true, &missing));

        assert!(!should_offer_model_pull_for_update(true, true, &missing));
        assert!(!should_offer_model_pull_for_update(false, false, &missing));

        let other = CoreError::Domain(KboltError::InvalidInput("bad input".to_string()));
        assert!(!should_offer_model_pull_for_update(false, true, &other));
    }

    #[test]
    fn update_model_missing_guidance_adds_no_embed_hint() {
        let missing = CoreError::Domain(KboltError::ModelNotAvailable {
            name: "test-model".to_string(),
        });
        let rewritten = with_update_model_missing_guidance(missing);
        let message = rewritten.to_string();
        assert!(message.contains("kbolt models pull"));
        assert!(message.contains("--no-embed"));

        let unchanged = CoreError::Domain(KboltError::InvalidInput("bad input".to_string()));
        let rewritten_other = with_update_model_missing_guidance(unchanged);
        assert_eq!(rewritten_other.to_string(), "invalid input: bad input");
    }

    #[test]
    fn collection_add_model_pull_prompt_offer_requires_index_and_tty_and_model_error() {
        let missing = CoreError::Domain(KboltError::ModelNotAvailable {
            name: "test-model".to_string(),
        });
        assert!(should_offer_model_pull_for_collection_add(
            false, true, &missing
        ));

        assert!(!should_offer_model_pull_for_collection_add(
            true, true, &missing
        ));
        assert!(!should_offer_model_pull_for_collection_add(
            false, false, &missing
        ));

        let other = CoreError::Domain(KboltError::InvalidInput("bad input".to_string()));
        assert!(!should_offer_model_pull_for_collection_add(
            false, true, &other
        ));
    }

    #[test]
    fn collection_add_model_missing_guidance_adds_no_index_hint() {
        let missing = CoreError::Domain(KboltError::ModelNotAvailable {
            name: "test-model".to_string(),
        });
        let rewritten = with_collection_add_model_missing_guidance(missing);
        let message = rewritten.to_string();
        assert!(message.contains("kbolt models pull"));
        assert!(message.contains("--no-index"));

        let unchanged = CoreError::Domain(KboltError::InvalidInput("bad input".to_string()));
        let rewritten_other = with_collection_add_model_missing_guidance(unchanged);
        assert_eq!(rewritten_other.to_string(), "invalid input: bad input");
    }

    #[test]
    fn first_run_models_hint_visibility_respects_context() {
        assert!(should_show_first_run_models_hint(
            &Command::Status,
            OutputFormat::Cli,
            true,
            false
        ));
        assert!(!should_show_first_run_models_hint(
            &Command::Status,
            OutputFormat::Cli,
            true,
            true
        ));
        assert!(!should_show_first_run_models_hint(
            &Command::Status,
            OutputFormat::Cli,
            false,
            false
        ));
        assert!(!should_show_first_run_models_hint(
            &Command::Mcp,
            OutputFormat::Cli,
            true,
            false
        ));
        assert!(!should_show_first_run_models_hint(
            &Command::Status,
            OutputFormat::Json,
            true,
            false
        ));
    }

    #[test]
    fn output_format_validation_rejects_json_for_mcp() {
        let err = ensure_supported_output_format(OutputFormat::Json, &Command::Mcp)
            .expect_err("json output should not be supported for mcp");
        assert!(
            err.to_string()
                .contains("--format json is not supported for the mcp command"),
            "unexpected error: {err}"
        );

        ensure_supported_output_format(OutputFormat::Cli, &Command::Mcp)
            .expect("cli output should remain valid for mcp");
    }
}

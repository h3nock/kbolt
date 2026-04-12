use std::path::PathBuf;

use clap::{ArgGroup, Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Cli,
    Json,
}

#[derive(Debug, Parser)]
#[command(name = "kbolt", version, about = "local-first retrieval engine")]
pub struct Cli {
    #[arg(short = 's', long = "space", value_name = "name")]
    pub space: Option<String>,

    #[arg(
        short = 'f',
        long = "format",
        value_enum,
        default_value_t = OutputFormat::Cli
    )]
    pub format: OutputFormat,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    #[command(about = "Check system configuration and model readiness")]
    Doctor,
    #[command(about = "Configure kbolt with a local inference stack")]
    Setup(SetupArgs),
    #[command(about = "Manage local llama-server processes")]
    Local(LocalArgs),
    #[command(about = "Create, list, and manage spaces")]
    Space(SpaceArgs),
    #[command(about = "Add, list, and manage document collections")]
    Collection(CollectionArgs),
    #[command(about = "Manage file ignore patterns for collections")]
    Ignore(IgnoreArgs),
    #[command(about = "Show configured model bindings")]
    Models(ModelsArgs),
    #[command(about = "Run retrieval benchmarks")]
    Eval(EvalArgs),
    #[command(about = "Manage automatic re-indexing schedules")]
    Schedule(ScheduleArgs),
    #[command(about = "Start the MCP server for AI agent integration")]
    Mcp,
    #[command(about = "Search indexed documents")]
    Search(SearchArgs),
    #[command(about = "Re-scan and re-index collections")]
    Update(UpdateArgs),
    #[command(about = "Show index status, disk usage, and model readiness")]
    Status,
    #[command(about = "List files in a collection")]
    Ls(LsArgs),
    #[command(about = "Retrieve a document by path or docid")]
    Get(GetArgs),
    #[command(about = "Retrieve multiple documents at once")]
    MultiGet(MultiGetArgs),
}

#[derive(Debug, Args)]
pub struct SpaceArgs {
    #[command(subcommand)]
    pub command: SpaceCommand,
}

#[derive(Debug, Args)]
pub struct SetupArgs {
    #[command(subcommand)]
    pub command: SetupCommand,
}

#[derive(Debug, Args)]
pub struct LocalArgs {
    #[command(subcommand)]
    pub command: LocalCommand,
}

#[derive(Debug, Args)]
pub struct CollectionArgs {
    #[command(subcommand)]
    pub command: CollectionCommand,
}

#[derive(Debug, Args)]
pub struct IgnoreArgs {
    #[command(subcommand)]
    pub command: IgnoreCommand,
}

#[derive(Debug, Args)]
pub struct ModelsArgs {
    #[command(subcommand)]
    pub command: ModelsCommand,
}

#[derive(Debug, Args)]
pub struct EvalArgs {
    #[command(subcommand)]
    pub command: EvalCommand,
}

#[derive(Debug, Args, PartialEq, Eq)]
pub struct EvalImportArgs {
    #[command(subcommand)]
    pub dataset: EvalImportCommand,
}

#[derive(Debug, Args, PartialEq, Eq)]
pub struct EvalRunArgs {
    #[arg(long, value_name = "path")]
    pub file: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct ScheduleArgs {
    #[command(subcommand)]
    pub command: ScheduleCommand,
}

#[derive(Debug, Args, PartialEq, Eq)]
pub struct UpdateArgs {
    #[arg(long = "collection", value_delimiter = ',')]
    pub collections: Vec<String>,
    #[arg(long)]
    pub no_embed: bool,
    #[arg(long)]
    pub dry_run: bool,
    #[arg(long)]
    pub verbose: bool,
}

#[derive(Debug, Args, PartialEq, Eq)]
pub struct LsArgs {
    pub collection: String,
    pub prefix: Option<String>,
    #[arg(long)]
    pub all: bool,
}

#[derive(Debug, Args, PartialEq, Eq)]
pub struct GetArgs {
    pub identifier: String,
    #[arg(long)]
    pub offset: Option<usize>,
    #[arg(long)]
    pub limit: Option<usize>,
}

#[derive(Debug, Args, PartialEq, Eq)]
pub struct MultiGetArgs {
    #[arg(value_delimiter = ',')]
    pub locators: Vec<String>,
    #[arg(long, default_value_t = 20)]
    pub max_files: usize,
    #[arg(long, default_value_t = 51_200)]
    pub max_bytes: usize,
}

#[derive(Debug, Args, PartialEq)]
pub struct SearchArgs {
    pub query: String,
    #[arg(long = "collection", value_delimiter = ',')]
    pub collections: Vec<String>,
    #[arg(long, default_value_t = 10)]
    pub limit: usize,
    #[arg(long, default_value_t = 0.0)]
    pub min_score: f32,
    #[arg(long)]
    pub deep: bool,
    #[arg(long)]
    pub keyword: bool,
    #[arg(long)]
    pub semantic: bool,
    #[arg(long, conflicts_with = "rerank")]
    pub no_rerank: bool,
    #[arg(long, conflicts_with = "no_rerank")]
    pub rerank: bool,
    #[arg(long)]
    pub debug: bool,
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum SpaceCommand {
    #[command(about = "Create a new space")]
    Add {
        name: String,
        #[arg(long)]
        description: Option<String>,
        #[arg(long)]
        strict: bool,
        dirs: Vec<PathBuf>,
    },
    #[command(about = "Set a space description")]
    Describe { name: String, text: String },
    #[command(about = "Rename a space")]
    Rename { old: String, new: String },
    #[command(about = "Remove a space and all its data")]
    Remove { name: String },
    #[command(about = "Show the active space")]
    Current,
    #[command(about = "Get or set the default space")]
    Default { name: Option<String> },
    #[command(about = "List all spaces")]
    List,
    #[command(about = "Show details about a space")]
    Info { name: String },
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum SetupCommand {
    #[command(about = "Set up local embedder and reranker using llama-server")]
    Local,
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum LocalCommand {
    #[command(about = "Show local server status")]
    Status,
    #[command(about = "Start local inference servers")]
    Start,
    #[command(about = "Stop local inference servers")]
    Stop,
    #[command(about = "Enable a local feature (e.g. deep search)")]
    Enable { feature: LocalFeature },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LocalFeature {
    Deep,
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum CollectionCommand {
    #[command(about = "Add a directory as a document collection")]
    Add {
        path: PathBuf,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        description: Option<String>,
        #[arg(long, value_delimiter = ',')]
        extensions: Option<Vec<String>>,
        #[arg(long)]
        no_index: bool,
    },
    #[command(about = "List all collections")]
    List,
    #[command(about = "Show details about a collection")]
    Info { name: String },
    #[command(about = "Set a collection description")]
    Describe { name: String, text: String },
    #[command(about = "Rename a collection")]
    Rename { old: String, new: String },
    #[command(about = "Remove a collection and its indexed data")]
    Remove { name: String },
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum IgnoreCommand {
    #[command(about = "Show ignore patterns for a collection")]
    Show { collection: String },
    #[command(about = "Add an ignore pattern to a collection")]
    Add { collection: String, pattern: String },
    #[command(about = "Remove an ignore pattern from a collection")]
    Remove { collection: String, pattern: String },
    #[command(about = "Open ignore patterns in an editor")]
    Edit { collection: String },
    #[command(about = "List all collections with ignore patterns")]
    List,
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum ModelsCommand {
    #[command(about = "List configured models and their status")]
    List,
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum EvalCommand {
    #[command(about = "Run a retrieval evaluation")]
    Run(EvalRunArgs),
    #[command(about = "Import a benchmark dataset")]
    Import(EvalImportArgs),
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum EvalImportCommand {
    #[command(
        about = "import a canonical BEIR dataset from an extracted directory",
        long_about = "Import a canonical BEIR dataset from an extracted directory.\n\nExpected source layout:\n  corpus.jsonl\n  queries.jsonl\n  qrels/test.tsv\n\nThis command always imports the test split."
    )]
    Beir(EvalImportBeirArgs),
}

#[derive(Debug, Args, PartialEq, Eq)]
pub struct EvalImportBeirArgs {
    #[arg(long, value_name = "name")]
    pub dataset: String,
    #[arg(long, value_name = "dir")]
    pub source: PathBuf,
    #[arg(long, value_name = "dir")]
    pub output: PathBuf,
    #[arg(long, value_name = "name")]
    pub collection: Option<String>,
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum ScheduleCommand {
    #[command(about = "Create a new re-indexing schedule")]
    Add(ScheduleAddArgs),
    #[command(about = "Show schedule status and last run info")]
    Status,
    #[command(about = "Remove a schedule")]
    Remove(ScheduleRemoveArgs),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ScheduleDayArg {
    Mon,
    Tue,
    Wed,
    Thu,
    Fri,
    Sat,
    Sun,
}

#[derive(Debug, Args, PartialEq, Eq)]
#[command(group(
    ArgGroup::new("trigger")
        .required(true)
        .args(["every", "at"])
))]
pub struct ScheduleAddArgs {
    #[arg(long, conflicts_with = "at")]
    pub every: Option<String>,
    #[arg(long, conflicts_with = "every")]
    pub at: Option<String>,
    #[arg(long = "on", value_delimiter = ',', requires = "at", value_enum)]
    pub on: Vec<ScheduleDayArg>,
    #[arg(long)]
    pub space: Option<String>,
    #[arg(long = "collection", requires = "space")]
    pub collections: Vec<String>,
}

#[derive(Debug, Args, PartialEq, Eq)]
#[command(group(
    ArgGroup::new("selector")
        .required(true)
        .args(["id", "all", "space"])
))]
pub struct ScheduleRemoveArgs {
    pub id: Option<String>,
    #[arg(long, conflicts_with_all = ["id", "space", "collections"])]
    pub all: bool,
    #[arg(long, conflicts_with = "id")]
    pub space: Option<String>,
    #[arg(long = "collection", requires = "space", conflicts_with = "id")]
    pub collections: Vec<String>,
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use clap::Parser;

    use super::{
        Cli, CollectionCommand, Command, EvalCommand, EvalImportArgs, EvalImportBeirArgs,
        EvalImportCommand, EvalRunArgs, GetArgs, LocalCommand, LocalFeature, MultiGetArgs,
        OutputFormat, ScheduleAddArgs, ScheduleCommand, ScheduleDayArg, ScheduleRemoveArgs,
        SearchArgs, SetupCommand, SpaceCommand, UpdateArgs,
    };

    fn parse<const N: usize>(args: [&str; N]) -> Cli {
        Cli::try_parse_from(args).expect("parse cli")
    }

    #[test]
    fn parses_output_format_variants() {
        let parsed = parse(["kbolt", "status"]);
        assert_eq!(parsed.format, OutputFormat::Cli);
        let parsed = parse(["kbolt", "--format", "json", "status"]);
        assert_eq!(parsed.format, OutputFormat::Json);
    }

    #[test]
    fn parses_doctor_command() {
        let parsed = parse(["kbolt", "doctor"]);
        assert!(matches!(parsed.command, Command::Doctor));
    }

    #[test]
    fn parses_setup_local_command() {
        let parsed = parse(["kbolt", "setup", "local"]);
        assert!(matches!(
            parsed.command,
            Command::Setup(args) if args.command == SetupCommand::Local
        ));
    }

    #[test]
    fn parses_local_enable_deep_command() {
        let parsed = parse(["kbolt", "local", "enable", "deep"]);
        assert!(matches!(
            parsed.command,
            Command::Local(args)
                if args.command == LocalCommand::Enable {
                    feature: LocalFeature::Deep
                }
        ));
    }

    #[test]
    fn parses_global_space_override() {
        let parsed = parse(["kbolt", "--space", "work", "space", "current"]);
        assert_eq!(parsed.space.as_deref(), Some("work"));
        assert!(matches!(
            parsed.command,
            Command::Space(space) if space.command == SpaceCommand::Current
        ));
    }

    #[test]
    fn parses_collection_add_with_options() {
        let parsed = parse([
            "kbolt",
            "collection",
            "add",
            "/tmp/work-api",
            "--name",
            "api",
            "--description",
            "api docs",
            "--extensions",
            "rs,md",
            "--no-index",
        ]);
        assert_eq!(parsed.space, None);

        assert!(matches!(
            parsed.command,
            Command::Collection(collection)
                if collection.command
                    == CollectionCommand::Add {
                        path: PathBuf::from("/tmp/work-api"),
                        name: Some("api".to_string()),
                        description: Some("api docs".to_string()),
                        extensions: Some(vec!["rs".to_string(), "md".to_string()]),
                        no_index: true
                    }
        ));
    }

    #[test]
    fn parses_update_with_defaults() {
        let parsed = parse(["kbolt", "update"]);
        assert_eq!(parsed.space, None);
        assert!(matches!(
            parsed.command,
            Command::Update(UpdateArgs {
                collections,
                no_embed: false,
                dry_run: false,
                verbose: false,
            }) if collections.is_empty()
        ));
    }

    #[test]
    fn parses_update_with_flags() {
        let parsed = parse([
            "kbolt",
            "--space",
            "work",
            "update",
            "--collection",
            "api,wiki",
            "--no-embed",
            "--dry-run",
            "--verbose",
        ]);
        assert_eq!(parsed.space.as_deref(), Some("work"));
        assert!(matches!(
            parsed.command,
            Command::Update(UpdateArgs {
                collections,
                no_embed: true,
                dry_run: true,
                verbose: true,
            }) if collections == vec!["api".to_string(), "wiki".to_string()]
        ));
    }

    #[test]
    fn parses_get_with_options() {
        let parsed = parse(["kbolt", "get", "api/src/lib.rs"]);
        assert_eq!(parsed.space, None);
        assert!(matches!(
            parsed.command,
            Command::Get(GetArgs {
                identifier,
                offset: None,
                limit: None,
            }) if identifier == "api/src/lib.rs"
        ));

        let parsed = parse([
            "kbolt", "--space", "work", "get", "#abc123", "--offset", "10", "--limit", "25",
        ]);
        assert_eq!(parsed.space.as_deref(), Some("work"));
        assert!(matches!(
            parsed.command,
            Command::Get(GetArgs {
                identifier,
                offset: Some(10),
                limit: Some(25),
            }) if identifier == "#abc123"
        ));
    }

    #[test]
    fn parses_multi_get_with_options() {
        let parsed = parse(["kbolt", "multi-get", "api/a.md,#abc123"]);
        assert_eq!(parsed.space, None);
        assert!(matches!(
            parsed.command,
            Command::MultiGet(MultiGetArgs {
                locators,
                max_files: 20,
                max_bytes: 51_200,
            }) if locators == vec!["api/a.md".to_string(), "#abc123".to_string()]
        ));

        let parsed = parse([
            "kbolt",
            "--space",
            "work",
            "multi-get",
            "api/a.md,api/b.md",
            "--max-files",
            "5",
            "--max-bytes",
            "1024",
        ]);
        assert_eq!(parsed.space.as_deref(), Some("work"));
        assert!(matches!(
            parsed.command,
            Command::MultiGet(MultiGetArgs {
                locators,
                max_files: 5,
                max_bytes: 1024,
            }) if locators == vec!["api/a.md".to_string(), "api/b.md".to_string()]
        ));
    }

    #[test]
    fn parses_search_with_defaults_and_flags() {
        let parsed = parse(["kbolt", "search", "alpha"]);
        assert_eq!(parsed.space, None);
        assert!(matches!(
            parsed.command,
            Command::Search(SearchArgs {
                query,
                collections,
                limit: 10,
                min_score,
                deep: false,
                keyword: false,
                semantic: false,
                no_rerank: false,
                rerank: false,
                debug: false,
            }) if query == "alpha" && collections.is_empty() && min_score == 0.0
        ));

        let parsed = parse([
            "kbolt",
            "--space",
            "work",
            "search",
            "alpha beta",
            "--collection",
            "api,wiki",
            "--limit",
            "7",
            "--min-score",
            "0.25",
            "--keyword",
            "--no-rerank",
            "--debug",
        ]);
        assert_eq!(parsed.space.as_deref(), Some("work"));
        assert!(matches!(
            parsed.command,
            Command::Search(SearchArgs {
                query,
                collections,
                limit: 7,
                min_score,
                deep: false,
                keyword: true,
                semantic: false,
                no_rerank: true,
                rerank: false,
                debug: true,
            }) if query == "alpha beta"
                && collections == vec!["api".to_string(), "wiki".to_string()]
                && min_score == 0.25
        ));
    }

    #[test]
    fn parses_search_rerank_opt_in_flag() {
        let parsed = parse(["kbolt", "search", "alpha", "--rerank"]);
        assert!(matches!(
            parsed.command,
            Command::Search(SearchArgs {
                rerank: true,
                no_rerank: false,
                ..
            })
        ));
    }

    #[test]
    fn parses_schedule_add_interval_and_weekly_variants() {
        let parsed = parse(["kbolt", "schedule", "add", "--every", "30m"]);
        assert!(matches!(
            parsed.command,
            Command::Schedule(schedule)
                if schedule.command
                    == ScheduleCommand::Add(ScheduleAddArgs {
                        every: Some("30m".to_string()),
                        at: None,
                        on: vec![],
                        space: None,
                        collections: vec![],
                    })
        ));

        let parsed = parse([
            "kbolt",
            "schedule",
            "add",
            "--at",
            "3pm",
            "--on",
            "mon,fri",
            "--space",
            "work",
            "--collection",
            "api",
            "--collection",
            "docs",
        ]);
        assert!(matches!(
            parsed.command,
            Command::Schedule(schedule)
                if schedule.command
                    == ScheduleCommand::Add(ScheduleAddArgs {
                        every: None,
                        at: Some("3pm".to_string()),
                        on: vec![ScheduleDayArg::Mon, ScheduleDayArg::Fri],
                        space: Some("work".to_string()),
                        collections: vec!["api".to_string(), "docs".to_string()],
                    })
        ));
    }

    #[test]
    fn parses_schedule_remove_selectors() {
        let parsed = parse(["kbolt", "schedule", "remove", "s2"]);
        assert!(matches!(
            parsed.command,
            Command::Schedule(schedule)
                if schedule.command
                    == ScheduleCommand::Remove(ScheduleRemoveArgs {
                        id: Some("s2".to_string()),
                        all: false,
                        space: None,
                        collections: vec![],
                    })
        ));

        let parsed = parse([
            "kbolt",
            "schedule",
            "remove",
            "--space",
            "work",
            "--collection",
            "api",
        ]);
        assert!(matches!(
            parsed.command,
            Command::Schedule(schedule)
                if schedule.command
                    == ScheduleCommand::Remove(ScheduleRemoveArgs {
                        id: None,
                        all: false,
                        space: Some("work".to_string()),
                        collections: vec!["api".to_string()],
                    })
        ));
    }

    #[test]
    fn parses_eval_run_with_optional_manifest_path() {
        let parsed = parse(["kbolt", "eval", "run"]);
        assert!(matches!(
            parsed.command,
            Command::Eval(eval) if eval.command == EvalCommand::Run(EvalRunArgs { file: None })
        ));

        let parsed = parse(["kbolt", "eval", "run", "--file", "/tmp/scifact.toml"]);
        assert!(matches!(
            parsed.command,
            Command::Eval(eval)
                if eval.command
                    == EvalCommand::Run(EvalRunArgs {
                        file: Some(PathBuf::from("/tmp/scifact.toml"))
                    })
        ));
    }

    #[test]
    fn parses_eval_import_beir_with_required_paths() {
        let parsed = parse([
            "kbolt",
            "eval",
            "import",
            "beir",
            "--dataset",
            "fiqa",
            "--source",
            "/tmp/fiqa-source",
            "--output",
            "/tmp/fiqa-bench",
        ]);

        let Command::Eval(eval) = parsed.command else {
            panic!("expected eval command");
        };
        assert_eq!(
            eval.command,
            EvalCommand::Import(EvalImportArgs {
                dataset: EvalImportCommand::Beir(EvalImportBeirArgs {
                    dataset: "fiqa".to_string(),
                    source: PathBuf::from("/tmp/fiqa-source"),
                    output: PathBuf::from("/tmp/fiqa-bench"),
                    collection: None,
                })
            })
        );
    }

    #[test]
    fn parses_eval_import_beir_with_collection_override() {
        let parsed = parse([
            "kbolt",
            "eval",
            "import",
            "beir",
            "--dataset",
            "fiqa",
            "--source",
            "/tmp/fiqa-source",
            "--output",
            "/tmp/fiqa-bench",
            "--collection",
            "finance",
        ]);

        let Command::Eval(eval) = parsed.command else {
            panic!("expected eval command");
        };
        assert_eq!(
            eval.command,
            EvalCommand::Import(EvalImportArgs {
                dataset: EvalImportCommand::Beir(EvalImportBeirArgs {
                    dataset: "fiqa".to_string(),
                    source: PathBuf::from("/tmp/fiqa-source"),
                    output: PathBuf::from("/tmp/fiqa-bench"),
                    collection: Some("finance".to_string()),
                })
            })
        );
    }
}

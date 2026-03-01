use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "kbolt", version, about = "local-first retrieval engine")]
pub struct Cli {
    #[arg(short = 's', long = "space", value_name = "name")]
    pub space: Option<String>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Space(SpaceArgs),
    Collection(CollectionArgs),
    Models(ModelsArgs),
    Mcp,
    Search(SearchArgs),
    Update(UpdateArgs),
    Status,
    Ls(LsArgs),
    Get(GetArgs),
    MultiGet(MultiGetArgs),
}

#[derive(Debug, Args)]
pub struct SpaceArgs {
    #[command(subcommand)]
    pub command: SpaceCommand,
}

#[derive(Debug, Args)]
pub struct CollectionArgs {
    #[command(subcommand)]
    pub command: CollectionCommand,
}

#[derive(Debug, Args)]
pub struct ModelsArgs {
    #[command(subcommand)]
    pub command: ModelsCommand,
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
    #[arg(long)]
    pub no_rerank: bool,
    #[arg(long)]
    pub debug: bool,
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum SpaceCommand {
    Add {
        name: String,
        #[arg(long)]
        description: Option<String>,
        #[arg(long)]
        strict: bool,
        dirs: Vec<PathBuf>,
    },
    Describe {
        name: String,
        text: String,
    },
    Rename {
        old: String,
        new: String,
    },
    Remove {
        name: String,
    },
    Current,
    Default { name: Option<String> },
    List,
    Info { name: String },
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum CollectionCommand {
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
    List,
    Info { name: String },
    Describe { name: String, text: String },
    Rename { old: String, new: String },
    Remove { name: String },
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum ModelsCommand {
    List,
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use clap::Parser;

    use super::{
        Cli, CollectionCommand, Command, GetArgs, LsArgs, ModelsCommand, MultiGetArgs, SearchArgs,
        SpaceCommand, UpdateArgs,
    };

    #[test]
    fn parses_space_current_command() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "current"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(space.command, SpaceCommand::Current),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_space_add_with_description() {
        let parsed =
            Cli::try_parse_from(["kbolt", "space", "add", "work", "--description", "work docs"])
                .expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(
                space.command,
                SpaceCommand::Add {
                    name: "work".to_string(),
                    description: Some("work docs".to_string()),
                    strict: false,
                    dirs: vec![]
                }
            ),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_space_add_with_dirs() {
        let parsed = Cli::try_parse_from([
            "kbolt",
            "space",
            "add",
            "work",
            "/tmp/work-api",
            "/tmp/work-wiki",
        ])
        .expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(
                space.command,
                SpaceCommand::Add {
                    name: "work".to_string(),
                    description: None,
                    strict: false,
                    dirs: vec![
                        PathBuf::from("/tmp/work-api"),
                        PathBuf::from("/tmp/work-wiki"),
                    ]
                }
            ),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_space_add_with_strict() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "add", "work", "--strict", "/tmp/work-api"])
            .expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(
                space.command,
                SpaceCommand::Add {
                    name: "work".to_string(),
                    description: None,
                    strict: true,
                    dirs: vec![PathBuf::from("/tmp/work-api")]
                }
            ),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_space_default_without_name() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "default"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(space.command, SpaceCommand::Default { name: None }),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_space_describe() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "describe", "work", "updated docs"])
            .expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(
                space.command,
                SpaceCommand::Describe {
                    name: "work".to_string(),
                    text: "updated docs".to_string()
                }
            ),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_space_default_with_name() {
        let parsed =
            Cli::try_parse_from(["kbolt", "space", "default", "work"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(
                space.command,
                SpaceCommand::Default {
                    name: Some("work".to_string())
                }
            ),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_space_rename() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "rename", "work", "team"])
            .expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(
                space.command,
                SpaceCommand::Rename {
                    old: "work".to_string(),
                    new: "team".to_string()
                }
            ),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_space_remove() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "remove", "work"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(
                space.command,
                SpaceCommand::Remove {
                    name: "work".to_string()
                }
            ),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_space_list() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "list"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(space.command, SpaceCommand::List),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_space_info() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "info", "work"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(
                space.command,
                SpaceCommand::Info {
                    name: "work".to_string()
                }
            ),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_global_space_override() {
        let parsed = Cli::try_parse_from(["kbolt", "--space", "work", "space", "current"])
            .expect("parse cli");
        assert_eq!(parsed.space.as_deref(), Some("work"));

        match parsed.command {
            Command::Space(space) => assert_eq!(space.command, SpaceCommand::Current),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_collection_list() {
        let parsed = Cli::try_parse_from(["kbolt", "collection", "list"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Collection(collection) => {
                assert_eq!(collection.command, CollectionCommand::List)
            }
            Command::Space(_) => panic!("unexpected space command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_collection_add_with_options() {
        let parsed = Cli::try_parse_from([
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
        ])
        .expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Collection(collection) => assert_eq!(
                collection.command,
                CollectionCommand::Add {
                    path: PathBuf::from("/tmp/work-api"),
                    name: Some("api".to_string()),
                    description: Some("api docs".to_string()),
                    extensions: Some(vec!["rs".to_string(), "md".to_string()]),
                    no_index: true
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_collection_info() {
        let parsed =
            Cli::try_parse_from(["kbolt", "collection", "info", "api"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Collection(collection) => assert_eq!(
                collection.command,
                CollectionCommand::Info {
                    name: "api".to_string()
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_collection_describe() {
        let parsed =
            Cli::try_parse_from(["kbolt", "collection", "describe", "api", "backend docs"])
                .expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Collection(collection) => assert_eq!(
                collection.command,
                CollectionCommand::Describe {
                    name: "api".to_string(),
                    text: "backend docs".to_string()
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_collection_rename() {
        let parsed = Cli::try_parse_from(["kbolt", "collection", "rename", "api", "backend"])
            .expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Collection(collection) => assert_eq!(
                collection.command,
                CollectionCommand::Rename {
                    old: "api".to_string(),
                    new: "backend".to_string()
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_collection_remove() {
        let parsed =
            Cli::try_parse_from(["kbolt", "collection", "remove", "api"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Collection(collection) => assert_eq!(
                collection.command,
                CollectionCommand::Remove {
                    name: "api".to_string()
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_update_with_defaults() {
        let parsed = Cli::try_parse_from(["kbolt", "update"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Update(update) => assert_eq!(
                update,
                UpdateArgs {
                    collections: vec![],
                    no_embed: false,
                    dry_run: false,
                    verbose: false,
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_update_with_flags() {
        let parsed = Cli::try_parse_from([
            "kbolt",
            "--space",
            "work",
            "update",
            "--collection",
            "api,wiki",
            "--no-embed",
            "--dry-run",
            "--verbose",
        ])
        .expect("parse cli");
        assert_eq!(parsed.space.as_deref(), Some("work"));

        match parsed.command {
            Command::Update(update) => assert_eq!(
                update,
                UpdateArgs {
                    collections: vec!["api".to_string(), "wiki".to_string()],
                    no_embed: true,
                    dry_run: true,
                    verbose: true,
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_status_command_with_and_without_space() {
        let parsed = Cli::try_parse_from(["kbolt", "status"]).expect("parse cli");
        assert_eq!(parsed.space, None);
        match parsed.command {
            Command::Status => {}
            Command::Space(_) => panic!("unexpected space command"),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }

        let parsed = Cli::try_parse_from(["kbolt", "--space", "work", "status"])
            .expect("parse cli");
        assert_eq!(parsed.space.as_deref(), Some("work"));
        match parsed.command {
            Command::Status => {}
            Command::Space(_) => panic!("unexpected space command"),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_ls_with_defaults_and_options() {
        let parsed = Cli::try_parse_from(["kbolt", "ls", "api"]).expect("parse cli");
        assert_eq!(parsed.space, None);
        match parsed.command {
            Command::Ls(ls) => assert_eq!(
                ls,
                LsArgs {
                    collection: "api".to_string(),
                    prefix: None,
                    all: false
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }

        let parsed = Cli::try_parse_from(["kbolt", "--space", "work", "ls", "api", "src", "--all"])
            .expect("parse cli");
        assert_eq!(parsed.space.as_deref(), Some("work"));
        match parsed.command {
            Command::Ls(ls) => assert_eq!(
                ls,
                LsArgs {
                    collection: "api".to_string(),
                    prefix: Some("src".to_string()),
                    all: true
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_get_with_defaults_and_options() {
        let parsed = Cli::try_parse_from(["kbolt", "get", "api/src/lib.rs"]).expect("parse cli");
        assert_eq!(parsed.space, None);
        match parsed.command {
            Command::Get(get) => assert_eq!(
                get,
                GetArgs {
                    identifier: "api/src/lib.rs".to_string(),
                    offset: None,
                    limit: None
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }

        let parsed = Cli::try_parse_from([
            "kbolt",
            "--space",
            "work",
            "get",
            "#abc123",
            "--offset",
            "10",
            "--limit",
            "25",
        ])
        .expect("parse cli");
        assert_eq!(parsed.space.as_deref(), Some("work"));
        match parsed.command {
            Command::Get(get) => assert_eq!(
                get,
                GetArgs {
                    identifier: "#abc123".to_string(),
                    offset: Some(10),
                    limit: Some(25)
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_multi_get_with_defaults_and_options() {
        let parsed = Cli::try_parse_from(["kbolt", "multi-get", "api/a.md,#abc123"])
            .expect("parse cli");
        assert_eq!(parsed.space, None);
        match parsed.command {
            Command::MultiGet(args) => assert_eq!(
                args,
                MultiGetArgs {
                    locators: vec!["api/a.md".to_string(), "#abc123".to_string()],
                    max_files: 20,
                    max_bytes: 51_200
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }

        let parsed = Cli::try_parse_from([
            "kbolt",
            "--space",
            "work",
            "multi-get",
            "api/a.md,api/b.md",
            "--max-files",
            "5",
            "--max-bytes",
            "1024",
        ])
        .expect("parse cli");
        assert_eq!(parsed.space.as_deref(), Some("work"));
        match parsed.command {
            Command::MultiGet(args) => assert_eq!(
                args,
                MultiGetArgs {
                    locators: vec!["api/a.md".to_string(), "api/b.md".to_string()],
                    max_files: 5,
                    max_bytes: 1024
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_models_list() {
        let parsed = Cli::try_parse_from(["kbolt", "models", "list"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Models(models) => assert_eq!(models.command, ModelsCommand::List),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }

    #[test]
    fn parses_search_with_defaults_and_flags() {
        let parsed = Cli::try_parse_from(["kbolt", "search", "alpha"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Search(search) => assert_eq!(
                search,
                SearchArgs {
                    query: "alpha".to_string(),
                    collections: vec![],
                    limit: 10,
                    min_score: 0.0,
                    deep: false,
                    keyword: false,
                    semantic: false,
                    no_rerank: false,
                    debug: false
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
        }

        let parsed = Cli::try_parse_from([
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
        ])
        .expect("parse cli");
        assert_eq!(parsed.space.as_deref(), Some("work"));

        match parsed.command {
            Command::Search(search) => assert_eq!(
                search,
                SearchArgs {
                    query: "alpha beta".to_string(),
                    collections: vec!["api".to_string(), "wiki".to_string()],
                    limit: 7,
                    min_score: 0.25,
                    deep: false,
                    keyword: true,
                    semantic: false,
                    no_rerank: true,
                    debug: true
                }
            ),
            Command::Space(_) => panic!("unexpected space command"),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Mcp => panic!("unexpected mcp command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
        }
    }

    #[test]
    fn parses_mcp_command() {
        let parsed = Cli::try_parse_from(["kbolt", "mcp"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Mcp => {}
            Command::Space(_) => panic!("unexpected space command"),
            Command::Collection(_) => panic!("unexpected collection command"),
            Command::Models(_) => panic!("unexpected models command"),
            Command::Update(_) => panic!("unexpected update command"),
            Command::Status => panic!("unexpected status command"),
            Command::Ls(_) => panic!("unexpected ls command"),
            Command::Get(_) => panic!("unexpected get command"),
            Command::MultiGet(_) => panic!("unexpected multi-get command"),
            Command::Search(_) => panic!("unexpected search command"),
        }
    }
}

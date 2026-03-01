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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use clap::Parser;

    use super::{Cli, CollectionCommand, Command, SpaceCommand};

    #[test]
    fn parses_space_current_command() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "current"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(space.command, SpaceCommand::Current),
            Command::Collection(_) => panic!("unexpected collection command"),
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
        }
    }

    #[test]
    fn parses_space_default_without_name() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "default"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(space.command, SpaceCommand::Default { name: None }),
            Command::Collection(_) => panic!("unexpected collection command"),
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
        }
    }

    #[test]
    fn parses_space_list() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "list"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(space.command, SpaceCommand::List),
            Command::Collection(_) => panic!("unexpected collection command"),
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
        }
    }
}

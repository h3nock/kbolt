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
}

#[derive(Debug, Args)]
pub struct SpaceArgs {
    #[command(subcommand)]
    pub command: SpaceCommand,
}

#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum SpaceCommand {
    Add {
        name: String,
        #[arg(long)]
        description: Option<String>,
    },
    Describe {
        name: String,
        text: String,
    },
    Current,
    Default { name: Option<String> },
    List,
    Info { name: String },
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Command, SpaceCommand};

    #[test]
    fn parses_space_current_command() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "current"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(space.command, SpaceCommand::Current),
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
                    description: Some("work docs".to_string())
                }
            ),
        }
    }

    #[test]
    fn parses_space_default_without_name() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "default"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(space.command, SpaceCommand::Default { name: None }),
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
        }
    }

    #[test]
    fn parses_space_list() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "list"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(space.command, SpaceCommand::List),
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
        }
    }

    #[test]
    fn parses_global_space_override() {
        let parsed = Cli::try_parse_from(["kbolt", "--space", "work", "space", "current"])
            .expect("parse cli");
        assert_eq!(parsed.space.as_deref(), Some("work"));

        match parsed.command {
            Command::Space(space) => assert_eq!(space.command, SpaceCommand::Current),
        }
    }
}

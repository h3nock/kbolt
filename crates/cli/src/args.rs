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
    Current,
    Default { name: Option<String> },
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
    fn parses_space_default_without_name() {
        let parsed = Cli::try_parse_from(["kbolt", "space", "default"]).expect("parse cli");
        assert_eq!(parsed.space, None);

        match parsed.command {
            Command::Space(space) => assert_eq!(space.command, SpaceCommand::Default { name: None }),
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
    fn parses_global_space_override() {
        let parsed = Cli::try_parse_from(["kbolt", "--space", "work", "space", "current"])
            .expect("parse cli");
        assert_eq!(parsed.space.as_deref(), Some("work"));

        match parsed.command {
            Command::Space(space) => assert_eq!(space.command, SpaceCommand::Current),
        }
    }
}

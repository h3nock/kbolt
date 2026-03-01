use clap::Parser;
use kbolt_cli::args::{Cli, Command, SpaceCommand};
use kbolt_cli::CliAdapter;
use kbolt_core::engine::Engine;
use kbolt_core::Result;

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let engine = Engine::new(None)?;
    let mut adapter = CliAdapter::new(engine);

    match cli.command {
        Command::Space(space) => match space.command {
            SpaceCommand::Add { name, description } => {
                let line = adapter.space_add(&name, description.as_deref())?;
                println!("{line}");
            }
            SpaceCommand::Describe { name, text } => {
                let line = adapter.space_describe(&name, &text)?;
                println!("{line}");
            }
            SpaceCommand::Current => {
                let line = adapter.space_current(cli.space.as_deref())?;
                println!("{line}");
            }
            SpaceCommand::Default { name } => {
                let line = adapter.space_default(name.as_deref())?;
                println!("{line}");
            }
            SpaceCommand::List => {
                let line = adapter.space_list()?;
                println!("{line}");
            }
            SpaceCommand::Info { name } => {
                let line = adapter.space_info(&name)?;
                println!("{line}");
            }
        },
    }

    Ok(())
}

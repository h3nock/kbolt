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
            SpaceCommand::Current => {
                let line = adapter.space_current(cli.space.as_deref())?;
                println!("{line}");
            }
            SpaceCommand::Default { name } => {
                let line = adapter.space_default(name.as_deref())?;
                println!("{line}");
            }
        },
    }

    Ok(())
}

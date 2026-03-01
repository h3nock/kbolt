use clap::Parser;
use kbolt_cli::args::{Cli, CollectionCommand, Command, SpaceCommand};
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
            SpaceCommand::Add {
                name,
                description,
                dirs,
            } => {
                let line = adapter.space_add(&name, description.as_deref(), &dirs)?;
                println!("{line}");
            }
            SpaceCommand::Describe { name, text } => {
                let line = adapter.space_describe(&name, &text)?;
                println!("{line}");
            }
            SpaceCommand::Rename { old, new } => {
                let line = adapter.space_rename(&old, &new)?;
                println!("{line}");
            }
            SpaceCommand::Remove { name } => {
                let line = adapter.space_remove(&name)?;
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
        Command::Collection(collection) => match collection.command {
            CollectionCommand::Add {
                path,
                name,
                description,
                extensions,
                no_index,
            } => {
                let line = adapter.collection_add(
                    cli.space.as_deref(),
                    &path,
                    name.as_deref(),
                    description.as_deref(),
                    extensions.as_deref(),
                    no_index,
                )?;
                println!("{line}");
            }
            CollectionCommand::List => {
                let line = adapter.collection_list(cli.space.as_deref())?;
                println!("{line}");
            }
            CollectionCommand::Info { name } => {
                let line = adapter.collection_info(cli.space.as_deref(), &name)?;
                println!("{line}");
            }
            CollectionCommand::Describe { name, text } => {
                let line = adapter.collection_describe(cli.space.as_deref(), &name, &text)?;
                println!("{line}");
            }
            CollectionCommand::Rename { old, new } => {
                let line = adapter.collection_rename(cli.space.as_deref(), &old, &new)?;
                println!("{line}");
            }
            CollectionCommand::Remove { name } => {
                let line = adapter.collection_remove(cli.space.as_deref(), &name)?;
                println!("{line}");
            }
        },
    }

    Ok(())
}

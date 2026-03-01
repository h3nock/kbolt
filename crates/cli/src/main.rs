use clap::Parser;
use kbolt_cli::args::{Cli, CollectionCommand, Command, IgnoreCommand, ModelsCommand, SpaceCommand};
use kbolt_cli::CliAdapter;
use kbolt_core::engine::Engine;
use kbolt_core::Result;
use kbolt_mcp::stdio;
use kbolt_mcp::McpAdapter;

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    if matches!(cli.command, Command::Mcp) {
        let engine = Engine::new(None)?;
        let adapter = McpAdapter::new(engine);
        return stdio::run_stdio(&adapter);
    }

    let engine = Engine::new(None)?;
    let mut adapter = CliAdapter::new(engine);

    match cli.command {
        Command::Space(space) => match space.command {
            SpaceCommand::Add {
                name,
                description,
                strict,
                dirs,
            } => {
                let line = adapter.space_add(&name, description.as_deref(), strict, &dirs)?;
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
        Command::Ignore(ignore) => match ignore.command {
            IgnoreCommand::Show { collection } => {
                let line = adapter.ignore_show(cli.space.as_deref(), &collection)?;
                println!("{line}");
            }
            IgnoreCommand::Add {
                collection,
                pattern,
            } => {
                let line = adapter.ignore_add(cli.space.as_deref(), &collection, &pattern)?;
                println!("{line}");
            }
            IgnoreCommand::Remove {
                collection,
                pattern,
            } => {
                let line = adapter.ignore_remove(cli.space.as_deref(), &collection, &pattern)?;
                println!("{line}");
            }
            IgnoreCommand::List => {
                let line = adapter.ignore_list(cli.space.as_deref())?;
                println!("{line}");
            }
        },
        Command::Models(models) => match models.command {
            ModelsCommand::List => {
                let line = adapter.models_list()?;
                println!("{line}");
            }
            ModelsCommand::Pull => {
                let line = adapter.models_pull()?;
                println!("{line}");
            }
        },
        Command::Mcp => unreachable!("mcp command handled before adapter setup"),
        Command::Search(search) => {
            let line = adapter.search(
                cli.space.as_deref(),
                &search.query,
                &search.collections,
                search.limit,
                search.min_score,
                search.deep,
                search.keyword,
                search.semantic,
                search.no_rerank,
                search.debug,
            )?;
            println!("{line}");
        }
        Command::Update(update) => {
            let line = adapter.update(
                cli.space.as_deref(),
                &update.collections,
                update.no_embed,
                update.dry_run,
                update.verbose,
            )?;
            println!("{line}");
        }
        Command::Status => {
            let line = adapter.status(cli.space.as_deref())?;
            println!("{line}");
        }
        Command::Ls(ls) => {
            let line = adapter.ls(
                cli.space.as_deref(),
                &ls.collection,
                ls.prefix.as_deref(),
                ls.all,
            )?;
            println!("{line}");
        }
        Command::Get(get) => {
            let line = adapter.get(
                cli.space.as_deref(),
                &get.identifier,
                get.offset,
                get.limit,
            )?;
            println!("{line}");
        }
        Command::MultiGet(get) => {
            let line = adapter.multi_get(
                cli.space.as_deref(),
                &get.locators,
                get.max_files,
                get.max_bytes,
            )?;
            println!("{line}");
        }
    }

    Ok(())
}

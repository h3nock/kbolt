use clap::Parser;
use kbolt_cli::args::{
    Cli, CollectionCommand, Command, IgnoreCommand, ModelsCommand, SearchArgs, SpaceCommand,
};
use kbolt_cli::CliAdapter;
use kbolt_core::error::CoreError;
use kbolt_core::engine::Engine;
use kbolt_core::Result;
use kbolt_mcp::stdio;
use kbolt_mcp::McpAdapter;
use kbolt_types::KboltError;

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
            IgnoreCommand::Edit { collection } => {
                let line = adapter.ignore_edit(cli.space.as_deref(), &collection)?;
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
            let requested_mode = requested_search_mode(&search);
            let run_search = |deep: bool, keyword: bool, semantic: bool| {
                adapter.search(
                    cli.space.as_deref(),
                    &search.query,
                    &search.collections,
                    search.limit,
                    search.min_score,
                    deep,
                    keyword,
                    semantic,
                    search.no_rerank,
                    search.debug,
                )
            };

            match run_search(search.deep, search.keyword, search.semantic) {
                Ok(line) => {
                    println!("{line}");
                }
                Err(err) if is_model_not_available_error(&err) && stdin_stdout_are_tty() => {
                    if prompt_pull_models()? {
                        let pull_report = adapter.models_pull()?;
                        println!("{pull_report}");
                        let retried = run_search(search.deep, search.keyword, search.semantic)?;
                        println!("{retried}");
                    } else if requested_mode == RequestedSearchMode::Auto {
                        println!("models unavailable; falling back to keyword mode");
                        let fallback = run_search(false, true, false)?;
                        println!("{fallback}");
                    } else {
                        return Err(err);
                    }
                }
                Err(err) => return Err(err),
            }
        }
        Command::Update(update) => {
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
                    println!("{line}");
                }
                Err(err)
                    if should_offer_model_pull_for_update(
                        update.no_embed,
                        stdin_stdout_are_tty(),
                        &err,
                    ) =>
                {
                    if prompt_pull_models()? {
                        let pull_report = adapter.models_pull()?;
                        println!("{pull_report}");
                        let retried = run_update(false)?;
                        println!("{retried}");
                    } else {
                        return Err(err);
                    }
                }
                Err(err) => return Err(err),
            }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestedSearchMode {
    Auto,
    Deep,
    Keyword,
    Semantic,
}

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

fn is_model_not_available_error(err: &CoreError) -> bool {
    matches!(err, CoreError::Domain(KboltError::ModelNotAvailable { .. }))
}

fn stdin_stdout_are_tty() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

fn prompt_pull_models() -> Result<bool> {
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

#[cfg(test)]
mod tests {
    use super::{
        is_model_not_available_error, parse_pull_confirmation, requested_search_mode,
        should_offer_model_pull_for_update, RequestedSearchMode,
    };
    use kbolt_cli::args::SearchArgs;
    use kbolt_core::error::CoreError;
    use kbolt_types::KboltError;

    fn search_args() -> SearchArgs {
        SearchArgs {
            query: "query".to_string(),
            collections: Vec::new(),
            limit: 10,
            min_score: 0.0,
            deep: false,
            keyword: false,
            semantic: false,
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
}

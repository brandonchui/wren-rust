use crate::wren::Wren;

mod ast;
mod codegen;
mod parser;
mod scanner;
mod token;
mod wren;

enum Command {
    Repl,
    RunFile(String),
    Usage,
}

fn main() {
    let args = std::env::args().collect::<Vec<String>>();

    let mut wren = Wren::new();

    match parse_args(&args) {
        Command::Repl => run_prompt(),
        Command::RunFile(path) => wren.run_file(&path),
        Command::Usage => println!("Usage: wren [script]"),
    }
}

fn parse_args(args: &[String]) -> Command {
    if args.len() > 2 {
        Command::Usage
    } else if args.len() == 1 {
        Command::Repl
    } else {
        Command::RunFile(args[1].clone())
    }
}

// Probably does not make sense currently to implment a REPL for this compiler project.
fn run_prompt() {
    //TODO
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_args_runs_repl() {
        let args = vec!["wren".to_string()];
        assert!(matches!(parse_args(&args), Command::Repl));
    }

    #[test]
    fn test_one_arg_runs_file() {
        let args = vec!["wren".to_string(), "script.wren".to_string()];
        assert!(matches!(parse_args(&args), Command::RunFile(_)));
    }

    #[test]
    fn test_too_many_args_shows_usage() {
        let args = vec!["wren".to_string(), "a".to_string(), "b".to_string()];
        assert!(matches!(parse_args(&args), Command::Usage));
    }
}

use clap::{Parser, arg};

use crate::wren::Wren;

mod ast;
mod codegen;
mod parser;
mod scanner;
mod token;
mod wren;

#[derive(Parser, Debug)]
struct Args {
    file: String,

    #[arg(long)]
    ast: bool,
}

fn main() {
    let args = Args::parse();

    let mut wren = Wren::new();

    let source = std::fs::read_to_string(args.file);

    if args.ast {
        wren.debug = true;
    }

    match source {
        Ok(src) => wren.run_file(&src),
        Err(_) => panic!("File error."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_parse_file_only() {
        let args = Args::try_parse_from(["wren", "script.wren"]).unwrap();
        assert_eq!(args.file, "script.wren");
        assert!(!args.ast);
    }

    #[test]
    fn test_args_parse_with_ast_flag() {
        let args = Args::try_parse_from(["wren", "script.wren", "--ast"]).unwrap();
        assert_eq!(args.file, "script.wren");
        assert!(args.ast);
    }

    #[test]
    fn test_args_parse_ast_flag_before_file() {
        let args = Args::try_parse_from(["wren", "--ast", "script.wren"]).unwrap();
        assert_eq!(args.file, "script.wren");
        assert!(args.ast);
    }

    #[test]
    fn test_args_parse_missing_file_fails() {
        let result = Args::try_parse_from(["wren"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_args_parse_unknown_flag_fails() {
        let result = Args::try_parse_from(["wren", "script.wren", "--unknown"]);
        assert!(result.is_err());
    }
}

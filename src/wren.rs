use inkwell::context::Context;

use crate::{codegen::CodeGen, parser::Parser, scanner::Scanner};

pub struct Wren {
    pub had_error: bool,
    pub debug: bool,
}

impl Wren {
    pub fn new() -> Self {
        Wren {
            had_error: false,
            debug: false,
        }
    }
    pub fn report(&mut self, line: u32, loc: String, msg: String) {
        println!("[line {} Error {}: {}]", line, loc, msg);
        self.had_error = true;
    }

    pub fn error(&mut self, line: u32, msg: String) {
        self.report(line, "".to_string(), msg);
    }

    pub fn run_file(&mut self, s: &str) {
        // println!("Running {s}");

        if self.had_error {
            //TODO error code?
            return;
        }

        let mut scan = Scanner::new(s);
        scan.scan_tokens();

        if !scan.errors.is_empty() {
            for (line, msg) in &scan.errors {
                self.error(*line, msg.clone());
            }
            return;
        }
        let tokens = scan.tokens;

        let mut parser = Parser::new(tokens);

        match parser.parse() {
            Ok(expr) => {
                if self.debug {
                    println!("{:?}", expr);
                }

                // Generate IR
                let context = Context::create();
                let mut codegen = CodeGen::new(&context);

                codegen.compile(&expr);
                codegen.print_ir();
                let result = codegen.jit_run();
                println!("Result: {}", result);
            }
            Err(e) => {
                self.error(e.line, e.message);
            }
        }
    }
}

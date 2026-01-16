use crate::scanner::Scanner;

pub struct Wren {
    pub had_error: bool,
}

impl Wren {
    pub fn new() -> Self {
        Wren { had_error: false }
    }
    pub fn report(&mut self, line: u32, loc: String, msg: String) {
        println!("[line {} Error {}: {}]", line, loc, msg);
        self.had_error = true;
    }

    pub fn error(&mut self, line: u32, msg: String) {
        self.report(line, "".to_string(), msg);
    }

    pub fn run_file(&mut self, s: &str) {
        println!("Running {s}");

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

        for token in &tokens {
            println!("{token}");
        }
    }
}

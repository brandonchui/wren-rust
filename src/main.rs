fn main() {
    let args = std::env::args().collect::<Vec<String>>();

    if args.len() > 2 {
        println!("Usage: wren [script]");
    } else if args.len() == 1 {
        // If we have no args given from the user, run the REPL
        run_prompt();
    } else {
        run_file(&args[1]);
    }
}

fn run_file(s: &str) {
    //TODO
    println!("Running {s}");
}

fn run_prompt() {
    //TODO
}

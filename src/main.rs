use interpreter::{lexer::LexerError, typechecking::TypeError};
use std::env;

mod interpreter;

fn print_lexer_errors(errors: &[LexerError]) {
    for lexer_error in errors {
        match lexer_error.error {
            interpreter::lexer::ErrorType::UnrecognizedToken(bad_token) => {
                eprintln!(
                    "Syntax Error: Unrecognized token '{}' at line: {}, column: {}",
                    bad_token, lexer_error.line, lexer_error.column
                );
            }
        }
    }
}

fn print_parsing_errors(errors: &str) {
    eprintln!("Parse Error: {}", errors);
}

fn print_type_errors(errors: &[TypeError]) {
    for err in errors {
        eprintln!("Type Error: {:#?}", err);
    }
}

fn main() {
    let args = env::args().collect::<Vec<String>>();
    let source = {
        if args.len() > 1 {
            let filepath = args.get(1).unwrap();
            std::fs::read_to_string(filepath).unwrap()
        } else {
            r#"
                fn main(args: [string]) -> void {
                    print("Hello, world!\n");
                }
            "#
            .to_string()
        }
    };

    let tokens = match interpreter::tokenize(&source) {
        Err(errors) => {
            print_lexer_errors(&errors);
            return;
        }
        Ok(tokens) => tokens,
    };
    println!("{:#?}", tokens);

    let program = match interpreter::parse(&tokens) {
        Err(errors) => {
            print_parsing_errors(errors);
            return;
        }
        Ok(program) => program,
    };
    println!("{:#?}", program);

    if let Err(type_errors) = interpreter::typechecking::typecheck_program(&program) {
        print_type_errors(&type_errors);
        return;
    }

    println!("\n{:?}", interpreter::run(program, &["test-program"]));
}

use crate::interpreter::eval;
use interpreter::lexer::LexerError;
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

    let tokens = interpreter::tokenize(&source);
    if let Err(errors) = tokens {
        print_lexer_errors(&errors);
        return;
    }
    let tokens = tokens.unwrap();
    println!("{:#?}", tokens);

    let parse_tree = interpreter::parse(&tokens);
    if let Err(errors) = parse_tree {
        print_parsing_errors(errors);
        return;
    }
    let top_level_declarations = parse_tree.unwrap();
    println!("{:#?}", top_level_declarations);

    println!("\n{:?}", eval(&["test-program"], top_level_declarations));
}

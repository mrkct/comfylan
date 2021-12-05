use interpreter::lexer::LexerError;

use crate::interpreter::eval;

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
    let source = r#"
    fn factorial(x: int) -> int {
        var result: int = 1;
        while (x > 0) {
            result = result * x;
            x = x - 1;
        }

        return result;
    }

    fn main(argc: int, argv: [string]) -> void {
        return factorial(7);
    }
    "#;

    let tokens = interpreter::tokenize(source);
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

    println!("Execution Result:");
    println!("{:?}", eval(1, &["test-program"], top_level_declarations));
}

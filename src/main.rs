use interpreter::lexer::LexerError;

mod interpreter;

fn print_lexer_errors(errors: &Vec<LexerError>) {
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
            result *= x;
            x -= 1;
        }

        return result;
    }

    fn main(argc: int) -> void {
        println("The factorial of 7 is ", factorial(7));
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
    let parse_tree = parse_tree.unwrap();
    println!("{:#?}", parse_tree);
}

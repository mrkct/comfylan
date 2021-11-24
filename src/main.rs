mod lexer;
mod parser;

fn main() {
    let source = r#"
    
    fn factorial(x: int) -> int {
        var result = 1;
        while (x > 0) {
            result = result * x;
            x = x - 1;
        }

        return result;
    }

    fn main() -> void {

    }
    "#;

    let tokens = lexer::tokenize(source);
    if let Err(errors) = tokens {
        for lexer_error in errors {
            match lexer_error.error {
                lexer::ErrorType::UnrecognizedToken(bad_token) => {
                    eprintln!(
                        "Syntax Error: Unrecognized token '{}' at line: {}, column: {}",
                        bad_token, lexer_error.line, lexer_error.column
                    );
                }
            }
        }
        return;
    }
    let tokens = tokens.unwrap();
    println!("{:?}", tokens);

    let ast = parser::parse(&tokens);
    if let Err(error) = ast {
        eprintln!("Parse Error: {}", error);
        return;
    }
    let ast = ast.unwrap();
    println!("{:?}", ast);
}

mod interpreter;

fn main() {
    let source = r#"
    fn factorial(x: int) -> int {
        var result = 1;
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
        return;
    }
    let tokens = tokens.unwrap();
    println!("{:#?}", tokens);

    let ast = interpreter::parse(&tokens);
    if let Err(error) = ast {
        eprintln!("Parse Error: {}", error);
        return;
    }
    let ast = ast.unwrap();
    println!("{:#?}", ast);
}

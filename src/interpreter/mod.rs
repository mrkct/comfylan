pub mod ast;
pub mod environment;
pub mod evaluator;
pub mod lexer;
pub mod parser;
pub mod typechecking;

pub fn tokenize(source: &str) -> Result<Vec<lexer::Token>, Vec<lexer::LexerError>> {
    let mut collected_tokens = vec![];
    let mut collected_errors = vec![];
    for token_or_error in lexer::Lexer::new(source) {
        match token_or_error {
            Err(error) => collected_errors.push(error),
            Ok(token) if collected_errors.is_empty() => collected_tokens.push(token),
            _ => {}
        }
    }

    if collected_errors.is_empty() {
        Ok(collected_tokens)
    } else {
        Err(collected_errors)
    }
}

pub fn parse<'a>(
    tokens: &'a [lexer::Token<'a>],
) -> Result<Vec<parser::RootFunctionDeclaration<'a>>, &'static str> {
    match parser::Parser::new(tokens).parse_program() {
        Some(ast_nodes) => Ok(ast_nodes),
        None => Err("Parsing Error: I don't have any other info for you"),
    }
}

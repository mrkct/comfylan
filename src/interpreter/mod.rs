use self::{ast::*, environment::*, evaluator::*, typechecking::*};
use std::{cell::RefCell, rc::Rc};

pub mod ast;
pub mod environment;
pub mod evaluator;
pub mod lexer;
mod native;
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

pub fn parse<'a>(tokens: &'a [lexer::Token<'a>]) -> Result<Program, &'static str> {
    parser::Parser::new(tokens)
        .parse_program()
        .ok_or("Parse Error")
}

pub fn run(program: Program, args: &[&str]) -> Result<ImmediateValue, EvaluationError> {
    let env = Env::empty();
    native::fill_values_env_with_native_functions(&env);
    for (function_name, function_decl) in program.function_declarations {
        env.declare(
            &function_name,
            FunctionDeclaration::make_closure_immediate_value(function_decl, &env),
        );
    }

    const FAKE_SOURCE_INFO: SourceInfo = SourceInfo {
        line: 0,
        column: 0,
        offset_in_source: 0,
    };
    Expression::FunctionCall(
        FAKE_SOURCE_INFO,
        None,
        Box::new(Expression::Identifier(FAKE_SOURCE_INFO, "main".to_string())),
        vec![Expression::Value(ImmediateValue::Array(
            Type::String,
            Rc::new(RefCell::new(
                args.iter()
                    .map(|v| ImmediateValue::String(v.to_string()))
                    .collect::<Vec<_>>(),
            )),
        ))],
    )
    .eval(&env)
}

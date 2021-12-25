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

pub fn parse<'a>(tokens: &'a [lexer::Token<'a>]) -> Result<Vec<TopLevelDeclaration>, &'static str> {
    parser::Parser::new(tokens)
        .parse_program()
        .ok_or("Parse Error")
}

fn fill_env_with_top_level_declarations(
    env: &Rc<Env<ImmediateValue>>,
    top_level_declarations: &[TopLevelDeclaration],
) {
    for decl in top_level_declarations {
        match decl {
            TopLevelDeclaration::Function(_, _, name, argnames, code) => {
                env.declare(
                    name,
                    ImmediateValue::Closure(
                        Type::Integer,
                        Rc::clone(env),
                        argnames.to_vec(),
                        code.clone(),
                    ),
                );
            }
        }
    }
}

pub fn eval(
    argv: &[&str],
    top_level_declarations: Vec<TopLevelDeclaration>,
) -> Result<ImmediateValue, EvaluationError> {
    const FAKE_SOURCE_INFO: SourceInfo = SourceInfo {
        line: 0,
        column: 0,
        offset_in_source: 0,
    };
    let root_env = Env::empty();
    let mut type_env = Env::empty();

    native::fill_env_with_native_functions(&root_env, &type_env);
    fill_env_with_top_level_declarations(&root_env, &top_level_declarations);

    if let Err(type_errors) = typecheck_program(&mut type_env, &top_level_declarations) {
        for error in type_errors {
            println!("TypeError: {:?}", error);
        }
        return Ok(ImmediateValue::Void);
    }

    Expression::FunctionCall(
        FAKE_SOURCE_INFO,
        None,
        Box::new(Expression::Identifier(FAKE_SOURCE_INFO, "main".to_string())),
        vec![Expression::Value(ImmediateValue::Array(
            Type::String,
            Rc::new(RefCell::new(
                argv.iter()
                    .map(|v| ImmediateValue::String(v.to_string()))
                    .collect::<Vec<_>>(),
            )),
        ))],
    )
    .eval(&root_env)
}

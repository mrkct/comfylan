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
    top_level_declarations: Vec<TopLevelDeclaration>,
) {
    for decl in top_level_declarations.into_iter() {
        match decl {
            TopLevelDeclaration::Function(_, _, name, args, _, code) => {
                let argnames = args
                    .iter()
                    .map(|(argname, _)| argname.clone())
                    .collect::<Vec<_>>();
                env.declare(
                    &name,
                    ImmediateValue::Closure(
                        Type::Integer,
                        Rc::clone(env),
                        argnames,
                        Box::new(code),
                    ),
                    true,
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

    native::fill_env_with_native_functions(&root_env);
    fill_env_with_top_level_declarations(&root_env, top_level_declarations);

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

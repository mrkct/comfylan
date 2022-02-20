use std::fmt::Display;

use self::{ast::*, environment::*, interpreter::*, typechecking::*};

mod ast;
mod environment;
mod interpreter;
mod lexer;
mod parser;
mod typechecking;

impl<'a> Display for lexer::LexerError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "In line {}:{} Syntax Error: {:?}",
            self.line, self.column, self.error
        ))
    }
}

impl Display for typechecking::TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("In line ?:? Type Error: {:?}", self))
    }
}

impl Display for interpreter::EvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("In line ?:? Runtime Error: {:?}", self))
    }
}

pub fn run_interpreter(source: &str, args: &[&str]) -> Option<i64> {
    fn ok_and_print_errors<T, E: Display>(r: Result<T, Vec<E>>) -> Option<T> {
        match r {
            Ok(v) => Some(v),
            Err(errors) => {
                for e in errors {
                    eprintln!("{}", e);
                }
                None
            }
        }
    }

    fn parse<'a>(tokens: &'a [lexer::Token<'a>]) -> Result<Program, Vec<String>> {
        parser::Parser::new(tokens)
            .parse_program()
            .ok_or_else(|| vec!["Parse Error".to_string()])
    }

    fn run(program: Program, args: &[&str]) -> Result<i64, Vec<interpreter::EvaluationError>> {
        interpreter::run(program, args).map_err(|e| vec![e])
    }

    let tokens = ok_and_print_errors(lexer::tokenize(source))?;
    let program = ok_and_print_errors(parse(&tokens))?;
    ok_and_print_errors(typechecking::typecheck_program(
        &program,
        define_interpreter_specific_builtins_types,
    ))?;
    ok_and_print_errors(run(program, args))
}

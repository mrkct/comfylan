mod evaluator;
mod native;
mod value;
use super::{ast::*, Env, Type};
pub(super) use evaluator::EvaluationError;
use std::collections::HashMap;
use value::*;

pub fn define_interpreter_specific_builtins_types(
    user_types: &mut HashMap<String, Type>,
    type_env: &Env<Type>,
) {
    native::declare_native_types(user_types);
    native::fill_type_env_with_native_functions(type_env);
}

pub fn run(program: Program, args: &[&str]) -> Result<i64, EvaluationError> {
    let env = Env::empty();
    native::fill_values_env_with_native_functions(&env);
    native::fill_global_env_with_builtin_constants(&env);
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

    let mut subsystem = native::get_default_system_game_engine_subsystem();
    Expression::FunctionCall(
        FAKE_SOURCE_INFO,
        None,
        Box::new(Expression::Value(Value::Variable("main".to_string()))),
        vec![Expression::ArrayInitializer(
            FAKE_SOURCE_INFO,
            None,
            args.iter()
                .map(|v| Box::new(Expression::Value(Value::String(v.to_string()))))
                .collect(),
        )],
    )
    .eval(&mut subsystem, &env)
    .map(|v| {
        if let InternalValue::Integer(x) = v {
            x
        } else {
            0
        }
    })
}

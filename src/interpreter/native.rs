use crate::interpreter::{ast::ImmediateValue, environment::Env, typechecking::Type};
use std::rc::Rc;

use super::{ast::SourceInfo, evaluator::EvaluationError};

pub fn fill_env_with_native_functions(env: &Rc<Env<ImmediateValue>>) {
    // I/O
    env.declare(
        "print",
        ImmediateValue::NativeFunction(
            Type::Closure(vec![Type::Array(Box::new(Type::Any))], Box::new(Type::Void)),
            native_print,
        ),
        true,
    );

    // Array operations
    env.declare(
        "len",
        ImmediateValue::NativeFunction(
            Type::Closure(
                vec![Type::Array(Box::new(Type::Any))],
                Box::new(Type::Integer),
            ),
            native_array_len,
        ),
        true,
    );
}

fn print_immediate_value(v: &ImmediateValue) {
    match v {
        ImmediateValue::Integer(x) => print!("{}", x),
        ImmediateValue::FloatingPoint(x) => print!("{}", x),
        ImmediateValue::String(s) => print!("{}", s),
        ImmediateValue::Boolean(b) => print!("{}", b),
        ImmediateValue::Array(_, a) => {
            print!("[");
            let a_borrow = a.borrow();
            let mut iter = a_borrow.iter();
            if let Some(v) = iter.next() {
                print_immediate_value(v);
            }
            for v in iter {
                print!(", ");
                print_immediate_value(v);
            }
            print!("]");
        }
        ImmediateValue::Closure(ftype, _, _, _) => print!("[@Closure {:?}]", ftype),
        ImmediateValue::NativeFunction(ftype, _) => print!("[@NativeFunction {:?}]", ftype),
        ImmediateValue::Void => print!("[Void]"),
    }
}

fn native_print(args: Vec<ImmediateValue>) -> Result<ImmediateValue, EvaluationError> {
    for arg in args {
        print_immediate_value(&arg);
    }
    Ok(ImmediateValue::Void)
}

fn native_array_len(args: Vec<ImmediateValue>) -> Result<ImmediateValue, EvaluationError> {
    match args.get(0) {
        Some(ImmediateValue::Array(_, array)) => {
            Ok(ImmediateValue::Integer(array.borrow().len() as i64))
        },
        _ => panic!("Typechecker failed! Native function 'len' was called with an argument that is not an array")
    }
}

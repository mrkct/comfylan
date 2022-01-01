use crate::interpreter::{ast::ImmediateValue, environment::Env, typechecking::Type};
use std::{cell::RefCell, rc::Rc};

use super::evaluator::EvaluationError;

pub fn fill_env_with_native_functions(env: &Rc<Env<ImmediateValue>>, type_env: &Rc<Env<Type>>) {
    let declare = |f, name, args: Vec<Type>, return_type| {
        let signature = Type::Closure(args, Box::new(return_type));
        env.declare(name, ImmediateValue::NativeFunction(signature.clone(), f));
        type_env.declare(name, signature);
    };

    // I/O
    declare(native_print, "print", vec![], Type::Void);

    // Array operations
    declare(
        native_array_len,
        "len",
        vec![Type::Array(Box::new(Type::Any))],
        Type::Integer,
    );
    declare(
        native_array_insert,
        "insert",
        vec![Type::Array(Box::new(Type::Any)), Type::Integer, Type::Any],
        Type::Void,
    );
    declare(
        native_array_remove,
        "remove",
        vec![Type::Array(Box::new(Type::Any)), Type::Integer],
        Type::Void,
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
        ImmediateValue::Struct(_, fields) => {
            print!("{{");
            fields.borrow().iter().for_each(|(key, val)| {
                print!("{}=", key);
                print_immediate_value(val);
            });
            print!("}}");
        }
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

fn validate_array_index(
    array: &Rc<RefCell<Vec<ImmediateValue>>>,
    index: i64,
) -> Result<usize, EvaluationError> {
    let array_len = array.borrow().len();
    if index < 0 {
        return Err(EvaluationError::ArrayIndexOutOfBounds(array_len, index));
    }

    let usize_index: usize = index
        .try_into()
        .map_err(|_| EvaluationError::ArrayIndexOutOfBounds(array_len, index))?;
    if usize_index > array_len {
        return Err(EvaluationError::ArrayIndexOutOfBounds(array_len, index));
    }
    Ok(usize_index)
}

fn native_array_insert(args: Vec<ImmediateValue>) -> Result<ImmediateValue, EvaluationError> {
    match (args.get(0), args.get(1), args.get(2)) {
        (Some(ImmediateValue::Array(_, array)), Some(ImmediateValue::Integer(index)), Some(v)) => {
            let i = validate_array_index(array, *index)?;
            array.borrow_mut().insert(i, v.clone())
        }
        _ => panic!("Typechecker failed! Native function 'insert' was called with bad arguments"),
    }
    Ok(ImmediateValue::Void)
}

fn native_array_remove(args: Vec<ImmediateValue>) -> Result<ImmediateValue, EvaluationError> {
    match (args.get(0), args.get(1)) {
        (Some(ImmediateValue::Array(_, array)), Some(ImmediateValue::Integer(index))) => {
            let i = validate_array_index(array, *index)?;
            Ok(array.borrow_mut().remove(i))
        }
        _ => panic!("Typechecker failed! Native function 'remove' was called with bad arguments"),
    }
}

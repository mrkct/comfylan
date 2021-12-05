use crate::interpreter::{ast::ImmediateValue, environment::Env, typechecking::Type};
use std::rc::Rc;

pub fn fill_env_with_native_functions(env: &Rc<Env<ImmediateValue>>) {
    env.declare(
        "print",
        ImmediateValue::NativeFunction(
            Type::Closure(vec![Type::Array(Box::new(Type::Any))], Box::new(Type::Void)),
            native_print,
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
            for v in a.borrow().iter() {
                print_immediate_value(v);
                print!(", ");
            }
            print!("]");
        }
        ImmediateValue::Closure(ftype, _, _, _) => print!("[@Closure {:?}]", ftype),
        ImmediateValue::NativeFunction(ftype, _) => print!("[@NativeFunction {:?}]", ftype),
        ImmediateValue::Void => print!("[Void]"),
    }
}

fn native_print(args: Vec<ImmediateValue>) -> ImmediateValue {
    for arg in args {
        print_immediate_value(&arg);
    }
    ImmediateValue::Void
}

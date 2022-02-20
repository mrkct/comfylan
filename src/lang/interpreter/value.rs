use crate::lang::interpreter::{evaluator::*, native::*};
use crate::lang::{ast::*, environment::*, typechecking::*};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

type InternalArrayRepresentation = Rc<RefCell<Vec<InternalValue>>>;
type InternalStructRepresentation = Rc<RefCell<HashMap<String, InternalValue>>>;

#[derive(Debug, PartialEq, Clone)]
pub enum InternalValue {
    Integer(i64),
    FloatingPoint(f64),
    String(String),
    Boolean(bool),
    Closure(Type, Rc<Env<InternalValue>>, Vec<String>, Block),
    Array(Type, InternalArrayRepresentation),
    NativeFunction(NativeFunction),
    Struct(Type, InternalStructRepresentation),
    Void,
}

impl InternalValue {
    pub fn get_type(&self) -> Type {
        match self {
            InternalValue::Integer(_) => Type::Integer,
            InternalValue::FloatingPoint(_) => Type::FloatingPoint,
            InternalValue::String(_) => Type::String,
            InternalValue::Boolean(_) => Type::Boolean,
            InternalValue::Array(array_type, _) => Type::Array(Box::new(array_type.clone())),
            InternalValue::Closure(signature, _, _, _)
            | InternalValue::NativeFunction(NativeFunction { signature, .. }) => signature.clone(),
            InternalValue::Void => Type::Void,
            InternalValue::Struct(t, _) => t.clone(),
        }
    }

    pub fn add(&self, other: &InternalValue) -> Result<InternalValue, EvaluationError> {
        match (self, other) {
            (InternalValue::Integer(x), InternalValue::Integer(y)) => {
                Ok(InternalValue::Integer(x + y))
            }
            (InternalValue::FloatingPoint(x), InternalValue::Integer(y)) => {
                Ok(InternalValue::FloatingPoint(x + *y as f64))
            }
            (InternalValue::Integer(x), InternalValue::FloatingPoint(y)) => {
                Ok(InternalValue::FloatingPoint(*x as f64 + y))
            }
            (InternalValue::FloatingPoint(x), InternalValue::FloatingPoint(y)) => {
                Ok(InternalValue::FloatingPoint(*x + *y))
            }
            (InternalValue::String(left), InternalValue::String(right)) => {
                Ok(InternalValue::String(left.to_owned() + right))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot add values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn sub(&self, other: &InternalValue) -> Result<InternalValue, EvaluationError> {
        match (self, other) {
            (InternalValue::Integer(x), InternalValue::Integer(y)) => {
                Ok(InternalValue::Integer(x - y))
            }
            (InternalValue::FloatingPoint(x), InternalValue::Integer(y)) => {
                Ok(InternalValue::FloatingPoint(x - *y as f64))
            }
            (InternalValue::Integer(x), InternalValue::FloatingPoint(y)) => {
                Ok(InternalValue::FloatingPoint(*x as f64 - y))
            }
            (InternalValue::FloatingPoint(x), InternalValue::FloatingPoint(y)) => {
                Ok(InternalValue::FloatingPoint(*x - *y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot subtract values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn mul(&self, other: &InternalValue) -> Result<InternalValue, EvaluationError> {
        match (self, other) {
            (InternalValue::Integer(x), InternalValue::Integer(y)) => {
                Ok(InternalValue::Integer(x * y))
            }
            (InternalValue::FloatingPoint(x), InternalValue::Integer(y)) => {
                Ok(InternalValue::FloatingPoint(x * *y as f64))
            }
            (InternalValue::Integer(x), InternalValue::FloatingPoint(y)) => {
                Ok(InternalValue::FloatingPoint(*x as f64 * y))
            }
            (InternalValue::FloatingPoint(x), InternalValue::FloatingPoint(y)) => {
                Ok(InternalValue::FloatingPoint(*x * *y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot multiply values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn div(&self, other: &InternalValue) -> Result<InternalValue, EvaluationError> {
        match (self, other) {
            (InternalValue::Integer(_), InternalValue::Integer(0)) => {
                Err(EvaluationError::DivisionByZero)
            }
            (InternalValue::Integer(x), InternalValue::Integer(y)) => {
                Ok(InternalValue::Integer(x / y))
            }
            (InternalValue::FloatingPoint(x), InternalValue::Integer(y)) => {
                Ok(InternalValue::FloatingPoint(x / *y as f64))
            }
            (InternalValue::Integer(x), InternalValue::FloatingPoint(y)) => {
                Ok(InternalValue::FloatingPoint(*x as f64 / y))
            }
            (InternalValue::FloatingPoint(x), InternalValue::FloatingPoint(y)) => {
                Ok(InternalValue::FloatingPoint(*x / *y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot divide values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn equal(&self, other: &InternalValue) -> Result<InternalValue, EvaluationError> {
        match (self, other) {
            (InternalValue::Integer(x), InternalValue::Integer(y)) => {
                Ok(InternalValue::Boolean(x == y))
            }
            (InternalValue::Boolean(x), InternalValue::Boolean(y)) => {
                Ok(InternalValue::Boolean(x == y))
            }
            (InternalValue::String(x), InternalValue::String(y)) => {
                Ok(InternalValue::Boolean(x == y))
            }
            (InternalValue::Struct(t1, v1), InternalValue::Struct(t2, v2)) if t1 == t2 => {
                Ok(InternalValue::Boolean(v1 == v2))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot compare values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn not_equal(&self, other: &InternalValue) -> Result<InternalValue, EvaluationError> {
        match (self, other) {
            (InternalValue::Integer(x), InternalValue::Integer(y)) => {
                Ok(InternalValue::Boolean(x != y))
            }
            (InternalValue::Boolean(x), InternalValue::Boolean(y)) => {
                Ok(InternalValue::Boolean(x != y))
            }
            (InternalValue::String(x), InternalValue::String(y)) => {
                Ok(InternalValue::Boolean(x != y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot compare values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn greater_than(&self, other: &InternalValue) -> Result<InternalValue, EvaluationError> {
        match (self, other) {
            (InternalValue::Integer(x), InternalValue::Integer(y)) => {
                Ok(InternalValue::Boolean(x > y))
            }
            (InternalValue::FloatingPoint(x), InternalValue::FloatingPoint(y)) => {
                Ok(InternalValue::Boolean(x > y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot compare values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn greater_than_equal(
        &self,
        other: &InternalValue,
    ) -> Result<InternalValue, EvaluationError> {
        match (self, other) {
            (InternalValue::Integer(x), InternalValue::Integer(y)) => {
                Ok(InternalValue::Boolean(x >= y))
            }
            (InternalValue::FloatingPoint(x), InternalValue::FloatingPoint(y)) => {
                Ok(InternalValue::Boolean(x >= y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot compare values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn boolean_and(&self, other: &InternalValue) -> Result<InternalValue, EvaluationError> {
        match (self, other) {
            (InternalValue::Boolean(x), InternalValue::Boolean(y)) => {
                Ok(InternalValue::Boolean(*x && *y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot apply 'and' to values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn boolean_or(&self, other: &InternalValue) -> Result<InternalValue, EvaluationError> {
        match (self, other) {
            (InternalValue::Boolean(x), InternalValue::Boolean(y)) => {
                Ok(InternalValue::Boolean(*x || *y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot apply 'or' to values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn boolean_xor(&self, other: &InternalValue) -> Result<InternalValue, EvaluationError> {
        match (self, other) {
            (InternalValue::Boolean(x), InternalValue::Boolean(y)) => {
                Ok(InternalValue::Boolean(*x ^ *y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot apply 'xor' to values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn boolean_nor(&self, other: &InternalValue) -> Result<InternalValue, EvaluationError> {
        match (self, other) {
            (InternalValue::Boolean(x), InternalValue::Boolean(y)) => {
                Ok(InternalValue::Boolean(!(*x || *y)))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot apply 'nor' to values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn indexing(&self, other: &InternalValue) -> Result<InternalValue, EvaluationError> {
        match (self, other) {
            (InternalValue::Array(_, array), InternalValue::Integer(index)) => {
                if let Some(value) = array.borrow().get(*index as usize) {
                    Ok(value.clone())
                } else {
                    Err(EvaluationError::ArrayIndexOutOfBounds(
                        array.borrow().len(),
                        *index,
                    ))
                }
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot index with values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }
}

impl FunctionDeclaration {
    pub fn make_closure_immediate_value(f: Self, env: &Rc<Env<InternalValue>>) -> InternalValue {
        let (arg_names, arg_types): (Vec<_>, Vec<_>) = f.args.into_iter().unzip();
        InternalValue::Closure(
            Type::Closure(arg_types, Box::new(f.return_type)),
            Rc::clone(env),
            arg_names,
            f.block,
        )
    }
}

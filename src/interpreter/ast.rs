use crate::interpreter::{environment::Env, typechecking::Type};
use std::{cell::RefCell, rc::Rc};

use super::evaluator::EvaluationError;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct SourceInfo {
    pub line: u64,
    pub column: u64,
    pub offset_in_source: usize,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanEqual,
    LessThan,
    LessThanEqual,
    And,
    Or,
    Nor,
    Xor,
    Indexing,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum AssignmentOperator {
    Equal,
    AddEqual,
    SubEqual,
    MulEqual,
    DivEqual,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum UnaryOperator {
    Not,
    Negation,
}

type InternalArrayRepresentation = Rc<RefCell<Vec<ImmediateValue>>>;

#[derive(Debug, PartialEq, Clone)]
pub enum ImmediateValue {
    Integer(i64),
    FloatingPoint(f64),
    String(String),
    Boolean(bool),
    Closure(Type, Rc<Env<ImmediateValue>>, Vec<String>, Box<Statement>),
    Array(Type, InternalArrayRepresentation),
    NativeFunction(
        Type,
        fn(Vec<ImmediateValue>) -> Result<ImmediateValue, EvaluationError>,
    ),
    Void,
}

#[derive(Debug, PartialEq, Clone)]
pub enum LValue {
    Identifier(String),
    IndexInArray(InternalArrayRepresentation, i64),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    Value(ImmediateValue),
    Identifier(SourceInfo, String),
    BinaryOperation(
        SourceInfo,
        Option<Type>,
        Box<Expression>,
        BinaryOperator,
        Box<Expression>,
    ),
    UnaryOperation(SourceInfo, Option<Type>, UnaryOperator, Box<Expression>),
    ArrayInitializer(SourceInfo, Option<Type>, Vec<Box<Expression>>),
    FunctionCall(SourceInfo, Option<Type>, Box<Expression>, Vec<Expression>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    Declaration(SourceInfo, String, Option<Type>, bool, Expression),
    Assignment(SourceInfo, Expression, AssignmentOperator, Expression),
    If(
        SourceInfo,
        Expression,
        Box<Statement>,
        Option<Box<Statement>>,
    ),
    While(SourceInfo, Expression, Box<Statement>),
    For(
        SourceInfo,
        Box<Statement>,
        Expression,
        Box<Statement>,
        Box<Statement>,
    ),
    InLineExpression(SourceInfo, Expression),
    Return(SourceInfo, Expression),
    Block(SourceInfo, Vec<Statement>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum TopLevelDeclaration {
    Function(SourceInfo, Type, String, Vec<String>, Statement),
}

impl ImmediateValue {
    pub fn get_type(&self) -> Type {
        match self {
            ImmediateValue::Integer(_) => Type::Integer,
            ImmediateValue::FloatingPoint(_) => Type::FloatingPoint,
            ImmediateValue::String(_) => Type::String,
            ImmediateValue::Boolean(_) => Type::Boolean,
            ImmediateValue::Array(array_type, _) => Type::Array(Box::new(array_type.clone())),
            ImmediateValue::Closure(functype, _, _, _)
            | ImmediateValue::NativeFunction(functype, _) => functype.clone(),
            ImmediateValue::Void => Type::Void,
        }
    }

    pub fn add(&self, other: &ImmediateValue) -> Result<ImmediateValue, EvaluationError> {
        match (self, other) {
            (ImmediateValue::Integer(x), ImmediateValue::Integer(y)) => {
                Ok(ImmediateValue::Integer(x + y))
            }
            (ImmediateValue::FloatingPoint(x), ImmediateValue::Integer(y)) => {
                Ok(ImmediateValue::FloatingPoint(x + *y as f64))
            }
            (ImmediateValue::Integer(x), ImmediateValue::FloatingPoint(y)) => {
                Ok(ImmediateValue::FloatingPoint(*x as f64 + y))
            }
            (ImmediateValue::FloatingPoint(x), ImmediateValue::FloatingPoint(y)) => {
                Ok(ImmediateValue::FloatingPoint(*x + *y))
            }
            (ImmediateValue::String(left), ImmediateValue::String(right)) => {
                Ok(ImmediateValue::String(left.to_owned() + right))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot add values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn sub(&self, other: &ImmediateValue) -> Result<ImmediateValue, EvaluationError> {
        match (self, other) {
            (ImmediateValue::Integer(x), ImmediateValue::Integer(y)) => {
                Ok(ImmediateValue::Integer(x - y))
            }
            (ImmediateValue::FloatingPoint(x), ImmediateValue::Integer(y)) => {
                Ok(ImmediateValue::FloatingPoint(x - *y as f64))
            }
            (ImmediateValue::Integer(x), ImmediateValue::FloatingPoint(y)) => {
                Ok(ImmediateValue::FloatingPoint(*x as f64 - y))
            }
            (ImmediateValue::FloatingPoint(x), ImmediateValue::FloatingPoint(y)) => {
                Ok(ImmediateValue::FloatingPoint(*x - *y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot subtract values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn mul(&self, other: &ImmediateValue) -> Result<ImmediateValue, EvaluationError> {
        match (self, other) {
            (ImmediateValue::Integer(x), ImmediateValue::Integer(y)) => {
                Ok(ImmediateValue::Integer(x * y))
            }
            (ImmediateValue::FloatingPoint(x), ImmediateValue::Integer(y)) => {
                Ok(ImmediateValue::FloatingPoint(x * *y as f64))
            }
            (ImmediateValue::Integer(x), ImmediateValue::FloatingPoint(y)) => {
                Ok(ImmediateValue::FloatingPoint(*x as f64 * y))
            }
            (ImmediateValue::FloatingPoint(x), ImmediateValue::FloatingPoint(y)) => {
                Ok(ImmediateValue::FloatingPoint(*x * *y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot multiply values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn div(&self, other: &ImmediateValue) -> Result<ImmediateValue, EvaluationError> {
        match (self, other) {
            (ImmediateValue::Integer(_), ImmediateValue::Integer(0)) => {
                Err(EvaluationError::DivisionByZero)
            }
            (ImmediateValue::Integer(x), ImmediateValue::Integer(y)) => {
                Ok(ImmediateValue::Integer(x / y))
            }
            (ImmediateValue::FloatingPoint(x), ImmediateValue::Integer(y)) => {
                Ok(ImmediateValue::FloatingPoint(x / *y as f64))
            }
            (ImmediateValue::Integer(x), ImmediateValue::FloatingPoint(y)) => {
                Ok(ImmediateValue::FloatingPoint(*x as f64 / y))
            }
            (ImmediateValue::FloatingPoint(x), ImmediateValue::FloatingPoint(y)) => {
                Ok(ImmediateValue::FloatingPoint(*x / *y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot divide values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn equal(&self, other: &ImmediateValue) -> Result<ImmediateValue, EvaluationError> {
        match (self, other) {
            (ImmediateValue::Integer(x), ImmediateValue::Integer(y)) => {
                Ok(ImmediateValue::Boolean(x == y))
            }
            (ImmediateValue::Boolean(x), ImmediateValue::Boolean(y)) => {
                Ok(ImmediateValue::Boolean(x == y))
            }
            (ImmediateValue::String(x), ImmediateValue::String(y)) => {
                Ok(ImmediateValue::Boolean(x == y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot compare values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn not_equal(&self, other: &ImmediateValue) -> Result<ImmediateValue, EvaluationError> {
        match (self, other) {
            (ImmediateValue::Integer(x), ImmediateValue::Integer(y)) => {
                Ok(ImmediateValue::Boolean(x != y))
            }
            (ImmediateValue::Boolean(x), ImmediateValue::Boolean(y)) => {
                Ok(ImmediateValue::Boolean(x != y))
            }
            (ImmediateValue::String(x), ImmediateValue::String(y)) => {
                Ok(ImmediateValue::Boolean(x != y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot compare values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn greater_than(&self, other: &ImmediateValue) -> Result<ImmediateValue, EvaluationError> {
        match (self, other) {
            (ImmediateValue::Integer(x), ImmediateValue::Integer(y)) => {
                Ok(ImmediateValue::Boolean(x > y))
            }
            (ImmediateValue::FloatingPoint(x), ImmediateValue::FloatingPoint(y)) => {
                Ok(ImmediateValue::Boolean(x > y))
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
        other: &ImmediateValue,
    ) -> Result<ImmediateValue, EvaluationError> {
        match (self, other) {
            (ImmediateValue::Integer(x), ImmediateValue::Integer(y)) => {
                Ok(ImmediateValue::Boolean(x >= y))
            }
            (ImmediateValue::FloatingPoint(x), ImmediateValue::FloatingPoint(y)) => {
                Ok(ImmediateValue::Boolean(x >= y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot compare values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn boolean_and(&self, other: &ImmediateValue) -> Result<ImmediateValue, EvaluationError> {
        match (self, other) {
            (ImmediateValue::Boolean(x), ImmediateValue::Boolean(y)) => {
                Ok(ImmediateValue::Boolean(*x && *y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot apply 'and' to values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn boolean_or(&self, other: &ImmediateValue) -> Result<ImmediateValue, EvaluationError> {
        match (self, other) {
            (ImmediateValue::Boolean(x), ImmediateValue::Boolean(y)) => {
                Ok(ImmediateValue::Boolean(*x || *y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot apply 'or' to values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn boolean_xor(&self, other: &ImmediateValue) -> Result<ImmediateValue, EvaluationError> {
        match (self, other) {
            (ImmediateValue::Boolean(x), ImmediateValue::Boolean(y)) => {
                Ok(ImmediateValue::Boolean(*x ^ *y))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot apply 'xor' to values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn boolean_nor(&self, other: &ImmediateValue) -> Result<ImmediateValue, EvaluationError> {
        match (self, other) {
            (ImmediateValue::Boolean(x), ImmediateValue::Boolean(y)) => {
                Ok(ImmediateValue::Boolean(!(*x || *y)))
            }
            (bad_val1, bad_val2) => Err(EvaluationError::FatalError(format!(
                "Cannot apply 'nor' to values of type {:?} and {:?}",
                bad_val1.get_type(),
                bad_val2.get_type()
            ))),
        }
    }

    pub fn indexing(&self, other: &ImmediateValue) -> Result<ImmediateValue, EvaluationError> {
        match (self, other) {
            (ImmediateValue::Array(_, array), ImmediateValue::Integer(index)) => {
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

impl LValue {
    const INTERNAL: SourceInfo = SourceInfo {
        line: 0,
        column: 0,
        offset_in_source: 0,
    };

    pub fn to_expression(&self) -> Expression {
        match self {
            LValue::Identifier(s) => Expression::Identifier(LValue::INTERNAL, s.clone()),
            LValue::IndexInArray(array, index) => Expression::BinaryOperation(
                LValue::INTERNAL,
                None,
                Box::new(Expression::Value(ImmediateValue::Array(
                    Type::Any,
                    Rc::clone(array),
                ))),
                BinaryOperator::Indexing,
                Box::new(Expression::Value(ImmediateValue::Integer(*index))),
            ),
        }
    }
}

use crate::interpreter::{environment::Env, typechecking::Type};
use std::{cell::RefCell, rc::Rc};

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
    NativeFunction(Type, fn(Vec<ImmediateValue>) -> ImmediateValue),
    Void,
}

#[derive(Debug, PartialEq, Clone)]
pub enum LValue {
    Identifier(String),
    IndexInArray(InternalArrayRepresentation, usize),
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
    Function(
        SourceInfo,
        Option<Type>,
        String,
        Vec<(String, Type)>,
        Type,
        Statement,
    ),
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
}

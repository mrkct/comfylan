use crate::interpreter::{environment::Env, typechecking::Type};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

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
type InternalStructRepresentation = Rc<RefCell<HashMap<String, ImmediateValue>>>;

#[derive(Debug, PartialEq, Clone)]
pub enum ImmediateValue {
    Integer(i64),
    FloatingPoint(f64),
    String(String),
    Boolean(bool),
    Closure(Type, Rc<Env<ImmediateValue>>, Vec<String>, Block),
    Array(Type, InternalArrayRepresentation),
    NativeFunction(
        Type,
        fn(Vec<ImmediateValue>) -> Result<ImmediateValue, EvaluationError>,
    ),
    Struct(Type, InternalStructRepresentation),
    Void,
}

#[derive(Debug, PartialEq, Clone)]
pub enum LValue {
    Identifier(String),
    IndexInArray(InternalArrayRepresentation, i64),
    Accessor(InternalStructRepresentation, String),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    Value(ImmediateValue),
    Identifier(SourceInfo, String),
    Accessor(SourceInfo, Box<Expression>, String),
    BinaryOperation(
        SourceInfo,
        Option<Type>,
        Box<Expression>,
        BinaryOperator,
        Box<Expression>,
    ),
    UnaryOperation(SourceInfo, Option<Type>, UnaryOperator, Box<Expression>),
    ArrayInitializer(SourceInfo, Option<Type>, Vec<Box<Expression>>),
    StructInitializer(SourceInfo, String, HashMap<String, Box<Expression>>),
    FunctionCall(SourceInfo, Option<Type>, Box<Expression>, Vec<Expression>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    Declaration(Declaration),
    Assignment(Assignment),
    If(If),
    While(While),
    For(For),
    StatementExpression(StatementExpression),
    Return(Return),
    Block(Block),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Declaration {
    pub _info: SourceInfo,
    pub name: String,
    pub expected_type: Option<Type>,
    pub immutable: bool,
    pub rvalue: Expression,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Assignment {
    pub _info: SourceInfo,
    pub lvalue: Expression, // FIXME: Change the type to be LValue directly
    pub operator: AssignmentOperator,
    pub rvalue: Expression,
}

#[derive(Debug, PartialEq, Clone)]
pub struct If {
    pub _info: SourceInfo,
    pub condition: Expression,
    pub branch_true: Block,
    pub branch_false: Option<Block>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct While {
    pub _info: SourceInfo,
    pub condition: Expression,
    pub body: Block,
}

#[derive(Debug, PartialEq, Clone)]
pub struct For {
    pub _info: SourceInfo,
    pub pre: Block,
    pub condition: Expression,
    pub post: Block,
    pub body: Block,
}

#[derive(Debug, PartialEq, Clone)]
pub struct StatementExpression {
    pub _info: SourceInfo,
    pub expression: Expression,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Return {
    pub _info: SourceInfo,
    pub expression: Expression,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Block {
    pub _info: SourceInfo,
    pub statements: Vec<Statement>,
}

impl From<Declaration> for Statement {
    fn from(value: Declaration) -> Self {
        Statement::Declaration(value)
    }
}

impl From<Assignment> for Statement {
    fn from(value: Assignment) -> Self {
        Statement::Assignment(value)
    }
}

impl From<If> for Statement {
    fn from(value: If) -> Self {
        Statement::If(value)
    }
}

impl From<While> for Statement {
    fn from(value: While) -> Self {
        Statement::While(value)
    }
}

impl From<For> for Statement {
    fn from(value: For) -> Self {
        Statement::For(value)
    }
}

impl From<StatementExpression> for Statement {
    fn from(value: StatementExpression) -> Self {
        Statement::StatementExpression(value)
    }
}

impl From<Return> for Statement {
    fn from(value: Return) -> Self {
        Statement::Return(value)
    }
}

impl From<Block> for Statement {
    fn from(value: Block) -> Self {
        Statement::Block(value)
    }
}

#[derive(Debug, PartialEq)]
pub struct Program {
    pub type_declarations: HashMap<String, TypeDeclaration>,
    pub function_declarations: HashMap<String, FunctionDeclaration>,
}

#[derive(Debug, PartialEq)]
pub struct TypeDeclaration {
    pub info: SourceInfo,
    pub name: String,
    pub fields: HashMap<String, Type>,
}

#[derive(Debug, PartialEq)]
pub struct FunctionDeclaration {
    pub info: SourceInfo,
    pub name: String,
    pub block: Block,
    pub return_type: Type,
    pub args: Vec<(String, Type)>,
}

impl FunctionDeclaration {
    pub fn make_closure_immediate_value(f: Self, env: &Rc<Env<ImmediateValue>>) -> ImmediateValue {
        let (arg_names, arg_types): (Vec<_>, Vec<_>) = f.args.into_iter().unzip();
        ImmediateValue::Closure(
            Type::Closure(arg_types, Box::new(f.return_type)),
            Rc::clone(env),
            arg_names,
            f.block,
        )
    }
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
            ImmediateValue::Struct(t, _) => t.clone(),
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
            (ImmediateValue::Struct(t1, v1), ImmediateValue::Struct(t2, v2)) if t1 == t2 => {
                Ok(ImmediateValue::Boolean(v1 == v2))
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
            LValue::Accessor(struct_repr, field) => Expression::Value(
                struct_repr
                    .borrow()
                    .get(field)
                    .expect("typechecker fail")
                    .clone(),
            ),
        }
    }
}

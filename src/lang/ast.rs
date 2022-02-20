use crate::lang::typechecking::Type;
use std::collections::HashMap;

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

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Integer(i64),
    FloatingPoint(f64),
    Boolean(bool),
    String(String),
    Variable(String),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    Value(Value),
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

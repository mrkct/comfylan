#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Integer,
    FloatingPoint,
    String,
    Boolean,
    Array(Box<Type>),
    Closure(Vec<Type>, Box<Type>),
}
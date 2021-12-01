use super::environment::Env;
use super::typechecking::Type;
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone)]
pub enum ImmediateValue {
    Integer(i64),
    FloatingPoint(f64),
    String(String),
    Boolean(bool),
    Closure(Rc<Env<ImmediateValue>>, Vec<String>),
    Array(Type, Vec<ImmediateValue>),
}

impl ImmediateValue {
    pub fn get_type(&self) -> Type {
        match self {
            ImmediateValue::Integer(_) => Type::Integer,
            ImmediateValue::FloatingPoint(_) => Type::FloatingPoint,
            ImmediateValue::String(_) => Type::String,
            ImmediateValue::Boolean(_) => Type::Boolean,
            _ => panic!("not implemented"),
        }
    }
}

pub enum Expression {
    Value(ImmediateValue),
    Identifier(String),
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Equal(Box<Expression>, Box<Expression>),
    NotEqual(Box<Expression>, Box<Expression>),
    GreaterThan(Box<Expression>, Box<Expression>),
    GreaterThanEqual(Box<Expression>, Box<Expression>),
    LessThan(Box<Expression>, Box<Expression>),
    LessThanEqual(Box<Expression>, Box<Expression>),
    And(Box<Expression>, Box<Expression>),
    Or(Box<Expression>, Box<Expression>),
    Nor(Box<Expression>, Box<Expression>),
    Xor(Box<Expression>, Box<Expression>),
    Not(Box<Expression>),
    FunctionCall(Box<Expression>, Vec<Box<Expression>>),
}

#[derive(Debug, PartialEq)]
pub enum ExpressionEvalError {
    InternalTypecheckingError(String),
    SymbolNotFound(String),
    DivisionByZero,
    NotImplemented,
}

impl Expression {
    pub fn eval(&self, env: &Env<ImmediateValue>) -> Result<ImmediateValue, ExpressionEvalError> {
        match self {
            Expression::Value(immediate_value) => Ok(immediate_value.clone()),
            Expression::Add(left, right) => match (left.eval(env), right.eval(env)) {
                (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::Integer(y))) => {
                    Ok(ImmediateValue::Integer(x + y))
                }
                (Ok(ImmediateValue::FloatingPoint(x)), Ok(ImmediateValue::Integer(y))) => {
                    Ok(ImmediateValue::FloatingPoint(x + y as f64))
                }
                (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::FloatingPoint(y))) => {
                    Ok(ImmediateValue::FloatingPoint(x as f64 + y))
                }
                (Ok(ImmediateValue::FloatingPoint(x)), Ok(ImmediateValue::FloatingPoint(y))) => {
                    Ok(ImmediateValue::FloatingPoint(x + y))
                }
                (Ok(ImmediateValue::String(left)), Ok(ImmediateValue::String(right))) => {
                    Ok(ImmediateValue::String(left + &right))
                }
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => {
                    Err(ExpressionEvalError::InternalTypecheckingError(format!(
                        "Cannot add values of types {:?} and {:?}",
                        left.get_type(),
                        right.get_type()
                    )))
                }
            },
            Expression::Sub(left, right) => match (left.eval(env), right.eval(env)) {
                (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::Integer(y))) => {
                    Ok(ImmediateValue::Integer(x - y))
                }
                (Ok(ImmediateValue::FloatingPoint(x)), Ok(ImmediateValue::Integer(y))) => {
                    Ok(ImmediateValue::FloatingPoint(x - y as f64))
                }
                (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::FloatingPoint(y))) => {
                    Ok(ImmediateValue::FloatingPoint(x as f64 - y))
                }
                (Ok(ImmediateValue::FloatingPoint(x)), Ok(ImmediateValue::FloatingPoint(y))) => {
                    Ok(ImmediateValue::FloatingPoint(x - y))
                }
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => {
                    Err(ExpressionEvalError::InternalTypecheckingError(format!(
                        "Cannot subtract values of types {:?} and {:?}",
                        left.get_type(),
                        right.get_type()
                    )))
                }
            },
            Expression::Mul(left, right) => match (left.eval(env), right.eval(env)) {
                (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::Integer(y))) => {
                    Ok(ImmediateValue::Integer(x * y))
                }
                (Ok(ImmediateValue::FloatingPoint(x)), Ok(ImmediateValue::Integer(y))) => {
                    Ok(ImmediateValue::FloatingPoint(x * y as f64))
                }
                (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::FloatingPoint(y))) => {
                    Ok(ImmediateValue::FloatingPoint(x as f64 * y))
                }
                (Ok(ImmediateValue::FloatingPoint(x)), Ok(ImmediateValue::FloatingPoint(y))) => {
                    Ok(ImmediateValue::FloatingPoint(x * y))
                }
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => {
                    Err(ExpressionEvalError::InternalTypecheckingError(format!(
                        "Cannot multiply values of types {:?} and {:?}",
                        left.get_type(),
                        right.get_type()
                    )))
                }
            },
            Expression::Div(left, right) => match (left.eval(env), right.eval(env)) {
                (
                    Ok(ImmediateValue::Integer(_) | ImmediateValue::FloatingPoint(_)),
                    Ok(ImmediateValue::Integer(0) | ImmediateValue::FloatingPoint(0.0)),
                ) => Err(ExpressionEvalError::DivisionByZero),
                (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::Integer(y))) => {
                    Ok(ImmediateValue::Integer(x / y))
                }
                (Ok(ImmediateValue::FloatingPoint(x)), Ok(ImmediateValue::Integer(y))) => {
                    Ok(ImmediateValue::FloatingPoint(x / y as f64))
                }
                (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::FloatingPoint(y))) => {
                    Ok(ImmediateValue::FloatingPoint(x as f64 / y))
                }
                (Ok(ImmediateValue::FloatingPoint(x)), Ok(ImmediateValue::FloatingPoint(y))) => {
                    Ok(ImmediateValue::FloatingPoint(x / y))
                }
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => {
                    Err(ExpressionEvalError::InternalTypecheckingError(format!(
                        "Cannot divide values of types {:?} and {:?}",
                        left.get_type(),
                        right.get_type()
                    )))
                }
            },
            Expression::Equal(left, right) => match (left.eval(env), right.eval(env)) {
                (Ok(ImmediateValue::Integer(left)), Ok(ImmediateValue::Integer(right))) => {
                    Ok(ImmediateValue::Boolean(left == right))
                }
                (Ok(ImmediateValue::String(left)), Ok(ImmediateValue::String(right))) => {
                    Ok(ImmediateValue::Boolean(left == right))
                }
                (Ok(ImmediateValue::Boolean(left)), Ok(ImmediateValue::Boolean(right))) => {
                    Ok(ImmediateValue::Boolean(left == right))
                }
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => {
                    Err(ExpressionEvalError::InternalTypecheckingError(format!(
                        "Cannot compare types {:?} and {:?} with '=='",
                        left.get_type(),
                        right.get_type()
                    )))
                }
            },
            Expression::NotEqual(left, right) => match (left.eval(env), right.eval(env)) {
                (Ok(ImmediateValue::Integer(left)), Ok(ImmediateValue::Integer(right))) => {
                    Ok(ImmediateValue::Boolean(left != right))
                }
                (Ok(ImmediateValue::String(left)), Ok(ImmediateValue::String(right))) => {
                    Ok(ImmediateValue::Boolean(left != right))
                }
                (Ok(ImmediateValue::Boolean(left)), Ok(ImmediateValue::Boolean(right))) => {
                    Ok(ImmediateValue::Boolean(left != right))
                }
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => {
                    Err(ExpressionEvalError::InternalTypecheckingError(format!(
                        "Cannot compare types {:?} and {:?} with '!='",
                        left.get_type(),
                        right.get_type()
                    )))
                }
            },
            Expression::GreaterThan(left, right) => match (left.eval(env), right.eval(env)) {
                (Ok(ImmediateValue::Integer(left)), Ok(ImmediateValue::Integer(right))) => {
                    Ok(ImmediateValue::Boolean(left > right))
                }
                (
                    Ok(ImmediateValue::FloatingPoint(left)),
                    Ok(ImmediateValue::FloatingPoint(right)),
                ) => Ok(ImmediateValue::Boolean(left > right)),
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => {
                    Err(ExpressionEvalError::InternalTypecheckingError(format!(
                        "Cannot compare types {:?} and {:?} with '>'",
                        left.get_type(),
                        right.get_type()
                    )))
                }
            },
            Expression::GreaterThanEqual(left, right) => match (left.eval(env), right.eval(env)) {
                (Ok(ImmediateValue::Integer(left)), Ok(ImmediateValue::Integer(right))) => {
                    Ok(ImmediateValue::Boolean(left >= right))
                }
                (
                    Ok(ImmediateValue::FloatingPoint(left)),
                    Ok(ImmediateValue::FloatingPoint(right)),
                ) => Ok(ImmediateValue::Boolean(left >= right)),
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => {
                    Err(ExpressionEvalError::InternalTypecheckingError(format!(
                        "Cannot compare types {:?} and {:?} with '>='",
                        left.get_type(),
                        right.get_type()
                    )))
                }
            },
            Expression::LessThan(left, right) => match (left.eval(env), right.eval(env)) {
                (Ok(ImmediateValue::Integer(left)), Ok(ImmediateValue::Integer(right))) => {
                    Ok(ImmediateValue::Boolean(left < right))
                }
                (
                    Ok(ImmediateValue::FloatingPoint(left)),
                    Ok(ImmediateValue::FloatingPoint(right)),
                ) => Ok(ImmediateValue::Boolean(left < right)),
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => {
                    Err(ExpressionEvalError::InternalTypecheckingError(format!(
                        "Cannot compare types {:?} and {:?} with '<'",
                        left.get_type(),
                        right.get_type()
                    )))
                }
            },
            Expression::LessThanEqual(left, right) => match (left.eval(env), right.eval(env)) {
                (Ok(ImmediateValue::Integer(left)), Ok(ImmediateValue::Integer(right))) => {
                    Ok(ImmediateValue::Boolean(left <= right))
                }
                (
                    Ok(ImmediateValue::FloatingPoint(left)),
                    Ok(ImmediateValue::FloatingPoint(right)),
                ) => Ok(ImmediateValue::Boolean(left <= right)),
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => {
                    Err(ExpressionEvalError::InternalTypecheckingError(format!(
                        "Cannot compare types {:?} and {:?} with '<='",
                        left.get_type(),
                        right.get_type()
                    )))
                }
            },
            Expression::And(left, right) => match (left.eval(env), right.eval(env)) {
                (Ok(ImmediateValue::Boolean(left)), Ok(ImmediateValue::Boolean(right))) => {
                    Ok(ImmediateValue::Boolean(left && right))
                }
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => {
                    Err(ExpressionEvalError::InternalTypecheckingError(format!(
                        "Cannot compare types {:?} and {:?} with 'and'",
                        left.get_type(),
                        right.get_type()
                    )))
                }
            },
            Expression::Or(left, right) => match (left.eval(env), right.eval(env)) {
                (Ok(ImmediateValue::Boolean(left)), Ok(ImmediateValue::Boolean(right))) => {
                    Ok(ImmediateValue::Boolean(left || right))
                }
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => {
                    Err(ExpressionEvalError::InternalTypecheckingError(format!(
                        "Cannot compare types {:?} and {:?} with 'or'",
                        left.get_type(),
                        right.get_type()
                    )))
                }
            },
            Expression::Xor(left, right) => match (left.eval(env), right.eval(env)) {
                (Ok(ImmediateValue::Boolean(left)), Ok(ImmediateValue::Boolean(right))) => {
                    Ok(ImmediateValue::Boolean(left ^ right))
                }
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => {
                    Err(ExpressionEvalError::InternalTypecheckingError(format!(
                        "Cannot compare types {:?} and {:?} with 'xor'",
                        left.get_type(),
                        right.get_type()
                    )))
                }
            },
            Expression::Nor(left, right) => match (left.eval(env), right.eval(env)) {
                (Ok(ImmediateValue::Boolean(left)), Ok(ImmediateValue::Boolean(right))) => {
                    Ok(ImmediateValue::Boolean(!(left || right)))
                }
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => {
                    Err(ExpressionEvalError::InternalTypecheckingError(format!(
                        "Cannot compare types {:?} and {:?} with 'xor'",
                        left.get_type(),
                        right.get_type()
                    )))
                }
            },
            Expression::Not(left) => match left.eval(env) {
                Ok(ImmediateValue::Boolean(value)) => Ok(ImmediateValue::Boolean(!value)),
                Err(error) => Err(error),
                Ok(value) => Err(ExpressionEvalError::InternalTypecheckingError(format!(
                    "Cannot negate a value of type {:?}",
                    value.get_type(),
                ))),
            },
            Expression::Identifier(name) => match env.lookup(name) {
                Some(value) => Ok(value),
                None => Err(ExpressionEvalError::SymbolNotFound(name.to_string())),
            },
            _ => Err(ExpressionEvalError::NotImplemented),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intval(i: i64) -> Box<Expression> {
        Box::new(Expression::Value(ImmediateValue::Integer(i)))
    }
    fn floatval(i: f64) -> Box<Expression> {
        Box::new(Expression::Value(ImmediateValue::FloatingPoint(i)))
    }
    fn ident(s: &str) -> Box<Expression> {
        Box::new(Expression::Identifier(s.to_string()))
    }
    fn boolval(b: bool) -> Box<Expression> {
        Box::new(Expression::Value(ImmediateValue::Boolean(b)))
    }

    #[test]
    fn simple_addition() {
        let e = Expression::Add(intval(1), intval(2));
        let env = Env::root_env(&[]);
        assert_eq!(e.eval(&env), Ok(ImmediateValue::Integer(3)));
    }

    #[test]
    fn simple_addition_with_identifier() {
        let e = Expression::Add(intval(1), Box::new(Expression::Identifier("x".to_string())));
        let env = Env::root_env(&[]);
        let _ = env.declare("x", ImmediateValue::Integer(2), true);
        assert_eq!(e.eval(&env), Ok(ImmediateValue::Integer(3)));
    }

    #[test]
    fn divide_by_zero() {
        let e = Expression::Div(intval(1), intval(0));
        let env = Env::root_env(&[]);
        assert_eq!(e.eval(&env), Err(ExpressionEvalError::DivisionByZero));
    }

    #[test]
    fn complex_integer_math_expression_without_identifiers() {
        let e = Expression::Sub(
            Box::new(Expression::Mul(intval(3), intval(7))),
            Box::new(Expression::Div(intval(77), intval(10))),
        );
        let env = Env::root_env(&[]);
        assert_eq!(e.eval(&env), Ok(ImmediateValue::Integer(14)));
    }

    #[test]
    fn int_and_float_automatic_casting() {
        let e = Expression::Sub(
            Box::new(Expression::Mul(intval(3), floatval(2.0))),
            intval(3),
        );
        let env = Env::root_env(&[]);
        assert_eq!(e.eval(&env), Ok(ImmediateValue::FloatingPoint(3.0)));
    }

    #[test]
    fn equality() {
        let e = Expression::Sub(
            Box::new(Expression::Mul(intval(3), floatval(2.0))),
            intval(3),
        );
        let env = Env::root_env(&[]);
        assert_eq!(e.eval(&env), Ok(ImmediateValue::FloatingPoint(3.0)));
    }

    #[test]
    fn very_complex_bool_expression_without_identifiers() {
        // (false nor 1 > 5) or ((7.30 - 7.28 < 0.05 or 10 < 11 or 11 <= 11 or 11 >= 11 or 10 > 10) and (not (1 > 3 ) xor false))
        let e = Box::new(Expression::Or(
            // (false nor 1 > 5)
            Box::new(Expression::Nor(
                boolval(false),
                Box::new(Expression::GreaterThan(intval(1), intval(5))),
            )),
            // ((7.30 - 7.28 < 0.05 or 10 < 11 or 11 <= 11 or 11 >= 11 or 10 > 10) and (not (1 > 3 ) xor false))
            Box::new(Expression::And(
                Box::new(Expression::Or(
                    // 7.30 - 7.28 < 0.05
                    Box::new(Expression::LessThan(
                        Box::new(Expression::Sub(floatval(7.30), floatval(7.28))),
                        floatval(0.05),
                    )),
                    Box::new(Expression::Or(
                        // 10 < 11
                        Box::new(Expression::LessThan(intval(10), intval(11))),
                        Box::new(Expression::Or(
                            // 11 <= 11
                            Box::new(Expression::LessThanEqual(intval(11), intval(11))),
                            Box::new(Expression::Or(
                                // 11 >= 11
                                Box::new(Expression::GreaterThanEqual(intval(11), intval(11))),
                                // 10 > 10
                                Box::new(Expression::GreaterThan(intval(10), intval(10))),
                            )),
                        )),
                    )),
                )),
                // (not (1 > 3 ) xor false)
                Box::new(Expression::Xor(
                    Box::new(Expression::Not(Box::new(Expression::GreaterThan(
                        intval(1),
                        intval(3),
                    )))),
                    boolval(false),
                )),
            )),
        ));
        let env = Env::root_env(&[]);
        assert_eq!(e.eval(&env), Ok(ImmediateValue::Boolean(true)));
    }
}

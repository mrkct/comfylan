use crate::interpreter::{ast::*, environment::*};

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Void,
    Integer,
    FloatingPoint,
    String,
    Boolean,
    Array(Box<Type>),
    Closure(Vec<Type>, Box<Type>),
    UserDefined(String),
    Any,
}

impl Type {
    pub fn is_subtype_of(&self, other: &Type) -> bool {
        // TODO: Actually implement this
        self == other || other == &Type::Any
    }
}

impl BinaryOperator {
    pub fn get_type(&self, left: &Type, right: &Type) -> Option<Type> {
        match self {
            BinaryOperator::Add => match (left, right) {
                (Type::Integer, Type::Integer) => Some(Type::Integer),
                (Type::FloatingPoint, Type::Integer | Type::FloatingPoint) => {
                    Some(Type::FloatingPoint)
                }
                (Type::Integer, Type::FloatingPoint) => Some(Type::FloatingPoint),
                (Type::String, Type::String) => Some(Type::String),
                _ => None,
            },
            BinaryOperator::Sub | BinaryOperator::Mul | BinaryOperator::Div => {
                match (left, right) {
                    (Type::Integer, Type::Integer) => Some(Type::Integer),
                    (Type::FloatingPoint, Type::Integer | Type::FloatingPoint) => {
                        Some(Type::FloatingPoint)
                    }
                    (Type::Integer, Type::FloatingPoint) => Some(Type::FloatingPoint),
                    _ => None,
                }
            }
            BinaryOperator::LessThan
            | BinaryOperator::LessThanEqual
            | BinaryOperator::GreaterThan
            | BinaryOperator::GreaterThanEqual => match (left, right) {
                (Type::Integer | Type::FloatingPoint, Type::Integer | Type::FloatingPoint) => {
                    Some(Type::Boolean)
                }
                _ => None,
            },
            BinaryOperator::Equal | BinaryOperator::NotEqual => {
                if left == right {
                    Some(Type::Boolean)
                } else {
                    None
                }
            }
            BinaryOperator::Or
            | BinaryOperator::And
            | BinaryOperator::Nor
            | BinaryOperator::Xor => match (left, right) {
                (Type::Boolean, Type::Boolean) => Some(Type::Boolean),
                _ => None,
            },
            BinaryOperator::Indexing => match (left, right) {
                (Type::Array(array_type), Type::Integer) => Some(*array_type.clone()),
                _ => None,
            },
        }
    }
}

impl UnaryOperator {
    pub fn get_type(&self, value: &Type) -> Option<Type> {
        match (self, value) {
            (UnaryOperator::Negation, Type::Integer) => Some(Type::Integer),
            (UnaryOperator::Not, Type::Boolean) => Some(Type::Boolean),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeError {
    UndeclaredSymbolInExpression(String),
    MismatchedTypes(Type, Type),
    CannotApplyBinaryOperation(Type, BinaryOperator, Type),
    CannotApplyUnaryOperation(UnaryOperator, Type),
    WrongArgumentNumberToFunctionCall(usize, usize),
}

pub fn eval_type_of_expression(
    env: &Env<Type>,
    expression: &Expression,
) -> Result<Type, Vec<TypeError>> {
    match expression {
        Expression::Identifier(_, symbol) => env
            .cloning_lookup(symbol)
            .ok_or_else(|| vec![TypeError::UndeclaredSymbolInExpression(symbol.clone())]),
        Expression::Value(v) => Ok(v.get_type()),
        Expression::ArrayInitializer(_, Some(expected_type), expressions) => {
            match eval_types_or_collect_errors(env, expressions) {
                Err(errors) => Err(errors),
                Ok(types) => {
                    let type_errors = types
                        .iter()
                        .filter_map(|t| {
                            if t.is_subtype_of(expected_type) {
                                None
                            } else {
                                Some(TypeError::MismatchedTypes(expected_type.clone(), t.clone()))
                            }
                        })
                        .collect::<Vec<_>>();
                    if type_errors.is_empty() {
                        Ok(Type::Array(Box::new(expected_type.clone())))
                    } else {
                        Err(type_errors)
                    }
                }
            }
        }
        Expression::ArrayInitializer(_, None, expressions) => {
            match eval_types_or_collect_errors(env, expressions) {
                Err(errors) => Err(errors),
                Ok(_types) => {
                    // TODO: Find out the closest parent type between all of the types
                    panic!("Type inference is not implemented yet");
                }
            }
        }
        Expression::BinaryOperation(_, expected_type, left, op, right) => {
            match (
                eval_type_of_expression(env, left),
                eval_type_of_expression(env, right),
            ) {
                (Ok(left), Ok(right)) => match (expected_type, op.get_type(&left, &right)) {
                    (_, None) => Err(vec![TypeError::CannotApplyBinaryOperation(
                        left, *op, right,
                    )]),
                    (None, Some(t)) => Ok(t),
                    (Some(expected), Some(actual)) => {
                        if expected == &actual {
                            Ok(actual)
                        } else {
                            Err(vec![TypeError::MismatchedTypes(expected.clone(), actual)])
                        }
                    }
                },
                (error @ Err(_), Ok(_)) | (Ok(_), error @ Err(_)) => error,
                (Err(mut left), Err(mut right)) => {
                    left.append(&mut right);
                    Err(left)
                }
            }
        }
        Expression::UnaryOperation(_, expected_type, op, expr) => {
            match eval_type_of_expression(env, expr) {
                Ok(t) => match (expected_type, op.get_type(&t)) {
                    (_, None) => Err(vec![TypeError::CannotApplyUnaryOperation(*op, t)]),
                    (None, Some(t)) => Ok(t),
                    (Some(expected), Some(actual)) => {
                        if expected == &actual {
                            Ok(actual)
                        } else {
                            Err(vec![TypeError::MismatchedTypes(expected.clone(), actual)])
                        }
                    }
                },
                error => error,
            }
        }
        Expression::FunctionCall(_, _, function, args) => {
            match eval_type_of_expression(env, function) {
                Ok(Type::Closure(arg_types, return_type)) => {
                    if arg_types.len() != args.len() {
                        return Err(vec![TypeError::WrongArgumentNumberToFunctionCall(
                            arg_types.len(),
                            args.len(),
                        )]);
                    }

                    let mut errors = vec![];
                    for (expected, expr) in arg_types.iter().zip(args) {
                        match (expected, eval_type_of_expression(env, expr)) {
                            (expected, Ok(actual)) if expected == &actual => {}
                            (expected, Ok(actual)) => {
                                errors.push(TypeError::MismatchedTypes(expected.clone(), actual));
                            }
                            (_, Err(mut error)) => {
                                errors.append(&mut error);
                            }
                        }
                    }
                    if errors.is_empty() {
                        Ok(*return_type)
                    } else {
                        Err(errors)
                    }
                }
                Ok(not_a_closure_type) => Err(vec![TypeError::MismatchedTypes(
                    Type::Closure(vec![], Box::new(Type::Any)),
                    not_a_closure_type,
                )]),
                error => error,
            }
        }
    }
}

fn eval_types_or_collect_errors(
    env: &Env<Type>,
    expressions: &[Box<Expression>],
) -> Result<Vec<Type>, Vec<TypeError>> {
    let mut collected_types = vec![];
    collected_types.reserve_exact(expressions.len());
    let mut collected_errors = vec![];

    for e in expressions {
        match eval_type_of_expression(env, e) {
            Ok(t) if collected_errors.is_empty() => collected_types.push(t),
            Err(mut errors) => collected_errors.append(&mut errors),
            _ => {}
        }
    }

    if collected_errors.is_empty() {
        Ok(collected_types)
    } else {
        Err(collected_errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INFO: SourceInfo = SourceInfo {
        line: 0,
        column: 0,
        offset_in_source: 0,
    };

    #[test]
    fn simple_expression() {
        let e = Expression::BinaryOperation(
            INFO,
            None,
            Box::new(Expression::Value(ImmediateValue::Integer(2))),
            BinaryOperator::Add,
            Box::new(Expression::Value(ImmediateValue::FloatingPoint(3.14))),
        );
        let env = Env::empty();
        assert_eq!(eval_type_of_expression(&env, &e), Ok(Type::FloatingPoint));
    }

    #[test]
    fn add_integer_and_bool_var_fails() {
        let env = Env::empty();
        env.declare("x", Type::Boolean, false);

        let e = Expression::BinaryOperation(
            INFO,
            None,
            Box::new(Expression::Value(ImmediateValue::Integer(2))),
            BinaryOperator::Add,
            Box::new(Expression::Identifier(INFO, "x".to_string())),
        );
        assert_eq!(
            eval_type_of_expression(&env, &e),
            Err(vec![TypeError::CannotApplyBinaryOperation(
                Type::Integer,
                BinaryOperator::Add,
                Type::Boolean
            )])
        );
    }

    #[test]
    fn or_between_bool_and_index_of_array_of_bools() {
        let env = Env::empty();
        env.declare("x", Type::Array(Box::new(Type::Boolean)), false);

        let e = Expression::BinaryOperation(
            INFO,
            None,
            Box::new(Expression::Value(ImmediateValue::Boolean(true))),
            BinaryOperator::Or,
            Box::new(Expression::BinaryOperation(
                INFO,
                None,
                Box::new(Expression::Identifier(INFO, "x".to_string())),
                BinaryOperator::Indexing,
                Box::new(Expression::Value(ImmediateValue::Integer(1))),
            )),
        );
        assert_eq!(eval_type_of_expression(&env, &e), Ok(Type::Boolean));
    }
}

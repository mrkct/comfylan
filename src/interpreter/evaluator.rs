use super::typechecking::Type;
use crate::interpreter::{
    ast::*,
    environment::{Env, EnvError},
};
use std::{cell::RefCell, rc::Rc};

#[derive(Debug, PartialEq)]
pub enum EvaluationError {
    DivisionByZero(SourceInfo),
    ArrayIndexOutOfBounds(SourceInfo, usize, usize),
    NotImplemented,
}

impl Expression {
    pub fn eval(&self, env: &Env<ImmediateValue>) -> Result<ImmediateValue, EvaluationError> {
        match self {
            Expression::Value(immediate_value) => Ok(immediate_value.clone()),
            Expression::BinaryOperation(info, _, left, BinaryOperator::Add, right) => {
                match (left.eval(env), right.eval(env)) {
                    (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::Integer(y))) => {
                        Ok(ImmediateValue::Integer(x + y))
                    }
                    (Ok(ImmediateValue::FloatingPoint(x)), Ok(ImmediateValue::Integer(y))) => {
                        Ok(ImmediateValue::FloatingPoint(x + y as f64))
                    }
                    (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::FloatingPoint(y))) => {
                        Ok(ImmediateValue::FloatingPoint(x as f64 + y))
                    }
                    (
                        Ok(ImmediateValue::FloatingPoint(x)),
                        Ok(ImmediateValue::FloatingPoint(y)),
                    ) => Ok(ImmediateValue::FloatingPoint(x + y)),
                    (Ok(ImmediateValue::String(left)), Ok(ImmediateValue::String(right))) => {
                        Ok(ImmediateValue::String(left + &right))
                    }
                    (Err(error), _) | (_, Err(error)) => Err(error),
                    (Ok(left), Ok(right)) => panic!(
                        "[{:?}]: Cannot add values of types {:?} and {:?}",
                        info,
                        left.get_type(),
                        right.get_type()
                    ),
                }
            }
            Expression::BinaryOperation(info, _, left, BinaryOperator::Sub, right) => {
                match (left.eval(env), right.eval(env)) {
                    (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::Integer(y))) => {
                        Ok(ImmediateValue::Integer(x - y))
                    }
                    (Ok(ImmediateValue::FloatingPoint(x)), Ok(ImmediateValue::Integer(y))) => {
                        Ok(ImmediateValue::FloatingPoint(x - y as f64))
                    }
                    (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::FloatingPoint(y))) => {
                        Ok(ImmediateValue::FloatingPoint(x as f64 - y))
                    }
                    (
                        Ok(ImmediateValue::FloatingPoint(x)),
                        Ok(ImmediateValue::FloatingPoint(y)),
                    ) => Ok(ImmediateValue::FloatingPoint(x - y)),
                    (Err(error), _) | (_, Err(error)) => Err(error),
                    (Ok(left), Ok(right)) => panic!(
                        "[{:?}]: Cannot subtract values of types {:?} and {:?}",
                        info,
                        left.get_type(),
                        right.get_type()
                    ),
                }
            }
            Expression::BinaryOperation(info, _, left, BinaryOperator::Mul, right) => {
                match (left.eval(env), right.eval(env)) {
                    (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::Integer(y))) => {
                        Ok(ImmediateValue::Integer(x * y))
                    }
                    (Ok(ImmediateValue::FloatingPoint(x)), Ok(ImmediateValue::Integer(y))) => {
                        Ok(ImmediateValue::FloatingPoint(x * y as f64))
                    }
                    (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::FloatingPoint(y))) => {
                        Ok(ImmediateValue::FloatingPoint(x as f64 * y))
                    }
                    (
                        Ok(ImmediateValue::FloatingPoint(x)),
                        Ok(ImmediateValue::FloatingPoint(y)),
                    ) => Ok(ImmediateValue::FloatingPoint(x * y)),
                    (Err(error), _) | (_, Err(error)) => Err(error),
                    (Ok(left), Ok(right)) => panic!(
                        "[{:?}]: Cannot multiply values of types {:?} and {:?}",
                        info,
                        left.get_type(),
                        right.get_type()
                    ),
                }
            }
            Expression::BinaryOperation(info, _, left, BinaryOperator::Div, right) => {
                match (left.eval(env), right.eval(env)) {
                    (
                        Ok(ImmediateValue::Integer(_) | ImmediateValue::FloatingPoint(_)),
                        Ok(ImmediateValue::Integer(0) | ImmediateValue::FloatingPoint(0.0)),
                    ) => Err(EvaluationError::DivisionByZero(*info)),
                    (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::Integer(y))) => {
                        Ok(ImmediateValue::Integer(x / y))
                    }
                    (Ok(ImmediateValue::FloatingPoint(x)), Ok(ImmediateValue::Integer(y))) => {
                        Ok(ImmediateValue::FloatingPoint(x / y as f64))
                    }
                    (Ok(ImmediateValue::Integer(x)), Ok(ImmediateValue::FloatingPoint(y))) => {
                        Ok(ImmediateValue::FloatingPoint(x as f64 / y))
                    }
                    (
                        Ok(ImmediateValue::FloatingPoint(x)),
                        Ok(ImmediateValue::FloatingPoint(y)),
                    ) => Ok(ImmediateValue::FloatingPoint(x / y)),
                    (Err(error), _) | (_, Err(error)) => Err(error),
                    (Ok(left), Ok(right)) => panic!(
                        "[{:?}]: Cannot divide values of types {:?} and {:?}",
                        info,
                        left.get_type(),
                        right.get_type()
                    ),
                }
            }
            Expression::BinaryOperation(info, _, left, BinaryOperator::Equal, right) => {
                match (left.eval(env), right.eval(env)) {
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
                    (Ok(left), Ok(right)) => panic!(
                        "[{:?}]: Cannot compare values of types {:?} and {:?}",
                        info,
                        left.get_type(),
                        right.get_type()
                    ),
                }
            }
            Expression::BinaryOperation(info, _, left, BinaryOperator::NotEqual, right) => {
                match (left.eval(env), right.eval(env)) {
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
                    (Ok(left), Ok(right)) => panic!(
                        "[{:?}]: Cannot compare values of types {:?} and {:?}",
                        info,
                        left.get_type(),
                        right.get_type()
                    ),
                }
            }
            Expression::BinaryOperation(info, _, left, BinaryOperator::GreaterThan, right) => {
                match (left.eval(env), right.eval(env)) {
                    (Ok(ImmediateValue::Integer(left)), Ok(ImmediateValue::Integer(right))) => {
                        Ok(ImmediateValue::Boolean(left > right))
                    }
                    (
                        Ok(ImmediateValue::FloatingPoint(left)),
                        Ok(ImmediateValue::FloatingPoint(right)),
                    ) => Ok(ImmediateValue::Boolean(left > right)),
                    (Err(error), _) | (_, Err(error)) => Err(error),
                    (Ok(left), Ok(right)) => panic!(
                        "[{:?}]: Cannot compare values of types {:?} and {:?}",
                        info,
                        left.get_type(),
                        right.get_type()
                    ),
                }
            }
            Expression::BinaryOperation(info, _, left, BinaryOperator::GreaterThanEqual, right) => {
                match (left.eval(env), right.eval(env)) {
                    (Ok(ImmediateValue::Integer(left)), Ok(ImmediateValue::Integer(right))) => {
                        Ok(ImmediateValue::Boolean(left >= right))
                    }
                    (
                        Ok(ImmediateValue::FloatingPoint(left)),
                        Ok(ImmediateValue::FloatingPoint(right)),
                    ) => Ok(ImmediateValue::Boolean(left >= right)),
                    (Err(error), _) | (_, Err(error)) => Err(error),
                    (Ok(left), Ok(right)) => panic!(
                        "[{:?}]: Cannot compare values of types {:?} and {:?}",
                        info,
                        left.get_type(),
                        right.get_type()
                    ),
                }
            }
            Expression::BinaryOperation(info, _, left, BinaryOperator::LessThan, right) => {
                match (left.eval(env), right.eval(env)) {
                    (Ok(ImmediateValue::Integer(left)), Ok(ImmediateValue::Integer(right))) => {
                        Ok(ImmediateValue::Boolean(left < right))
                    }
                    (
                        Ok(ImmediateValue::FloatingPoint(left)),
                        Ok(ImmediateValue::FloatingPoint(right)),
                    ) => Ok(ImmediateValue::Boolean(left < right)),
                    (Err(error), _) | (_, Err(error)) => Err(error),
                    (Ok(left), Ok(right)) => panic!(
                        "[{:?}]: Cannot compare values of types {:?} and {:?}",
                        info,
                        left.get_type(),
                        right.get_type()
                    ),
                }
            }
            Expression::BinaryOperation(info, _, left, BinaryOperator::LessThanEqual, right) => {
                match (left.eval(env), right.eval(env)) {
                    (Ok(ImmediateValue::Integer(left)), Ok(ImmediateValue::Integer(right))) => {
                        Ok(ImmediateValue::Boolean(left <= right))
                    }
                    (
                        Ok(ImmediateValue::FloatingPoint(left)),
                        Ok(ImmediateValue::FloatingPoint(right)),
                    ) => Ok(ImmediateValue::Boolean(left <= right)),
                    (Err(error), _) | (_, Err(error)) => Err(error),
                    (Ok(left), Ok(right)) => panic!(
                        "[{:?}]: Cannot compare values of types {:?} and {:?}",
                        info,
                        left.get_type(),
                        right.get_type()
                    ),
                }
            }
            Expression::BinaryOperation(info, _, left, BinaryOperator::And, right) => {
                match (left.eval(env), right.eval(env)) {
                    (Ok(ImmediateValue::Boolean(left)), Ok(ImmediateValue::Boolean(right))) => {
                        Ok(ImmediateValue::Boolean(left && right))
                    }
                    (Err(error), _) | (_, Err(error)) => Err(error),
                    (Ok(left), Ok(right)) => panic!(
                    "[{:?}]: Cannot apply boolean operator 'and' on values of types {:?} and {:?}",
                    info,
                    left.get_type(),
                    right.get_type()
                ),
                }
            }
            Expression::BinaryOperation(info, _, left, BinaryOperator::Or, right) => {
                match (left.eval(env), right.eval(env)) {
                    (Ok(ImmediateValue::Boolean(left)), Ok(ImmediateValue::Boolean(right))) => {
                        Ok(ImmediateValue::Boolean(left || right))
                    }
                    (Err(error), _) | (_, Err(error)) => Err(error),
                    (Ok(left), Ok(right)) => panic!(
                    "[{:?}]: Cannot apply boolean operator 'or' on values of types {:?} and {:?}",
                    info,
                    left.get_type(),
                    right.get_type()
                ),
                }
            }
            Expression::BinaryOperation(info, _, left, BinaryOperator::Xor, right) => {
                match (left.eval(env), right.eval(env)) {
                    (Ok(ImmediateValue::Boolean(left)), Ok(ImmediateValue::Boolean(right))) => {
                        Ok(ImmediateValue::Boolean(left ^ right))
                    }
                    (Err(error), _) | (_, Err(error)) => Err(error),
                    (Ok(left), Ok(right)) => panic!(
                    "[{:?}]: Cannot apply boolean operator 'xor' on values of types {:?} and {:?}",
                    info,
                    left.get_type(),
                    right.get_type()
                ),
                }
            }
            Expression::BinaryOperation(info, _, left, BinaryOperator::Nor, right) => {
                match (left.eval(env), right.eval(env)) {
                    (Ok(ImmediateValue::Boolean(left)), Ok(ImmediateValue::Boolean(right))) => {
                        Ok(ImmediateValue::Boolean(!(left || right)))
                    }
                    (Err(error), _) | (_, Err(error)) => Err(error),
                    (Ok(left), Ok(right)) => panic!(
                    "[{:?}]: Cannot apply boolean operator 'nor' on values of types {:?} and {:?}",
                    info,
                    left.get_type(),
                    right.get_type()
                ),
                }
            }
            Expression::BinaryOperation(info, _, left, BinaryOperator::Indexing, right) => {
                match (left.eval(env), right.eval(env)) {
                    (Ok(ImmediateValue::Array(_, array)), Ok(ImmediateValue::Integer(index))) => {
                        if let Some(value) = array.borrow().get(index as usize) {
                            Ok(value.clone())
                        } else {
                            Err(EvaluationError::ArrayIndexOutOfBounds(
                                *info,
                                array.borrow().len(),
                                index as usize,
                            ))
                        }
                    }
                    (Ok(not_an_array), Ok(not_an_index)) => {
                        panic!(
                            "[{:?}]: Cannot index a value of type {:?} with a value of type {:?}",
                            info,
                            not_an_array.get_type(),
                            not_an_index.get_type()
                        )
                    }
                    (Err(e), _) => Err(e),
                    (_, Err(e)) => Err(e),
                }
            }
            Expression::UnaryOperation(info, _, UnaryOperator::Not, left) => match left.eval(env) {
                Ok(ImmediateValue::Boolean(value)) => Ok(ImmediateValue::Boolean(!value)),
                Err(error) => Err(error),
                Ok(value) => panic!(
                    "[{:?}]: Cannot apply boolean operator 'not' on value of type {:?}",
                    info,
                    value.get_type(),
                ),
            },
            Expression::FunctionCall(info, _, function, args) => match function.eval(env) {
                Ok(ImmediateValue::Closure(_, closure_env, args_names, statements)) => {
                    let function_context_env = Env::create_child(&closure_env);
                    for (argname, argexpr) in args_names.iter().zip(args.iter()) {
                        let argvalue = argexpr.eval(env)?;
                        print!("{}:{:?}", argname, argvalue);
                        function_context_env.declare(argname, argvalue, false);
                    }
                    statements
                        .eval(&function_context_env)
                        .map(|v| v.expect("Function returning void was used in an expression"))
                }
                Ok(value) => panic!(
                    "[{:?}]: Cannot use value of types {:?} as a function",
                    info,
                    value.get_type(),
                ),
                error => error,
            },
            Expression::Identifier(info, name) => match env.cloning_lookup(name) {
                Some(value) => Ok(value),
                None => panic!(
                    "[{:?}]: Symbol '{}' was not found in the current scope",
                    info, name
                ),
            },
            _ => Err(EvaluationError::NotImplemented),
        }
    }

    pub fn eval_to_lvalue(&self, env: &Env<ImmediateValue>) -> Result<LValue, EvaluationError> {
        match self {
            Expression::Identifier(_, symbol) => Ok(LValue::Identifier(symbol.clone())),
            Expression::BinaryOperation(info, _, array, BinaryOperator::Indexing, index) => {
                let array = array.eval(env)?;
                let index = index.eval(env)?;
                match (array, index) {
                    (ImmediateValue::Array(_, arr), ImmediateValue::Integer(index)) => {
                        Ok(LValue::IndexInArray(arr, index as usize))
                    }
                    (not_an_array, not_an_index) => {
                        panic!(
                            "[{:?}]: Cannot index a value of type {:?} with a value of type {:?} and use it as an lvalue",
                            info,
                            not_an_array.get_type(),
                            not_an_index.get_type()
                        );
                    }
                }
            }
            other => panic!("Expression {:?} is not an lvalue", other),
        }
    }
}

impl Statement {
    pub fn eval(
        &self,
        env: &Rc<Env<ImmediateValue>>,
    ) -> Result<Option<ImmediateValue>, EvaluationError> {
        match self {
            Statement::Declaration(_, symbol, _, immutable, expr) => match expr.eval(env) {
                Ok(value) => {
                    env.declare(symbol, value, *immutable);
                    Ok(None)
                }
                Err(error) => Err(error),
            },
            Statement::Assignment(info, left, operator, right) => {
                let rvalue = right.eval(env)?;
                let lvalue = left.eval_to_lvalue(env)?;
                match lvalue {
                    LValue::Identifier(symbol) => {
                        if env.assign(&symbol, rvalue).is_err() {
                            panic!("Assignment to non-declared variable {}", symbol);
                        }
                        Ok(None)
                    }
                    LValue::IndexInArray(array, index) => {
                        let array_len = array.borrow().len();
                        match array.borrow_mut().get_mut(index) {
                            Some(array_cell) => {
                                *array_cell = rvalue;
                                Ok(None)
                            }
                            None => Err(EvaluationError::ArrayIndexOutOfBounds(
                                *info, array_len, index,
                            )),
                        }
                    }
                }
            }
            Statement::Block(_, statements) => {
                let child_env = Env::create_child(env);
                let mut last_returned_value = None;
                for statement in statements {
                    last_returned_value = statement.eval(&child_env)?;
                }
                Ok(last_returned_value)
            }
            Statement::Return(_, expr) => Ok(Some(expr.eval(env)?)),
            Statement::If(info, condition, branch_true, branch_false) => {
                match condition.eval(env) {
                    Ok(ImmediateValue::Boolean(true)) => branch_true.eval(env),
                    Ok(ImmediateValue::Boolean(false)) => match branch_false {
                        Some(branch_false) => branch_false.eval(env),
                        None => Ok(None),
                    },
                    Err(error) => Err(error),
                    Ok(value) => panic!(
                        "[{:?}]: Expected type boolean in 'if' condition, got {:?} instead",
                        info,
                        value.get_type()
                    ),
                }
            }
            Statement::While(info, condition, repeat) => {
                loop {
                    match condition.eval(env) {
                        Ok(ImmediateValue::Boolean(true)) => {
                            repeat.eval(env)?;
                        }
                        Ok(ImmediateValue::Boolean(false)) => {
                            break;
                        }
                        Ok(value) => {
                            panic!("[{:?}]: Expected type boolean in 'while' condition, got {:?} instead", info, value.get_type());
                        }
                        Err(error) => {
                            return Err(error);
                        }
                    }
                }
                Ok(None)
            }
            Statement::For(info, pre, condition, post, repeat) => {
                pre.eval(env)?;
                loop {
                    match condition.eval(env) {
                        Ok(ImmediateValue::Boolean(true)) => {
                            repeat.eval(env)?;
                            post.eval(env)?;
                        }
                        Ok(ImmediateValue::Boolean(false)) => {
                            break;
                        }
                        Ok(value) => {
                            panic!("[{:?}]: Expected type boolean in 'for' condition, got {:?} instead", info, value.get_type());
                        }
                        Err(error) => {
                            return Err(error);
                        }
                    }
                }
                post.eval(env)?;
                Ok(None)
            }
            Statement::InLineExpression(_, expr) => {
                expr.eval(env)?;
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::matches;

    const INFO: SourceInfo = SourceInfo {
        line: 0,
        column: 0,
        offset_in_source: 0,
    };

    fn intval(i: i64) -> Box<Expression> {
        Box::new(Expression::Value(ImmediateValue::Integer(i)))
    }
    fn floatval(i: f64) -> Box<Expression> {
        Box::new(Expression::Value(ImmediateValue::FloatingPoint(i)))
    }
    fn ident(s: &str) -> Box<Expression> {
        Box::new(Expression::Identifier(INFO, s.to_string()))
    }
    fn boolval(b: bool) -> Box<Expression> {
        Box::new(Expression::Value(ImmediateValue::Boolean(b)))
    }
    fn binop(left: Box<Expression>, op: BinaryOperator, right: Box<Expression>) -> Box<Expression> {
        Box::new(Expression::BinaryOperation(INFO, None, left, op, right))
    }
    fn array(atype: Type, values: Vec<ImmediateValue>) -> Box<Expression> {
        Box::new(Expression::Value(ImmediateValue::Array(
            atype,
            Rc::new(RefCell::new(values)),
        )))
    }
    fn declare(name: &str, val: i64) -> Box<Statement> {
        Box::new(Statement::Declaration(
            INFO,
            name.to_string(),
            None,
            false,
            *intval(val),
        ))
    }

    #[test]
    fn simple_addition() {
        let e = binop(intval(1), BinaryOperator::Add, intval(2));
        let env = Env::root_env(&[]);
        assert_eq!(e.eval(&env), Ok(ImmediateValue::Integer(3)));
    }

    #[test]
    fn simple_addition_with_identifier() {
        let e = binop(intval(1), BinaryOperator::Add, ident("x"));
        let env = Env::root_env(&[]);
        let _ = env.declare("x", ImmediateValue::Integer(2), true);
        assert_eq!(e.eval(&env), Ok(ImmediateValue::Integer(3)));
    }

    #[test]
    fn divide_by_zero() {
        let e = Expression::BinaryOperation(INFO, None, intval(1), BinaryOperator::Div, intval(0));
        let env = Env::root_env(&[]);
        assert!(matches!(
            e.eval(&env),
            Err(EvaluationError::DivisionByZero(_))
        ));
    }

    #[test]
    fn complex_integer_math_expression_without_identifiers() {
        let e = binop(
            binop(intval(3), BinaryOperator::Mul, intval(7)),
            BinaryOperator::Sub,
            binop(intval(77), BinaryOperator::Div, intval(10)),
        );
        let env = Env::root_env(&[]);
        assert_eq!(e.eval(&env), Ok(ImmediateValue::Integer(14)));
    }

    #[test]
    fn int_and_float_automatic_casting() {
        let e = binop(
            binop(intval(3), BinaryOperator::Mul, floatval(2.0)),
            BinaryOperator::Sub,
            intval(3),
        );
        let env = Env::root_env(&[]);
        assert_eq!(e.eval(&env), Ok(ImmediateValue::FloatingPoint(3.0)));
    }

    #[test]
    fn equality() {
        let e = binop(
            binop(intval(3), BinaryOperator::Mul, floatval(2.0)),
            BinaryOperator::Sub,
            intval(3),
        );
        let env = Env::root_env(&[]);
        assert_eq!(e.eval(&env), Ok(ImmediateValue::FloatingPoint(3.0)));
    }

    #[test]
    fn very_complex_bool_expression_without_identifiers() {
        // (false nor 1 > 5) or ((7.30 - 7.28 < 0.05 or 10 < 11 or 11 <= 11 or 11 >= 11 or 10 > 10) and (not (1 > 3 ) xor false))
        let e = binop(
            // (false nor 1 > 5)
            binop(
                boolval(false),
                BinaryOperator::Nor,
                binop(intval(1), BinaryOperator::GreaterThan, intval(5)),
            ),
            BinaryOperator::Or,
            // ((7.30 - 7.28 < 0.05 or 10 < 11 or 11 <= 11 or 11 >= 11 or 10 > 10) and (not (1 > 3 ) xor false))
            binop(
                binop(
                    // 7.30 - 7.28 < 0.05
                    binop(
                        binop(floatval(7.30), BinaryOperator::Sub, floatval(7.28)),
                        BinaryOperator::LessThan,
                        floatval(0.05),
                    ),
                    BinaryOperator::Or,
                    binop(
                        // 10 < 11
                        binop(intval(10), BinaryOperator::LessThan, intval(11)),
                        BinaryOperator::Or,
                        binop(
                            // 11 <= 11
                            binop(intval(11), BinaryOperator::LessThanEqual, intval(11)),
                            BinaryOperator::Or,
                            binop(
                                // 11 >= 11
                                binop(intval(11), BinaryOperator::GreaterThanEqual, intval(11)),
                                BinaryOperator::Or,
                                // 10 > 10
                                binop(intval(10), BinaryOperator::GreaterThan, intval(10)),
                            ),
                        ),
                    ),
                ),
                BinaryOperator::And,
                // (not (1 > 3 ) xor false)
                binop(
                    Box::new(Expression::UnaryOperation(
                        INFO,
                        None,
                        UnaryOperator::Not,
                        binop(intval(1), BinaryOperator::GreaterThan, intval(3)),
                    )),
                    BinaryOperator::Xor,
                    boolval(false),
                ),
            ),
        );
        let env = Env::root_env(&[]);
        assert_eq!(e.eval(&env), Ok(ImmediateValue::Boolean(true)));
    }

    #[test]
    fn array_indexing() {
        let e = binop(
            array(
                Type::Integer,
                vec![
                    ImmediateValue::Integer(7),
                    ImmediateValue::Integer(8),
                    ImmediateValue::Integer(9),
                ],
            ),
            BinaryOperator::Indexing,
            intval(2),
        );
        let env = Env::root_env(&[]);
        assert_eq!(e.eval(&env), Ok(ImmediateValue::Integer(9)));
    }

    #[test]
    fn array_indexing_out_of_bounds() {
        let e = binop(
            array(
                Type::Integer,
                vec![
                    ImmediateValue::Integer(7),
                    ImmediateValue::Integer(8),
                    ImmediateValue::Integer(9),
                ],
            ),
            BinaryOperator::Indexing,
            intval(3),
        );
        let env = Env::root_env(&[]);
        assert!(matches!(
            e.eval(&env),
            Err(EvaluationError::ArrayIndexOutOfBounds(_, 3, 3))
        ));
    }

    #[test]
    fn array_cell_as_lvalue() {
        let env = Env::root_env(&[]);
        env.declare(
            "array",
            ImmediateValue::Array(
                Type::Integer,
                Rc::new(RefCell::new(vec![
                    ImmediateValue::Integer(7),
                    ImmediateValue::Integer(8),
                    ImmediateValue::Integer(9), // this element will be incremented by 1
                ])),
            ),
            false,
        );

        let program = Statement::Block(
            INFO,
            vec![Statement::Assignment(
                INFO,
                *binop(
                    ident("array"),
                    BinaryOperator::Indexing,
                    binop(intval(1), BinaryOperator::Add, intval(1)),
                ),
                AssignmentOperator::Equal,
                *binop(
                    intval(1),
                    BinaryOperator::Add,
                    binop(
                        ident("array"),
                        BinaryOperator::Indexing,
                        binop(intval(1), BinaryOperator::Add, intval(1)),
                    ),
                ),
            )],
        );

        assert_eq!(program.eval(&env), Ok(None));
        assert_eq!(
            env.cloning_lookup("array"),
            Some(ImmediateValue::Array(
                Type::Integer,
                Rc::new(RefCell::new(vec![
                    ImmediateValue::Integer(7),
                    ImmediateValue::Integer(8),
                    ImmediateValue::Integer(10)
                ]))
            ))
        );
    }

    #[test]
    fn sum_of_first_10_values() {
        let program = Statement::Block(
            INFO,
            vec![
                *declare("x", 0),
                Statement::For(
                    INFO,
                    declare("i", 0),
                    *binop(ident("i"), BinaryOperator::LessThan, intval(10)),
                    Box::new(Statement::Assignment(
                        INFO,
                        *ident("i"),
                        AssignmentOperator::Equal,
                        *binop(ident("i"), BinaryOperator::Add, intval(1)),
                    )),
                    Box::new(Statement::Assignment(
                        INFO,
                        *ident("x"),
                        AssignmentOperator::Equal,
                        *binop(ident("x"), BinaryOperator::Add, ident("i")),
                    )),
                ),
                Statement::Return(INFO, *ident("x")),
            ],
        );
        let env = Env::root_env(&[]);
        assert_eq!(program.eval(&env), Ok(Some(ImmediateValue::Integer(45))));
    }

    #[test]
    fn simple_function_call() {
        let env = Env::root_env(&[]);
        let function = Box::new(Statement::Block(
            INFO,
            vec![Statement::Return(
                INFO,
                *binop(ident("x"), BinaryOperator::Add, intval(1)),
            )],
        ));
        env.declare(
            "add_one",
            ImmediateValue::Closure(Type::Void, Rc::clone(&env), vec!["x".to_string()], function),
            true,
        );
        let program = Statement::Declaration(
            INFO,
            "result".to_string(),
            None,
            true,
            Expression::FunctionCall(INFO, None, ident("add_one"), vec![*intval(2)]),
        );

        assert_eq!(program.eval(&env), Ok(None));
        assert_eq!(
            env.cloning_lookup("result"),
            Some(ImmediateValue::Integer(3))
        );
    }

    #[test]
    fn recursive_factorial_function_call() {
        let env = Env::root_env(&[]);
        let function = Box::new(Statement::Block(
            INFO,
            vec![Statement::If(
                INFO,
                *binop(ident("x"), BinaryOperator::Equal, intval(0)),
                Box::new(Statement::Return(INFO, *intval(1))),
                Some(Box::new(Statement::Return(
                    INFO,
                    *binop(
                        Box::new(Expression::FunctionCall(
                            INFO,
                            None,
                            ident("factorial"),
                            vec![*binop(ident("x"), BinaryOperator::Sub, intval(1))],
                        )),
                        BinaryOperator::Mul,
                        ident("x"),
                    ),
                ))),
            )],
        ));
        env.declare(
            "factorial",
            ImmediateValue::Closure(Type::Void, Rc::clone(&env), vec!["x".to_string()], function),
            true,
        );
        let program = Statement::Declaration(
            INFO,
            "result".to_string(),
            None,
            true,
            Expression::FunctionCall(INFO, None, ident("factorial"), vec![*intval(7)]),
        );

        assert_eq!(program.eval(&env), Ok(None));
        assert_eq!(
            env.cloning_lookup("result"),
            Some(ImmediateValue::Integer(5040))
        );
    }
}

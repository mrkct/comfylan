use crate::interpreter::{ast::*, environment::*};
use std::{borrow::BorrowMut, rc::Rc};

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
    CannotFindCommonType(Vec<Type>),
    MismatchedReturnType(Type, Type),
}

fn find_closest_common_parent_type(types: &[Type]) -> Option<Type> {
    // TODO: This is a super stupid and bad implementation
    if types.is_empty() {
        return None;
    }

    let mut highest_type = types.first().unwrap();
    for t in types {
        if highest_type.is_subtype_of(t) {
            highest_type = t;
        }
    }

    for t in types {
        if !t.is_subtype_of(highest_type) {
            return None;
        }
    }
    Some(highest_type.clone())
}

fn eval_type_of_expression(
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

fn typecheck_or_collect_errors(
    env: &mut Rc<Env<Type>>,
    statement: &Statement,
    errors: &mut Vec<TypeError>,
) {
    if let Err(mut e) = typecheck_statement(env, statement) {
        errors.append(&mut e);
    }
}

fn typecheck_statement(
    env: &mut Rc<Env<Type>>,
    statement: &Statement,
) -> Result<Option<Type>, Vec<TypeError>> {
    let verify_expr_type_or_collect_errors =
        |env, expr, expected_type, errors: &mut Vec<TypeError>| {
            match eval_type_of_expression(env, expr) {
                Ok(t) if t.is_subtype_of(&expected_type) => {}
                Ok(t) => {
                    errors.push(TypeError::MismatchedTypes(expected_type, t));
                }
                Err(mut e) => {
                    errors.append(&mut e);
                }
            };
        };

    match statement {
        Statement::Declaration(_, name, expected_type, _, expression) => {
            let actual_type = eval_type_of_expression(env, expression)?;
            match (expected_type, actual_type) {
                (Some(expected), actual) if actual.is_subtype_of(expected) => {
                    env.borrow_mut().declare(name, expected.clone());
                    Ok(None)
                }
                (Some(expected), actual) => {
                    Err(vec![TypeError::MismatchedTypes(expected.clone(), actual)])
                }
                (None, actual) => {
                    env.borrow_mut().declare(name, actual);
                    Ok(None)
                }
            }
        }
        Statement::Assignment(_, lvalue, op, rvalue) => {
            match (
                eval_type_of_expression(env, lvalue),
                eval_type_of_expression(env, rvalue),
            ) {
                (Ok(left), Ok(right)) => match op {
                    AssignmentOperator::Equal => {
                        if right.is_subtype_of(&left) {
                            Ok(None)
                        } else {
                            Err(vec![TypeError::MismatchedTypes(left, right)])
                        }
                    }
                    AssignmentOperator::AddEqual => {
                        if BinaryOperator::Add.get_type(&left, &right).is_some() {
                            Ok(None)
                        } else {
                            Err(vec![TypeError::CannotApplyBinaryOperation(
                                left,
                                BinaryOperator::Add,
                                right,
                            )])
                        }
                    }
                    AssignmentOperator::SubEqual => {
                        if BinaryOperator::Sub.get_type(&left, &right).is_some() {
                            Ok(None)
                        } else {
                            Err(vec![TypeError::CannotApplyBinaryOperation(
                                left,
                                BinaryOperator::Sub,
                                right,
                            )])
                        }
                    }
                    AssignmentOperator::MulEqual => {
                        if BinaryOperator::Mul.get_type(&left, &right).is_some() {
                            Ok(None)
                        } else {
                            Err(vec![TypeError::CannotApplyBinaryOperation(
                                left,
                                BinaryOperator::Mul,
                                right,
                            )])
                        }
                    }
                    AssignmentOperator::DivEqual => {
                        if BinaryOperator::Div.get_type(&left, &right).is_some() {
                            Ok(None)
                        } else {
                            Err(vec![TypeError::CannotApplyBinaryOperation(
                                left,
                                BinaryOperator::Div,
                                right,
                            )])
                        }
                    }
                },
                (Err(e), Ok(_)) | (Ok(_), Err(e)) => Err(e),
                (Err(mut e1), Err(mut e2)) => {
                    e1.append(&mut e2);
                    Err(e1)
                }
            }
        }
        Statement::If(_, expected_bool_expr, branch_true, branch_false) => {
            let mut errors = vec![];
            verify_expr_type_or_collect_errors(env, expected_bool_expr, Type::Boolean, &mut errors);
            {
                let mut child = Env::create_child(env);
                typecheck_or_collect_errors(&mut child, branch_true, &mut errors);
            }
            {
                let mut child = Env::create_child(env);
                if let Some(branch_false) = branch_false {
                    typecheck_or_collect_errors(&mut child, branch_false, &mut errors);
                }
            }

            if errors.is_empty() {
                Ok(None)
            } else {
                Err(errors)
            }
        }
        Statement::While(_, expected_bool_expr, repeating_statement) => {
            let mut errors = vec![];
            verify_expr_type_or_collect_errors(env, expected_bool_expr, Type::Boolean, &mut errors);
            {
                let mut child = Env::create_child(env);
                typecheck_or_collect_errors(&mut child, repeating_statement, &mut errors);
            }

            if errors.is_empty() {
                Ok(None)
            } else {
                Err(errors)
            }
        }
        Statement::For(_, pre, expected_bool_expr, post, repeating_statement) => {
            let mut errors = vec![];
            {
                let mut child = Env::create_child(env);
                typecheck_or_collect_errors(&mut child, pre, &mut errors);
                verify_expr_type_or_collect_errors(
                    &mut child,
                    expected_bool_expr,
                    Type::Boolean,
                    &mut errors,
                );
                typecheck_or_collect_errors(&mut child, post, &mut errors);
                typecheck_or_collect_errors(&mut child, repeating_statement, &mut errors);
            }
            if errors.is_empty() {
                Ok(None)
            } else {
                Err(errors)
            }
        }
        Statement::InLineExpression(_, expr) => match eval_type_of_expression(env, expr) {
            Ok(_) => Ok(None),
            Err(errors) => Err(errors),
        },
        Statement::Return(_, expr) => match eval_type_of_expression(env, expr) {
            Ok(t) => Ok(Some(t)),
            Err(errors) => Err(errors),
        },
        Statement::Block(_, statements) => {
            let mut collected_errors = vec![];
            let mut collected_return_types = vec![];

            for statement in statements {
                match typecheck_statement(env, statement) {
                    Ok(Some(t)) => collected_return_types.push(t),
                    Ok(None) => {}
                    Err(mut errors) => collected_errors.append(&mut errors),
                }
            }

            if collected_errors.is_empty() {
                if collected_return_types.is_empty() {
                    Ok(Some(Type::Void))
                } else if let Some(common_type) =
                    find_closest_common_parent_type(&collected_return_types)
                {
                    Ok(Some(common_type))
                } else {
                    Err(vec![TypeError::CannotFindCommonType(
                        collected_return_types,
                    )])
                }
            } else {
                Err(collected_errors)
            }
        }
    }
}

fn typecheck_top_level_declaration(
    env: &Rc<Env<Type>>,
    def: &TopLevelDeclaration,
) -> Result<(), Vec<TypeError>> {
    match def {
        TopLevelDeclaration::Function(
            _,
            Type::Closure(arg_types, return_type),
            _,
            arg_names,
            statement,
        ) => {
            let mut child = Env::create_child(env);
            for (argname, argtype) in arg_names.iter().zip(arg_types.iter()) {
                child.declare(argname, argtype.clone());
            }
            match typecheck_statement(&mut child, statement) {
                Ok(Some(t)) if t.is_subtype_of(return_type) => Ok(()),
                Ok(Some(t)) => Err(vec![TypeError::MismatchedReturnType(
                    *return_type.clone(),
                    t,
                )]),
                Err(errors) => Err(errors),
                _ => unreachable!(),
            }
        }
        _ => unreachable!(),
    }
}

pub fn typecheck_program(
    global_env: &mut Rc<Env<Type>>,
    top_level_declarations: &[TopLevelDeclaration],
) -> Result<(), Vec<TypeError>> {
    for decl in top_level_declarations {
        match decl {
            TopLevelDeclaration::Function(_, ftype, name, _, _) => {
                global_env.declare(name, ftype.clone());
            }
        }
    }

    let mut errors = vec![];
    for decl in top_level_declarations {
        if let Err(mut e) = typecheck_top_level_declaration(global_env, decl) {
            errors.append(&mut e);
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
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
        env.declare("x", Type::Boolean);

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
        env.declare("x", Type::Array(Box::new(Type::Boolean)));

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

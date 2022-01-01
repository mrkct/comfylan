use crate::interpreter::{ast::*, environment::*};
use std::{borrow::BorrowMut, collections::HashMap, rc::Rc};

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
    Struct(String, HashMap<String, Type>),
    Any,
}

impl Type {
    pub fn is_subtype_of(&self, other: &Type) -> bool {
        // TODO: Actually implement this
        self == other || other == &Type::Any
    }
}

impl From<StructDeclaration> for Type {
    fn from(struct_declaration: StructDeclaration) -> Self {
        let StructDeclaration { name, fields, .. } = struct_declaration;
        Type::Struct(name, HashMap::from_iter(fields))
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
    NoSuchFieldInStruct(Type, String),
    CannotAccessFieldInNonStructType(Type, String),
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

trait Typecheckable {
    fn typecheck(&self, env: &mut Rc<Env<Type>>) -> (Option<Type>, Option<Vec<TypeError>>);
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
        Expression::Accessor(_, struct_expr, field_name) => {
            match eval_type_of_expression(env, &struct_expr) {
                Ok(Type::Struct(struct_name, fields)) => {
                    if let Some(field_type) = fields.get(field_name) {
                        Ok(field_type.clone())
                    } else {
                        Err(vec![TypeError::NoSuchFieldInStruct(
                            Type::Struct(struct_name, fields),
                            field_name.clone(),
                        )])
                    }
                }
                Ok(not_a_struct) => Err(vec![TypeError::CannotAccessFieldInNonStructType(
                    not_a_struct,
                    field_name.clone(),
                )]),
                errors => errors,
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

fn verify_type_or_collect_errors(
    env: &mut Rc<Env<Type>>,
    expr: &Expression,
    expected_type: Type,
    errors: &mut Vec<TypeError>,
) {
    match eval_type_of_expression(env, expr) {
        Ok(t) if t.is_subtype_of(&expected_type) => {}
        Ok(t) => {
            errors.push(TypeError::MismatchedTypes(expected_type, t));
        }
        Err(mut e) => {
            errors.append(&mut e);
        }
    };
}

impl Typecheckable for Declaration {
    fn typecheck(&self, env: &mut Rc<Env<Type>>) -> (Option<Type>, Option<Vec<TypeError>>) {
        let actual_type = eval_type_of_expression(env, &self.rvalue);
        match (&self.expected_type, actual_type) {
            (_, Err(errors)) => (Some(Type::Void), Some(errors)),
            (Some(expected), Ok(actual)) if actual.is_subtype_of(&expected) => {
                env.borrow_mut().declare(&self.name, expected.clone());
                (Some(Type::Void), None)
            }
            (None, Ok(actual)) => {
                env.borrow_mut().declare(&self.name, actual);
                (Some(Type::Void), None)
            }
            (Some(expected), Ok(actual)) => (
                Some(Type::Void),
                Some(vec![TypeError::MismatchedTypes(expected.clone(), actual)]),
            ),
        }
    }
}

impl Typecheckable for Assignment {
    fn typecheck(&self, env: &mut Rc<Env<Type>>) -> (Option<Type>, Option<Vec<TypeError>>) {
        match (
            eval_type_of_expression(env, &self.lvalue),
            eval_type_of_expression(env, &self.rvalue),
        ) {
            (Ok(left), Ok(right)) => {
                let errors = match self.operator {
                    AssignmentOperator::Equal => {
                        if right.is_subtype_of(&left) {
                            None
                        } else {
                            Some(vec![TypeError::MismatchedTypes(left, right)])
                        }
                    }
                    AssignmentOperator::AddEqual => {
                        if BinaryOperator::Add.get_type(&left, &right).is_some() {
                            None
                        } else {
                            Some(vec![TypeError::CannotApplyBinaryOperation(
                                left,
                                BinaryOperator::Add,
                                right,
                            )])
                        }
                    }
                    AssignmentOperator::SubEqual => {
                        if BinaryOperator::Sub.get_type(&left, &right).is_some() {
                            None
                        } else {
                            Some(vec![TypeError::CannotApplyBinaryOperation(
                                left,
                                BinaryOperator::Sub,
                                right,
                            )])
                        }
                    }
                    AssignmentOperator::MulEqual => {
                        if BinaryOperator::Mul.get_type(&left, &right).is_some() {
                            None
                        } else {
                            Some(vec![TypeError::CannotApplyBinaryOperation(
                                left,
                                BinaryOperator::Mul,
                                right,
                            )])
                        }
                    }
                    AssignmentOperator::DivEqual => {
                        if BinaryOperator::Div.get_type(&left, &right).is_some() {
                            None
                        } else {
                            Some(vec![TypeError::CannotApplyBinaryOperation(
                                left,
                                BinaryOperator::Div,
                                right,
                            )])
                        }
                    }
                };
                (Some(Type::Void), errors)
            }
            (Err(e), Ok(_)) | (Ok(_), Err(e)) => (Some(Type::Void), Some(e)),
            (Err(mut e1), Err(mut e2)) => {
                e1.append(&mut e2);
                (Some(Type::Void), Some(e1))
            }
        }
    }
}

impl Typecheckable for If {
    fn typecheck(&self, env: &mut Rc<Env<Type>>) -> (Option<Type>, Option<Vec<TypeError>>) {
        let mut collected_errors = vec![];
        verify_type_or_collect_errors(env, &self.condition, Type::Boolean, &mut collected_errors);
        {
            let mut child = Env::create_child(env);
            if let (_, Some(mut errors)) = self.branch_true.typecheck(&mut child) {
                collected_errors.append(&mut errors);
            }
        }
        {
            let mut child = Env::create_child(env);
            if let Some(branch_false) = &self.branch_false {
                if let (_, Some(mut errors)) = branch_false.typecheck(&mut child) {
                    collected_errors.append(&mut errors);
                }
            }
        }

        (
            Some(Type::Void),
            if collected_errors.is_empty() {
                None
            } else {
                Some(collected_errors)
            },
        )
    }
}

impl Typecheckable for For {
    fn typecheck(&self, env: &mut Rc<Env<Type>>) -> (Option<Type>, Option<Vec<TypeError>>) {
        let mut collected_errors = vec![];
        let mut child = Env::create_child(env);
        if let (_, Some(mut errors)) = self.pre.typecheck(&mut child) {
            collected_errors.append(&mut errors);
        }

        verify_type_or_collect_errors(env, &self.condition, Type::Boolean, &mut collected_errors);
        if let (_, Some(mut errors)) = self.post.typecheck(&mut child) {
            collected_errors.append(&mut errors);
        }
        if let (_, Some(mut errors)) = self.body.typecheck(&mut child) {
            collected_errors.append(&mut errors);
        }

        (
            Some(Type::Void),
            if collected_errors.is_empty() {
                None
            } else {
                Some(collected_errors)
            },
        )
    }
}

impl Typecheckable for While {
    fn typecheck(&self, env: &mut Rc<Env<Type>>) -> (Option<Type>, Option<Vec<TypeError>>) {
        let mut collected_errors = vec![];
        let mut child = Env::create_child(env);
        verify_type_or_collect_errors(env, &self.condition, Type::Boolean, &mut collected_errors);
        if let (_, Some(mut errors)) = self.body.typecheck(&mut child) {
            collected_errors.append(&mut errors);
        }

        (
            Some(Type::Void),
            if collected_errors.is_empty() {
                None
            } else {
                Some(collected_errors)
            },
        )
    }
}

impl Typecheckable for Block {
    fn typecheck(&self, env: &mut Rc<Env<Type>>) -> (Option<Type>, Option<Vec<TypeError>>) {
        let mut collected_errors = vec![];
        let mut collected_return_types = vec![];

        for statement in &self.statements {
            match statement.typecheck(env) {
                (Some(Type::Void), None) => {}
                (Some(t), None) => collected_return_types.push(t),
                (Some(t), Some(mut errors)) => {
                    collected_errors.append(&mut errors);
                    collected_return_types.push(t);
                }
                _ => unreachable!(),
            }
        }

        let return_type = {
            if collected_return_types.is_empty() {
                Some(Type::Void)
            } else if let Some(common_type) =
                find_closest_common_parent_type(&collected_return_types)
            {
                Some(common_type)
            } else {
                collected_errors.push(TypeError::CannotFindCommonType(collected_return_types));
                None
            }
        };

        (
            return_type,
            if collected_errors.is_empty() {
                None
            } else {
                Some(collected_errors)
            },
        )
    }
}

impl Typecheckable for StatementExpression {
    fn typecheck(&self, env: &mut Rc<Env<Type>>) -> (Option<Type>, Option<Vec<TypeError>>) {
        if let Err(e) = eval_type_of_expression(env, &self.expression) {
            (Some(Type::Void), Some(e))
        } else {
            (Some(Type::Void), None)
        }
    }
}

impl Typecheckable for Return {
    fn typecheck(&self, env: &mut Rc<Env<Type>>) -> (Option<Type>, Option<Vec<TypeError>>) {
        match eval_type_of_expression(env, &self.expression) {
            Ok(t) => (Some(t), None),
            Err(e) => (None, Some(e)),
        }
    }
}

impl Typecheckable for Statement {
    fn typecheck(&self, env: &mut Rc<Env<Type>>) -> (Option<Type>, Option<Vec<TypeError>>) {
        match self {
            Statement::Declaration(x) => x.typecheck(env),
            Statement::Assignment(x) => x.typecheck(env),
            Statement::If(x) => x.typecheck(env),
            Statement::For(x) => x.typecheck(env),
            Statement::While(x) => x.typecheck(env),
            Statement::Return(x) => x.typecheck(env),
            Statement::Block(x) => x.typecheck(env),
            Statement::StatementExpression(x) => x.typecheck(env),
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
            match statement.typecheck(&mut child) {
                (Some(t), None) if t.is_subtype_of(return_type) => Ok(()),
                (Some(t), maybe_errors) => {
                    let mismatched_return_type =
                        TypeError::MismatchedReturnType(*return_type.clone(), t);
                    if let Some(mut errors) = maybe_errors {
                        errors.push(mismatched_return_type);
                        Err(errors)
                    } else {
                        Err(vec![mismatched_return_type])
                    }
                }
                _ => unreachable!(),
            }
        }
        TopLevelDeclaration::Function(_, _, _, _, _) => unreachable!(),
        TopLevelDeclaration::StructDeclaration(_) => Ok(()),
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
            TopLevelDeclaration::StructDeclaration(s @ StructDeclaration { ref name, .. }) => {
                global_env.declare(name, s.clone().into());
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

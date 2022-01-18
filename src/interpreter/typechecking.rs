use crate::interpreter::{ast::*, environment::*, native};
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
    TypeReference(String),
    Struct(String, HashMap<String, Type>),
    VarArgs(Box<Type>),
    Any,
}

impl Type {
    pub fn is_subtype_of(&self, _user_types: &HashMap<String, Type>, other: &Type) -> bool {
        if other == &Type::Any || self == other {
            return true;
        }
        match (self, other) {
            (Type::Array(inner1), Type::Array(inner2)) => inner1.is_subtype_of(_user_types, inner2),
            _ => false,
        }
    }
}

impl From<&TypeDeclaration> for Type {
    fn from(struct_declaration: &TypeDeclaration) -> Self {
        let TypeDeclaration { name, fields, .. } = struct_declaration;
        Type::Struct(name.to_string(), fields.clone())
    }
}

impl From<&FunctionDeclaration> for Type {
    fn from(function_declaration: &FunctionDeclaration) -> Self {
        Type::Closure(
            function_declaration
                .args
                .clone()
                .into_iter()
                .map(|(_, t)| t)
                .collect(),
            Box::new(function_declaration.return_type.clone()),
        )
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
    UndeclaredType(String),
    MissingFieldInStructDeclaration(String, String),
    TooManyFieldsInStructDeclaration(String),
}

fn find_closest_common_parent_type(
    user_types: &HashMap<String, Type>,
    types: &[Type],
) -> Option<Type> {
    // TODO: This is a super stupid and bad implementation
    if types.is_empty() {
        return None;
    }

    let mut highest_type = types.first().unwrap();
    for t in types {
        if highest_type.is_subtype_of(user_types, t) {
            highest_type = t;
        }
    }

    for t in types {
        if !t.is_subtype_of(user_types, highest_type) {
            return None;
        }
    }
    Some(highest_type.clone())
}

trait Typecheckable {
    fn typecheck(
        &self,
        user_types: &HashMap<String, Type>,
        env: &mut Rc<Env<Type>>,
    ) -> (Option<Type>, Option<Vec<TypeError>>);
}

fn eval_type_of_expression(
    user_types: &HashMap<String, Type>,
    env: &Env<Type>,
    expression: &Expression,
) -> Result<Type, Vec<TypeError>> {
    match expression {
        Expression::Identifier(_, symbol) => env
            .cloning_lookup(symbol)
            .ok_or_else(|| vec![TypeError::UndeclaredSymbolInExpression(symbol.clone())]),
        Expression::Value(v) => Ok(v.get_type()),
        Expression::ArrayInitializer(_, Some(expected_type), expressions) => {
            match eval_types_or_collect_errors(user_types, env, expressions) {
                Err(errors) => Err(errors),
                Ok(types) => {
                    let type_errors = types
                        .iter()
                        .filter_map(|t| {
                            if t.is_subtype_of(user_types, expected_type) {
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
            match eval_types_or_collect_errors(user_types, env, expressions) {
                Err(errors) => Err(errors),
                Ok(_types) => {
                    // TODO: Find out the closest parent type between all of the types
                    panic!("Type inference is not implemented yet");
                }
            }
        }
        Expression::BinaryOperation(_, expected_type, left, op, right) => {
            match (
                eval_type_of_expression(user_types, env, left),
                eval_type_of_expression(user_types, env, right),
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
            match eval_type_of_expression(user_types, env, expr) {
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
            match eval_type_of_expression(user_types, env, function) {
                Ok(Type::Closure(arg_types, return_type)) => {
                    let mut collected_errors = vec![];

                    let arg_types_len = arg_types.len();
                    let args_len = args.len();
                    let mut actual_args = args.iter().peekable();

                    for expected in arg_types {
                        if let Type::VarArgs(internal) = expected {
                            while let Some(e) = actual_args.peek() {
                                match eval_type_of_expression(user_types, env, e) {
                                    Ok(actual) if actual.is_subtype_of(user_types, &internal) => {
                                        actual_args.next();
                                    }
                                    Ok(_) => {
                                        break;
                                    }
                                    Err(mut errors) => {
                                        collected_errors.append(&mut errors);
                                    }
                                }
                            }
                        } else {
                            let actual = actual_args.next().ok_or_else(|| {
                                vec![TypeError::WrongArgumentNumberToFunctionCall(
                                    arg_types_len,
                                    args_len,
                                )]
                            })?;

                            match eval_type_of_expression(user_types, env, actual) {
                                Ok(actual) if actual.is_subtype_of(user_types, &expected) => {}
                                Ok(actual) => {
                                    collected_errors
                                        .push(TypeError::MismatchedTypes(expected, actual));
                                }
                                Err(mut errors) => {
                                    collected_errors.append(&mut errors);
                                }
                            }
                        }
                    }

                    if collected_errors.is_empty() {
                        Ok(*return_type)
                    } else {
                        Err(collected_errors)
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
            match eval_type_of_expression(user_types, env, struct_expr) {
                Ok(Type::TypeReference(struct_type_name)) => {
                    match user_types.get(&struct_type_name) {
                        Some(Type::Struct(_, fields)) => {
                            if let Some(field_type) = fields.get(field_name) {
                                Ok(field_type.clone())
                            } else {
                                Err(vec![TypeError::NoSuchFieldInStruct(
                                    Type::TypeReference(struct_type_name),
                                    field_name.to_string(),
                                )])
                            }
                        }
                        Some(_) => panic!(
                            "there is another user-defined type with accessors other than structs?"
                        ),
                        None => Err(vec![TypeError::UndeclaredType(struct_type_name)]),
                    }
                }
                Ok(not_a_struct) => Err(vec![TypeError::CannotAccessFieldInNonStructType(
                    not_a_struct,
                    field_name.to_string(),
                )]),
                Err(errors) => Err(errors),
            }
        }
        Expression::StructInitializer(_, struct_type_name, expected_fields) => {
            match user_types.get(struct_type_name) {
                Some(Type::Struct(_, actual_fields)) => {
                    let mut collected_errors = vec![];
                    for (key, expected_type) in actual_fields {
                        if let Some(expr) = expected_fields.get(key) {
                            match (
                                eval_type_of_expression(user_types, env, expr),
                                expected_type,
                            ) {
                                (Ok(t1), t2) if t1.is_subtype_of(user_types, t2) => {}
                                (Ok(t1), t2) => {
                                    collected_errors
                                        .push(TypeError::MismatchedTypes(t2.clone(), t1));
                                }
                                (Err(mut errors), _) => {
                                    collected_errors.append(&mut errors);
                                }
                            }
                        } else {
                            collected_errors.push(TypeError::MissingFieldInStructDeclaration(
                                struct_type_name.to_string(),
                                key.to_string(),
                            ));
                        }
                    }
                    if actual_fields.keys().count() > expected_fields.keys().count() {
                        collected_errors.push(TypeError::TooManyFieldsInStructDeclaration(
                            struct_type_name.to_string(),
                        ));
                    }

                    if collected_errors.is_empty() {
                        Ok(Type::TypeReference(struct_type_name.clone()))
                    } else {
                        Err(collected_errors)
                    }
                }
                Some(_) => panic!("user_types contains a non-struct type"),
                None => Err(vec![TypeError::UndeclaredType(struct_type_name.clone())]),
            }
        }
    }
}

fn eval_types_or_collect_errors(
    user_types: &HashMap<String, Type>,
    env: &Env<Type>,
    expressions: &[Box<Expression>],
) -> Result<Vec<Type>, Vec<TypeError>> {
    let mut collected_types = vec![];
    collected_types.reserve_exact(expressions.len());
    let mut collected_errors = vec![];

    for e in expressions {
        match eval_type_of_expression(user_types, env, e) {
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
    user_types: &HashMap<String, Type>,
    env: &mut Rc<Env<Type>>,
    expr: &Expression,
    expected_type: Type,
    errors: &mut Vec<TypeError>,
) {
    match eval_type_of_expression(user_types, env, expr) {
        Ok(t) if t.is_subtype_of(user_types, &expected_type) => {}
        Ok(t) => {
            errors.push(TypeError::MismatchedTypes(expected_type, t));
        }
        Err(mut e) => {
            errors.append(&mut e);
        }
    };
}

impl Typecheckable for Declaration {
    fn typecheck(
        &self,
        user_types: &HashMap<String, Type>,
        env: &mut Rc<Env<Type>>,
    ) -> (Option<Type>, Option<Vec<TypeError>>) {
        let actual_type = eval_type_of_expression(user_types, env, &self.rvalue);
        match (&self.expected_type, actual_type) {
            (_, Err(errors)) => (Some(Type::Void), Some(errors)),
            (Some(expected), Ok(actual)) if actual.is_subtype_of(user_types, expected) => {
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
    fn typecheck(
        &self,
        user_types: &HashMap<String, Type>,
        env: &mut Rc<Env<Type>>,
    ) -> (Option<Type>, Option<Vec<TypeError>>) {
        match (
            eval_type_of_expression(user_types, env, &self.lvalue),
            eval_type_of_expression(user_types, env, &self.rvalue),
        ) {
            (Ok(left), Ok(right)) => {
                let errors = match self.operator {
                    AssignmentOperator::Equal => {
                        if right.is_subtype_of(user_types, &left) {
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
    fn typecheck(
        &self,
        user_types: &HashMap<String, Type>,
        env: &mut Rc<Env<Type>>,
    ) -> (Option<Type>, Option<Vec<TypeError>>) {
        let mut collected_errors = vec![];
        verify_type_or_collect_errors(
            user_types,
            env,
            &self.condition,
            Type::Boolean,
            &mut collected_errors,
        );
        {
            let mut child = Env::create_child(env);
            if let (_, Some(mut errors)) = self.branch_true.typecheck(user_types, &mut child) {
                collected_errors.append(&mut errors);
            }
        }
        {
            let mut child = Env::create_child(env);
            if let Some(branch_false) = &self.branch_false {
                if let (_, Some(mut errors)) = branch_false.typecheck(user_types, &mut child) {
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
    fn typecheck(
        &self,
        user_types: &HashMap<String, Type>,
        env: &mut Rc<Env<Type>>,
    ) -> (Option<Type>, Option<Vec<TypeError>>) {
        let mut collected_errors = vec![];
        let mut child = Env::create_child(env);
        if let (_, Some(mut errors)) = self.pre.typecheck(user_types, &mut child) {
            collected_errors.append(&mut errors);
        }

        verify_type_or_collect_errors(
            user_types,
            env,
            &self.condition,
            Type::Boolean,
            &mut collected_errors,
        );
        if let (_, Some(mut errors)) = self.post.typecheck(user_types, &mut child) {
            collected_errors.append(&mut errors);
        }
        if let (_, Some(mut errors)) = self.body.typecheck(user_types, &mut child) {
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
    fn typecheck(
        &self,
        user_types: &HashMap<String, Type>,
        env: &mut Rc<Env<Type>>,
    ) -> (Option<Type>, Option<Vec<TypeError>>) {
        let mut collected_errors = vec![];
        let mut child = Env::create_child(env);
        verify_type_or_collect_errors(
            user_types,
            env,
            &self.condition,
            Type::Boolean,
            &mut collected_errors,
        );
        if let (_, Some(mut errors)) = self.body.typecheck(user_types, &mut child) {
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
    fn typecheck(
        &self,
        user_types: &HashMap<String, Type>,
        env: &mut Rc<Env<Type>>,
    ) -> (Option<Type>, Option<Vec<TypeError>>) {
        let mut collected_errors = vec![];
        let mut collected_return_types = vec![];

        for statement in &self.statements {
            match statement.typecheck(user_types, env) {
                (Some(Type::Void), None) => {}
                (Some(t), None) => collected_return_types.push(t),
                (Some(t), Some(mut errors)) => {
                    collected_errors.append(&mut errors);
                    collected_return_types.push(t);
                }
                (None, Some(mut errors)) => {
                    collected_errors.append(&mut errors);
                }
                (None, None) => {}
            }
        }

        let return_type = {
            if collected_return_types.is_empty() {
                Some(Type::Void)
            } else if let Some(common_type) =
                find_closest_common_parent_type(user_types, &collected_return_types)
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
    fn typecheck(
        &self,
        user_types: &HashMap<String, Type>,
        env: &mut Rc<Env<Type>>,
    ) -> (Option<Type>, Option<Vec<TypeError>>) {
        if let Err(e) = eval_type_of_expression(user_types, env, &self.expression) {
            (Some(Type::Void), Some(e))
        } else {
            (Some(Type::Void), None)
        }
    }
}

impl Typecheckable for Return {
    fn typecheck(
        &self,
        user_types: &HashMap<String, Type>,
        env: &mut Rc<Env<Type>>,
    ) -> (Option<Type>, Option<Vec<TypeError>>) {
        match eval_type_of_expression(user_types, env, &self.expression) {
            Ok(t) => (Some(t), None),
            Err(e) => (None, Some(e)),
        }
    }
}

impl Typecheckable for Statement {
    fn typecheck(
        &self,
        user_types: &HashMap<String, Type>,
        env: &mut Rc<Env<Type>>,
    ) -> (Option<Type>, Option<Vec<TypeError>>) {
        match self {
            Statement::Declaration(x) => x.typecheck(user_types, env),
            Statement::Assignment(x) => x.typecheck(user_types, env),
            Statement::If(x) => x.typecheck(user_types, env),
            Statement::For(x) => x.typecheck(user_types, env),
            Statement::While(x) => x.typecheck(user_types, env),
            Statement::Return(x) => x.typecheck(user_types, env),
            Statement::Block(x) => x.typecheck(user_types, env),
            Statement::StatementExpression(x) => x.typecheck(user_types, env),
        }
    }
}

impl Typecheckable for FunctionDeclaration {
    fn typecheck(
        &self,
        user_types: &HashMap<String, Type>,
        env: &mut Rc<Env<Type>>,
    ) -> (Option<Type>, Option<Vec<TypeError>>) {
        let mut child = Env::create_child(env);
        for (argname, argtype) in &self.args {
            child.declare(argname, argtype.clone());
        }
        match self.block.typecheck(user_types, &mut child) {
            (Some(t), maybe_errors) if t.is_subtype_of(user_types, &self.return_type) => {
                (None, maybe_errors)
            }
            (Some(t), maybe_errors) => {
                let mismatched_return_type =
                    TypeError::MismatchedReturnType(self.return_type.clone(), t);
                if let Some(mut errors) = maybe_errors {
                    errors.push(mismatched_return_type);
                    (None, Some(errors))
                } else {
                    (None, Some(vec![mismatched_return_type]))
                }
            }
            (None, errors @ Some(_)) => (None, errors),
            _ => unreachable!(),
        }
    }
}

pub fn typecheck_program(program: &Program) -> Result<(), Vec<TypeError>> {
    let mut user_types: HashMap<String, Type> = HashMap::new();
    let mut type_env = Env::empty();
    native::fill_type_env_with_native_functions(&type_env);
    for (name, type_decl) in &program.type_declarations {
        user_types.insert(name.to_string(), type_decl.into());
    }
    for (name, func_decl) in &program.function_declarations {
        type_env.declare(name, func_decl.into());
    }

    let mut collected_errors = vec![];
    for func_decl in program.function_declarations.values() {
        if let (_, Some(mut errors)) = func_decl.typecheck(&user_types, &mut type_env) {
            collected_errors.append(&mut errors);
        }
    }

    if collected_errors.is_empty() {
        Ok(())
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
        assert_eq!(
            eval_type_of_expression(&HashMap::new(), &env, &e),
            Ok(Type::FloatingPoint)
        );
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
            eval_type_of_expression(&HashMap::new(), &env, &e),
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
        assert_eq!(
            eval_type_of_expression(&HashMap::new(), &env, &e),
            Ok(Type::Boolean)
        );
    }

    #[test]
    fn undeclared_variable_in_expr() {
        let e = Expression::BinaryOperation(
            INFO,
            None,
            Box::new(Expression::Value(ImmediateValue::Integer(1))),
            BinaryOperator::Add,
            Box::new(Expression::Identifier(INFO, "x".to_string())),
        );
        assert_eq!(
            eval_type_of_expression(&HashMap::new(), &Env::empty(), &e),
            Err(vec![TypeError::UndeclaredSymbolInExpression(
                "x".to_string()
            )])
        );
    }

    #[test]
    fn return_struct() {
        let program = FunctionDeclaration {
            info: INFO,
            name: "get_point".to_string(),
            args: vec![],
            return_type: Type::TypeReference("Point".to_string()),
            block: Block {
                _info: INFO,
                statements: vec![Statement::Return(Return {
                    _info: INFO,
                    expression: Expression::Identifier(INFO, "point_struct".to_string()),
                })],
            },
        };

        let mut env = Env::empty();
        env.declare("point_struct", Type::TypeReference("Point".to_string()));
        let user_types = HashMap::from([(
            "Point".to_string(),
            Type::Struct("Point".to_string(), HashMap::new()),
        )]);

        assert!(program.typecheck(&user_types, &mut env).1.is_none());
    }

    #[test]
    fn return_struct_field() {
        let program = FunctionDeclaration {
            info: INFO,
            name: "get_point".to_string(),
            args: vec![],
            return_type: Type::Integer,
            block: Block {
                _info: INFO,
                statements: vec![Statement::Return(Return {
                    _info: INFO,
                    expression: Expression::Accessor(
                        INFO,
                        Box::new(Expression::Identifier(INFO, "point_struct".to_string())),
                        "x".to_string(),
                    ),
                })],
            },
        };

        let mut env = Env::empty();
        env.declare("point_struct", Type::TypeReference("Point".to_string()));

        let user_types = HashMap::from([(
            "Point".to_string(),
            Type::Struct(
                "Point".to_string(),
                HashMap::from([("x".to_string(), Type::Integer)]),
            ),
        )]);

        assert!(program.typecheck(&user_types, &mut env).1.is_none());
    }

    #[test]
    fn call_function_with_var_args() {
        let program = Statement::StatementExpression(StatementExpression {
            _info: INFO,
            expression: Expression::FunctionCall(
                INFO,
                None,
                Box::new(Expression::Identifier(INFO, "my_func".to_string())),
                vec![
                    Expression::Value(ImmediateValue::Integer(0)),
                    Expression::Value(ImmediateValue::FloatingPoint(0.1)),
                    Expression::Value(ImmediateValue::FloatingPoint(0.2)),
                    Expression::Value(ImmediateValue::FloatingPoint(0.3)),
                    Expression::Value(ImmediateValue::String("4".to_string())),
                    // Intentionally empty
                ],
            ),
        });

        let mut env = Env::empty();
        env.declare(
            "my_func",
            Type::Closure(
                vec![
                    Type::Integer,
                    Type::VarArgs(Box::new(Type::FloatingPoint)),
                    Type::String,
                    Type::VarArgs(Box::new(Type::Integer)),
                ],
                Box::new(Type::Void),
            ),
        );

        assert!(program.typecheck(&HashMap::new(), &mut env).1.is_none());
    }
}

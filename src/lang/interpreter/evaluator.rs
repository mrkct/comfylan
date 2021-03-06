use super::{
    native::{GameEngineSubsystem, NativeFunction},
    value::InternalValue,
};
use crate::lang::{ast::*, environment::Env, typechecking::Type};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

#[derive(Debug, PartialEq)]
pub enum EvaluationError {
    DivisionByZero,
    ArrayIndexOutOfBounds(usize, i64),
    NativeSpecific(String),
    FatalError(String),
}

impl Expression {
    pub fn eval(
        &self,
        subsystem: &mut dyn GameEngineSubsystem,
        env: &Env<InternalValue>,
    ) -> Result<InternalValue, EvaluationError> {
        match self {
            Expression::Value(Value::Variable(name)) => match env.cloning_lookup(name) {
                Some(value) => Ok(value),
                None => panic!("Symbol '{}' was not found in the current scope", name),
            },
            Expression::Value(Value::Integer(x)) => Ok(InternalValue::Integer(*x)),
            Expression::Value(Value::FloatingPoint(x)) => Ok(InternalValue::FloatingPoint(*x)),
            Expression::Value(Value::Boolean(x)) => Ok(InternalValue::Boolean(*x)),
            Expression::Value(Value::String(s)) => Ok(InternalValue::String(s.clone())),
            Expression::BinaryOperation(_info, _, left, operator, right) => {
                match (left.eval(subsystem, env), right.eval(subsystem, env)) {
                    (Ok(left), Ok(right)) => match operator {
                        BinaryOperator::Add => left.add(&right),
                        BinaryOperator::Sub => left.sub(&right),
                        BinaryOperator::Mul => left.mul(&right),
                        BinaryOperator::Div => left.div(&right),
                        BinaryOperator::Equal => left.equal(&right),
                        BinaryOperator::NotEqual => left.not_equal(&right),
                        BinaryOperator::GreaterThan => left.greater_than(&right),
                        BinaryOperator::GreaterThanEqual => left.greater_than_equal(&right),
                        BinaryOperator::LessThan => right.greater_than(&left),
                        BinaryOperator::LessThanEqual => right.greater_than_equal(&left),
                        BinaryOperator::And => left.boolean_and(&right),
                        BinaryOperator::Or => left.boolean_or(&right),
                        BinaryOperator::Nor => left.boolean_nor(&right),
                        BinaryOperator::Xor => left.boolean_xor(&right),
                        BinaryOperator::Indexing => left.indexing(&right),
                    },
                    (error, _) => error,
                }
            }
            Expression::UnaryOperation(info, _, UnaryOperator::Not, left) => {
                match left.eval(subsystem, env) {
                    Ok(InternalValue::Boolean(value)) => Ok(InternalValue::Boolean(!value)),
                    Err(error) => Err(error),
                    Ok(value) => panic!(
                        "[{:?}]: Cannot apply boolean operator 'not' on value of type {:?}",
                        info,
                        value.get_type(),
                    ),
                }
            }
            Expression::UnaryOperation(info, _, UnaryOperator::Negation, left) => {
                match left.eval(subsystem, env) {
                    Ok(InternalValue::Integer(x)) => Ok(InternalValue::Integer(-x)),
                    Ok(InternalValue::FloatingPoint(f)) => Ok(InternalValue::FloatingPoint(-f)),
                    Ok(cant_negate) => {
                        panic!(
                            "[{:?}]: Cannot negate a value of type {:?}",
                            *info,
                            cant_negate.get_type()
                        );
                    }
                    err => err,
                }
            }
            Expression::FunctionCall(info, _, function, args) => {
                match function.eval(subsystem, env) {
                    Ok(InternalValue::Closure(_, closure_env, args_names, statements)) => {
                        let function_context_env = Env::create_child(&closure_env);
                        for (argname, argexpr) in args_names.iter().zip(args.iter()) {
                            let argvalue = argexpr.eval(subsystem, env)?;
                            function_context_env.declare(argname, argvalue);
                        }

                        statements
                            .eval(subsystem, &function_context_env)
                            .map(|v| v.unwrap_or(InternalValue::Void))
                    }
                    Ok(InternalValue::NativeFunction(NativeFunction { callback, .. })) => {
                        let mut evaluated_args = vec![];
                        evaluated_args.reserve_exact(args.len());
                        for e in args {
                            let value = e.eval(subsystem, env)?;
                            evaluated_args.push(value);
                        }
                        callback(subsystem, evaluated_args)
                    }
                    Ok(value) => panic!(
                        "[{:?}]: Cannot use value of types {:?} as a function",
                        info,
                        value.get_type(),
                    ),
                    error => error,
                }
            }
            Expression::ArrayInitializer(_, _, values) => {
                let mut array = vec![];
                array.reserve_exact(values.len());
                for e in values {
                    let value = e.eval(subsystem, env)?;
                    array.push(value);
                }
                Ok(InternalValue::Array(
                    Type::Array(Box::new(Type::Integer)),
                    Rc::new(RefCell::new(array)),
                ))
            }
            Expression::Accessor(info, struct_expr, field) => {
                match struct_expr.eval(subsystem, env) {
                    Ok(InternalValue::Struct(struct_type, contents)) => {
                        if let Some(v) = contents.borrow().get(field) {
                            Ok(v.clone())
                        } else {
                            panic!(
                                "[{:?}]: Tried to access field '{}' of a struct with type {:?}",
                                info, field, struct_type
                            );
                        }
                    }
                    Ok(not_a_struct) => {
                        panic!(
                            "[{:?}]: Tried to access field '{}' of a non-struct value ({:?})",
                            info, field, not_a_struct
                        );
                    }
                    errors => errors,
                }
            }
            Expression::StructInitializer(_, struct_type_name, fields) => {
                let mut f = HashMap::new();
                for (field_name, expr) in fields {
                    let value = expr.eval(subsystem, env)?;
                    f.insert(field_name.clone(), value);
                }
                Ok(InternalValue::Struct(
                    Type::TypeReference(struct_type_name.clone()),
                    Rc::new(RefCell::new(f)),
                ))
            }
        }
    }
}

trait Evaluable {
    fn eval(
        &self,
        subsystem: &mut dyn GameEngineSubsystem,
        env: &Rc<Env<InternalValue>>,
    ) -> Result<Option<InternalValue>, EvaluationError>;
}

impl Evaluable for Declaration {
    fn eval(
        &self,
        subsystem: &mut dyn GameEngineSubsystem,
        env: &Rc<Env<InternalValue>>,
    ) -> Result<Option<InternalValue>, EvaluationError> {
        match self.rvalue.eval(subsystem, env) {
            Ok(value) => {
                env.declare(&self.name, value);
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }
}

impl Evaluable for Assignment {
    fn eval(
        &self,
        subsystem: &mut dyn GameEngineSubsystem,
        env: &Rc<Env<InternalValue>>,
    ) -> Result<Option<InternalValue>, EvaluationError> {
        let rvalue = self.rvalue.eval(subsystem, env)?;

        let calculate_value_to_store = |left: InternalValue, operator, right| match operator {
            AssignmentOperator::Equal => Ok(right),
            AssignmentOperator::AddEqual => left.add(&right),
            AssignmentOperator::SubEqual => left.sub(&right),
            AssignmentOperator::MulEqual => left.mul(&right),
            AssignmentOperator::DivEqual => left.div(&right),
        };

        match &self.lvalue {
            Expression::Value(Value::Variable(symbol)) => {
                let left = env
                    .cloning_lookup(symbol)
                    .expect("Assignment to non-declared variable");
                let to_store = calculate_value_to_store(left, self.operator, rvalue)?;
                if env.assign(symbol, to_store).is_err() {
                    panic!("Assignment to non-declared variable {}", symbol);
                }
                Ok(None)
            }
            Expression::BinaryOperation(_, _, array, BinaryOperator::Indexing, index) => {
                let array = array.eval(subsystem, env)?;
                if let InternalValue::Array(_, array) = array {
                    let array_len = array.borrow().len();
                    let index = index.eval(subsystem, env).map(|v| {
                        if let InternalValue::Integer(x) = v {
                            x
                        } else {
                            panic!("Cannot index array with value {:?}", v);
                        }
                    })?;
                    let uindex: usize = index
                        .try_into()
                        .map_err(|_| EvaluationError::ArrayIndexOutOfBounds(array_len, index))?;

                    match array.borrow_mut().get_mut(uindex) {
                        Some(array_cell) => {
                            *array_cell = calculate_value_to_store(
                                array_cell.clone(),
                                self.operator,
                                rvalue,
                            )?;
                            Ok(None)
                        }
                        None => Err(EvaluationError::ArrayIndexOutOfBounds(array_len, index)),
                    }
                } else {
                    panic!("Cannot use array indexing on value {:?}", array);
                }
            }
            Expression::Accessor(info, struct_expr, field) => {
                match struct_expr.eval(subsystem, env) {
                    Ok(InternalValue::Struct(_, struct_repr)) => {
                        let left = struct_repr
                            .borrow()
                            .get(field)
                            .unwrap_or_else(|| {
                                panic!("Cannot access field {:?} in {:?}", field, struct_repr)
                            })
                            .clone();
                        let to_store = calculate_value_to_store(left, self.operator, rvalue)?;
                        struct_repr.borrow_mut().insert(field.to_string(), to_store);
                        Ok(None)
                    }
                    Ok(not_a_struct) => {
                        panic!(
                            "[{:?}]: Cannot access field {} of value {:?} and use it as an lvalue",
                            info, field, not_a_struct
                        );
                    }
                    Err(errors) => Err(errors),
                }
            }
            other => panic!("Expression {:?} is not an lvalue", other),
        }
    }
}

impl Evaluable for Block {
    fn eval(
        &self,
        subsystem: &mut dyn GameEngineSubsystem,
        env: &Rc<Env<InternalValue>>,
    ) -> Result<Option<InternalValue>, EvaluationError> {
        let mut last_returned_value = None;
        for statement in &self.statements {
            last_returned_value = statement.eval(subsystem, env)?;
            if last_returned_value.is_some() {
                break;
            }
        }
        Ok(last_returned_value)
    }
}

impl Evaluable for Return {
    fn eval(
        &self,
        subsystem: &mut dyn GameEngineSubsystem,
        env: &Rc<Env<InternalValue>>,
    ) -> Result<Option<InternalValue>, EvaluationError> {
        Ok(Some(self.expression.eval(subsystem, env)?))
    }
}

impl Evaluable for If {
    fn eval(
        &self,
        subsystem: &mut dyn GameEngineSubsystem,
        env: &Rc<Env<InternalValue>>,
    ) -> Result<Option<InternalValue>, EvaluationError> {
        match self.condition.eval(subsystem, env) {
            Ok(InternalValue::Boolean(true)) => {
                self.branch_true.eval(subsystem, &Env::create_child(env))
            }
            Ok(InternalValue::Boolean(false)) => match &self.branch_false {
                Some(branch_false) => branch_false.eval(subsystem, &Env::create_child(env)),
                None => Ok(None),
            },
            Err(error) => Err(error),
            Ok(value) => panic!(
                "[{:?}]: Expected type boolean in 'if' condition, got {:?} instead",
                self._info,
                value.get_type()
            ),
        }
    }
}

impl Evaluable for While {
    fn eval(
        &self,
        subsystem: &mut dyn GameEngineSubsystem,
        env: &Rc<Env<InternalValue>>,
    ) -> Result<Option<InternalValue>, EvaluationError> {
        loop {
            match self.condition.eval(subsystem, env) {
                Ok(InternalValue::Boolean(true)) => {
                    let block_result = self.body.eval(subsystem, &Env::create_child(env))?;
                    if block_result.is_some() {
                        return Ok(block_result);
                    }
                }
                Ok(InternalValue::Boolean(false)) => {
                    break;
                }
                Ok(value) => {
                    panic!(
                        "[{:?}]: Expected type boolean in 'while' condition, got {:?} instead",
                        self._info,
                        value.get_type()
                    );
                }
                Err(error) => {
                    return Err(error);
                }
            }
        }
        Ok(None)
    }
}

impl Evaluable for For {
    fn eval(
        &self,
        subsystem: &mut dyn GameEngineSubsystem,
        env: &Rc<Env<InternalValue>>,
    ) -> Result<Option<InternalValue>, EvaluationError> {
        let child_env = Env::create_child(env);
        if let v @ Some(_) = self.pre.eval(subsystem, &child_env)? {
            return Ok(v);
        }

        loop {
            match self.condition.eval(subsystem, &child_env) {
                Ok(InternalValue::Boolean(true)) => {
                    if let v @ Some(_) =
                        self.body.eval(subsystem, &Env::create_child(&child_env))?
                    {
                        return Ok(v);
                    }
                    if let v @ Some(_) = self.post.eval(subsystem, &child_env)? {
                        return Ok(v);
                    }
                }
                Ok(InternalValue::Boolean(false)) => {
                    break;
                }
                Ok(value) => {
                    panic!(
                        "[{:?}]: Expected type boolean in 'for' condition, got {:?} instead",
                        self._info,
                        value.get_type()
                    );
                }
                Err(error) => {
                    return Err(error);
                }
            }
        }
        Ok(None)
    }
}

impl Evaluable for StatementExpression {
    fn eval(
        &self,
        subsystem: &mut dyn GameEngineSubsystem,
        env: &Rc<Env<InternalValue>>,
    ) -> Result<Option<InternalValue>, EvaluationError> {
        self.expression.eval(subsystem, env)?;
        Ok(None)
    }
}

impl Evaluable for Statement {
    fn eval(
        &self,
        subsystem: &mut dyn GameEngineSubsystem,
        env: &Rc<Env<InternalValue>>,
    ) -> Result<Option<InternalValue>, EvaluationError> {
        match self {
            Statement::Declaration(x) => x.eval(subsystem, env),
            Statement::Assignment(x) => x.eval(subsystem, env),
            Statement::If(x) => x.eval(subsystem, env),
            Statement::For(x) => x.eval(subsystem, env),
            Statement::While(x) => x.eval(subsystem, env),
            Statement::Block(x) => x.eval(subsystem, env),
            Statement::StatementExpression(x) => x.eval(subsystem, env),
            Statement::Return(x) => x.eval(subsystem, env),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::lang::interpreter::native::MockGameEngineSubsystem;

    use super::*;
    use std::{collections::HashMap, matches};

    const INFO: SourceInfo = SourceInfo {
        line: 0,
        column: 0,
        offset_in_source: 0,
    };

    fn intval(i: i64) -> Box<Expression> {
        Box::new(Expression::Value(Value::Integer(i)))
    }
    fn floatval(i: f64) -> Box<Expression> {
        Box::new(Expression::Value(Value::FloatingPoint(i)))
    }
    fn ident(s: &str) -> Box<Expression> {
        Box::new(Expression::Value(Value::Variable(s.to_string())))
    }
    fn boolval(b: bool) -> Box<Expression> {
        Box::new(Expression::Value(Value::Boolean(b)))
    }
    fn binop(left: Box<Expression>, op: BinaryOperator, right: Box<Expression>) -> Box<Expression> {
        Box::new(Expression::BinaryOperation(INFO, None, left, op, right))
    }
    fn array(atype: Type, values: Vec<Value>) -> Box<Expression> {
        Box::new(Expression::ArrayInitializer(
            INFO,
            Some(atype),
            values
                .into_iter()
                .map(|v| Box::new(Expression::Value(v)))
                .collect(),
        ))
    }
    fn declare(name: &str, val: i64) -> Box<Statement> {
        Box::new(Statement::Declaration(Declaration {
            _info: INFO,
            name: name.to_string(),
            expected_type: None,
            immutable: false,
            rvalue: *intval(val),
        }))
    }

    #[test]
    fn simple_addition() {
        let e = binop(intval(1), BinaryOperator::Add, intval(2));
        let env = Env::empty();
        assert_eq!(
            e.eval(&mut MockGameEngineSubsystem::new(), &env),
            Ok(InternalValue::Integer(3))
        );
    }

    #[test]
    fn simple_addition_with_identifier() {
        let e = binop(intval(1), BinaryOperator::Add, ident("x"));
        let env = Env::empty();
        let _ = env.declare("x", InternalValue::Integer(2));
        assert_eq!(
            e.eval(&mut MockGameEngineSubsystem::new(), &env),
            Ok(InternalValue::Integer(3))
        );
    }

    #[test]
    fn divide_by_zero() {
        let e = Expression::BinaryOperation(INFO, None, intval(1), BinaryOperator::Div, intval(0));
        let env = Env::empty();
        assert!(matches!(
            e.eval(&mut MockGameEngineSubsystem::new(), &env),
            Err(EvaluationError::DivisionByZero)
        ));
    }

    #[test]
    fn complex_integer_math_expression_without_identifiers() {
        let e = binop(
            binop(intval(3), BinaryOperator::Mul, intval(7)),
            BinaryOperator::Sub,
            binop(intval(77), BinaryOperator::Div, intval(10)),
        );
        let env = Env::empty();
        assert_eq!(
            e.eval(&mut MockGameEngineSubsystem::new(), &env),
            Ok(InternalValue::Integer(14))
        );
    }

    #[test]
    fn int_and_float_automatic_casting() {
        let e = binop(
            binop(intval(3), BinaryOperator::Mul, floatval(2.0)),
            BinaryOperator::Sub,
            intval(3),
        );
        let env = Env::empty();
        assert_eq!(
            e.eval(&mut MockGameEngineSubsystem::new(), &env),
            Ok(InternalValue::FloatingPoint(3.0))
        );
    }

    #[test]
    fn equality() {
        let e = binop(
            binop(intval(3), BinaryOperator::Mul, floatval(2.0)),
            BinaryOperator::Sub,
            intval(3),
        );
        let env = Env::empty();
        assert_eq!(
            e.eval(&mut MockGameEngineSubsystem::new(), &env),
            Ok(InternalValue::FloatingPoint(3.0))
        );
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
        let env = Env::empty();
        assert_eq!(
            e.eval(&mut MockGameEngineSubsystem::new(), &env),
            Ok(InternalValue::Boolean(true))
        );
    }

    #[test]
    fn array_indexing() {
        let e = binop(
            array(
                Type::Integer,
                vec![Value::Integer(7), Value::Integer(8), Value::Integer(9)],
            ),
            BinaryOperator::Indexing,
            intval(2),
        );
        let env = Env::empty();
        assert_eq!(
            e.eval(&mut MockGameEngineSubsystem::new(), &env),
            Ok(InternalValue::Integer(9))
        );
    }

    #[test]
    fn array_indexing_out_of_bounds() {
        let e = binop(
            array(
                Type::Integer,
                vec![Value::Integer(7), Value::Integer(8), Value::Integer(9)],
            ),
            BinaryOperator::Indexing,
            intval(3),
        );
        let env = Env::empty();
        assert!(matches!(
            e.eval(&mut MockGameEngineSubsystem::new(), &env),
            Err(EvaluationError::ArrayIndexOutOfBounds(3, 3))
        ));
    }

    #[test]
    fn array_cell_as_lvalue() {
        let env = Env::empty();
        env.declare(
            "array",
            InternalValue::Array(
                Type::Integer,
                Rc::new(RefCell::new(vec![
                    InternalValue::Integer(7),
                    InternalValue::Integer(8),
                    InternalValue::Integer(9), // this element will be incremented by 1
                ])),
            ),
        );

        let program = Block {
            _info: INFO,
            statements: vec![Statement::Assignment(Assignment {
                _info: INFO,
                lvalue: *binop(
                    ident("array"),
                    BinaryOperator::Indexing,
                    binop(intval(1), BinaryOperator::Add, intval(1)),
                ),
                operator: AssignmentOperator::Equal,
                rvalue: *binop(
                    intval(1),
                    BinaryOperator::Add,
                    binop(
                        ident("array"),
                        BinaryOperator::Indexing,
                        binop(intval(1), BinaryOperator::Add, intval(1)),
                    ),
                ),
            })],
        };

        assert_eq!(
            program.eval(&mut MockGameEngineSubsystem::new(), &env),
            Ok(None)
        );
        assert_eq!(
            env.cloning_lookup("array"),
            Some(InternalValue::Array(
                Type::Integer,
                Rc::new(RefCell::new(vec![
                    InternalValue::Integer(7),
                    InternalValue::Integer(8),
                    InternalValue::Integer(10)
                ]))
            ))
        );
    }

    #[test]
    fn sum_of_first_10_values() {
        let program = Statement::Block(Block {
            _info: INFO,
            statements: vec![
                *declare("x", 0),
                Statement::For(For {
                    _info: INFO,
                    pre: Block {
                        _info: INFO,
                        statements: vec![*declare("i", 0)],
                    },
                    condition: *binop(ident("i"), BinaryOperator::LessThan, intval(10)),
                    post: Block {
                        _info: INFO,
                        statements: vec![Statement::Assignment(Assignment {
                            _info: INFO,
                            lvalue: *ident("i"),
                            operator: AssignmentOperator::Equal,
                            rvalue: *binop(ident("i"), BinaryOperator::Add, intval(1)),
                        })],
                    },
                    body: Block {
                        _info: INFO,
                        statements: vec![Statement::Assignment(Assignment {
                            _info: INFO,
                            lvalue: *ident("x"),
                            operator: AssignmentOperator::Equal,
                            rvalue: *binop(ident("x"), BinaryOperator::Add, ident("i")),
                        })],
                    },
                }),
                Statement::Return(Return {
                    _info: INFO,
                    expression: *ident("x"),
                }),
            ],
        });
        let env = Env::empty();
        assert_eq!(
            program.eval(&mut MockGameEngineSubsystem::new(), &env),
            Ok(Some(InternalValue::Integer(45)))
        );
    }

    #[test]
    fn simple_function_call() {
        let env = Env::empty();
        let function = Block {
            _info: INFO,
            statements: vec![Statement::Return(Return {
                _info: INFO,
                expression: *binop(ident("x"), BinaryOperator::Add, intval(1)),
            })],
        };
        env.declare(
            "add_one",
            InternalValue::Closure(Type::Void, Rc::clone(&env), vec!["x".to_string()], function),
        );
        let program = Statement::Declaration(Declaration {
            _info: INFO,
            name: "result".to_string(),
            expected_type: None,
            immutable: true,
            rvalue: Expression::FunctionCall(INFO, None, ident("add_one"), vec![*intval(2)]),
        });

        assert_eq!(
            program.eval(&mut MockGameEngineSubsystem::new(), &env),
            Ok(None)
        );
        assert_eq!(
            env.cloning_lookup("result"),
            Some(InternalValue::Integer(3))
        );
    }

    #[test]
    fn recursive_factorial_function_call() {
        let env = Env::empty();
        let function = Block {
            _info: INFO,
            statements: vec![Statement::If(If {
                _info: INFO,
                condition: *binop(ident("x"), BinaryOperator::Equal, intval(0)),
                branch_true: Block {
                    _info: INFO,
                    statements: vec![Statement::Return(Return {
                        _info: INFO,
                        expression: *intval(1),
                    })],
                },
                branch_false: Some(Block {
                    _info: INFO,
                    statements: vec![Statement::Return(Return {
                        _info: INFO,
                        expression: *binop(
                            Box::new(Expression::FunctionCall(
                                INFO,
                                None,
                                ident("factorial"),
                                vec![*binop(ident("x"), BinaryOperator::Sub, intval(1))],
                            )),
                            BinaryOperator::Mul,
                            ident("x"),
                        ),
                    })],
                }),
            })],
        };
        env.declare(
            "factorial",
            InternalValue::Closure(Type::Void, Rc::clone(&env), vec!["x".to_string()], function),
        );
        let program = Declaration {
            _info: INFO,
            name: "result".to_string(),
            expected_type: None,
            immutable: true,
            rvalue: Expression::FunctionCall(INFO, None, ident("factorial"), vec![*intval(7)]),
        };

        assert_eq!(
            program.eval(&mut MockGameEngineSubsystem::new(), &env),
            Ok(None)
        );
        assert_eq!(
            env.cloning_lookup("result"),
            Some(InternalValue::Integer(5040))
        );
    }

    #[test]
    fn assignment_operators() {
        let env = Env::empty();

        // var x = 1; x += 1;
        let program = Block {
            _info: INFO,
            statements: vec![
                *declare("x", 1),
                Statement::Assignment(Assignment {
                    _info: INFO,
                    lvalue: Expression::Value(Value::Variable("x".to_string())),
                    operator: AssignmentOperator::AddEqual,
                    rvalue: *intval(1),
                }),
            ],
        };
        assert!(program
            .eval(&mut MockGameEngineSubsystem::new(), &env)
            .is_ok());
        assert_eq!(env.cloning_lookup("x"), Some(InternalValue::Integer(2)));

        // var x = 3; x *= 3;
        let program = Block {
            _info: INFO,
            statements: vec![
                *declare("x", 3),
                Statement::Assignment(Assignment {
                    _info: INFO,
                    lvalue: Expression::Value(Value::Variable("x".to_string())),
                    operator: AssignmentOperator::MulEqual,
                    rvalue: *intval(3),
                }),
            ],
        };
        assert!(program
            .eval(&mut MockGameEngineSubsystem::new(), &env)
            .is_ok());
        assert_eq!(env.cloning_lookup("x"), Some(InternalValue::Integer(9)));

        // var x = 3; x -= 3;
        let program = Block {
            _info: INFO,
            statements: vec![
                *declare("x", 3),
                Statement::Assignment(Assignment {
                    _info: INFO,
                    lvalue: Expression::Value(Value::Variable("x".to_string())),
                    operator: AssignmentOperator::SubEqual,
                    rvalue: *intval(3),
                }),
            ],
        };
        assert!(program
            .eval(&mut MockGameEngineSubsystem::new(), &env)
            .is_ok());
        assert_eq!(env.cloning_lookup("x"), Some(InternalValue::Integer(0)));

        // var x = 3; x /= 3
        let program = Block {
            _info: INFO,
            statements: vec![
                *declare("x", 3),
                Statement::Assignment(Assignment {
                    _info: INFO,
                    lvalue: Expression::Value(Value::Variable("x".to_string())),
                    operator: AssignmentOperator::DivEqual,
                    rvalue: *intval(3),
                }),
            ],
        };
        assert!(program
            .eval(&mut MockGameEngineSubsystem::new(), &env)
            .is_ok());
        assert_eq!(env.cloning_lookup("x"), Some(InternalValue::Integer(1)));
    }

    #[test]
    fn read_struct_accessor() {
        let env = Env::empty();
        let program = Block {
            _info: INFO,
            statements: vec![Statement::Return(Return {
                _info: INFO,
                expression: Expression::Accessor(INFO, ident("point"), "x".to_string()),
            })],
        };
        env.declare(
            "point",
            InternalValue::Struct(
                Type::Struct(
                    "Point".to_string(),
                    HashMap::from([("x".to_string(), Type::Integer)]),
                ),
                Rc::from(RefCell::new(HashMap::from([(
                    "x".to_string(),
                    InternalValue::Integer(1),
                )]))),
            ),
        );

        assert_eq!(
            program.eval(&mut MockGameEngineSubsystem::new(), &env),
            Ok(Some(InternalValue::Integer(1)))
        );
    }

    #[test]
    fn assign_struct_accessor() {
        let env = Env::empty();
        let program = Block {
            _info: INFO,
            statements: vec![Statement::Assignment(Assignment {
                _info: INFO,
                lvalue: Expression::Accessor(INFO, ident("point"), "x".to_string()),
                rvalue: Expression::Value(Value::Integer(2)),
                operator: AssignmentOperator::AddEqual,
            })],
        };
        let struct_type = Type::Struct(
            "Point".to_string(),
            HashMap::from([("x".to_string(), Type::Integer)]),
        );
        env.declare(
            "point",
            InternalValue::Struct(
                struct_type.clone(),
                Rc::from(RefCell::new(HashMap::from([(
                    "x".to_string(),
                    InternalValue::Integer(1),
                )]))),
            ),
        );

        assert_eq!(
            program.eval(&mut MockGameEngineSubsystem::new(), &env),
            Ok(None)
        );
        assert_eq!(
            env.cloning_lookup("point"),
            Some(InternalValue::Struct(
                struct_type,
                Rc::from(RefCell::new(HashMap::from([(
                    "x".to_string(),
                    InternalValue::Integer(3)
                )])))
            ))
        );
    }
}

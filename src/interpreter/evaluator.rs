use super::typechecking::Type;
use crate::interpreter::{ast::*, environment::Env};
use std::{cell::RefCell, rc::Rc};

#[derive(Debug, PartialEq)]
pub enum EvaluationError {
    DivisionByZero,
    ArrayIndexOutOfBounds(usize, i64),
    FatalError(String),
}

impl Expression {
    pub fn eval(&self, env: &Env<ImmediateValue>) -> Result<ImmediateValue, EvaluationError> {
        match self {
            Expression::Value(immediate_value) => Ok(immediate_value.clone()),
            Expression::BinaryOperation(_info, _, left, operator, right) => {
                match (left.eval(env), right.eval(env)) {
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
            Expression::UnaryOperation(info, _, UnaryOperator::Not, left) => match left.eval(env) {
                Ok(ImmediateValue::Boolean(value)) => Ok(ImmediateValue::Boolean(!value)),
                Err(error) => Err(error),
                Ok(value) => panic!(
                    "[{:?}]: Cannot apply boolean operator 'not' on value of type {:?}",
                    info,
                    value.get_type(),
                ),
            },
            Expression::UnaryOperation(info, _, UnaryOperator::Negation, left) => {
                match left.eval(env) {
                    Ok(ImmediateValue::Integer(x)) => Ok(ImmediateValue::Integer(-x)),
                    Ok(ImmediateValue::FloatingPoint(f)) => Ok(ImmediateValue::FloatingPoint(-f)),
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
            Expression::FunctionCall(info, _, function, args) => match function.eval(env) {
                Ok(ImmediateValue::Closure(_, closure_env, args_names, statements)) => {
                    let function_context_env = Env::create_child(&closure_env);
                    for (argname, argexpr) in args_names.iter().zip(args.iter()) {
                        let argvalue = argexpr.eval(env)?;
                        function_context_env.declare(argname, argvalue);
                    }

                    statements
                        .eval(&function_context_env)
                        .map(|v| v.unwrap_or(ImmediateValue::Void))
                }
                Ok(ImmediateValue::NativeFunction(_, native_function)) => {
                    let mut evaluated_args = vec![];
                    evaluated_args.reserve_exact(args.len());
                    for e in args {
                        let value = e.eval(env)?;
                        evaluated_args.push(value);
                    }
                    native_function(evaluated_args)
                }
                Ok(value) => panic!(
                    "[{:?}]: Cannot use value of types {:?} as a function",
                    info,
                    value.get_type(),
                ),
                error => error,
            },
            Expression::ArrayInitializer(_, _, values) => {
                let mut array = vec![];
                array.reserve_exact(values.len());
                for e in values {
                    let value = e.eval(env)?;
                    array.push(value);
                }
                Ok(ImmediateValue::Array(
                    Type::Array(Box::new(Type::Integer)),
                    Rc::new(RefCell::new(array)),
                ))
            }
            Expression::Identifier(info, name) => match env.cloning_lookup(name) {
                Some(value) => Ok(value),
                None => panic!(
                    "[{:?}]: Symbol '{}' was not found in the current scope",
                    info, name
                ),
            },
            Expression::Accessor(info, struct_expr, field) => match struct_expr.eval(env) {
                Ok(ImmediateValue::Struct(struct_type, contents)) => {
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
            },
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
                        Ok(LValue::IndexInArray(arr, index))
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
            Expression::Accessor(info, struct_expr, field) => match struct_expr.eval(env) {
                Ok(ImmediateValue::Struct(_, struct_repr)) => {
                    Ok(LValue::Accessor(struct_repr, field.to_string()))
                }
                Ok(not_a_struct) => {
                    panic!(
                        "[{:?}]: Cannot access field {} of value {:?} and use it as an lvalue",
                        info, field, not_a_struct
                    );
                }
                Err(errors) => Err(errors),
            },
            other => panic!("Expression {:?} is not an lvalue", other),
        }
    }
}

trait Evaluable {
    fn eval(
        &self,
        env: &Rc<Env<ImmediateValue>>,
    ) -> Result<Option<ImmediateValue>, EvaluationError>;
}

impl Evaluable for Declaration {
    fn eval(
        &self,
        env: &Rc<Env<ImmediateValue>>,
    ) -> Result<Option<ImmediateValue>, EvaluationError> {
        match self.rvalue.eval(env) {
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
        env: &Rc<Env<ImmediateValue>>,
    ) -> Result<Option<ImmediateValue>, EvaluationError> {
        let lvalue = self.lvalue.eval_to_lvalue(env)?;

        let rvalue = match self.operator {
            AssignmentOperator::Equal => self.rvalue.eval(env)?,
            AssignmentOperator::AddEqual => {
                let right = self.rvalue.eval(env)?;
                lvalue.to_expression().eval(env)?.add(&right)?
            }
            AssignmentOperator::SubEqual => {
                let right = self.rvalue.eval(env)?;
                lvalue.to_expression().eval(env)?.sub(&right)?
            }
            AssignmentOperator::MulEqual => {
                let right = self.rvalue.eval(env)?;
                lvalue.to_expression().eval(env)?.mul(&right)?
            }
            AssignmentOperator::DivEqual => {
                let right = self.rvalue.eval(env)?;
                lvalue.to_expression().eval(env)?.div(&right)?
            }
        };

        match lvalue {
            LValue::Identifier(symbol) => {
                if env.assign(&symbol, rvalue).is_err() {
                    panic!("Assignment to non-declared variable {}", symbol);
                }
                Ok(None)
            }
            LValue::IndexInArray(array, index) => {
                let array_len = array.borrow().len();
                let uindex: usize = index
                    .try_into()
                    .map_err(|_| EvaluationError::ArrayIndexOutOfBounds(array_len, index))?;
                match array.borrow_mut().get_mut(uindex) {
                    Some(array_cell) => {
                        *array_cell = rvalue;
                        Ok(None)
                    }
                    None => Err(EvaluationError::ArrayIndexOutOfBounds(array_len, index)),
                }
            }
            LValue::Accessor(struct_object, field) => {
                struct_object.borrow_mut().insert(field, rvalue);
                Ok(None)
            }
        }
    }
}

impl Evaluable for Block {
    fn eval(
        &self,
        env: &Rc<Env<ImmediateValue>>,
    ) -> Result<Option<ImmediateValue>, EvaluationError> {
        let mut last_returned_value = None;
        for statement in &self.statements {
            last_returned_value = statement.eval(env)?;
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
        env: &Rc<Env<ImmediateValue>>,
    ) -> Result<Option<ImmediateValue>, EvaluationError> {
        Ok(Some(self.expression.eval(env)?))
    }
}

impl Evaluable for If {
    fn eval(
        &self,
        env: &Rc<Env<ImmediateValue>>,
    ) -> Result<Option<ImmediateValue>, EvaluationError> {
        match self.condition.eval(env) {
            Ok(ImmediateValue::Boolean(true)) => self.branch_true.eval(&Env::create_child(env)),
            Ok(ImmediateValue::Boolean(false)) => match &self.branch_false {
                Some(branch_false) => branch_false.eval(&Env::create_child(env)),
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
        env: &Rc<Env<ImmediateValue>>,
    ) -> Result<Option<ImmediateValue>, EvaluationError> {
        loop {
            match self.condition.eval(env) {
                Ok(ImmediateValue::Boolean(true)) => {
                    let block_result = self.body.eval(&Env::create_child(env))?;
                    if block_result.is_some() {
                        return Ok(block_result);
                    }
                }
                Ok(ImmediateValue::Boolean(false)) => {
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
        env: &Rc<Env<ImmediateValue>>,
    ) -> Result<Option<ImmediateValue>, EvaluationError> {
        let child_env = Env::create_child(env);
        if let v @ Some(_) = self.pre.eval(&child_env)? {
            return Ok(v);
        }

        loop {
            match self.condition.eval(&child_env) {
                Ok(ImmediateValue::Boolean(true)) => {
                    if let v @ Some(_) = self.body.eval(&Env::create_child(&child_env))? {
                        return Ok(v);
                    }
                    if let v @ Some(_) = self.post.eval(&child_env)? {
                        return Ok(v);
                    }
                }
                Ok(ImmediateValue::Boolean(false)) => {
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
        env: &Rc<Env<ImmediateValue>>,
    ) -> Result<Option<ImmediateValue>, EvaluationError> {
        self.expression.eval(env)?;
        Ok(None)
    }
}

impl Evaluable for Statement {
    fn eval(
        &self,
        env: &Rc<Env<ImmediateValue>>,
    ) -> Result<Option<ImmediateValue>, EvaluationError> {
        match self {
            Statement::Declaration(x) => x.eval(env),
            Statement::Assignment(x) => x.eval(env),
            Statement::If(x) => x.eval(env),
            Statement::For(x) => x.eval(env),
            Statement::While(x) => x.eval(env),
            Statement::Block(x) => x.eval(env),
            Statement::StatementExpression(x) => x.eval(env),
            Statement::Return(x) => x.eval(env),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashMap, matches};

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
        assert_eq!(e.eval(&env), Ok(ImmediateValue::Integer(3)));
    }

    #[test]
    fn simple_addition_with_identifier() {
        let e = binop(intval(1), BinaryOperator::Add, ident("x"));
        let env = Env::empty();
        let _ = env.declare("x", ImmediateValue::Integer(2));
        assert_eq!(e.eval(&env), Ok(ImmediateValue::Integer(3)));
    }

    #[test]
    fn divide_by_zero() {
        let e = Expression::BinaryOperation(INFO, None, intval(1), BinaryOperator::Div, intval(0));
        let env = Env::empty();
        assert!(matches!(e.eval(&env), Err(EvaluationError::DivisionByZero)));
    }

    #[test]
    fn complex_integer_math_expression_without_identifiers() {
        let e = binop(
            binop(intval(3), BinaryOperator::Mul, intval(7)),
            BinaryOperator::Sub,
            binop(intval(77), BinaryOperator::Div, intval(10)),
        );
        let env = Env::empty();
        assert_eq!(e.eval(&env), Ok(ImmediateValue::Integer(14)));
    }

    #[test]
    fn int_and_float_automatic_casting() {
        let e = binop(
            binop(intval(3), BinaryOperator::Mul, floatval(2.0)),
            BinaryOperator::Sub,
            intval(3),
        );
        let env = Env::empty();
        assert_eq!(e.eval(&env), Ok(ImmediateValue::FloatingPoint(3.0)));
    }

    #[test]
    fn equality() {
        let e = binop(
            binop(intval(3), BinaryOperator::Mul, floatval(2.0)),
            BinaryOperator::Sub,
            intval(3),
        );
        let env = Env::empty();
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
        let env = Env::empty();
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
        let env = Env::empty();
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
        let env = Env::empty();
        assert!(matches!(
            e.eval(&env),
            Err(EvaluationError::ArrayIndexOutOfBounds(3, 3))
        ));
    }

    #[test]
    fn array_cell_as_lvalue() {
        let env = Env::empty();
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
        assert_eq!(program.eval(&env), Ok(Some(ImmediateValue::Integer(45))));
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
            ImmediateValue::Closure(Type::Void, Rc::clone(&env), vec!["x".to_string()], function),
        );
        let program = Statement::Declaration(Declaration {
            _info: INFO,
            name: "result".to_string(),
            expected_type: None,
            immutable: true,
            rvalue: Expression::FunctionCall(INFO, None, ident("add_one"), vec![*intval(2)]),
        });

        assert_eq!(program.eval(&env), Ok(None));
        assert_eq!(
            env.cloning_lookup("result"),
            Some(ImmediateValue::Integer(3))
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
            ImmediateValue::Closure(Type::Void, Rc::clone(&env), vec!["x".to_string()], function),
        );
        let program = Declaration {
            _info: INFO,
            name: "result".to_string(),
            expected_type: None,
            immutable: true,
            rvalue: Expression::FunctionCall(INFO, None, ident("factorial"), vec![*intval(7)]),
        };

        assert_eq!(program.eval(&env), Ok(None));
        assert_eq!(
            env.cloning_lookup("result"),
            Some(ImmediateValue::Integer(5040))
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
                    lvalue: Expression::Identifier(INFO, "x".to_string()),
                    operator: AssignmentOperator::AddEqual,
                    rvalue: *intval(1),
                }),
            ],
        };
        assert!(program.eval(&env).is_ok());
        assert_eq!(env.cloning_lookup("x"), Some(ImmediateValue::Integer(2)));

        // var x = 3; x *= 3;
        let program = Block {
            _info: INFO,
            statements: vec![
                *declare("x", 3),
                Statement::Assignment(Assignment {
                    _info: INFO,
                    lvalue: Expression::Identifier(INFO, "x".to_string()),
                    operator: AssignmentOperator::MulEqual,
                    rvalue: *intval(3),
                }),
            ],
        };
        assert!(program.eval(&env).is_ok());
        assert_eq!(env.cloning_lookup("x"), Some(ImmediateValue::Integer(9)));

        // var x = 3; x -= 3;
        let program = Block {
            _info: INFO,
            statements: vec![
                *declare("x", 3),
                Statement::Assignment(Assignment {
                    _info: INFO,
                    lvalue: Expression::Identifier(INFO, "x".to_string()),
                    operator: AssignmentOperator::SubEqual,
                    rvalue: *intval(3),
                }),
            ],
        };
        assert!(program.eval(&env).is_ok());
        assert_eq!(env.cloning_lookup("x"), Some(ImmediateValue::Integer(0)));

        // var x = 3; x /= 3
        let program = Block {
            _info: INFO,
            statements: vec![
                *declare("x", 3),
                Statement::Assignment(Assignment {
                    _info: INFO,
                    lvalue: Expression::Identifier(INFO, "x".to_string()),
                    operator: AssignmentOperator::DivEqual,
                    rvalue: *intval(3),
                }),
            ],
        };
        assert!(program.eval(&env).is_ok());
        assert_eq!(env.cloning_lookup("x"), Some(ImmediateValue::Integer(1)));
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
            ImmediateValue::Struct(
                Type::Struct(
                    "Point".to_string(),
                    HashMap::from([("x".to_string(), Type::Integer)]),
                ),
                Rc::from(RefCell::new(HashMap::from([(
                    "x".to_string(),
                    ImmediateValue::Integer(1),
                )]))),
            ),
        );

        assert_eq!(program.eval(&env), Ok(Some(ImmediateValue::Integer(1))));
    }

    #[test]
    fn assign_struct_accessor() {
        let env = Env::empty();
        let program = Block {
            _info: INFO,
            statements: vec![Statement::Assignment(Assignment {
                _info: INFO,
                lvalue: Expression::Accessor(INFO, ident("point"), "x".to_string()),
                rvalue: Expression::Value(ImmediateValue::Integer(2)),
                operator: AssignmentOperator::AddEqual,
            })],
        };
        let struct_type = Type::Struct(
            "Point".to_string(),
            HashMap::from([("x".to_string(), Type::Integer)]),
        );
        env.declare(
            "point",
            ImmediateValue::Struct(
                struct_type.clone(),
                Rc::from(RefCell::new(HashMap::from([(
                    "x".to_string(),
                    ImmediateValue::Integer(1),
                )]))),
            ),
        );

        assert_eq!(program.eval(&env), Ok(None));
        assert_eq!(
            env.cloning_lookup("point"),
            Some(ImmediateValue::Struct(
                struct_type,
                Rc::from(RefCell::new(HashMap::from([(
                    "x".to_string(),
                    ImmediateValue::Integer(3)
                )])))
            ))
        );
    }
}

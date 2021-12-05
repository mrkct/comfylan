use crate::interpreter::{
    ast::*,
    lexer::{Token, TokenKind},
    typechecking::Type,
};

const TODO_INFO: SourceInfo = SourceInfo {
    line: 0,
    column: 0,
    offset_in_source: 0,
};

/*
Program := (<Function>|<Type-Declaration>)*
Function := fn identifier ( <Empty>|<Arg> (,<Arg>)* ) -> <Type> <Block>
Type := identifier | [ Type ]
Arg := identifier : <Type>
Block := { Statement* }
Statement := (<Block> | <LetDeclaration> | <VarDeclaration> | <Assignment> | <IfExpr> | <WhileExpr> | <ForExpr> | <Expr>) ;
LetDeclaration := let identifier (: <Type>) = <Expr>
VarDeclaration := var identifier (: <Type>) = <Expr>
Assignment := identifier = <Expr>
IfExpr := if ( <Expr> ) <Block> (else <Block>)
WhileExpr := while ( <Expr> ) <Block>
Return := return <Expr>
ForExpr := for <Statement>|<Block> (<Expr>) <Statement>|<Block> <Block>
Expr := P1Expr
P1Expr := P2Expr ( (and|or|xor|nor) P2Expr)*
P2Expr := P3Expr ( (==|!=|<=|>=|<|>) P3Expr )*
P3Expr := P4Expr ( (+|-) P4Expr )*
P4Expr := <P5Expr> ( (*|/) <P5Expr> )*
P5Expr := - <Value> | <Value>[<Expr>] | <Value>( <Expr> (, <Expr>)* )
Value := <Constant> | '(' <Expr> ')' | identifier
Constant := integer | floating_point | string | KeywordTrue | KeywordFalse
*/

pub struct Parser<'a> {
    tokens: &'a [Token<'a>],
}

macro_rules! try_consume {
    ($tokens:expr, $token_kind:pat) => {{
        match $tokens.first() {
            Some(
                t @ Token {
                    kind: $token_kind, ..
                },
            ) => {
                $tokens = &$tokens[1..];
                Some(t)
            }
            _ => None,
        }
    }};
}

macro_rules! rewinding_if_none {
    ($self:expr, $e:block) => {
        (|| {
            let saved_tokens = $self.tokens;
            match { (|| $e)() } {
                None => {
                    $self.tokens = saved_tokens;
                    None
                }
                some_value => some_value,
            }
        })()
    };
}

macro_rules! parse_expression_level {
    ($name:ident, $operators:pat, $lower_level_parse_method:ident) => {
        fn $name(&mut self) -> Option<Box<Expression>> {
            let first_value = self.$lower_level_parse_method()?;

            let mut parse_operator_value = || -> Option<(&Token, Box<Expression>)> {
                rewinding_if_none!(self, {
                    let operator = try_consume!(self.tokens, $operators)?;
                    if let Some(value) = self.$lower_level_parse_method() {
                        return Some((operator, value));
                    }
                    None
                })
            };

            let mut left = first_value;
            while let Some((operator, right)) = parse_operator_value() {
                left = Box::new(Expression::BinaryOperation(
                    TODO_INFO,
                    None,
                    left,
                    operator.as_binary_operator(),
                    right,
                ));
            }
            Some(left)
        }
    };
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token<'a>]) -> Parser<'a> {
        Parser { tokens }
    }

    fn parse_token_separated_list_of<T>(
        &mut self,
        _separator: TokenKind,
        parser: fn(&mut Self) -> Option<T>,
    ) -> Option<Vec<T>> {
        let mut collected_parameters = vec![];
        if let Some(p) = parser(&mut *self) {
            collected_parameters.push(p);
            while let Some(p) = rewinding_if_none!(self, {
                try_consume!(self.tokens, _separator)?;
                parser(&mut *self)
            }) {
                collected_parameters.push(p)
            }
        }
        Some(collected_parameters)
    }

    pub fn parse_program(&mut self) -> Option<Vec<TopLevelDeclaration>> {
        let mut collected_functions = vec![];
        while let Some(function) = self.parse_root_function_declaration() {
            collected_functions.push(function);
        }

        Some(collected_functions)
    }

    fn parse_value(&mut self) -> Option<Box<Expression>> {
        macro_rules! try_match_single_token_to_immediate_value {
            ($p:pat) => {
                if let Some(token) = try_consume!(self.tokens, $p) {
                    return Some(Box::new(Expression::Value(token.as_immediate_value())));
                }
            };
        }

        rewinding_if_none!(self, {
            try_match_single_token_to_immediate_value!(TokenKind::Integer(_));
            try_match_single_token_to_immediate_value!(TokenKind::String(_));
            try_match_single_token_to_immediate_value!(TokenKind::FloatingPoint(_));
            try_match_single_token_to_immediate_value!(TokenKind::KeywordTrue);
            try_match_single_token_to_immediate_value!(TokenKind::KeywordFalse);

            if let Some(token) = try_consume!(self.tokens, TokenKind::Identifier(_)) {
                return Some(Box::new(Expression::Identifier(
                    TODO_INFO,
                    token.clone_identifiers_string(),
                )));
            }

            if let Some(node) = rewinding_if_none!(self, {
                try_consume!(self.tokens, TokenKind::OpenRoundBracket)?;
                let node = self.parse_expr()?;
                try_consume!(self.tokens, TokenKind::CloseRoundBracket)?;
                Some(node)
            }) {
                return Some(node);
            }

            None
        })
    }

    fn parse_expr(&mut self) -> Option<Box<Expression>> {
        self.parse_p1expr()
    }

    parse_expression_level!(
        parse_p1expr,
        TokenKind::KeywordAnd
            | TokenKind::KeywordOr
            | TokenKind::KeywordXor
            | TokenKind::KeywordNor,
        parse_p2expr
    );
    parse_expression_level!(
        parse_p2expr,
        TokenKind::EqualEqual
            | TokenKind::ExclamationMarkEqual
            | TokenKind::LessThanEqual
            | TokenKind::GreaterThanEqual
            | TokenKind::LessThan
            | TokenKind::GreaterThan,
        parse_p3expr
    );
    parse_expression_level!(
        parse_p3expr,
        TokenKind::Plus | TokenKind::Minus,
        parse_p4expr
    );
    parse_expression_level!(
        parse_p4expr,
        TokenKind::Star | TokenKind::Slash,
        parse_p5expr
    );

    fn parse_p5expr(&mut self) -> Option<Box<Expression>> {
        let parsing_methods = &[
            Self::parse_unary_minus,
            Self::parse_array_indexing,
            Self::parse_function_call,
            Self::parse_value,
        ];

        for parser in parsing_methods {
            if let Some(node) = parser(self) {
                return Some(node);
            }
        }
        None
    }

    fn parse_unary_minus(&mut self) -> Option<Box<Expression>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::Minus)?;
            let value = self.parse_value()?;
            Some(Box::new(Expression::UnaryOperation(
                TODO_INFO,
                None,
                UnaryOperator::Negation,
                value,
            )))
        })
    }

    fn parse_array_indexing(&mut self) -> Option<Box<Expression>> {
        rewinding_if_none!(self, {
            let indexable = self.parse_value()?;
            try_consume!(self.tokens, TokenKind::OpenSquareBracket)?;
            let index = self.parse_expr()?;
            try_consume!(self.tokens, TokenKind::CloseSquareBracket)?;
            Some(Box::new(Expression::BinaryOperation(
                TODO_INFO,
                None,
                indexable,
                BinaryOperator::Indexing,
                index,
            )))
        })
    }

    fn parse_function_call(&mut self) -> Option<Box<Expression>> {
        rewinding_if_none!(self, {
            let function = self.parse_value()?;
            try_consume!(self.tokens, TokenKind::OpenRoundBracket)?;
            let arguments = self.parse_token_separated_list_of(TokenKind::Comma, |myself| {
                myself.parse_expr().map(|x| *x)
            })?;
            try_consume!(self.tokens, TokenKind::CloseRoundBracket)?;
            Some(Box::new(Expression::FunctionCall(
                TODO_INFO, None, function, arguments,
            )))
        })
    }

    fn parse_let_declaration(&mut self) -> Option<Box<Statement>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordLet)?;
            let name = try_consume!(self.tokens, TokenKind::Identifier(_))?;
            try_consume!(self.tokens, TokenKind::Colon)?;
            let declared_type = self.parse_type()?;
            try_consume!(self.tokens, TokenKind::Equal)?;
            let expr = self.parse_expr()?;
            Some(Box::new(Statement::Declaration(
                TODO_INFO,
                name.clone_identifiers_string(),
                Some(*declared_type),
                true,
                *expr,
            )))
        })
    }

    fn parse_var_declaration(&mut self) -> Option<Box<Statement>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordVar)?;
            let name = try_consume!(self.tokens, TokenKind::Identifier(_))?;
            try_consume!(self.tokens, TokenKind::Colon)?;
            let declared_type = self.parse_type()?;
            try_consume!(self.tokens, TokenKind::Equal)?;
            let expr = self.parse_expr()?;
            Some(Box::new(Statement::Declaration(
                TODO_INFO,
                name.clone_identifiers_string(),
                Some(*declared_type),
                false,
                *expr,
            )))
        })
    }

    fn parse_assignment_operator(&mut self) -> Option<AssignmentOperator> {
        let operator_token = try_consume!(
            self.tokens,
            TokenKind::Equal
                | TokenKind::PlusEqual
                | TokenKind::MinusEqual
                | TokenKind::StarEqual
                | TokenKind::SlashEqual
        )?;
        Some(operator_token.as_assignment_operator())
    }

    fn parse_assignment(&mut self) -> Option<Box<Statement>> {
        rewinding_if_none!(self, {
            // FIXME: Only a subset of expressions are valid lvalues
            let lvalue = self.parse_expr()?;
            let operator = self.parse_assignment_operator()?;
            let rvalue = self.parse_expr()?;
            Some(Box::new(Statement::Assignment(
                TODO_INFO, *lvalue, operator, *rvalue,
            )))
        })
    }

    fn parse_if(&mut self) -> Option<Box<Statement>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordIf)?;
            try_consume!(self.tokens, TokenKind::OpenRoundBracket)?;
            let condition = self.parse_expr()?;
            try_consume!(self.tokens, TokenKind::CloseRoundBracket)?;
            let true_branch = self.parse_block()?;
            match try_consume!(self.tokens, TokenKind::KeywordElse) {
                None => Some(Box::new(Statement::If(
                    TODO_INFO,
                    *condition,
                    true_branch,
                    None,
                ))),
                _ => {
                    let false_branch = self.parse_block()?;
                    Some(Box::new(Statement::If(
                        TODO_INFO,
                        *condition,
                        true_branch,
                        Some(false_branch),
                    )))
                }
            }
        })
    }

    fn parse_while(&mut self) -> Option<Box<Statement>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordWhile)?;
            try_consume!(self.tokens, TokenKind::OpenRoundBracket)?;
            let condition = self.parse_expr()?;
            try_consume!(self.tokens, TokenKind::CloseRoundBracket)?;
            let loop_block = self.parse_block()?;
            Some(Box::new(Statement::While(
                TODO_INFO, *condition, loop_block,
            )))
        })
    }

    fn parse_for(&mut self) -> Option<Box<Statement>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordFor)?;
            let pre = self.parse_block()?;
            let condition = self.parse_expr()?;
            let post = self.parse_block()?;
            let loop_block = self.parse_block()?;
            Some(Box::new(Statement::For(
                TODO_INFO, pre, *condition, post, loop_block,
            )))
        })
    }

    fn parse_return(&mut self) -> Option<Box<Statement>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordReturn)?;
            let expression = self.parse_expr()?;
            Some(Box::new(Statement::Return(TODO_INFO, *expression)))
        })
    }

    fn parse_type(&mut self) -> Option<Box<Type>> {
        if let Some(array_of_type) = rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::OpenSquareBracket)?;
            let inner_type = self.parse_type()?;
            try_consume!(self.tokens, TokenKind::CloseSquareBracket)?;
            Some(Box::new(Type::Array(inner_type)))
        }) {
            Some(array_of_type)
        } else {
            let base_type = try_consume!(self.tokens, TokenKind::Identifier(_))?;
            let type_name = base_type.clone_identifiers_string();
            Some(match type_name.as_str() {
                "int" => Box::new(Type::Integer),
                "float" => Box::new(Type::FloatingPoint),
                "bool" => Box::new(Type::Boolean),
                "string" => Box::new(Type::String),
                _ => Box::new(Type::UserDefined(type_name)),
            })
        }
    }

    fn parse_root_function_declaration(&mut self) -> Option<TopLevelDeclaration> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordFn)?;
            let function_name = try_consume!(self.tokens, TokenKind::Identifier(_))?;
            try_consume!(self.tokens, TokenKind::OpenRoundBracket)?;

            let collected_parameters =
                self.parse_token_separated_list_of(TokenKind::Comma, |myself| {
                    let arg_name = try_consume!(myself.tokens, TokenKind::Identifier(_))?;
                    try_consume!(myself.tokens, TokenKind::Colon)?;
                    let arg_type = myself.parse_type()?;
                    Some((arg_name.clone_identifiers_string(), *arg_type))
                })?;

            try_consume!(self.tokens, TokenKind::CloseRoundBracket)?;
            try_consume!(self.tokens, TokenKind::MinusGreaterThan)?;
            let return_type = self.parse_type()?;
            let function_code = self.parse_block()?;
            Some(TopLevelDeclaration::Function(
                TODO_INFO,
                None,
                function_name.clone_identifiers_string(),
                collected_parameters,
                *return_type,
                *function_code,
            ))
        })
    }

    fn parse_statement_expr(&mut self) -> Option<Box<Statement>> {
        let expr = self.parse_expr()?;
        Some(Box::new(Statement::InLineExpression(TODO_INFO, *expr)))
    }

    fn parse_statement(&mut self) -> Option<Box<Statement>> {
        let semicolon_terminated_statements = [
            Self::parse_let_declaration,
            Self::parse_var_declaration,
            Self::parse_assignment,
            Self::parse_return,
            Self::parse_statement_expr,
        ];
        let block_terminated_statements = [
            Self::parse_block,
            Self::parse_while,
            Self::parse_if,
            Self::parse_for,
        ];

        for parsing_method in semicolon_terminated_statements {
            if let Some(node) = rewinding_if_none!(self, {
                let node = parsing_method(self)?;
                try_consume!(self.tokens, TokenKind::SemiColon)?;
                Some(node)
            }) {
                return Some(node);
            }
        }

        for parsing_method in block_terminated_statements {
            if let Some(node) = rewinding_if_none!(self, {
                let node = parsing_method(self)?;
                Some(node)
            }) {
                return Some(node);
            }
        }

        None
    }

    fn parse_block(&mut self) -> Option<Box<Statement>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::OpenCurlyBracket)?;
            let mut statements = vec![];
            while let Some(s) = self.parse_statement() {
                statements.push(*s);
            }
            try_consume!(self.tokens, TokenKind::CloseCurlyBracket)?;
            Some(Box::new(Statement::Block(TODO_INFO, statements)))
        })
    }
}

impl Token<'_> {
    pub fn as_assignment_operator(&self) -> AssignmentOperator {
        match self.kind {
            TokenKind::Equal => AssignmentOperator::Equal,
            TokenKind::PlusEqual => AssignmentOperator::AddEqual,
            TokenKind::MinusEqual => AssignmentOperator::SubEqual,
            TokenKind::StarEqual => AssignmentOperator::MulEqual,
            TokenKind::SlashEqual => AssignmentOperator::DivEqual,
            _ => panic!("Token {:?} was parsed successfully in an assignment but there is no corresponding assignment operator", self)
        }
    }

    pub fn as_binary_operator(&self) -> BinaryOperator {
        match self.kind {
            TokenKind::Plus => BinaryOperator::Add,
            TokenKind::Minus => BinaryOperator::Sub,
            TokenKind::Star => BinaryOperator::Mul,
            TokenKind::Slash => BinaryOperator::Div,
            TokenKind::KeywordAnd => BinaryOperator::And,
            TokenKind::KeywordOr => BinaryOperator::Or,
            TokenKind::KeywordNor => BinaryOperator::Nor,
            TokenKind::KeywordXor => BinaryOperator::Xor,
            TokenKind::EqualEqual => BinaryOperator::Equal,
            TokenKind::ExclamationMarkEqual => BinaryOperator::NotEqual,
            TokenKind::GreaterThan => BinaryOperator::GreaterThan,
            TokenKind::GreaterThanEqual => BinaryOperator::GreaterThanEqual,
            TokenKind::LessThan => BinaryOperator::LessThan,
            TokenKind::LessThanEqual => BinaryOperator::LessThanEqual,
            _ => panic!("Token {:?} was parsed successfully in a binary expression but there is no corresponding binary operator", self)
        }
    }

    pub fn as_immediate_value(&self) -> ImmediateValue {
        match self.kind {
            TokenKind::Integer(x) => ImmediateValue::Integer(x),
            TokenKind::FloatingPoint(f) => ImmediateValue::FloatingPoint(f),
            TokenKind::KeywordTrue => ImmediateValue::Boolean(true),
            TokenKind::KeywordFalse => ImmediateValue::Boolean(false),
            TokenKind::String(s) => ImmediateValue::String(s.to_string()),
            _ => panic!("Token {:?} was used successfully in a parse but expected to be converted to an immediate value", self)
        }
    }

    pub fn clone_identifiers_string(&self) -> String {
        match self.kind {
            TokenKind::Identifier(s) => s.to_string(),
            _ => panic!(
                "Token {:?} was used successfully in a parse but expected to be an identifier",
                self
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const I: SourceInfo = SourceInfo {
        line: 0,
        column: 0,
        offset_in_source: 0,
    };

    fn tok(kind: TokenKind) -> Token {
        Token {
            kind,
            line: 0,
            column: 0,
        }
    }

    fn intval(x: i64) -> Box<Expression> {
        Box::new(Expression::Value(ImmediateValue::Integer(x)))
    }

    fn floatval(x: f64) -> Box<Expression> {
        Box::new(Expression::Value(ImmediateValue::FloatingPoint(x)))
    }

    fn stringval(s: &str) -> Box<Expression> {
        Box::new(Expression::Value(ImmediateValue::String(s.to_string())))
    }

    fn ident(s: &str) -> Box<Expression> {
        Box::new(Expression::Identifier(I, s.to_string()))
    }

    fn boolval(b: bool) -> Box<Expression> {
        Box::new(Expression::Value(ImmediateValue::Boolean(b)))
    }

    fn unaryop(op: UnaryOperator, e: Box<Expression>) -> Box<Expression> {
        Box::new(Expression::UnaryOperation(I, None, op, e))
    }

    fn binop(left: Box<Expression>, op: BinaryOperator, right: Box<Expression>) -> Box<Expression> {
        Box::new(Expression::BinaryOperation(I, None, left, op, right))
    }

    fn functioncall(name: &str, values: Vec<Expression>) -> Box<Expression> {
        Box::new(Expression::FunctionCall(I, None, ident(name), values))
    }

    #[test]
    fn try_parse_value() {
        let tokens = [
            tok(TokenKind::Integer(1)),
            tok(TokenKind::FloatingPoint(3.14)),
            tok(TokenKind::String("Hello")),
            tok(TokenKind::Identifier("count")),
            tok(TokenKind::Ampersand),
        ];
        let mut parser = Parser::new(&tokens);

        assert_eq!(parser.parse_value(), Some(intval(1)));
        assert_eq!(parser.parse_value(), Some(floatval(3.14)));
        assert_eq!(parser.parse_value(), Some(stringval("Hello")));
        assert_eq!(parser.parse_value(), Some(ident("count")));
        assert_eq!(parser.parse_value(), None);
    }

    #[test]
    fn parse_p5expr() {
        let tokens = [tok(TokenKind::Integer(1))];

        let mut parser = Parser::new(&tokens);
        assert_eq!(parser.parse_p5expr(), Some(intval(1)));

        let tokens = [tok(TokenKind::Minus), tok(TokenKind::Integer(1))];
        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p5expr(),
            Some(unaryop(UnaryOperator::Negation, intval(1)))
        );

        let tokens = [
            tok(TokenKind::Identifier("values")),
            tok(TokenKind::OpenSquareBracket),
            tok(TokenKind::Integer(1)),
            tok(TokenKind::CloseSquareBracket),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p5expr(),
            Some(binop(ident("values"), BinaryOperator::Indexing, intval(1)))
        );

        let tokens = [
            tok(TokenKind::Identifier("myfunc")),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::Integer(1)),
            tok(TokenKind::Comma),
            tok(TokenKind::Integer(2)),
            tok(TokenKind::CloseRoundBracket),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p5expr(),
            Some(functioncall("myfunc", vec![*intval(1), *intval(2)]))
        );
    }

    #[test]
    fn parse_p4expr() {
        let tokens = [
            tok(TokenKind::Integer(1)),
            tok(TokenKind::Star),
            tok(TokenKind::Integer(2)),
            tok(TokenKind::Slash),
            tok(TokenKind::Integer(3)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p4expr(),
            Some(binop(
                binop(intval(1), BinaryOperator::Mul, intval(2)),
                BinaryOperator::Div,
                intval(3)
            ))
        );
    }

    #[test]
    fn parse_p3expr() {
        let tokens = [
            tok(TokenKind::Integer(1)),
            tok(TokenKind::Plus),
            tok(TokenKind::Integer(2)),
            tok(TokenKind::Star),
            tok(TokenKind::Integer(3)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p3expr(),
            Some(binop(
                intval(1),
                BinaryOperator::Add,
                binop(intval(2), BinaryOperator::Mul, intval(3))
            ))
        );
    }

    #[test]
    fn parse_p2expr() {
        let tokens = [
            tok(TokenKind::Integer(1)),
            tok(TokenKind::Plus),
            tok(TokenKind::Integer(2)),
            tok(TokenKind::Star),
            tok(TokenKind::Integer(3)),
            tok(TokenKind::EqualEqual),
            tok(TokenKind::Integer(4)),
            tok(TokenKind::Plus),
            tok(TokenKind::Integer(3)),
            tok(TokenKind::LessThanEqual),
            tok(TokenKind::Integer(10)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p2expr(),
            Some(binop(
                binop(
                    binop(
                        intval(1),
                        BinaryOperator::Add,
                        binop(intval(2), BinaryOperator::Mul, intval(3))
                    ),
                    BinaryOperator::Equal,
                    binop(intval(4), BinaryOperator::Add, intval(3))
                ),
                BinaryOperator::LessThanEqual,
                intval(10)
            ))
        );
    }

    #[test]
    fn parse_p1expr() {
        let tokens = [
            tok(TokenKind::Integer(1)),
            tok(TokenKind::LessThan),
            tok(TokenKind::Integer(2)),
            tok(TokenKind::KeywordAnd),
            tok(TokenKind::Integer(3)),
            tok(TokenKind::LessThan),
            tok(TokenKind::Integer(99)),
            tok(TokenKind::KeywordOr),
            tok(TokenKind::Integer(4)),
            tok(TokenKind::EqualEqual),
            tok(TokenKind::Integer(5)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p1expr(),
            Some(binop(
                binop(
                    binop(intval(1), BinaryOperator::LessThan, intval(2)),
                    BinaryOperator::And,
                    binop(intval(3), BinaryOperator::LessThan, intval(99))
                ),
                BinaryOperator::Or,
                binop(intval(4), BinaryOperator::Equal, intval(5))
            ))
        );
    }

    #[test]
    fn parse_very_complex_expression() {
        // 2 + 3 * 5 * -1 - -2 == ((1 / 2.0) + "hello") and (x xor y) or (true != false nor 0.1 <= .1234 >= .123)
        let tokens = [
            tok(TokenKind::Integer(2)),
            tok(TokenKind::Plus),
            tok(TokenKind::Integer(3)),
            tok(TokenKind::Star),
            tok(TokenKind::Integer(5)),
            tok(TokenKind::Star),
            tok(TokenKind::Minus),
            tok(TokenKind::Integer(1)),
            tok(TokenKind::Minus),
            tok(TokenKind::Minus),
            tok(TokenKind::Integer(2)),
            tok(TokenKind::EqualEqual),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::Integer(1)),
            tok(TokenKind::Slash),
            tok(TokenKind::FloatingPoint(2.0)),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::Plus),
            tok(TokenKind::String("hello")),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::KeywordAnd),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::Identifier("x")),
            tok(TokenKind::KeywordXor),
            tok(TokenKind::Identifier("y")),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::KeywordOr),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::KeywordTrue),
            tok(TokenKind::ExclamationMarkEqual),
            tok(TokenKind::KeywordFalse),
            tok(TokenKind::KeywordNor),
            tok(TokenKind::FloatingPoint(0.1)),
            tok(TokenKind::LessThanEqual),
            tok(TokenKind::FloatingPoint(0.1234)),
            tok(TokenKind::GreaterThanEqual),
            tok(TokenKind::FloatingPoint(0.123)),
            tok(TokenKind::CloseRoundBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_expr(),
            Some(binop(
                binop(
                    binop(
                        binop(
                            binop(
                                intval(2),
                                BinaryOperator::Add,
                                binop(
                                    binop(intval(3), BinaryOperator::Mul, intval(5)),
                                    BinaryOperator::Mul,
                                    unaryop(UnaryOperator::Negation, intval(1))
                                )
                            ),
                            BinaryOperator::Sub,
                            unaryop(UnaryOperator::Negation, intval(2))
                        ),
                        BinaryOperator::Equal,
                        binop(
                            binop(intval(1), BinaryOperator::Div, floatval(2.0)),
                            BinaryOperator::Add,
                            stringval("hello")
                        )
                    ),
                    BinaryOperator::And,
                    binop(ident("x"), BinaryOperator::Xor, ident("y"))
                ),
                BinaryOperator::Or,
                binop(
                    binop(boolval(true), BinaryOperator::NotEqual, boolval(false)),
                    BinaryOperator::Nor,
                    binop(
                        binop(
                            floatval(0.1),
                            BinaryOperator::LessThanEqual,
                            floatval(0.1234)
                        ),
                        BinaryOperator::GreaterThanEqual,
                        floatval(0.123)
                    )
                )
            ))
        );
    }

    #[test]
    fn parenthesis_precedence() {
        let tokens = [
            tok(TokenKind::Integer(1)),
            tok(TokenKind::Star),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::Integer(2)),
            tok(TokenKind::Plus),
            tok(TokenKind::Integer(3)),
            tok(TokenKind::CloseRoundBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_expr(),
            Some(binop(
                intval(1),
                BinaryOperator::Mul,
                binop(intval(2), BinaryOperator::Add, intval(3))
            ))
        );
    }

    #[test]
    fn parse_let_declaration() {
        let tokens = [
            tok(TokenKind::KeywordLet),
            tok(TokenKind::Identifier("myval")),
            tok(TokenKind::Colon),
            tok(TokenKind::Identifier("int")),
            tok(TokenKind::Equal),
            tok(TokenKind::Integer(1)),
            tok(TokenKind::Plus),
            tok(TokenKind::Integer(1)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_let_declaration(),
            Some(Box::new(Statement::Declaration(
                I,
                "myval".to_string(),
                Some(Type::Integer),
                true,
                *binop(intval(1), BinaryOperator::Add, intval(1))
            )))
        );
    }

    #[test]
    fn parse_var_declaration() {
        let tokens = [
            tok(TokenKind::KeywordVar),
            tok(TokenKind::Identifier("myval")),
            tok(TokenKind::Colon),
            tok(TokenKind::Identifier("int")),
            tok(TokenKind::Equal),
            tok(TokenKind::Integer(1)),
            tok(TokenKind::Plus),
            tok(TokenKind::Integer(1)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_var_declaration(),
            Some(Box::new(Statement::Declaration(
                I,
                "myval".to_string(),
                Some(Type::Integer),
                false,
                *binop(intval(1), BinaryOperator::Add, intval(1))
            )))
        );
    }

    #[test]
    fn parse_assignment() {
        let tokens = [
            tok(TokenKind::Identifier("x")),
            tok(TokenKind::Equal),
            tok(TokenKind::Identifier("x")),
            tok(TokenKind::Plus),
            tok(TokenKind::Integer(1)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_assignment(),
            Some(Box::new(Statement::Assignment(
                I,
                *ident("x"),
                AssignmentOperator::Equal,
                *binop(ident("x"), BinaryOperator::Add, intval(1))
            )))
        );
    }

    #[test]
    fn parse_statement() {
        let tokens = [
            tok(TokenKind::Identifier("x")),
            tok(TokenKind::Equal),
            tok(TokenKind::Integer(1)),
            tok(TokenKind::SemiColon),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_statement(),
            Some(Box::new(Statement::Assignment(
                I,
                *ident("x"),
                AssignmentOperator::Equal,
                *intval(1)
            )))
        );
    }

    #[test]
    fn parse_if_else() {
        let tokens = [
            tok(TokenKind::KeywordIf),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::KeywordTrue),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::OpenCurlyBracket),
            tok(TokenKind::CloseCurlyBracket),
            tok(TokenKind::KeywordElse),
            tok(TokenKind::OpenCurlyBracket),
            tok(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_if(),
            Some(Box::new(Statement::If(
                I,
                *boolval(true),
                Box::new(Statement::Block(I, vec![])),
                Some(Box::new(Statement::Block(I, vec![])))
            )))
        );
    }

    #[test]
    fn parse_if_no_else() {
        let tokens = [
            tok(TokenKind::KeywordIf),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::KeywordTrue),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::OpenCurlyBracket),
            tok(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_if(),
            Some(Box::new(Statement::If(
                I,
                *boolval(true),
                Box::new(Statement::Block(I, vec![])),
                None
            )))
        );
    }

    #[test]
    fn parse_while() {
        let tokens = [
            tok(TokenKind::KeywordWhile),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::KeywordTrue),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::OpenCurlyBracket),
            tok(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_while(),
            Some(Box::new(Statement::While(
                I,
                *boolval(true),
                Box::new(Statement::Block(I, vec![]))
            )))
        );
    }

    #[test]
    fn parse_for() {
        let tokens = [
            tok(TokenKind::KeywordFor),
            tok(TokenKind::OpenCurlyBracket),
            tok(TokenKind::CloseCurlyBracket),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::KeywordTrue),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::OpenCurlyBracket),
            tok(TokenKind::CloseCurlyBracket),
            tok(TokenKind::OpenCurlyBracket),
            tok(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_for(),
            Some(Box::new(Statement::For(
                I,
                Box::new(Statement::Block(I, vec![])),
                *boolval(true),
                Box::new(Statement::Block(I, vec![])),
                Box::new(Statement::Block(I, vec![])),
            )))
        );
    }

    #[test]
    fn parse_return() {
        let tokens = [tok(TokenKind::KeywordReturn), tok(TokenKind::Integer(1))];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_return(),
            Some(Box::new(Statement::Return(I, *intval(1))))
        );
    }

    #[test]
    fn parse_block() {
        let tokens = [
            // {
            tok(TokenKind::OpenCurlyBracket),
            // x = 1;
            tok(TokenKind::Identifier("x")),
            tok(TokenKind::Equal),
            tok(TokenKind::Integer(1)),
            tok(TokenKind::SemiColon),
            // let y = 2;
            tok(TokenKind::KeywordLet),
            tok(TokenKind::Identifier("y")),
            tok(TokenKind::Colon),
            tok(TokenKind::Identifier("int")),
            tok(TokenKind::Equal),
            tok(TokenKind::Integer(2)),
            tok(TokenKind::SemiColon),
            // }
            tok(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_block(),
            Some(Box::new(Statement::Block(
                I,
                vec![
                    Statement::Assignment(I, *ident("x"), AssignmentOperator::Equal, *intval(1)),
                    Statement::Declaration(
                        I,
                        "y".to_string(),
                        Some(Type::Integer),
                        true,
                        *intval(2)
                    )
                ]
            )))
        );
    }

    #[test]
    fn parse_array_of_array_of_ints() {
        let tokens = [
            tok(TokenKind::OpenSquareBracket),
            tok(TokenKind::OpenSquareBracket),
            tok(TokenKind::Identifier("int")),
            tok(TokenKind::CloseSquareBracket),
            tok(TokenKind::CloseSquareBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_type(),
            Some(Box::new(Type::Array(Box::new(Type::Array(Box::new(
                Type::Integer
            ))))))
        );
    }

    #[test]
    fn parse_function_declaration() {
        let tokens = [
            tok(TokenKind::KeywordFn),
            tok(TokenKind::Identifier("myfunc")),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::Identifier("arg1")),
            tok(TokenKind::Colon),
            tok(TokenKind::Identifier("int")),
            tok(TokenKind::Comma),
            tok(TokenKind::Identifier("arg2")),
            tok(TokenKind::Colon),
            tok(TokenKind::Identifier("bool")),
            tok(TokenKind::Comma),
            tok(TokenKind::Identifier("arg3")),
            tok(TokenKind::Colon),
            tok(TokenKind::Identifier("string")),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::MinusGreaterThan),
            tok(TokenKind::Identifier("CustomType")),
            tok(TokenKind::OpenCurlyBracket),
            tok(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_root_function_declaration(),
            Some(TopLevelDeclaration::Function(
                I,
                None,
                "myfunc".to_string(),
                vec![
                    ("arg1".to_string(), Type::Integer),
                    ("arg2".to_string(), Type::Boolean),
                    ("arg3".to_string(), Type::String)
                ],
                Type::UserDefined("CustomType".to_string()),
                Statement::Block(I, vec![])
            ))
        );
    }
}

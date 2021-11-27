use crate::lexer::{Token, TokenKind};

pub fn parse<'a>(tokens: &'a [Token<'a>]) -> Result<Box<ASTNode<'a>>, &'static str> {
    match Parser::new(tokens).parse_program() {
        Some(ast) => Ok(ast),
        None => Err("Parsing Error: I don't have any other info for you"),
    }
}

#[derive(Debug, PartialEq)]
pub enum ASTNode<'a> {
    Root(Vec<Box<ASTNode<'a>>>),
    Value(&'a Token<'a>),
    UnaryOperation(&'a Token<'a>, Box<ASTNode<'a>>),
    BinaryOperation(Box<ASTNode<'a>>, &'a Token<'a>, Box<ASTNode<'a>>),
    LetDeclaration(&'a Token<'a>, Box<ASTNode<'a>>),
    VarDeclaration(&'a Token<'a>, Box<ASTNode<'a>>),
    Assignment(Box<ASTNode<'a>>, &'a Token<'a>, Box<ASTNode<'a>>),
    ArrayIndexing(Box<ASTNode<'a>>, Box<ASTNode<'a>>),
    Return(Box<ASTNode<'a>>),
    If(Box<ASTNode<'a>>, Box<ASTNode<'a>>, Option<Box<ASTNode<'a>>>),
    While(Box<ASTNode<'a>>, Box<ASTNode<'a>>),
    For(
        Box<ASTNode<'a>>,
        Box<ASTNode<'a>>,
        Box<ASTNode<'a>>,
        Box<ASTNode<'a>>,
    ),
    FunctionDeclaration(
        &'a Token<'a>,
        Vec<(&'a Token<'a>, Box<Type<'a>>)>,
        Box<Type<'a>>,
        Box<ASTNode<'a>>,
    ),
    FunctionCall(Box<ASTNode<'a>>, Vec<Box<ASTNode<'a>>>),
    Block(Vec<Box<ASTNode<'a>>>),
}

// TODO: More complex types
#[derive(Debug, PartialEq)]
pub enum Type<'a> {
    Base(&'a Token<'a>),
    Array(Box<Type<'a>>),
}

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
--cut-- ForExpr := for (<Statement>*) (<Expr>) (<Statement>*) <Block>
Expr := P1Expr
P1Expr := P2Expr ( (and|or|xor|nor) P2Expr)*
P2Expr := P3Expr ( (==|!=|<=|>=|<|>) P3Expr )*
P3Expr := P4Expr ( (+|-) P4Expr )*
P4Expr := <P5Expr> ( (*|/) <P5Expr> )*
P5Expr := - <Value> | <Value>[<Expr>] | <Value>( <Expr> (, <Expr>)* )
Value := <Constant> | '(' <Expr> ')' | identifier
Constant := integer | floating_point | string | KeywordTrue | KeywordFalse
*/

struct Parser<'a> {
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
        fn $name(&mut self) -> Option<Box<ASTNode<'a>>> {
            let first_value = self.$lower_level_parse_method()?;

            let mut parse_operator_value = || -> Option<(&Token, Box<ASTNode>)> {
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
                left = Box::new(ASTNode::BinaryOperation(left, operator, right));
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
        separator: TokenKind,
        parser: fn(&mut Self) -> Option<T>,
    ) -> Option<Vec<T>> {
        let mut collected_parameters = vec![];
        if let Some(p) = parser(&mut *self) {
            collected_parameters.push(p);
            while let Some(p) = rewinding_if_none!(self, {
                try_consume!(self.tokens, separator)?;
                parser(&mut *self)
            }) {
                collected_parameters.push(p)
            }
        }
        Some(collected_parameters)
    }

    pub fn parse_program(&mut self) -> Option<Box<ASTNode<'a>>> {
        let mut collected_functions = vec![];
        while let Some(function) = self.parse_function_declaration() {
            collected_functions.push(function);
        }

        Some(Box::new(ASTNode::Root(collected_functions)))
    }

    fn parse_value(&mut self) -> Option<Box<ASTNode<'a>>> {
        macro_rules! try_match_single_token_to_value {
            ($p:pat) => {
                if let Some(token) = try_consume!(self.tokens, $p) {
                    return Some(Box::new(ASTNode::Value(token)));
                }
            };
        }

        rewinding_if_none!(self, {
            try_match_single_token_to_value!(TokenKind::Integer(_));
            try_match_single_token_to_value!(TokenKind::String(_));
            try_match_single_token_to_value!(TokenKind::FloatingPoint(_));
            try_match_single_token_to_value!(TokenKind::Identifier(_));
            try_match_single_token_to_value!(TokenKind::KeywordTrue);
            try_match_single_token_to_value!(TokenKind::KeywordFalse);

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

    fn parse_expr(&mut self) -> Option<Box<ASTNode<'a>>> {
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

    fn parse_p5expr(&mut self) -> Option<Box<ASTNode<'a>>> {
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

    fn parse_unary_minus(&mut self) -> Option<Box<ASTNode<'a>>> {
        rewinding_if_none!(self, {
            let operator = try_consume!(self.tokens, TokenKind::Minus)?;
            let value = self.parse_value()?;
            Some(Box::new(ASTNode::UnaryOperation(operator, value)))
        })
    }

    fn parse_array_indexing(&mut self) -> Option<Box<ASTNode<'a>>> {
        rewinding_if_none!(self, {
            let indexable = self.parse_value()?;
            try_consume!(self.tokens, TokenKind::OpenSquareBracket)?;
            let index = self.parse_expr()?;
            try_consume!(self.tokens, TokenKind::CloseSquareBracket)?;
            Some(Box::new(ASTNode::ArrayIndexing(indexable, index)))
        })
    }

    fn parse_function_call(&mut self) -> Option<Box<ASTNode<'a>>> {
        rewinding_if_none!(self, {
            let function = self.parse_value()?;
            try_consume!(self.tokens, TokenKind::OpenRoundBracket)?;
            let arguments =
                self.parse_token_separated_list_of(TokenKind::Comma, |myself| myself.parse_expr())?;
            try_consume!(self.tokens, TokenKind::CloseRoundBracket)?;
            Some(Box::new(ASTNode::FunctionCall(function, arguments)))
        })
    }

    fn parse_let_declaration(&mut self) -> Option<Box<ASTNode<'a>>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordLet)?;
            let name = try_consume!(self.tokens, TokenKind::Identifier(_))?;
            try_consume!(self.tokens, TokenKind::Equal)?;
            let expr = self.parse_expr()?;
            Some(Box::new(ASTNode::LetDeclaration(name, expr)))
        })
    }

    fn parse_var_declaration(&mut self) -> Option<Box<ASTNode<'a>>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordVar)?;
            let name = try_consume!(self.tokens, TokenKind::Identifier(_))?;
            try_consume!(self.tokens, TokenKind::Equal)?;
            let expr = self.parse_expr()?;
            Some(Box::new(ASTNode::VarDeclaration(name, expr)))
        })
    }

    fn parse_assignment(&mut self) -> Option<Box<ASTNode<'a>>> {
        rewinding_if_none!(self, {
            // FIXME: Only a subset of expressions are valid lvalues
            let lvalue = self.parse_expr()?;
            let operator = try_consume!(
                self.tokens,
                TokenKind::Equal
                    | TokenKind::PlusEqual
                    | TokenKind::MinusEqual
                    | TokenKind::StarEqual
                    | TokenKind::SlashEqual
            )?;
            let rvalue = self.parse_expr()?;
            Some(Box::new(ASTNode::Assignment(lvalue, operator, rvalue)))
        })
    }

    fn parse_if(&mut self) -> Option<Box<ASTNode<'a>>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordIf)?;
            try_consume!(self.tokens, TokenKind::OpenRoundBracket)?;
            let condition = self.parse_expr()?;
            try_consume!(self.tokens, TokenKind::CloseRoundBracket)?;
            let true_branch = self.parse_block()?;
            match try_consume!(self.tokens, TokenKind::KeywordElse) {
                None => Some(Box::new(ASTNode::If(condition, true_branch, None))),
                _ => {
                    let false_branch = self.parse_block()?;
                    Some(Box::new(ASTNode::If(
                        condition,
                        true_branch,
                        Some(false_branch),
                    )))
                }
            }
        })
    }

    fn parse_while(&mut self) -> Option<Box<ASTNode<'a>>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordWhile)?;
            try_consume!(self.tokens, TokenKind::OpenRoundBracket)?;
            let condition = self.parse_expr()?;
            try_consume!(self.tokens, TokenKind::CloseRoundBracket)?;
            let loop_block = self.parse_block()?;
            Some(Box::new(ASTNode::While(condition, loop_block)))
        })
    }

    fn parse_for(&mut self) -> Option<Box<ASTNode<'a>>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordFor)?;
            let pre = self.parse_block()?;
            let condition = self.parse_expr()?;
            let post = self.parse_block()?;
            let loop_block = self.parse_block()?;
            Some(Box::new(ASTNode::For(pre, condition, post, loop_block)))
        })
    }

    fn parse_return(&mut self) -> Option<Box<ASTNode<'a>>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordReturn)?;
            let expression = self.parse_expr()?;
            Some(Box::new(ASTNode::Return(expression)))
        })
    }

    fn parse_type(&mut self) -> Option<Box<Type<'a>>> {
        if let Some(array_of_type) = rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::OpenSquareBracket)?;
            let inner_type = self.parse_type()?;
            try_consume!(self.tokens, TokenKind::CloseSquareBracket)?;
            Some(Box::new(Type::Array(inner_type)))
        }) {
            Some(array_of_type)
        } else {
            let base_type = try_consume!(self.tokens, TokenKind::Identifier(_))?;
            Some(Box::new(Type::Base(base_type)))
        }
    }

    fn parse_function_declaration(&mut self) -> Option<Box<ASTNode<'a>>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordFn)?;
            let function_name = try_consume!(self.tokens, TokenKind::Identifier(_))?;
            try_consume!(self.tokens, TokenKind::OpenRoundBracket)?;

            let collected_parameters =
                self.parse_token_separated_list_of(TokenKind::Comma, |myself| {
                    let arg_name = try_consume!(myself.tokens, TokenKind::Identifier(_))?;
                    try_consume!(myself.tokens, TokenKind::Colon)?;
                    let arg_type = myself.parse_type()?;
                    Some((arg_name, arg_type))
                })?;

            try_consume!(self.tokens, TokenKind::CloseRoundBracket)?;
            try_consume!(self.tokens, TokenKind::MinusGreaterThan)?;
            let return_type = self.parse_type()?;
            let function_code = self.parse_block()?;
            Some(Box::new(ASTNode::FunctionDeclaration(
                function_name,
                collected_parameters,
                return_type,
                function_code,
            )))
        })
    }

    fn parse_statement(&mut self) -> Option<Box<ASTNode<'a>>> {
        let semicolon_terminated_statements = [
            Self::parse_let_declaration,
            Self::parse_var_declaration,
            Self::parse_assignment,
            Self::parse_return,
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

    fn parse_block(&mut self) -> Option<Box<ASTNode<'a>>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::OpenCurlyBracket)?;
            let mut statements = vec![];
            while let Some(s) = self.parse_statement() {
                statements.push(s);
            }
            try_consume!(self.tokens, TokenKind::CloseCurlyBracket)?;
            Some(Box::new(ASTNode::Block(statements)))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(kind: TokenKind) -> Token {
        Token {
            kind: kind,
            line: 0,
            column: 0,
        }
    }

    fn binop<'a>(
        left: Box<ASTNode<'a>>,
        op: &'a Token<'a>,
        right: Box<ASTNode<'a>>,
    ) -> Box<ASTNode<'a>> {
        Box::new(ASTNode::BinaryOperation(left, op, right))
    }

    fn val<'a>(v: &'a Token<'a>) -> Box<ASTNode<'a>> {
        Box::new(ASTNode::Value(v))
    }

    macro_rules! basetype {
        ($val:expr) => {
            Box::new(Type::Base(&tok(TokenKind::Identifier($val))))
        };
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

        assert_eq!(parser.parse_value(), Some(val(&tok(TokenKind::Integer(1)))));
        assert_eq!(
            parser.parse_value(),
            Some(val(&tok(TokenKind::FloatingPoint(3.14))))
        );
        assert_eq!(
            parser.parse_value(),
            Some(val(&tok(TokenKind::String("Hello"))))
        );
        assert_eq!(
            parser.parse_value(),
            Some(val(&tok(TokenKind::Identifier("count"))))
        );
        assert_eq!(parser.parse_value(), None);
    }

    #[test]
    fn parse_p5expr() {
        let tokens = [tok(TokenKind::Integer(1))];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p5expr(),
            Some(val(&tok(TokenKind::Integer(1))))
        );

        let tokens = [tok(TokenKind::Minus), tok(TokenKind::Integer(1))];
        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p5expr(),
            Some(Box::new(ASTNode::UnaryOperation(
                &tok(TokenKind::Minus),
                val(&tok(TokenKind::Integer(1)))
            )))
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
            Some(Box::new(ASTNode::ArrayIndexing(
                val(&tok(TokenKind::Identifier("values"))),
                val(&tok(TokenKind::Integer(1)))
            )))
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
            Some(Box::new(ASTNode::FunctionCall(
                val(&tok(TokenKind::Identifier("myfunc"))),
                vec![
                    val(&tok(TokenKind::Integer(1))),
                    val(&tok(TokenKind::Integer(2)))
                ]
            )))
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
            tok(TokenKind::Plus),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p4expr(),
            Some(binop(
                binop(
                    val(&tok(TokenKind::Integer(1))),
                    &tok(TokenKind::Star),
                    val(&tok(TokenKind::Integer(2)))
                ),
                &tok(TokenKind::Slash),
                val(&tok(TokenKind::Integer(3)))
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
                val(&tok(TokenKind::Integer(1))),
                &tok(TokenKind::Plus),
                binop(
                    val(&tok(TokenKind::Integer(2))),
                    &tok(TokenKind::Star),
                    val(&tok(TokenKind::Integer(3)))
                )
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
                        val(&tok(TokenKind::Integer(1))),
                        &tok(TokenKind::Plus),
                        binop(
                            val(&tok(TokenKind::Integer(2))),
                            &tok(TokenKind::Star),
                            val(&tok(TokenKind::Integer(3)))
                        )
                    ),
                    &tok(TokenKind::EqualEqual),
                    binop(
                        val(&tok(TokenKind::Integer(4))),
                        &tok(TokenKind::Plus),
                        val(&tok(TokenKind::Integer(3)))
                    )
                ),
                &tok(TokenKind::LessThanEqual),
                val(&tok(TokenKind::Integer(10)))
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
                    binop(
                        val(&tok(TokenKind::Integer(1))),
                        &tok(TokenKind::LessThan),
                        val(&tok(TokenKind::Integer(2)))
                    ),
                    &tok(TokenKind::KeywordAnd),
                    binop(
                        val(&tok(TokenKind::Integer(3))),
                        &tok(TokenKind::LessThan),
                        val(&tok(TokenKind::Integer(99)))
                    )
                ),
                &tok(TokenKind::KeywordOr),
                binop(
                    val(&tok(TokenKind::Integer(4))),
                    &tok(TokenKind::EqualEqual),
                    val(&tok(TokenKind::Integer(5)))
                )
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
                                val(&tok(TokenKind::Integer(2))),
                                &tok(TokenKind::Plus),
                                binop(
                                    binop(
                                        val(&tok(TokenKind::Integer(3))),
                                        &tok(TokenKind::Star),
                                        val(&tok(TokenKind::Integer(5)))
                                    ),
                                    &tok(TokenKind::Star),
                                    Box::new(ASTNode::UnaryOperation(
                                        &tok(TokenKind::Minus),
                                        val(&tok(TokenKind::Integer(1)))
                                    ))
                                )
                            ),
                            &tok(TokenKind::Minus),
                            Box::new(ASTNode::UnaryOperation(
                                &tok(TokenKind::Minus),
                                val(&tok(TokenKind::Integer(2))),
                            ))
                        ),
                        &tok(TokenKind::EqualEqual),
                        binop(
                            binop(
                                val(&tok(TokenKind::Integer(1))),
                                &tok(TokenKind::Slash),
                                val(&tok(TokenKind::FloatingPoint(2.0)))
                            ),
                            &tok(TokenKind::Plus),
                            val(&tok(TokenKind::String("hello")))
                        )
                    ),
                    &tok(TokenKind::KeywordAnd),
                    binop(
                        val(&tok(TokenKind::Identifier("x"))),
                        &tok(TokenKind::KeywordXor),
                        val(&tok(TokenKind::Identifier("y")))
                    )
                ),
                &tok(TokenKind::KeywordOr),
                binop(
                    binop(
                        val(&tok(TokenKind::KeywordTrue)),
                        &tok(TokenKind::ExclamationMarkEqual),
                        val(&tok(TokenKind::KeywordFalse))
                    ),
                    &tok(TokenKind::KeywordNor),
                    binop(
                        binop(
                            val(&tok(TokenKind::FloatingPoint(0.1))),
                            &tok(TokenKind::LessThanEqual),
                            val(&tok(TokenKind::FloatingPoint(0.1234)))
                        ),
                        &tok(TokenKind::GreaterThanEqual),
                        val(&tok(TokenKind::FloatingPoint(0.123)))
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
                val(&tok(TokenKind::Integer(1))),
                &tok(TokenKind::Star),
                binop(
                    val(&tok(TokenKind::Integer(2))),
                    &tok(TokenKind::Plus),
                    val(&tok(TokenKind::Integer(3)))
                )
            ))
        );
    }

    #[test]
    fn parse_let_declaration() {
        let tokens = [
            tok(TokenKind::KeywordLet),
            tok(TokenKind::Identifier("myval")),
            tok(TokenKind::Equal),
            tok(TokenKind::Integer(1)),
            tok(TokenKind::Plus),
            tok(TokenKind::Integer(1)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_let_declaration(),
            Some(Box::new(ASTNode::LetDeclaration(
                &tok(TokenKind::Identifier("myval")),
                binop(
                    val(&tok(TokenKind::Integer(1))),
                    &tok(TokenKind::Plus),
                    val(&tok(TokenKind::Integer(1)))
                )
            )))
        );
    }

    #[test]
    fn parse_var_declaration() {
        let tokens = [
            tok(TokenKind::KeywordVar),
            tok(TokenKind::Identifier("myval")),
            tok(TokenKind::Equal),
            tok(TokenKind::Integer(1)),
            tok(TokenKind::Plus),
            tok(TokenKind::Integer(1)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_var_declaration(),
            Some(Box::new(ASTNode::VarDeclaration(
                &tok(TokenKind::Identifier("myval")),
                binop(
                    val(&tok(TokenKind::Integer(1))),
                    &tok(TokenKind::Plus),
                    val(&tok(TokenKind::Integer(1)))
                )
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
            Some(Box::new(ASTNode::Assignment(
                val(&tok(TokenKind::Identifier("x"))),
                &tok(TokenKind::Equal),
                binop(
                    val(&tok(TokenKind::Identifier("x"))),
                    &tok(TokenKind::Plus),
                    val(&tok(TokenKind::Integer(1)))
                )
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
            Some(Box::new(ASTNode::Assignment(
                val(&tok(TokenKind::Identifier("x"))),
                &tok(TokenKind::Equal),
                val(&tok(TokenKind::Integer(1)))
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
            Some(Box::new(ASTNode::If(
                val(&tok(TokenKind::KeywordTrue)),
                Box::new(ASTNode::Block(vec![])),
                Some(Box::new(ASTNode::Block(vec![])))
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
            Some(Box::new(ASTNode::If(
                val(&tok(TokenKind::KeywordTrue)),
                Box::new(ASTNode::Block(vec![])),
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
            Some(Box::new(ASTNode::While(
                val(&tok(TokenKind::KeywordTrue)),
                Box::new(ASTNode::Block(vec![]))
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
            Some(Box::new(ASTNode::For(
                Box::new(ASTNode::Block(vec![])),
                val(&tok(TokenKind::KeywordTrue)),
                Box::new(ASTNode::Block(vec![])),
                Box::new(ASTNode::Block(vec![])),
            )))
        );
    }

    #[test]
    fn parse_return() {
        let tokens = [tok(TokenKind::KeywordReturn), tok(TokenKind::Integer(1))];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_return(),
            Some(Box::new(ASTNode::Return(val(&tok(TokenKind::Integer(1))),)))
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
            tok(TokenKind::Equal),
            tok(TokenKind::Integer(2)),
            tok(TokenKind::SemiColon),
            // }
            tok(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_block(),
            Some(Box::new(ASTNode::Block(vec![
                Box::new(ASTNode::Assignment(
                    val(&tok(TokenKind::Identifier("x"))),
                    &tok(TokenKind::Equal),
                    val(&tok(TokenKind::Integer(1)))
                )),
                Box::new(ASTNode::LetDeclaration(
                    &tok(TokenKind::Identifier("y")),
                    val(&tok(TokenKind::Integer(2)))
                ))
            ])))
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
            Some(Box::new(Type::Array(Box::new(Type::Array(basetype!(
                "int"
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
            tok(TokenKind::Identifier("u32")),
            tok(TokenKind::Comma),
            tok(TokenKind::Identifier("arg2")),
            tok(TokenKind::Colon),
            tok(TokenKind::Identifier("bool")),
            tok(TokenKind::Comma),
            tok(TokenKind::Identifier("arg3")),
            tok(TokenKind::Colon),
            tok(TokenKind::Identifier("str")),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::MinusGreaterThan),
            tok(TokenKind::Identifier("ReturnType")),
            tok(TokenKind::OpenCurlyBracket),
            tok(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_function_declaration(),
            Some(Box::new(ASTNode::FunctionDeclaration(
                &tok(TokenKind::Identifier("myfunc")),
                vec![
                    (&tok(TokenKind::Identifier("arg1")), basetype!("u32")),
                    (&tok(TokenKind::Identifier("arg2")), basetype!("bool")),
                    (&tok(TokenKind::Identifier("arg3")), basetype!("str")),
                ],
                basetype!("ReturnType"),
                Box::new(ASTNode::Block(vec![]))
            )))
        );
    }
}

use crate::lexer::{Token, TokenKind};

#[derive(Debug, PartialEq)]
pub enum ASTNode<'a> {
    Value(&'a Token<'a>),
    UnaryOperation(&'a Token<'a>, Box<ASTNode<'a>>),
    BinaryOperation(Box<ASTNode<'a>>, &'a Token<'a>, Box<ASTNode<'a>>),
    LetDeclaration(&'a Token<'a>, Box<ASTNode<'a>>),
    VarDeclaration(&'a Token<'a>, Box<ASTNode<'a>>),
    Assignment(Box<ASTNode<'a>>, Box<ASTNode<'a>>),
    If(Box<ASTNode<'a>>, Box<ASTNode<'a>>, Option<Box<ASTNode<'a>>>),
    While(Box<ASTNode<'a>>, Box<ASTNode<'a>>),
    For(
        Vec<Box<ASTNode<'a>>>,
        Box<ASTNode<'a>>,
        Vec<Box<ASTNode<'a>>>,
    ),
    Block(Vec<Box<ASTNode<'a>>>),
}

/*
Program := (<Function>|<Type-Declaration>)*
Function := fn identifier ( <Empty>|<Arg> (,<Arg>)* ) :: <Type> <Block>
Type := identifier
Arg := identifier : <Type>
Block := { Statement* }
Statement := (<Block> | <LetDeclaration> | <VarDeclaration> | <Assignment> | <IfExpr> | <WhileExpr> | <ForExpr> | <Expr>) ;
LetDeclaration := let identifier (: <Type>) = <Expr>
VarDeclaration := var identifier (: <Type>) = <Expr>
Assignment := identifier = <Expr>
IfExpr := if ( <Expr> ) <Block> (else <Block>)
WhileExpr := while ( <Expr> ) <Block>
ForExpr := for (<Statement>*) (<Expr>) (<Statement>*) <Block>
Expr := P1Expr
P1Expr := P2Expr ( (and|or|xor|nor) P2Expr)*
P2Expr := P3Expr ( (==|!=|<=|>=|<|>) P3Expr )*
P3Expr := P4Expr ( (+|-) P4Expr )*
P4Expr := <Value> ( (*|/) <Value> )*
Value := '-'? (<Constant> | '(' <Expr> ')' | identifier
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
            match { $e } {
                Some(t) => Some(t),
                None => {
                    $self.tokens = saved_tokens;
                    None
                }
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

    fn parse_value(&mut self) -> Option<Box<ASTNode<'a>>> {
        macro_rules! try_match_single_token_to_value {
            ($p:pat) => {
                if let Some(token) = try_consume!(self.tokens, $p) {
                    return Some(Box::new(ASTNode::Value(token)));
                }
            };
        }

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
    }

    fn parse_expr(&mut self) -> Option<Box<ASTNode<'a>>> {
        self.parse_p1expr()
    }

    parse_expression_level!(
        parse_p4expr,
        TokenKind::Star | TokenKind::Slash,
        parse_value
    );
    parse_expression_level!(
        parse_p3expr,
        TokenKind::Plus | TokenKind::Minus,
        parse_p4expr
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
        parse_p1expr,
        TokenKind::KeywordAnd
            | TokenKind::KeywordOr
            | TokenKind::KeywordXor
            | TokenKind::KeywordNor,
        parse_p2expr
    );

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
            try_consume!(self.tokens, TokenKind::Equal)?;
            let rvalue = self.parse_expr()?;
            Some(Box::new(ASTNode::Assignment(lvalue, rvalue)))
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

    fn parse_statement(&mut self) -> Option<Box<ASTNode<'a>>> {
        let possible_statements = [
            Self::parse_let_declaration,
            Self::parse_var_declaration,
            Self::parse_assignment,
        ];

        for parsing_method in possible_statements {
            if let Some(node) = rewinding_if_none!(self, {
                let node = parsing_method(self)?;
                try_consume!(self.tokens, TokenKind::SemiColon)?;

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

    macro_rules! tok {
        ($p:expr) => {
            Token {
                kind: $p,
                line: 0,
                column: 0,
            }
        };
    }

    macro_rules! binop {
        ($left:expr, $op:expr, $right:expr) => {
            Box::new(ASTNode::BinaryOperation($left, $op, $right))
        };
    }

    macro_rules! val {
        ($val:expr) => {
            Box::new(ASTNode::Value($val))
        };
    }

    #[test]
    fn try_parse_value() {
        let tokens = [
            tok!(TokenKind::Integer(1)),
            tok!(TokenKind::FloatingPoint(3.14)),
            tok!(TokenKind::String("Hello")),
            tok!(TokenKind::Identifier("count")),
            tok!(TokenKind::Ampersand),
        ];
        let mut parser = Parser::new(&tokens);

        assert_eq!(
            parser.parse_value(),
            Some(val!(&tok!(TokenKind::Integer(1))))
        );
        assert_eq!(
            parser.parse_value(),
            Some(val!(&tok!(TokenKind::FloatingPoint(3.14))))
        );
        assert_eq!(
            parser.parse_value(),
            Some(val!(&tok!(TokenKind::String("Hello"))))
        );
        assert_eq!(
            parser.parse_value(),
            Some(val!(&tok!(TokenKind::Identifier("count"))))
        );
        assert_eq!(parser.parse_value(), None);
    }

    #[test]
    fn parse_p4expr() {
        let tokens = [
            tok!(TokenKind::Integer(1)),
            tok!(TokenKind::Star),
            tok!(TokenKind::Integer(2)),
            tok!(TokenKind::Slash),
            tok!(TokenKind::Integer(3)),
            tok!(TokenKind::Plus),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p4expr(),
            Some(binop!(
                binop!(
                    val!(&tok!(TokenKind::Integer(1))),
                    &tok!(TokenKind::Star),
                    val!(&tok!(TokenKind::Integer(2)))
                ),
                &tok!(TokenKind::Slash),
                val!(&tok!(TokenKind::Integer(3)))
            ))
        );
    }

    #[test]
    fn parse_p3expr() {
        let tokens = [
            tok!(TokenKind::Integer(1)),
            tok!(TokenKind::Plus),
            tok!(TokenKind::Integer(2)),
            tok!(TokenKind::Star),
            tok!(TokenKind::Integer(3)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p3expr(),
            Some(binop!(
                val!(&tok!(TokenKind::Integer(1))),
                &tok!(TokenKind::Plus),
                binop!(
                    val!(&tok!(TokenKind::Integer(2))),
                    &tok!(TokenKind::Star),
                    val!(&tok!(TokenKind::Integer(3)))
                )
            ))
        );
    }

    #[test]
    fn parse_p2expr() {
        let tokens = [
            tok!(TokenKind::Integer(1)),
            tok!(TokenKind::Plus),
            tok!(TokenKind::Integer(2)),
            tok!(TokenKind::Star),
            tok!(TokenKind::Integer(3)),
            tok!(TokenKind::EqualEqual),
            tok!(TokenKind::Integer(4)),
            tok!(TokenKind::Plus),
            tok!(TokenKind::Integer(3)),
            tok!(TokenKind::LessThanEqual),
            tok!(TokenKind::Integer(10)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p2expr(),
            Some(binop!(
                binop!(
                    binop!(
                        val!(&tok!(TokenKind::Integer(1))),
                        &tok!(TokenKind::Plus),
                        binop!(
                            val!(&tok!(TokenKind::Integer(2))),
                            &tok!(TokenKind::Star),
                            val!(&tok!(TokenKind::Integer(3)))
                        )
                    ),
                    &tok!(TokenKind::EqualEqual),
                    binop!(
                        val!(&tok!(TokenKind::Integer(4))),
                        &tok!(TokenKind::Plus),
                        val!(&tok!(TokenKind::Integer(3)))
                    )
                ),
                &tok!(TokenKind::LessThanEqual),
                val!(&tok!(TokenKind::Integer(10)))
            ))
        );
    }

    #[test]
    fn parse_p1expr() {
        let tokens = [
            tok!(TokenKind::Integer(1)),
            tok!(TokenKind::LessThan),
            tok!(TokenKind::Integer(2)),
            tok!(TokenKind::KeywordAnd),
            tok!(TokenKind::Integer(3)),
            tok!(TokenKind::LessThan),
            tok!(TokenKind::Integer(99)),
            tok!(TokenKind::KeywordOr),
            tok!(TokenKind::Integer(4)),
            tok!(TokenKind::EqualEqual),
            tok!(TokenKind::Integer(5)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p1expr(),
            Some(binop!(
                binop!(
                    binop!(
                        val!(&tok!(TokenKind::Integer(1))),
                        &tok!(TokenKind::LessThan),
                        val!(&tok!(TokenKind::Integer(2)))
                    ),
                    &tok!(TokenKind::KeywordAnd),
                    binop!(
                        val!(&tok!(TokenKind::Integer(3))),
                        &tok!(TokenKind::LessThan),
                        val!(&tok!(TokenKind::Integer(99)))
                    )
                ),
                &tok!(TokenKind::KeywordOr),
                binop!(
                    val!(&tok!(TokenKind::Integer(4))),
                    &tok!(TokenKind::EqualEqual),
                    val!(&tok!(TokenKind::Integer(5)))
                )
            ))
        );
    }

    #[test]
    fn parenthesis_precedence() {
        let tokens = [
            tok!(TokenKind::Integer(1)),
            tok!(TokenKind::Star),
            tok!(TokenKind::OpenRoundBracket),
            tok!(TokenKind::Integer(2)),
            tok!(TokenKind::Plus),
            tok!(TokenKind::Integer(3)),
            tok!(TokenKind::CloseRoundBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_expr(),
            Some(binop!(
                val!(&tok!(TokenKind::Integer(1))),
                &tok!(TokenKind::Star),
                binop!(
                    val!(&tok!(TokenKind::Integer(2))),
                    &tok!(TokenKind::Plus),
                    val!(&tok!(TokenKind::Integer(3)))
                )
            ))
        );
    }

    #[test]
    fn parse_let_declaration() {
        let tokens = [
            tok!(TokenKind::KeywordLet),
            tok!(TokenKind::Identifier("myval")),
            tok!(TokenKind::Equal),
            tok!(TokenKind::Integer(1)),
            tok!(TokenKind::Plus),
            tok!(TokenKind::Integer(1)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_let_declaration(),
            Some(Box::new(ASTNode::LetDeclaration(
                &tok!(TokenKind::Identifier("myval")),
                binop!(
                    val!(&tok!(TokenKind::Integer(1))),
                    &tok!(TokenKind::Plus),
                    val!(&tok!(TokenKind::Integer(1)))
                )
            )))
        );
    }

    #[test]
    fn parse_var_declaration() {
        let tokens = [
            tok!(TokenKind::KeywordVar),
            tok!(TokenKind::Identifier("myval")),
            tok!(TokenKind::Equal),
            tok!(TokenKind::Integer(1)),
            tok!(TokenKind::Plus),
            tok!(TokenKind::Integer(1)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_var_declaration(),
            Some(Box::new(ASTNode::VarDeclaration(
                &tok!(TokenKind::Identifier("myval")),
                binop!(
                    val!(&tok!(TokenKind::Integer(1))),
                    &tok!(TokenKind::Plus),
                    val!(&tok!(TokenKind::Integer(1)))
                )
            )))
        );
    }

    #[test]
    fn parse_assignment() {
        let tokens = [
            tok!(TokenKind::Identifier("x")),
            tok!(TokenKind::Equal),
            tok!(TokenKind::Identifier("x")),
            tok!(TokenKind::Plus),
            tok!(TokenKind::Integer(1)),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_assignment(),
            Some(Box::new(ASTNode::Assignment(
                val!(&tok!(TokenKind::Identifier("x"))),
                binop!(
                    val!(&tok!(TokenKind::Identifier("x"))),
                    &tok!(TokenKind::Plus),
                    val!(&tok!(TokenKind::Integer(1)))
                )
            )))
        );
    }

    #[test]
    fn parse_statement() {
        let tokens = [
            tok!(TokenKind::Identifier("x")),
            tok!(TokenKind::Equal),
            tok!(TokenKind::Integer(1)),
            tok!(TokenKind::SemiColon),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_statement(),
            Some(Box::new(ASTNode::Assignment(
                val!(&tok!(TokenKind::Identifier("x"))),
                val!(&tok!(TokenKind::Integer(1)))
            )))
        );
    }

    #[test]
    fn parse_if_else() {
        let tokens = [
            tok!(TokenKind::KeywordIf),
            tok!(TokenKind::OpenRoundBracket),
            tok!(TokenKind::KeywordTrue),
            tok!(TokenKind::CloseRoundBracket),
            tok!(TokenKind::OpenCurlyBracket),
            tok!(TokenKind::CloseCurlyBracket),
            tok!(TokenKind::KeywordElse),
            tok!(TokenKind::OpenCurlyBracket),
            tok!(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_if(),
            Some(Box::new(ASTNode::If(
                val!(&tok!(TokenKind::KeywordTrue)),
                Box::new(ASTNode::Block(vec![])),
                Some(Box::new(ASTNode::Block(vec![])))
            )))
        );
    }

    fn parse_if_no_else() {
        let tokens = [
            tok!(TokenKind::KeywordIf),
            tok!(TokenKind::OpenRoundBracket),
            tok!(TokenKind::KeywordTrue),
            tok!(TokenKind::CloseRoundBracket),
            tok!(TokenKind::OpenCurlyBracket),
            tok!(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_if(),
            Some(Box::new(ASTNode::If(
                val!(&tok!(TokenKind::KeywordTrue)),
                Box::new(ASTNode::Block(vec![])),
                None
            )))
        );
    }

    #[test]
    fn parse_while() {
        let tokens = [
            tok!(TokenKind::KeywordWhile),
            tok!(TokenKind::OpenRoundBracket),
            tok!(TokenKind::KeywordTrue),
            tok!(TokenKind::CloseRoundBracket),
            tok!(TokenKind::OpenCurlyBracket),
            tok!(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_while(),
            Some(Box::new(ASTNode::While(
                val!(&tok!(TokenKind::KeywordTrue)),
                Box::new(ASTNode::Block(vec![]))
            )))
        );
    }

    #[test]
    fn parse_block() {
        let tokens = [
            // {
            tok!(TokenKind::OpenCurlyBracket),
            // x = 1;
            tok!(TokenKind::Identifier("x")),
            tok!(TokenKind::Equal),
            tok!(TokenKind::Integer(1)),
            tok!(TokenKind::SemiColon),
            // let y = 2;
            tok!(TokenKind::KeywordLet),
            tok!(TokenKind::Identifier("y")),
            tok!(TokenKind::Equal),
            tok!(TokenKind::Integer(2)),
            tok!(TokenKind::SemiColon),
            // }
            tok!(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_block(),
            Some(Box::new(ASTNode::Block(vec![
                Box::new(ASTNode::Assignment(
                    val!(&tok!(TokenKind::Identifier("x"))),
                    val!(&tok!(TokenKind::Integer(1)))
                )),
                Box::new(ASTNode::LetDeclaration(
                    &tok!(TokenKind::Identifier("y")),
                    val!(&tok!(TokenKind::Integer(2)))
                ))
            ])))
        );
    }
}

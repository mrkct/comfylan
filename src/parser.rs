use crate::lexer::{Token, TokenKind};

#[derive(Debug, PartialEq)]
pub enum ASTNode<'a> {
    Value(&'a Token<'a>),
    UnaryOperation(&'a Token<'a>, Box<ASTNode<'a>>),
    BinaryOperation(Box<ASTNode<'a>>, &'a Token<'a>, Box<ASTNode<'a>>),
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
P1Expr := P2Expr ( (and|or) P2Expr)*
P2Expr := P3Expr ( (==|!=|<=|>=|<|>) P3Expr )*
P3Expr := P4Expr ( (*|/) P4Expr )*
P4Expr := <Value> ( (+|-) <Value> )*
Value := integer | floating_point | string | identifier | ( <Expr> )
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
    ($self:expr, $e:expr) => {{
        let saved_tokens = $self.tokens;
        match $e {
            Some(t) => Some(t),
            None => {
                $self.tokens = saved_tokens;
                None
            }
        }
    }};
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token<'a>]) -> Parser<'a> {
        Parser { tokens }
    }

    fn parse_value(&mut self) -> Option<ASTNode<'a>> {
        if let Some(token) = try_consume!(self.tokens, TokenKind::Integer(_)) {
            return Some(ASTNode::Value(token));
        }

        if let Some(token) = try_consume!(self.tokens, TokenKind::String(_)) {
            return Some(ASTNode::Value(token));
        }

        if let Some(token) = try_consume!(self.tokens, TokenKind::FloatingPoint(_)) {
            return Some(ASTNode::Value(token));
        }

        if let Some(token) = try_consume!(self.tokens, TokenKind::Identifier(_)) {
            return Some(ASTNode::Value(token));
        }

        None
    }

    fn parse_p4expr(&mut self) -> Option<Box<ASTNode<'a>>> {
        let first_value = self.parse_value()?;

        let mut parse_operator_value = || -> Option<(&Token, ASTNode)> {
            rewinding_if_none!(self, {
                let operator = try_consume!(self.tokens, TokenKind::Plus | TokenKind::Minus)?;
                if let Some(value) = self.parse_value() {
                    return Some((operator, value));
                }
                None
            })
        };

        let mut left = Box::new(first_value);
        while let Some((operator, right)) = parse_operator_value() {
            left = Box::new(ASTNode::BinaryOperation(left, operator, Box::new(right)));
        }
        Some(left)
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
            Some(ASTNode::Value(&tok!(TokenKind::Integer(1))))
        );
        assert_eq!(
            parser.parse_value(),
            Some(ASTNode::Value(&tok!(TokenKind::FloatingPoint(3.14))))
        );
        assert_eq!(
            parser.parse_value(),
            Some(ASTNode::Value(&tok!(TokenKind::String("Hello"))))
        );
        assert_eq!(
            parser.parse_value(),
            Some(ASTNode::Value(&tok!(TokenKind::Identifier("count"))))
        );
        assert_eq!(parser.parse_value(), None);
    }

    #[test]
    fn parse_p4expr() {
        let tokens = [
            tok!(TokenKind::Integer(1)),
            tok!(TokenKind::Plus),
            tok!(TokenKind::Integer(2)),
            tok!(TokenKind::Minus),
            tok!(TokenKind::Integer(3)),
            tok!(TokenKind::Star),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p4expr(),
            Some(binop!(
                binop!(
                    val!(&tok!(TokenKind::Integer(1))),
                    &tok!(TokenKind::Plus),
                    val!(&tok!(TokenKind::Integer(2)))
                ),
                &tok!(TokenKind::Minus),
                val!(&tok!(TokenKind::Integer(3)))
            ))
        );
    }
}

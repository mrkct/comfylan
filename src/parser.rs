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
Value := integer | floating_point | string | identifier
*/

type TokenStream<'a> = Vec<Token<'a>>;

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

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token<'a>]) -> Parser<'a> {
        Parser { tokens }
    }

    fn repeat_zero_or_more(
        &mut self,
        parse: fn(&mut Self) -> Option<ASTNode<'a>>,
    ) -> Vec<ASTNode<'a>> {
        let mut collected_nodes = vec![];
        while let Some(node) = parse(self) {
            collected_nodes.push(node);
        }
        collected_nodes
    }

    fn repeat_once_or_more(
        &mut self,
        parse: fn(&mut Self) -> Option<ASTNode<'a>>,
    ) -> Option<Vec<ASTNode<'a>>> {
        let first_node = parse(self);

        if let None = first_node {
            return None;
        }

        let mut collected_nodes = vec![first_node.unwrap()];
        while let Some(node) = parse(self) {
            collected_nodes.push(node);
        }
        Some(collected_nodes)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_parse_value() {
        let tokens = [
            Token {
                kind: TokenKind::Integer(1),
                line: 0,
                column: 0,
            },
            Token {
                kind: TokenKind::FloatingPoint(3.14),
                line: 0,
                column: 0,
            },
            Token {
                kind: TokenKind::String("Hello"),
                line: 0,
                column: 0,
            },
            Token {
                kind: TokenKind::Identifier("count"),
                line: 0,
                column: 0,
            },
            Token {
                kind: TokenKind::Ampersand,
                line: 0,
                column: 0,
            },
        ];
        let mut parser = Parser::new(&tokens);

        assert_eq!(
            parser.parse_value(),
            Some(ASTNode::Value(&Token {
                kind: TokenKind::Integer(1),
                line: 0,
                column: 0
            }))
        );
        assert_eq!(
            parser.parse_value(),
            Some(ASTNode::Value(&Token {
                kind: TokenKind::FloatingPoint(3.14),
                line: 0,
                column: 0
            }))
        );
        assert_eq!(
            parser.parse_value(),
            Some(ASTNode::Value(&Token {
                kind: TokenKind::String("Hello"),
                line: 0,
                column: 0
            }))
        );
        assert_eq!(
            parser.parse_value(),
            Some(ASTNode::Value(&Token {
                kind: TokenKind::Identifier("count"),
                line: 0,
                column: 0
            }))
        );
        assert_eq!(parser.parse_value(), None);
    }
}

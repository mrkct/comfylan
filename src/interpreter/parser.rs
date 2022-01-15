use crate::interpreter::{
    ast::*,
    lexer::{Token, TokenKind},
    typechecking::Type,
};
use std::collections::HashMap;

const TODO_INFO: SourceInfo = SourceInfo {
    line: 0,
    column: 0,
    offset_in_source: 0,
};

/*
Program := (<Function>|<Struct-Declaration>)*
Function := fn identifier ( <Empty>|<Arg> (,<Arg>)* ) -> <Type> <Block>
Struct-Declaration := struct <identifier> { <identifier> : <Type> (, <identifier> : <Type>)* }
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
P4Expr := <P5Expr> ( (*|/|.) <P5Expr> )*
P5Expr := (-|not) <P6Expr> | <P6Expr>
P6Expr := <P7Expr><P6Expr'>
P6Expr' := ( <',' separated list of <Expr>> )P6Expr' | [<Expr>]P6Expr' | '.' identifier | <Empty>
Value := <Constant> | '(' <Expr> ')' | '[' <',' separated list of <Expr>> ']' | <Value> '.' identifier | identifier
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
        separator: TokenKind,
        parser: fn(&mut Self) -> Option<T>,
    ) -> Option<Vec<T>> {
        let mut collected_parameters = vec![];
        if let Some(p) = parser(&mut *self) {
            collected_parameters.push(p);
            while let Some(p) = rewinding_if_none!(self, {
                // We need to do this check explicitly because try_consume takes a pattern, not a value
                // Putting 'separator' would accept anything, it has nothing to do with the 'separator' argument!
                let t = try_consume!(self.tokens, _)?;
                if t.kind != separator {
                    return None;
                }

                parser(&mut *self)
            }) {
                collected_parameters.push(p)
            }
        }
        Some(collected_parameters)
    }

    pub fn parse_program(&mut self) -> Option<Program> {
        let mut type_declarations = HashMap::new();
        let mut function_declarations = HashMap::new();

        loop {
            if let Some(type_declaration) = self.parse_struct_declaration() {
                type_declarations.insert(type_declaration.name.to_string(), type_declaration);
            } else if let Some(function_declaration) = self.parse_function_declaration() {
                function_declarations
                    .insert(function_declaration.name.to_string(), function_declaration);
            } else {
                break;
            }
        }

        Some(Program {
            type_declarations,
            function_declarations,
        })
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

            if let result @ Some(_) = self.parse_array_initialization() {
                return result;
            }

            None
        })
    }

    fn parse_array_initialization(&mut self) -> Option<Box<Expression>> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::OpenSquareBracket)?;
            let values =
                self.parse_token_separated_list_of(TokenKind::Comma, |myself| myself.parse_expr())?;
            try_consume!(self.tokens, TokenKind::CloseSquareBracket)?;
            Some(Box::new(Expression::ArrayInitializer(
                TODO_INFO, None, values,
            )))
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
        if let result @ Some(_) = self.parse_unary_expression() {
            result
        } else {
            self.parse_p6expr()
        }
    }

    fn parse_p6expr(&mut self) -> Option<Box<Expression>> {
        fn parse_p6expr_sub(myself: &mut Parser, prev_expr: Box<Expression>) -> Box<Expression> {
            // Function call
            if let Some(args) = rewinding_if_none!(myself, {
                try_consume!(myself.tokens, TokenKind::OpenRoundBracket)?;
                let args = myself.parse_token_separated_list_of(TokenKind::Comma, |myself| {
                    myself.parse_expr().map(|x| *x)
                })?;
                try_consume!(myself.tokens, TokenKind::CloseRoundBracket)?;
                Some(args)
            }) {
                return parse_p6expr_sub(
                    myself,
                    Box::new(Expression::FunctionCall(TODO_INFO, None, prev_expr, args)),
                );
            }

            // Array indexing
            if let Some(index) = rewinding_if_none!(myself, {
                try_consume!(myself.tokens, TokenKind::OpenSquareBracket)?;
                let index = myself.parse_expr()?;
                try_consume!(myself.tokens, TokenKind::CloseSquareBracket)?;
                Some(index)
            }) {
                return parse_p6expr_sub(
                    myself,
                    Box::new(Expression::BinaryOperation(
                        TODO_INFO,
                        None,
                        prev_expr,
                        BinaryOperator::Indexing,
                        index,
                    )),
                );
            }

            // Struct accessor
            if let Some(field) = rewinding_if_none!(myself, {
                try_consume!(myself.tokens, TokenKind::Period)?;
                let (_, field) = myself.try_consume_identifier()?;
                Some(field)
            }) {
                return parse_p6expr_sub(
                    myself,
                    Box::new(Expression::Accessor(TODO_INFO, prev_expr, field)),
                );
            }

            prev_expr
        }

        rewinding_if_none!(self, {
            let first_expr = self.parse_value()?;
            Some(parse_p6expr_sub(self, first_expr))
        })
    }

    fn parse_unary_expression(&mut self) -> Option<Box<Expression>> {
        rewinding_if_none!(self, {
            let operator = try_consume!(self.tokens, TokenKind::Minus | TokenKind::KeywordNot)?;
            let value = self.parse_value()?;
            Some(Box::new(Expression::UnaryOperation(
                TODO_INFO,
                None,
                match operator.kind {
                    TokenKind::Minus => UnaryOperator::Negation,
                    TokenKind::KeywordNot => UnaryOperator::Not,
                    _ => unreachable!(),
                },
                value,
            )))
        })
    }

    fn parse_let_declaration(&mut self) -> Option<Declaration> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordLet)?;
            let (_, name) = self.try_consume_identifier()?;
            let declared_type = rewinding_if_none!(self, {
                try_consume!(self.tokens, TokenKind::Colon)?;
                self.parse_type()
            });
            try_consume!(self.tokens, TokenKind::Equal)?;
            let expr = self.parse_expr()?;
            Some(Declaration {
                _info: TODO_INFO,
                name: name,
                expected_type: declared_type.map(|t| *t),
                immutable: true,
                rvalue: *expr,
            })
        })
    }

    fn parse_var_declaration(&mut self) -> Option<Declaration> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordVar)?;
            let (_, name) = self.try_consume_identifier()?;
            let declared_type = rewinding_if_none!(self, {
                try_consume!(self.tokens, TokenKind::Colon)?;
                self.parse_type()
            });
            try_consume!(self.tokens, TokenKind::Equal)?;
            let expr = self.parse_expr()?;
            Some(Declaration {
                _info: TODO_INFO,
                name: name,
                expected_type: declared_type.map(|t| *t),
                immutable: false,
                rvalue: *expr,
            })
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

    fn parse_assignment(&mut self) -> Option<Assignment> {
        rewinding_if_none!(self, {
            // FIXME: Only a subset of expressions are valid lvalues
            let lvalue = self.parse_expr()?;
            let operator = self.parse_assignment_operator()?;
            let rvalue = self.parse_expr()?;
            Some(Assignment {
                _info: TODO_INFO,
                lvalue: *lvalue,
                operator,
                rvalue: *rvalue,
            })
        })
    }

    fn parse_if(&mut self) -> Option<If> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordIf)?;
            try_consume!(self.tokens, TokenKind::OpenRoundBracket)?;
            let condition = self.parse_expr()?;
            try_consume!(self.tokens, TokenKind::CloseRoundBracket)?;
            let branch_true = self.parse_block()?;
            match try_consume!(self.tokens, TokenKind::KeywordElse) {
                None => Some(If {
                    _info: TODO_INFO,
                    condition: *condition,
                    branch_true,
                    branch_false: None,
                }),
                _ => {
                    let false_branch = self.parse_block()?;
                    Some(If {
                        _info: TODO_INFO,
                        condition: *condition,
                        branch_true,
                        branch_false: Some(false_branch),
                    })
                }
            }
        })
    }

    fn parse_while(&mut self) -> Option<While> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordWhile)?;
            try_consume!(self.tokens, TokenKind::OpenRoundBracket)?;
            let condition = self.parse_expr()?;
            try_consume!(self.tokens, TokenKind::CloseRoundBracket)?;
            let body = self.parse_block()?;
            Some(While {
                _info: TODO_INFO,
                condition: *condition,
                body,
            })
        })
    }

    fn parse_statement_list(&mut self) -> Option<Block> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::OpenRoundBracket)?;

            let statements = self.parse_token_separated_list_of(TokenKind::SemiColon, |myself| {
                let allowed_statements: [fn(&mut Self) -> Option<Statement>; 5] = [
                    |myself| Some(Self::parse_let_declaration(myself)?.into()),
                    |myself| Some(Self::parse_var_declaration(myself)?.into()),
                    |myself| Some(Self::parse_assignment(myself)?.into()),
                    |myself| Some(Self::parse_return(myself)?.into()),
                    |myself| Some(Self::parse_statement_expr(myself)?.into()),
                ];
                for parsing_method in allowed_statements {
                    if let r @ Some(_) = parsing_method(myself) {
                        return r;
                    }
                }
                None
            });

            if let Some(statements) = statements {
                try_consume!(self.tokens, TokenKind::SemiColon);
                try_consume!(self.tokens, TokenKind::CloseRoundBracket)?;
                Some(Block {
                    _info: TODO_INFO,
                    statements,
                })
            } else {
                None
            }
        })
    }

    fn parse_for(&mut self) -> Option<For> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordFor)?;
            let pre = self.parse_statement_list()?;
            let condition = rewinding_if_none!(self, {
                try_consume!(self.tokens, TokenKind::OpenRoundBracket)?;
                let condition = self.parse_expr()?;
                try_consume!(self.tokens, TokenKind::CloseRoundBracket)?;
                Some(condition)
            })?;
            let post = self.parse_statement_list()?;
            let body = self.parse_block()?;
            Some(For {
                _info: TODO_INFO,
                pre,
                condition: *condition,
                post,
                body,
            })
        })
    }

    fn parse_return(&mut self) -> Option<Return> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordReturn)?;
            let expression = self.parse_expr()?;
            Some(Return {
                _info: TODO_INFO,
                expression: *expression,
            })
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
                "void" => Box::new(Type::Void),
                _ => Box::new(Type::TypeReference(type_name)),
            })
        }
    }

    fn parse_function_declaration(&mut self) -> Option<FunctionDeclaration> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordFn)?;
            let (_, function_name) = self.try_consume_identifier()?;
            try_consume!(self.tokens, TokenKind::OpenRoundBracket)?;

            let collected_parameters =
                self.parse_token_separated_list_of(TokenKind::Comma, |myself| {
                    let (_, arg_name) = myself.try_consume_identifier()?;
                    try_consume!(myself.tokens, TokenKind::Colon)?;
                    let arg_type = myself.parse_type()?;
                    Some((arg_name, *arg_type))
                })?;

            try_consume!(self.tokens, TokenKind::CloseRoundBracket)?;
            try_consume!(self.tokens, TokenKind::MinusGreaterThan)?;
            let return_type = self.parse_type()?;
            let function_code = self.parse_block()?;

            Some(FunctionDeclaration {
                info: TODO_INFO,
                name: function_name,
                block: function_code,
                return_type: *return_type,
                args: collected_parameters,
            })
        })
    }

    fn try_consume_identifier(&mut self) -> Option<(&Token, String)> {
        if let Some(
            token
            @
            Token {
                kind: TokenKind::Identifier(name),
                ..
            },
        ) = try_consume!(self.tokens, TokenKind::Identifier(_))
        {
            return Some((token, name.to_string()));
        }
        None
    }

    fn parse_struct_declaration(&mut self) -> Option<TypeDeclaration> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::KeywordStruct)?;
            let (_name_token, name) = self.try_consume_identifier()?;
            try_consume!(self.tokens, TokenKind::OpenCurlyBracket)?;
            let fields = self.parse_token_separated_list_of(TokenKind::Comma, |myself| {
                let (_name_token, field_name) = myself.try_consume_identifier()?;
                try_consume!(myself.tokens, TokenKind::Colon)?;
                let field_type = myself.parse_type()?;
                Some((field_name, *field_type))
            })?;
            try_consume!(self.tokens, TokenKind::CloseCurlyBracket)?;

            Some(TypeDeclaration {
                info: TODO_INFO,
                name,
                fields: HashMap::from_iter(fields),
            })
        })
    }

    fn parse_statement_expr(&mut self) -> Option<StatementExpression> {
        let expr = self.parse_expr()?;
        Some(StatementExpression {
            _info: TODO_INFO,
            expression: *expr,
        })
    }

    fn parse_statement(&mut self) -> Option<Statement> {
        let semicolon_terminated_statements: [fn(&mut Self) -> Option<Statement>; 5] = [
            |myself| Some(Self::parse_let_declaration(myself)?.into()),
            |myself| Some(Self::parse_var_declaration(myself)?.into()),
            |myself| Some(Self::parse_assignment(myself)?.into()),
            |myself| Some(Self::parse_return(myself)?.into()),
            |myself| Some(Self::parse_statement_expr(myself)?.into()),
        ];
        let block_terminated_statements: [fn(&mut Self) -> Option<Statement>; 4] = [
            |myself| Some(Self::parse_block(myself)?.into()),
            |myself| Some(Self::parse_while(myself)?.into()),
            |myself| Some(Self::parse_if(myself)?.into()),
            |myself| Some(Self::parse_for(myself)?.into()),
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

    fn parse_block(&mut self) -> Option<Block> {
        rewinding_if_none!(self, {
            try_consume!(self.tokens, TokenKind::OpenCurlyBracket)?;
            let mut statements = vec![];
            while let Some(s) = self.parse_statement() {
                statements.push(s);
            }
            try_consume!(self.tokens, TokenKind::CloseCurlyBracket)?;
            Some(Block {
                _info: TODO_INFO,
                statements,
            })
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
            TokenKind::String(s) => {
                // FIXME: This does not support writing '\\n' (escaping the '\' character)
                ImmediateValue::String(s.replace(r#"\n"#, "\n").replace(r#"\t"#, "\t"))
            },
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

        let tokens = [
            tok(TokenKind::OpenSquareBracket),
            tok(TokenKind::Integer(1)),
            tok(TokenKind::Comma),
            tok(TokenKind::Integer(2)),
            tok(TokenKind::Comma),
            tok(TokenKind::Integer(2)),
            tok(TokenKind::Plus),
            tok(TokenKind::Integer(1)),
            tok(TokenKind::CloseSquareBracket),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_p5expr(),
            Some(Box::new(Expression::ArrayInitializer(
                TODO_INFO,
                None,
                vec![
                    intval(1),
                    intval(2),
                    binop(intval(2), BinaryOperator::Add, intval(1))
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
            Some(Declaration {
                _info: I,
                name: "myval".to_string(),
                expected_type: Some(Type::Integer),
                immutable: true,
                rvalue: *binop(intval(1), BinaryOperator::Add, intval(1))
            })
        );

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
            Some(Declaration {
                _info: I,
                name: "myval".to_string(),
                expected_type: None,
                immutable: true,
                rvalue: *binop(intval(1), BinaryOperator::Add, intval(1))
            })
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
            Some(Declaration {
                _info: I,
                name: "myval".to_string(),
                expected_type: Some(Type::Integer),
                immutable: false,
                rvalue: *binop(intval(1), BinaryOperator::Add, intval(1))
            })
        );

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
            Some(Declaration {
                _info: I,
                name: "myval".to_string(),
                expected_type: None,
                immutable: false,
                rvalue: *binop(intval(1), BinaryOperator::Add, intval(1))
            })
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
            Some(Assignment {
                _info: I,
                lvalue: *ident("x"),
                operator: AssignmentOperator::Equal,
                rvalue: *binop(ident("x"), BinaryOperator::Add, intval(1))
            })
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
            Some(Statement::Assignment(Assignment {
                _info: I,
                lvalue: *ident("x"),
                operator: AssignmentOperator::Equal,
                rvalue: *intval(1)
            }))
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
            Some(If {
                _info: I,
                condition: *boolval(true),
                branch_true: Block {
                    _info: I,
                    statements: vec![]
                },
                branch_false: Some(Block {
                    _info: I,
                    statements: vec![]
                })
            })
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
            Some(If {
                _info: I,
                condition: *boolval(true),
                branch_true: Block {
                    _info: I,
                    statements: vec![]
                },
                branch_false: None
            })
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
            Some(While {
                _info: I,
                condition: *boolval(true),
                body: Block {
                    _info: I,
                    statements: vec![]
                }
            })
        );
    }

    #[test]
    fn parse_for() {
        let tokens = [
            tok(TokenKind::KeywordFor),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::KeywordTrue),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::OpenCurlyBracket),
            tok(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_for(),
            Some(For {
                _info: I,
                pre: Block {
                    _info: I,
                    statements: vec![]
                },
                condition: *boolval(true),
                post: Block {
                    _info: I,
                    statements: vec![]
                },
                body: Block {
                    _info: I,
                    statements: vec![]
                }
            })
        );
    }

    #[test]
    fn parse_return() {
        let tokens = [tok(TokenKind::KeywordReturn), tok(TokenKind::Integer(1))];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_return(),
            Some(Return {
                _info: I,
                expression: *intval(1)
            })
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
            Some(Block {
                _info: I,
                statements: vec![
                    Statement::Assignment(Assignment {
                        _info: I,
                        lvalue: *ident("x"),
                        operator: AssignmentOperator::Equal,
                        rvalue: *intval(1)
                    }),
                    Statement::Declaration(Declaration {
                        _info: I,
                        name: "y".to_string(),
                        expected_type: Some(Type::Integer),
                        immutable: true,
                        rvalue: *intval(2)
                    })
                ]
            })
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
            tok(TokenKind::Identifier("int")),
            tok(TokenKind::OpenCurlyBracket),
            tok(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_function_declaration(),
            Some(FunctionDeclaration {
                info: I,
                name: "myfunc".into(),
                args: vec![
                    ("arg1".into(), Type::Integer),
                    ("arg2".into(), Type::Boolean),
                    ("arg3".into(), Type::String)
                ],
                return_type: Type::Integer,
                block: Block {
                    _info: I,
                    statements: vec![]
                }
            })
        );
    }

    #[test]
    fn parse_struct_declaration() {
        let tokens = [
            tok(TokenKind::KeywordStruct),
            tok(TokenKind::Identifier("Point")),
            tok(TokenKind::OpenCurlyBracket),
            tok(TokenKind::Identifier("x")),
            tok(TokenKind::Colon),
            tok(TokenKind::Identifier("int")),
            tok(TokenKind::Comma),
            tok(TokenKind::Identifier("y")),
            tok(TokenKind::Colon),
            tok(TokenKind::Identifier("float")),
            tok(TokenKind::CloseCurlyBracket),
        ];

        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_struct_declaration(),
            Some(TypeDeclaration {
                info: I,
                name: "Point".to_string(),
                fields: HashMap::from([
                    ("x".to_string(), Type::Integer),
                    ("y".to_string(), Type::FloatingPoint)
                ])
            })
        );
    }

    #[test]
    fn function_call_minus_1() {
        let tokens = vec![
            tok(TokenKind::Identifier("len")),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::Identifier("input")),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::Minus),
            tok(TokenKind::Integer(1)),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_expr(),
            Some(binop(
                functioncall("len", vec![*ident("input")]),
                BinaryOperator::Sub,
                intval(1)
            ))
        )
    }

    #[test]
    fn unary_not() {
        let tokens = vec![
            tok(TokenKind::KeywordNot),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::Identifier("len")),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::Identifier("array")),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::CloseRoundBracket),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_expr(),
            Some(Box::new(Expression::UnaryOperation(
                TODO_INFO,
                None,
                UnaryOperator::Not,
                Box::new(Expression::FunctionCall(
                    TODO_INFO,
                    None,
                    ident("len"),
                    vec![*ident("array")]
                ))
            )))
        )
    }

    #[test]
    fn parse_function_call_from_indexing() {
        let tokens = vec![
            tok(TokenKind::Identifier("x")),
            tok(TokenKind::OpenSquareBracket),
            tok(TokenKind::Integer(1)),
            tok(TokenKind::CloseSquareBracket),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::CloseRoundBracket),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_expr(),
            Some(Box::new(Expression::FunctionCall(
                I,
                None,
                binop(ident("x"), BinaryOperator::Indexing, intval(1)),
                vec![]
            )))
        );
    }

    #[test]
    fn parse_struct_accessor_operator() {
        let tokens = vec![
            tok(TokenKind::Identifier("player")),
            tok(TokenKind::Period),
            tok(TokenKind::Identifier("get_position")),
            tok(TokenKind::OpenRoundBracket),
            tok(TokenKind::CloseRoundBracket),
            tok(TokenKind::Period),
            tok(TokenKind::Identifier("x")),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_expr(),
            Some(Box::new(Expression::Accessor(
                I,
                Box::new(Expression::FunctionCall(
                    I,
                    None,
                    Box::new(Expression::Accessor(
                        I,
                        ident("player"),
                        "get_position".to_string()
                    )),
                    vec![]
                )),
                "x".to_string()
            )))
        );
    }

    #[test]
    fn chained_accessors() {
        let tokens = vec![
            tok(TokenKind::Identifier("a")),
            tok(TokenKind::Period),
            tok(TokenKind::Identifier("b")),
            tok(TokenKind::Period),
            tok(TokenKind::Identifier("c")),
        ];
        let mut parser = Parser::new(&tokens);
        assert_eq!(
            parser.parse_expr(),
            Some(Box::new(Expression::Accessor(
                I,
                Box::new(Expression::Accessor(I, ident("a"), "b".to_string())),
                "c".to_string()
            )))
        );
    }
}

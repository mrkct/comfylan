use lazy_static::lazy_static;
use regex::Regex;

#[derive(Debug, PartialEq, Clone)]
pub enum TokenKind<'a> {
    Integer(i64),
    FloatingPoint(f64),
    String(&'a str),
    Plus,
    Minus,
    Star,
    Slash,
    LessThan,
    GreaterThan,
    LessThanEqual,
    GreaterThanEqual,
    Equal,
    PlusEqual,
    MinusEqual,
    StarEqual,
    SlashEqual,
    EqualEqual,
    ExclamationMarkEqual,
    Ampersand,
    Pipe,
    Colon,
    Comma,
    Period,
    MinusGreaterThan,
    SemiColon,
    PipeGreaterThan,
    OpenRoundBracket,
    CloseRoundBracket,
    OpenSquareBracket,
    CloseSquareBracket,
    OpenCurlyBracket,
    CloseCurlyBracket,
    Identifier(&'a str),
    KeywordFn,
    KeywordType,
    KeywordVar,
    KeywordLet,
    KeywordIf,
    KeywordElse,
    KeywordFor,
    KeywordWhile,
    KeywordAnd,
    KeywordOr,
    KeywordNot,
    KeywordXor,
    KeywordNor,
    KeywordTrue,
    KeywordFalse,
    KeywordReturn,
    KeywordStruct,
    KeywordNew,
}

#[derive(Debug, PartialEq)]
pub struct Token<'a> {
    pub kind: TokenKind<'a>,
    pub line: u64,
    pub column: u64,
}

#[derive(Debug, PartialEq)]
pub enum ErrorType<'a> {
    UnrecognizedToken(&'a str),
}

#[derive(Debug, PartialEq)]
pub struct LexerError<'a> {
    pub error: ErrorType<'a>,
    pub line: u64,
    pub column: u64,
}

pub struct Lexer<'a> {
    remaining_string: &'a str,
    line: u64,
    column: u64,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &str) -> Lexer {
        Lexer {
            remaining_string: source,
            line: 0,
            column: 0,
        }
    }

    fn consume_left_whitespace(&mut self) {
        let mut find_start_of_non_whitespace = || {
            for (i, c) in self.remaining_string.char_indices() {
                if !c.is_whitespace() {
                    return i;
                }

                match c {
                    '\n' => {
                        self.column = 0;
                        self.line += 1;
                    }
                    _ => {
                        self.column += 1;
                    }
                };
            }
            self.remaining_string.len()
        };

        self.remaining_string = &self.remaining_string[find_start_of_non_whitespace()..];
    }

    fn consume_bad_token(&mut self) -> &'a str {
        for (i, c) in self.remaining_string.char_indices() {
            if c.is_whitespace() {
                let bad_token = &self.remaining_string[..i];
                self.remaining_string = &self.remaining_string[i..];
                return bad_token;
            }
            self.column += 1;
        }

        let bad_token = self.remaining_string;
        self.remaining_string = &self.remaining_string[self.remaining_string.len()..];
        bad_token
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token<'a>, LexerError<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.consume_left_whitespace();
        if self.remaining_string.is_empty() {
            return None;
        }

        lazy_static! {
            static ref FLOATING_POINT_REGEX: Regex = Regex::new(r#"^\d*\.\d+"#).unwrap();
            static ref INTEGER_REGEX: Regex = Regex::new(r#"^\d+"#).unwrap();
            static ref STRING_REGEX: Regex = Regex::new(r#"^"(\\"|[^"])*""#).unwrap();
            static ref IDENTIFIER_REGEX: Regex =
                Regex::new(r#"^([a-zA-Z_][_a-zA-Z0-9]*)"#).unwrap();
        }

        // NOTE: These needs to be ordered by descending length or it won't work
        const SYMBOLS: &[(&str, TokenKind)] = &[
            ("==", TokenKind::EqualEqual),
            ("<=", TokenKind::LessThanEqual),
            (">=", TokenKind::GreaterThanEqual),
            ("!=", TokenKind::ExclamationMarkEqual),
            ("|>", TokenKind::PipeGreaterThan),
            ("->", TokenKind::MinusGreaterThan),
            ("<", TokenKind::LessThan),
            (">", TokenKind::GreaterThan),
            ("+=", TokenKind::PlusEqual),
            ("-=", TokenKind::MinusEqual),
            ("*=", TokenKind::StarEqual),
            ("/=", TokenKind::SlashEqual),
            ("=", TokenKind::Equal),
            ("&", TokenKind::Ampersand),
            ("|", TokenKind::Pipe),
            ("+", TokenKind::Plus),
            ("-", TokenKind::Minus),
            ("*", TokenKind::Star),
            ("/", TokenKind::Slash),
            (",", TokenKind::Comma),
            (";", TokenKind::SemiColon),
            (":", TokenKind::Colon),
            (".", TokenKind::Period),
            ("(", TokenKind::OpenRoundBracket),
            (")", TokenKind::CloseRoundBracket),
            ("[", TokenKind::OpenSquareBracket),
            ("]", TokenKind::CloseSquareBracket),
            ("{", TokenKind::OpenCurlyBracket),
            ("}", TokenKind::CloseCurlyBracket),
        ];

        const KEYWORDS: &[(&str, TokenKind)] = &[
            ("fn", TokenKind::KeywordFn),
            ("type", TokenKind::KeywordType),
            ("var", TokenKind::KeywordVar),
            ("let", TokenKind::KeywordLet),
            ("if", TokenKind::KeywordIf),
            ("else", TokenKind::KeywordElse),
            ("for", TokenKind::KeywordFor),
            ("while", TokenKind::KeywordWhile),
            ("and", TokenKind::KeywordAnd),
            ("or", TokenKind::KeywordOr),
            ("not", TokenKind::KeywordNot),
            ("xor", TokenKind::KeywordXor),
            ("nor", TokenKind::KeywordNor),
            ("true", TokenKind::KeywordTrue),
            ("false", TokenKind::KeywordFalse),
            ("return", TokenKind::KeywordReturn),
            ("struct", TokenKind::KeywordStruct),
            ("new", TokenKind::KeywordNew),
        ];

        if let Some(captures) = FLOATING_POINT_REGEX.captures(self.remaining_string) {
            let string_capture = captures.get(0).unwrap().as_str();
            let token = Token {
                kind: TokenKind::FloatingPoint(string_capture.parse::<f64>().unwrap()),
                line: self.line,
                column: self.column,
            };
            self.column += string_capture.len() as u64;
            self.remaining_string = &self.remaining_string[string_capture.len()..];

            return Some(Ok(token));
        }

        if let Some(captures) = INTEGER_REGEX.captures(self.remaining_string) {
            let string_capture = captures.get(0).unwrap().as_str();
            let token = Token {
                kind: TokenKind::Integer(string_capture.parse::<i64>().unwrap()),
                line: self.line,
                column: self.column,
            };
            self.column += string_capture.len() as u64;
            self.remaining_string = &self.remaining_string[string_capture.len()..];

            return Some(Ok(token));
        }

        if let Some(captures) = STRING_REGEX.captures(self.remaining_string) {
            let string_capture = captures.get(0).unwrap().as_str();
            let token = Token {
                kind: TokenKind::String(&string_capture[1..string_capture.len() - 1]),
                line: self.line,
                column: self.column,
            };
            self.column += string_capture.len() as u64;
            self.remaining_string = &self.remaining_string[string_capture.len()..];

            return Some(Ok(token));
        }

        if let Some(captures) = IDENTIFIER_REGEX.captures(self.remaining_string) {
            let string_capture = captures.get(0).unwrap().as_str();
            let token = Token {
                kind: {
                    if let Some((_, token_kind)) = KEYWORDS
                        .iter()
                        .find(|(keyword, _)| *keyword == string_capture)
                    {
                        token_kind.clone()
                    } else {
                        TokenKind::Identifier(string_capture)
                    }
                },
                line: self.line,
                column: self.column,
            };
            self.column += string_capture.len() as u64;
            self.remaining_string = &self.remaining_string[string_capture.len()..];

            return Some(Ok(token));
        }

        for (symbol, token_kind) in SYMBOLS {
            if self.remaining_string.starts_with(symbol) {
                let token = Token {
                    kind: token_kind.clone(),
                    line: self.line,
                    column: self.column,
                };
                self.column += symbol.len() as u64;
                self.remaining_string = &self.remaining_string[symbol.len()..];
                return Some(Ok(token));
            }
        }

        let bad_token_line = self.line;
        let bad_token_column = self.column;
        let bad_token = self.consume_bad_token();

        Some(Err(LexerError {
            error: ErrorType::UnrecognizedToken(bad_token),
            line: bad_token_line,
            column: bad_token_column,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn good_tokens() {
        let lexer = Lexer::new(r#"1 33 55555 + - * / () "Hola Mundo!" 3.14 "esc\"apes" "#);
        let tokens = lexer.collect::<Vec<_>>();
        assert_eq!(
            tokens,
            vec![
                Ok(Token {
                    kind: TokenKind::Integer(1),
                    line: 0,
                    column: 0
                }),
                Ok(Token {
                    kind: TokenKind::Integer(33),
                    line: 0,
                    column: 2
                }),
                Ok(Token {
                    kind: TokenKind::Integer(55555),
                    line: 0,
                    column: 5
                }),
                Ok(Token {
                    kind: TokenKind::Plus,
                    line: 0,
                    column: 11
                }),
                Ok(Token {
                    kind: TokenKind::Minus,
                    line: 0,
                    column: 13
                }),
                Ok(Token {
                    kind: TokenKind::Star,
                    line: 0,
                    column: 15
                }),
                Ok(Token {
                    kind: TokenKind::Slash,
                    line: 0,
                    column: 17
                }),
                Ok(Token {
                    kind: TokenKind::OpenRoundBracket,
                    line: 0,
                    column: 19
                }),
                Ok(Token {
                    kind: TokenKind::CloseRoundBracket,
                    line: 0,
                    column: 20
                }),
                Ok(Token {
                    kind: TokenKind::String("Hola Mundo!"),
                    line: 0,
                    column: 22
                }),
                Ok(Token {
                    kind: TokenKind::FloatingPoint(3.14),
                    line: 0,
                    column: 36
                }),
                Ok(Token {
                    kind: TokenKind::String(r#"esc\"apes"#),
                    line: 0,
                    column: 41
                }),
            ]
        );
    }

    #[test]
    fn brackets() {
        let lexer = Lexer::new("() [] {} ( ) [ ] { }");
        let tokens = lexer.map(|t| t.unwrap().kind).collect::<Vec<_>>();
        assert_eq!(
            tokens,
            vec![
                TokenKind::OpenRoundBracket,
                TokenKind::CloseRoundBracket,
                TokenKind::OpenSquareBracket,
                TokenKind::CloseSquareBracket,
                TokenKind::OpenCurlyBracket,
                TokenKind::CloseCurlyBracket,
                TokenKind::OpenRoundBracket,
                TokenKind::CloseRoundBracket,
                TokenKind::OpenSquareBracket,
                TokenKind::CloseSquareBracket,
                TokenKind::OpenCurlyBracket,
                TokenKind::CloseCurlyBracket
            ]
        );
    }

    #[test]
    fn operators() {
        let lexer = Lexer::new("== = != < > <= >= & | |> ; :");
        let tokens = lexer.map(|t| t.unwrap().kind).collect::<Vec<_>>();
        assert_eq!(
            tokens,
            vec![
                TokenKind::EqualEqual,
                TokenKind::Equal,
                TokenKind::ExclamationMarkEqual,
                TokenKind::LessThan,
                TokenKind::GreaterThan,
                TokenKind::LessThanEqual,
                TokenKind::GreaterThanEqual,
                TokenKind::Ampersand,
                TokenKind::Pipe,
                TokenKind::PipeGreaterThan,
                TokenKind::SemiColon,
                TokenKind::Colon
            ]
        );
    }

    #[test]
    fn tokens_between_whitespace() {
        let lexer = Lexer::new("1\n2\t\t3\n  4 ");
        let tokens = lexer.collect::<Vec<_>>();
        assert_eq!(
            tokens,
            vec![
                Ok(Token {
                    kind: TokenKind::Integer(1),
                    line: 0,
                    column: 0
                }),
                Ok(Token {
                    kind: TokenKind::Integer(2),
                    line: 1,
                    column: 0
                }),
                Ok(Token {
                    kind: TokenKind::Integer(3),
                    line: 1,
                    column: 3
                }),
                Ok(Token {
                    kind: TokenKind::Integer(4),
                    line: 2,
                    column: 2
                })
            ]
        );
    }

    #[test]
    fn keywords_and_identifiers() {
        let lexer = Lexer::new(
            "fn fni type typee false falsex true truex if for while and or not struct new",
        );
        let tokens = lexer.map(|token| token.unwrap().kind).collect::<Vec<_>>();
        assert_eq!(
            tokens,
            vec![
                TokenKind::KeywordFn,
                TokenKind::Identifier("fni"),
                TokenKind::KeywordType,
                TokenKind::Identifier("typee"),
                TokenKind::KeywordFalse,
                TokenKind::Identifier("falsex"),
                TokenKind::KeywordTrue,
                TokenKind::Identifier("truex"),
                TokenKind::KeywordIf,
                TokenKind::KeywordFor,
                TokenKind::KeywordWhile,
                TokenKind::KeywordAnd,
                TokenKind::KeywordOr,
                TokenKind::KeywordNot,
                TokenKind::KeywordStruct,
                TokenKind::KeywordNew
            ]
        );
    }

    #[test]
    fn unexpected_token() {
        let mut lexer = Lexer::new(r#"1 £"#);

        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Integer(1),
                line: 0,
                column: 0
            }),)
        );
        assert_eq!(
            lexer.next(),
            Some(Err(LexerError {
                error: ErrorType::UnrecognizedToken("£"),
                line: 0,
                column: 2
            }))
        );
    }
}

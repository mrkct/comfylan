use lazy_static::lazy_static;
use regex::Regex;

#[derive(Debug, PartialEq)]
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
    EqualEqual,
    ExclamationMarkEqual,
    Ampersand,
    Pipe,
    Colon,
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
    KeywordFor,
    KeywordWhile,
    KeywordAnd,
    KeywordOr,
    KeywordNot,
    KeywordTrue,
    KeywordFalse,
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
            static ref FLOATING_POINT: Regex = Regex::new(r#"^\d*\.\d+"#).unwrap();
            static ref INTEGER: Regex = Regex::new(r#"^\d+"#).unwrap();
            static ref STRING: Regex = Regex::new(r#"^"(\\"|[^"])*""#).unwrap();
            static ref EQUAL_EQUAL: Regex = Regex::new(r#"^=="#).unwrap();
            static ref LESS_THAN_EQUAL: Regex = Regex::new(r#"^<="#).unwrap();
            static ref GREATER_THAN_EQUAL: Regex = Regex::new(r#"^>="#).unwrap();
            static ref EXCLAMATION_MARK_EQUAL: Regex = Regex::new(r#"^!="#).unwrap();
            static ref PIPE_GREATER_THAN: Regex = Regex::new(r#"^\|>"#).unwrap();
            static ref LESS_THAN: Regex = Regex::new(r#"^<"#).unwrap();
            static ref GREATER_THAN: Regex = Regex::new(r#"^>"#).unwrap();
            static ref EQUAL: Regex = Regex::new(r#"^="#).unwrap();
            static ref AMPERSAND: Regex = Regex::new(r#"^&"#).unwrap();
            static ref PIPE: Regex = Regex::new(r#"^\|"#).unwrap();
            static ref PLUS: Regex = Regex::new(r#"^\+"#).unwrap();
            static ref MINUS: Regex = Regex::new(r#"^-"#).unwrap();
            static ref STAR: Regex = Regex::new(r#"^\*"#).unwrap();
            static ref SLASH: Regex = Regex::new(r#"^/"#).unwrap();
            static ref COMMA: Regex = Regex::new(r#"^,"#).unwrap();
            static ref COLON: Regex = Regex::new(r#"^:"#).unwrap();
            static ref SEMICOLON: Regex = Regex::new(r#"^;"#).unwrap();
            static ref OPEN_ROUND_BRACKET: Regex = Regex::new(r#"^\("#).unwrap();
            static ref CLOSE_ROUND_BRACKET: Regex = Regex::new(r#"^\)"#).unwrap();
            static ref OPEN_SQUARE_BRACKET: Regex = Regex::new(r#"^\["#).unwrap();
            static ref CLOSE_SQUARE_BRACKET: Regex = Regex::new(r#"^\]"#).unwrap();
            static ref OPEN_CURLY_BRACKET: Regex = Regex::new(r#"^\{"#).unwrap();
            static ref CLOSE_CURLY_BRACKET: Regex = Regex::new(r#"^\}"#).unwrap();
            static ref KEYWORD_FN: Regex = Regex::new(r#"^fn(\s|$)"#).unwrap();
            static ref KEYWORD_TYPE: Regex = Regex::new(r#"^type(\s|$)"#).unwrap();
            static ref KEYWORD_VAR: Regex = Regex::new(r#"^var(\s|$)"#).unwrap();
            static ref KEYWORD_LET: Regex = Regex::new(r#"^let(\s|$)"#).unwrap();
            static ref KEYWORD_IF: Regex = Regex::new(r#"^if(\s|$)"#).unwrap();
            static ref KEYWORD_FOR: Regex = Regex::new(r#"^for(\s|$)"#).unwrap();
            static ref KEYWORD_WHILE: Regex = Regex::new(r#"^while(\s|$)"#).unwrap();
            static ref KEYWORD_AND: Regex = Regex::new(r#"^and(\s|$)"#).unwrap();
            static ref KEYWORD_OR: Regex = Regex::new(r#"^or(\s|$)"#).unwrap();
            static ref KEYWORD_NOT: Regex = Regex::new(r#"^not(\s|$)"#).unwrap();
            static ref KEYWORD_TRUE: Regex = Regex::new(r#"^true(\s|$)"#).unwrap();
            static ref KEYWORD_FALSE: Regex = Regex::new(r#"^false(\s|$)"#).unwrap();
            static ref IDENTIFIER: Regex = Regex::new(r#"^([a-zA-Z][a-zA-Z0-9]*)(\s|$)"#).unwrap();
            static ref TOKENS_REGEXEPS: Vec<(&'static Regex, fn(&str) -> TokenKind)> = vec![
                (&FLOATING_POINT, |fp| {
                    TokenKind::FloatingPoint(fp.parse::<f64>().unwrap())
                }),
                (&INTEGER, |i| {
                    TokenKind::Integer(i.parse::<i64>().unwrap())
                }),
                (&STRING, |s| { TokenKind::String(&s[1..s.len() - 1]) }),
                (&EQUAL_EQUAL, |_| { TokenKind::EqualEqual }),
                (&LESS_THAN_EQUAL, |_| { TokenKind::LessThanEqual }),
                (&GREATER_THAN_EQUAL, |_| { TokenKind::GreaterThanEqual }),
                (&EXCLAMATION_MARK_EQUAL, |_| {
                    TokenKind::ExclamationMarkEqual
                }),
                (&PIPE_GREATER_THAN, |_| { TokenKind::PipeGreaterThan }),
                (&LESS_THAN, |_| { TokenKind::LessThan }),
                (&GREATER_THAN, |_| { TokenKind::GreaterThan }),
                (&EQUAL, |_| { TokenKind::Equal }),
                (&AMPERSAND, |_| { TokenKind::Ampersand }),
                (&PIPE, |_| { TokenKind::Pipe }),
                (&PLUS, |_| { TokenKind::Plus }),
                (&MINUS, |_| { TokenKind::Minus }),
                (&STAR, |_| { TokenKind::Star }),
                (&SLASH, |_| { TokenKind::Slash }),
                (&COLON, |_| { TokenKind::Colon }),
                (&SEMICOLON, |_| { TokenKind::SemiColon }),
                (&OPEN_ROUND_BRACKET, |_| { TokenKind::OpenRoundBracket }),
                (&CLOSE_ROUND_BRACKET, |_| { TokenKind::CloseRoundBracket }),
                (&OPEN_SQUARE_BRACKET, |_| { TokenKind::OpenSquareBracket }),
                (&CLOSE_SQUARE_BRACKET, |_| { TokenKind::CloseSquareBracket }),
                (&OPEN_CURLY_BRACKET, |_| { TokenKind::OpenCurlyBracket }),
                (&CLOSE_CURLY_BRACKET, |_| { TokenKind::CloseCurlyBracket }),
                (&KEYWORD_FN, |_| { TokenKind::KeywordFn }),
                (&KEYWORD_TYPE, |_| { TokenKind::KeywordType }),
                (&KEYWORD_VAR, |_| { TokenKind::KeywordVar }),
                (&KEYWORD_LET, |_| { TokenKind::KeywordLet }),
                (&KEYWORD_IF, |_| { TokenKind::KeywordIf }),
                (&KEYWORD_FOR, |_| { TokenKind::KeywordFor }),
                (&KEYWORD_WHILE, |_| { TokenKind::KeywordWhile }),
                (&KEYWORD_AND, |_| { TokenKind::KeywordAnd }),
                (&KEYWORD_OR, |_| { TokenKind::KeywordOr }),
                (&KEYWORD_NOT, |_| { TokenKind::KeywordNot }),
                (&KEYWORD_TRUE, |_| { TokenKind::KeywordTrue }),
                (&KEYWORD_FALSE, |_| { TokenKind::KeywordFalse }),
                (&IDENTIFIER, |i| { TokenKind::Identifier(i.trim_end()) })
            ];
        };

        for (regex, converter) in TOKENS_REGEXEPS.iter() {
            if let Some(captures) = regex.captures(self.remaining_string) {
                let string_capture = captures.get(0).unwrap().as_str();
                let token = Token {
                    kind: converter(string_capture),
                    line: self.line,
                    column: self.column,
                };
                self.column += string_capture.len() as u64;
                self.remaining_string = &self.remaining_string[string_capture.len()..];

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
        let lexer = Lexer::new("fn fni type typee false falsex true truex if for while and or not");
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
                TokenKind::KeywordNot
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

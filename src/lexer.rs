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
    OpenParenthesis,
    CloseParethesis,
    Identifier(&'a str),
    KeywordFn,
    KeywordType
}

#[derive(Debug, PartialEq)]
pub struct Token<'a> {
    kind: TokenKind<'a>,
    line: u64,
    column: u64
}

#[derive(Debug, PartialEq)]
pub enum ErrorType<'a> {
    UnrecognizedToken(&'a str)
}

#[derive(Debug, PartialEq)]
pub struct LexerError<'a> {
    error: ErrorType<'a>,
    line: u64,
    column: u64,
}

pub struct Lexer<'a> {
    remaining_string: &'a str,
    line: u64,
    column: u64,
}   

impl<'a> Lexer<'a> {
    pub fn new(source: &str) -> Lexer {
        Lexer { remaining_string:source, line:0, column:0 }
    }

    fn consume_left_whitespace(&mut self) {
        let mut find_start_of_non_whitespace = || {
            for (i, c) in self.remaining_string.char_indices() {
                if !c.is_whitespace() {
                    return i
                }

                match c {
                    '\n' => { self.column = 0; self.line += 1; }
                    _ => { self.column += 1; }
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
            static ref PLUS: Regex = Regex::new(r#"^\+"#).unwrap();
            static ref MINUS: Regex = Regex::new(r#"^-"#).unwrap();
            static ref STAR: Regex = Regex::new(r#"^\*"#).unwrap();
            static ref SLASH: Regex = Regex::new(r#"^/"#).unwrap();
            static ref OPEN_PARENTHESIS: Regex = Regex::new(r#"^\("#).unwrap();
            static ref CLOSE_PARENTHESIS: Regex = Regex::new(r#"^\)"#).unwrap();
            static ref KEYWORD_FN: Regex = Regex::new(r#"^fn(\s|$)"#).unwrap();
            static ref KEYWORD_TYPE: Regex = Regex::new(r#"^type(\s|$)"#).unwrap();
            static ref IDENTIFIER: Regex = Regex::new(r#"^([a-zA-Z][a-zA-Z0-9]*)(\s|$)"#).unwrap();

            static ref TOKENS_REGEXEPS: Vec<(&'static Regex, fn(&str) -> TokenKind)> = vec![
                (&FLOATING_POINT, |fp| { TokenKind::FloatingPoint(fp.parse::<f64>().unwrap())}),
                (&INTEGER, |i| { TokenKind::Integer(i.parse::<i64>().unwrap()) }),
                (&STRING, |s| { TokenKind::String(&s[1..s.len()-1]) }),
                (&PLUS, |_| { TokenKind::Plus }),
                (&MINUS, |_| { TokenKind::Minus }),
                (&STAR, |_| { TokenKind::Star }),
                (&SLASH, |_| { TokenKind::Slash }),
                (&OPEN_PARENTHESIS, |_| { TokenKind::OpenParenthesis }),
                (&CLOSE_PARENTHESIS, |_| { TokenKind::CloseParethesis }),
                (&KEYWORD_FN, |_| { TokenKind::KeywordFn }),
                (&KEYWORD_TYPE, |_| { TokenKind::KeywordType }),
                (&IDENTIFIER, |i| { TokenKind::Identifier(i.trim_end())})
            ];
        };

        for (regex, converter) in TOKENS_REGEXEPS.iter() {
            if let Some(captures) = regex.captures(self.remaining_string) {                                
                let string_capture = captures.get(0).unwrap().as_str();
                let token = Token {kind: converter(string_capture), line: self.line, column: self.column};
                self.column += string_capture.len() as u64;
                self.remaining_string = &self.remaining_string[string_capture.len()..];

                return Some(Ok(token));
            }
        }

        let bad_token_line = self.line;
        let bad_token_column = self.column;
        let bad_token = self.consume_bad_token();

        Some(Err(LexerError {error: ErrorType::UnrecognizedToken(bad_token), line: bad_token_line, column: bad_token_column}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn good_tokens() {
        let lexer = Lexer::new(r#"1 33 55555 + - * / () "Hola Mundo!" 3.14 "esc\"apes"   "#);
        let tokens = lexer.collect::<Vec<_>>();
        assert_eq!(tokens, vec![
            Ok(Token {kind: TokenKind::Integer(1), line: 0, column: 0}),
            Ok(Token {kind: TokenKind::Integer(33), line: 0, column: 2}),
            Ok(Token {kind: TokenKind::Integer(55555), line: 0, column: 5}),
            Ok(Token {kind: TokenKind::Plus, line: 0, column: 11}),
            Ok(Token {kind: TokenKind::Minus, line: 0, column: 13}),
            Ok(Token {kind: TokenKind::Star, line: 0, column: 15}),
            Ok(Token {kind: TokenKind::Slash, line: 0, column: 17}),
            Ok(Token {kind: TokenKind::OpenParenthesis, line: 0, column: 19}),
            Ok(Token {kind: TokenKind::CloseParethesis, line: 0, column: 20}),
            Ok(Token {kind: TokenKind::String("Hola Mundo!"), line: 0, column: 22}),
            Ok(Token {kind: TokenKind::FloatingPoint(3.14), line: 0, column: 36}),
            Ok(Token {kind: TokenKind::String(r#"esc\"apes"#), line: 0, column: 41}),
        ]);
    }

    #[test]
    fn tokens_between_whitespace() {
        let lexer = Lexer::new("1\n2\t\t3\n  4 ");
        let tokens = lexer.collect::<Vec<_>>();
        assert_eq!(tokens, vec![
            Ok(Token { kind: TokenKind::Integer(1), line: 0, column: 0}),
            Ok(Token { kind: TokenKind::Integer(2), line: 1, column: 0}),
            Ok(Token { kind: TokenKind::Integer(3), line: 1, column: 3}),
            Ok(Token { kind: TokenKind::Integer(4), line: 2, column: 2})
        ]);
    }

    #[test]
    fn keywords_and_identifiers() {
        let lexer = Lexer::new("fn fni type typee");
        let tokens = lexer.map(|token| token.unwrap().kind).collect::<Vec<_>>();
        assert_eq!(tokens, vec![
            TokenKind::KeywordFn,
            TokenKind::Identifier("fni"),
            TokenKind::KeywordType,
            TokenKind::Identifier("typee")
        ]);
    }

    #[test]
    fn unexpected_token() {
        let mut lexer = Lexer::new(r#"1 £"#);

        assert_eq!(lexer.next(), Some(Ok(Token {kind: TokenKind::Integer(1), line: 0, column: 0}),));
        assert_eq!(lexer.next(), Some(Err(LexerError {error: ErrorType::UnrecognizedToken("£"), line: 0, column: 2}),));
    }
}

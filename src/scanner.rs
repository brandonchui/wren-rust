use std::collections::HashMap;

use crate::token::{Literal, Token, TokenType};

pub struct Scanner<'a> {
    source: &'a str,
    pub tokens: Vec<Token>,
    keywords: HashMap<&'static str, TokenType>,

    // Positions
    start: u32,
    current: u32,
    line: u32,

    // Errors
    pub errors: Vec<(u32, String)>,
    had_error: bool,
}

impl<'a> Scanner<'a> {
    pub fn new(src: &'a str) -> Self {
        Scanner {
            source: src,
            tokens: Vec::<Token>::new(),
            start: 0,
            current: 0,
            line: 1,
            errors: Vec::<(u32, String)>::new(),
            had_error: false,
            keywords: HashMap::from([
                ("class", TokenType::Class),
                ("if", TokenType::If),
                ("else", TokenType::Else),
                ("false", TokenType::False),
                ("for", TokenType::For),
                ("return", TokenType::Return),
                ("super", TokenType::Super),
                ("this", TokenType::This),
                ("true", TokenType::True),
                ("var", TokenType::Var),
                ("while", TokenType::While),
                ("as", TokenType::As),
                ("break", TokenType::Break),
                ("construct", TokenType::Construct),
                ("continue", TokenType::Continue),
                ("foreign", TokenType::Foreign),
                ("import", TokenType::Import),
                ("in", TokenType::In),
                ("is", TokenType::Is),
                ("null", TokenType::Null),
                ("static", TokenType::Static),
            ]),
        }
    }

    pub fn scan_tokens(&mut self) {
        while !self.is_at_end() {
            self.start = self.current;
            self.scan_token();
        }

        self.tokens
            .push(Token::new(TokenType::Eof, "".to_string(), None, self.line));
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.source.len() as u32
    }

    // Consumes the next character, but if it hits a single char token, add it to Vec
    fn scan_token(&mut self) {
        let c = self.advance();

        match c {
            '(' => self.add_token(TokenType::LeftParen, None),
            ')' => self.add_token(TokenType::RightParen, None),
            '{' => self.add_token(TokenType::LeftBrace, None),
            '}' => self.add_token(TokenType::RightBrace, None),
            '[' => self.add_token(TokenType::LeftBracket, None),
            ']' => self.add_token(TokenType::RightBracket, None),
            ',' => self.add_token(TokenType::Comma, None),
            ';' => self.add_token(TokenType::Semicolon, None),
            '~' => self.add_token(TokenType::Tilde, None),
            '?' => self.add_token(TokenType::Question, None),
            ':' => self.add_token(TokenType::Colon, None),
            '+' => self.add_token(TokenType::Plus, None),
            '-' => self.add_token(TokenType::Minus, None),
            '*' => self.add_token(TokenType::Star, None),
            '^' => self.add_token(TokenType::Caret, None),
            '"' => {
                self.add_string();
            }
            // Need peeking
            '!' => {
                if self.match_char('=') {
                    self.add_token(TokenType::BangEqual, None);
                } else {
                    self.add_token(TokenType::Bang, None);
                }
            }
            '=' => {
                if self.match_char('=') {
                    self.add_token(TokenType::EqualEqual, None);
                } else {
                    self.add_token(TokenType::Equal, None);
                }
            }
            '<' => {
                if self.match_char('=') {
                    self.add_token(TokenType::LessEqual, None);
                } else if self.match_char('<') {
                    self.add_token(TokenType::LessLess, None);
                } else {
                    self.add_token(TokenType::Less, None);
                }
            }
            '>' => {
                if self.match_char('=') {
                    self.add_token(TokenType::GreaterEqual, None);
                } else if self.match_char('>') {
                    self.add_token(TokenType::GreaterGreater, None);
                } else {
                    self.add_token(TokenType::Greater, None);
                }
            }
            '&' => {
                if self.match_char('&') {
                    self.add_token(TokenType::AmpAmp, None);
                } else {
                    self.add_token(TokenType::Ampersand, None);
                }
            }
            '|' => {
                if self.match_char('|') {
                    self.add_token(TokenType::PipePipe, None);
                } else {
                    self.add_token(TokenType::Pipe, None);
                }
            }
            '/' => {
                if self.match_char('/') {
                    while self.peek() != '\n' && !self.is_at_end() {
                        self.advance();
                    }
                } else {
                    self.add_token(TokenType::Slash, None);
                }
            }
            '.' => {
                if self.match_char('.') {
                    if self.match_char('.') {
                        // Found '...'
                        self.add_token(TokenType::DotDotDot, None);
                    } else {
                        // Found '..'
                        self.add_token(TokenType::DotDot, None);
                    }
                } else {
                    self.add_token(TokenType::Dot, None);
                }
            }
            // Not errors, but not intersting
            ' ' => {}
            '\t' => {}
            '\r' => {}
            '\n' => {
                self.line += 1;
            }
            // Default to push error out
            _ => {
                if self.is_digit(c) {
                    self.add_number();
                } else if c.is_alphabetic() || c == '_' {
                    self.identifier();
                } else {
                    self.errors
                        .push((self.line, "Unexpected character.".to_string()));
                    self.had_error = true;
                }
            }
        }
    }

    // Consume the next char in the source file and returns it.
    fn advance(&mut self) -> char {
        //BUG Check performance for this function, might be slow due to O(n)
        let ch = match self.source.chars().nth(self.current as usize) {
            Some(c) => c,
            None => unreachable!(),
        };
        self.current += 1;

        ch
    }

    // Recall that the scanner always advances after doing something, so the
    // current is always the unchecked char so far.
    fn match_char(&mut self, expected: char) -> bool {
        if self.peek() != expected {
            return false;
        }

        self.current += 1;
        true
    }

    // Looks at the next char without consuming (advancing)
    fn peek(&self) -> char {
        self.source
            .chars()
            .nth(self.current as usize)
            .unwrap_or('\0')
    }

    fn peek_next(&self) -> char {
        let next = (self.current + 1) as usize;
        self.source.chars().nth(next).unwrap_or('\0')
    }

    fn add_token(&mut self, token_kind: TokenType, literal: Option<Literal>) {
        let text = &self.source[self.start as usize..self.current as usize];
        self.tokens.push({
            let line = self.line;
            Token {
                kind: token_kind,
                lexeme: text.to_string(),
                literal,
                line,
            }
        });
    }

    fn add_string(&mut self) {
        // Find first `"` then keep going.
        while self.peek() != '"' && !self.is_at_end() {
            if self.peek() == '\n' {
                self.line += 1;
            }
            self.advance();
        }

        // There was not another '"', that is an error.
        if self.is_at_end() {
            self.errors
                .push((self.line, "Unterminated String.".to_string()));
            self.had_error = true;
            return;
        }

        // The last closing '"'
        self.advance();

        let begin = (self.start + 1) as usize;
        let end = (self.current - 1) as usize;

        let value = &self.source[begin..end];
        let literal = Some(Literal::StringLit(value.to_string()));
        self.add_token(TokenType::String, literal);
    }

    // This function probably already exist as part of the stl
    fn is_digit(&self, c: char) -> bool {
        c.is_ascii_digit()
    }

    fn add_number(&mut self) {
        while self.is_digit(self.peek()) {
            self.advance();
        }

        // Fractional
        if self.peek() == '.' && self.is_digit(self.peek_next()) {
            self.advance();

            while self.is_digit(self.peek()) {
                self.advance();
            }
        }

        let text = &self.source[self.start as usize..self.current as usize];
        let num: f64 = text.parse().unwrap_or(0.0);

        let literal = Some(Literal::Number(num));
        self.add_token(TokenType::Number, literal);
    }

    fn identifier(&mut self) {
        while self.peek().is_alphanumeric() || self.peek() == '_' {
            self.advance();
        }

        let text = &self.source[self.start as usize..self.current as usize];
        let token_type = self.keywords.get(text);

        match token_type {
            Some(tk) => self.add_token(*tk, None),
            None => self.add_token(TokenType::Identifier, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts;

    use super::*;

    // Helper to scan and return tokens
    fn scan(source: &str) -> Vec<Token> {
        let mut scanner = Scanner::new(source);
        scanner.scan_tokens();
        scanner.tokens
    }

    // Helper to get token types only
    fn scan_types(source: &str) -> Vec<TokenType> {
        scan(source).into_iter().map(|t| t.kind).collect()
    }

    // ==================== Single Character Tokens ====================

    #[test]
    fn test_single_char_tokens() {
        let types = scan_types("(){},;~?:+-*^[]");
        assert_eq!(
            types,
            vec![
                TokenType::LeftParen,
                TokenType::RightParen,
                TokenType::LeftBrace,
                TokenType::RightBrace,
                TokenType::Comma,
                TokenType::Semicolon,
                TokenType::Tilde,
                TokenType::Question,
                TokenType::Colon,
                TokenType::Plus,
                TokenType::Minus,
                TokenType::Star,
                TokenType::Caret,
                TokenType::LeftBracket,
                TokenType::RightBracket,
                TokenType::Eof,
            ]
        );
    }

    // ==================== Two Character Tokens ====================

    #[test]
    fn test_bang_and_bang_equal() {
        assert_eq!(scan_types("!"), vec![TokenType::Bang, TokenType::Eof]);
        assert_eq!(scan_types("!="), vec![TokenType::BangEqual, TokenType::Eof]);
    }

    #[test]
    fn test_equal_and_equal_equal() {
        assert_eq!(scan_types("="), vec![TokenType::Equal, TokenType::Eof]);
        assert_eq!(
            scan_types("=="),
            vec![TokenType::EqualEqual, TokenType::Eof]
        );
    }

    #[test]
    fn test_less_variants() {
        assert_eq!(scan_types("<"), vec![TokenType::Less, TokenType::Eof]);
        assert_eq!(scan_types("<="), vec![TokenType::LessEqual, TokenType::Eof]);
        assert_eq!(scan_types("<<"), vec![TokenType::LessLess, TokenType::Eof]);
    }

    #[test]
    fn test_greater_variants() {
        assert_eq!(scan_types(">"), vec![TokenType::Greater, TokenType::Eof]);
        assert_eq!(
            scan_types(">="),
            vec![TokenType::GreaterEqual, TokenType::Eof]
        );
        assert_eq!(
            scan_types(">>"),
            vec![TokenType::GreaterGreater, TokenType::Eof]
        );
    }

    #[test]
    fn test_ampersand_variants() {
        assert_eq!(scan_types("&"), vec![TokenType::Ampersand, TokenType::Eof]);
        assert_eq!(scan_types("&&"), vec![TokenType::AmpAmp, TokenType::Eof]);
    }

    #[test]
    fn test_pipe_variants() {
        assert_eq!(scan_types("|"), vec![TokenType::Pipe, TokenType::Eof]);
        assert_eq!(scan_types("||"), vec![TokenType::PipePipe, TokenType::Eof]);
    }

    #[test]
    fn test_dot_variants() {
        assert_eq!(scan_types("."), vec![TokenType::Dot, TokenType::Eof]);
        assert_eq!(scan_types(".."), vec![TokenType::DotDot, TokenType::Eof]);
        assert_eq!(
            scan_types("..."),
            vec![TokenType::DotDotDot, TokenType::Eof]
        );
    }

    // ==================== Comments ====================

    #[test]
    fn test_slash_token() {
        assert_eq!(scan_types("/"), vec![TokenType::Slash, TokenType::Eof]);
    }

    #[test]
    fn test_line_comment_ignored() {
        let types = scan_types("// this is a comment");
        assert_eq!(types, vec![TokenType::Eof]);
    }

    #[test]
    fn test_comment_before_code() {
        let types = scan_types("// comment\n+");
        assert_eq!(types, vec![TokenType::Plus, TokenType::Eof]);
    }

    // ==================== Whitespace ====================

    #[test]
    fn test_whitespace_ignored() {
        let types = scan_types("  +  -  ");
        assert_eq!(
            types,
            vec![TokenType::Plus, TokenType::Minus, TokenType::Eof]
        );
    }

    #[test]
    fn test_tabs_ignored() {
        let types = scan_types("\t+\t-\t");
        assert_eq!(
            types,
            vec![TokenType::Plus, TokenType::Minus, TokenType::Eof]
        );
    }

    #[test]
    fn test_newlines_increment_line() {
        let tokens = scan("(\n)");
        assert_eq!(tokens[0].line, 1); // (
        assert_eq!(tokens[1].line, 2); // )
    }

    // ==================== Numbers ====================

    #[test]
    fn test_integer() {
        let tokens = scan("123");
        assert_eq!(tokens[0].kind, TokenType::Number);
        assert_eq!(tokens[0].lexeme, "123");
        if let Some(Literal::Number(n)) = &tokens[0].literal {
            assert_eq!(*n, 123.0);
        } else {
            panic!("Expected number literal");
        }
    }

    #[test]
    fn test_decimal_number() {
        let tokens = scan("3.14159");
        assert_eq!(tokens[0].kind, TokenType::Number);
        if let Some(Literal::Number(n)) = &tokens[0].literal {
            assert!((n - 3.14159).abs() < 0.00001);
        } else {
            panic!("Expected number literal");
        }
    }

    #[test]
    fn test_number_followed_by_dot_method() {
        // "42.foo" should be: Number(42), Dot, Identifier(foo)
        let types = scan_types("42.foo");
        assert_eq!(
            types,
            vec![
                TokenType::Number,
                TokenType::Dot,
                TokenType::Identifier,
                TokenType::Eof,
            ]
        );
    }

    // ==================== Strings ====================

    #[test]
    fn test_simple_string() {
        let tokens = scan("\"hello\"");
        assert_eq!(tokens[0].kind, TokenType::String);
        assert_eq!(tokens[0].lexeme, "\"hello\"");
        if let Some(Literal::StringLit(s)) = &tokens[0].literal {
            assert_eq!(s, "hello");
        } else {
            panic!("Expected string literal");
        }
    }

    #[test]
    fn test_empty_string() {
        let tokens = scan("\"\"");
        assert_eq!(tokens[0].kind, TokenType::String);
        if let Some(Literal::StringLit(s)) = &tokens[0].literal {
            assert_eq!(s, "");
        } else {
            panic!("Expected string literal");
        }
    }

    #[test]
    fn test_multiline_string() {
        let tokens = scan("\"line1\nline2\"");
        assert_eq!(tokens[0].kind, TokenType::String);
        assert_eq!(tokens[1].line, 2); // EOF should be on line 2
    }

    #[test]
    fn test_unterminated_string_error() {
        let mut scanner = Scanner::new("\"unterminated");
        scanner.scan_tokens();
        assert!(scanner.had_error);
        assert!(!scanner.errors.is_empty());
    }

    // ==================== Identifiers ====================

    #[test]
    fn test_simple_identifier() {
        let tokens = scan("foo");
        assert_eq!(tokens[0].kind, TokenType::Identifier);
        assert_eq!(tokens[0].lexeme, "foo");
    }

    #[test]
    fn test_identifier_with_underscore() {
        let tokens = scan("_foo_bar");
        assert_eq!(tokens[0].kind, TokenType::Identifier);
        assert_eq!(tokens[0].lexeme, "_foo_bar");
    }

    #[test]
    fn test_identifier_with_numbers() {
        let tokens = scan("var2");
        assert_eq!(tokens[0].kind, TokenType::Identifier);
        assert_eq!(tokens[0].lexeme, "var2");
    }

    #[test]
    fn test_underscore_only() {
        let tokens = scan("_");
        assert_eq!(tokens[0].kind, TokenType::Identifier);
        assert_eq!(tokens[0].lexeme, "_");
    }

    // ==================== Keywords ====================

    #[test]
    fn test_all_keywords() {
        let types = scan_types(
            "class if else false for return super this true var while as break construct continue foreign import in is null static",
        );
        assert_eq!(
            types,
            vec![
                TokenType::Class,
                TokenType::If,
                TokenType::Else,
                TokenType::False,
                TokenType::For,
                TokenType::Return,
                TokenType::Super,
                TokenType::This,
                TokenType::True,
                TokenType::Var,
                TokenType::While,
                TokenType::As,
                TokenType::Break,
                TokenType::Construct,
                TokenType::Continue,
                TokenType::Foreign,
                TokenType::Import,
                TokenType::In,
                TokenType::Is,
                TokenType::Null,
                TokenType::Static,
                TokenType::Eof,
            ]
        );
    }

    #[test]
    fn test_keyword_prefix_is_identifier() {
        // "iff" is not "if" - should be identifier
        let tokens = scan("iff");
        assert_eq!(tokens[0].kind, TokenType::Identifier);
        assert_eq!(tokens[0].lexeme, "iff");
    }

    // ==================== Complex Expressions ====================

    #[test]
    fn test_arithmetic_expression() {
        let types = scan_types("3 + 4 * 2");
        assert_eq!(
            types,
            vec![
                TokenType::Number,
                TokenType::Plus,
                TokenType::Number,
                TokenType::Star,
                TokenType::Number,
                TokenType::Eof,
            ]
        );
    }

    #[test]
    fn test_parenthesized_expression() {
        let types = scan_types("(1 + 2) * 3");
        assert_eq!(
            types,
            vec![
                TokenType::LeftParen,
                TokenType::Number,
                TokenType::Plus,
                TokenType::Number,
                TokenType::RightParen,
                TokenType::Star,
                TokenType::Number,
                TokenType::Eof,
            ]
        );
    }

    #[test]
    fn test_comparison() {
        let types = scan_types("x <= 10 && y >= 5");
        assert_eq!(
            types,
            vec![
                TokenType::Identifier,
                TokenType::LessEqual,
                TokenType::Number,
                TokenType::AmpAmp,
                TokenType::Identifier,
                TokenType::GreaterEqual,
                TokenType::Number,
                TokenType::Eof,
            ]
        );
    }

    #[test]
    fn test_var_declaration() {
        let types = scan_types("var x = 42");
        assert_eq!(
            types,
            vec![
                TokenType::Var,
                TokenType::Identifier,
                TokenType::Equal,
                TokenType::Number,
                TokenType::Eof,
            ]
        );
    }

    #[test]
    fn test_if_statement() {
        let types = scan_types("if (x == 1) { return true }");
        assert_eq!(
            types,
            vec![
                TokenType::If,
                TokenType::LeftParen,
                TokenType::Identifier,
                TokenType::EqualEqual,
                TokenType::Number,
                TokenType::RightParen,
                TokenType::LeftBrace,
                TokenType::Return,
                TokenType::True,
                TokenType::RightBrace,
                TokenType::Eof,
            ]
        );
    }

    #[test]
    fn test_range_expression() {
        let types = scan_types("1..10");
        assert_eq!(
            types,
            vec![
                TokenType::Number,
                TokenType::DotDot,
                TokenType::Number,
                TokenType::Eof,
            ]
        );
    }

    #[test]
    fn test_exclusive_range() {
        let types = scan_types("1...10");
        assert_eq!(
            types,
            vec![
                TokenType::Number,
                TokenType::DotDotDot,
                TokenType::Number,
                TokenType::Eof,
            ]
        );
    }

    // ==================== Error Cases ====================

    #[test]
    fn test_unexpected_character_error() {
        let mut scanner = Scanner::new("@");
        scanner.scan_tokens();
        assert!(scanner.had_error);
        assert!(!scanner.errors.is_empty());
    }

    #[test]
    fn test_continues_after_error() {
        let mut scanner = Scanner::new("@ + 1");
        scanner.scan_tokens();
        assert!(scanner.had_error);
        // Should still have tokens after the error
        assert!(scanner.tokens.len() > 1);
    }
}

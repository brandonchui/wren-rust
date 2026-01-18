use core::fmt;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenType {
    // Single char tokens
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Ampersand,
    Pipe,
    Caret,
    Tilde,
    Question,
    Colon,
    Comma,
    Dot,
    Minus,
    Plus,
    Semicolon,
    Slash,
    Star,
    // Multi chars
    Bang,
    BangEqual,
    Equal,
    EqualEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    AmpAmp,
    PipePipe,
    DotDot,
    DotDotDot,
    LessLess,
    GreaterGreater,
    // Literals
    Identifier,
    String,
    Number,
    // Keywords
    Class,
    Else,
    False,
    For,
    If,
    Return,
    Super,
    This,
    True,
    Var,
    While,
    As,
    Break,
    Construct,
    Continue,
    Foreign,
    Import,
    In,
    Is,
    Null,
    Static,
    Eof,
}

// Holds the actual value from the lexeme, which is usually just the text representation only.
#[derive(Debug, Clone)]
pub enum Literal {
    Number(f64),
    StringLit(String),
}

impl Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Literal::Number(n) => write!(f, "{n}"),
            Literal::StringLit(s) => write!(f, "{s}"),
        }
    }
}

// Represents a single word, which holds some metadata on what the meaning behind the word consists of, which is generated from the Scanner.
#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenType,
    pub lexeme: String,
    pub literal: Option<Literal>,
    pub line: u32,
}

impl Token {
    pub fn new(kind: TokenType, lexeme: String, literal: Option<Literal>, line: u32) -> Self {
        Token {
            kind,
            lexeme,
            literal,
            line,
        }
    }
}

impl Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let lit = match &self.literal {
            Some(s) => s.to_string(),
            None => "".to_string(),
        };

        write!(f, "{:?} {} {}", self.kind, self.lexeme, lit)
    }
}

use crate::{
    ast::{Expr, Method, Stmt},
    token::{Literal, Token, TokenType},
};

// Parser uses this precedence and associativity:
// Lowest      =           RIGHT
//             ||          LEFT
//             &&          LEFT
//             == !=       LEFT
//             < > <= >=   LEFT
//             + -         LEFT  (term)
//             * /         LEFT  (factor)
//             ! -         RIGHT (Unary)
// Highest     () . []     LEFT  (primary, includes, number,string,bools,null,identifier)
//
// Left - while,
// Right, recursion
//
pub struct Parser {
    pub tokens: Vec<Token>,
    pub current: usize,
}

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
    pub line: u32,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, current: 0 }
    }

    pub fn parse(&mut self) -> Result<Vec<Stmt>, ParseError> {
        // match self.expression() {
        //     Ok(expr) => Ok(expr),
        //     Err(e) => Err(ParseError {
        //         message: e.message,
        //         line: e.line,
        //     }),
        // }
        let mut statements = Vec::<Stmt>::new();

        while !self.is_at_end() {
            statements.push(self.declaration()?);
        }

        Ok(statements)
    }
    // Declaration
    fn declaration(&mut self) -> Result<Stmt, ParseError> {
        if self.match_token_kind(TokenType::Class) {
            return self.class_declaration();
        }
        if self.match_token_kind(TokenType::Var) {
            return self.var_declaration();
        } else {
            return self.statement();
        }

        // Error check
        // ParseError{
        // self.synchronize();
        // ...}
    }

    fn var_declaration(&mut self) -> Result<Stmt, ParseError> {
        let token_name = self
            .consume(TokenType::Identifier, "Expect variable name.")?
            .clone();

        self.consume(TokenType::Equal, "")?;

        let var_expr = self.expression()?;

        // BUG Right now, we are requiring initialization when creating variables.

        Ok(Stmt::Var {
            name: token_name,
            initializer: Box::new(var_expr),
        })
    }

    // Statements
    fn statement(&mut self) -> Result<Stmt, ParseError> {
        if self.match_token_kind(TokenType::Return) {
            return self.return_statement();
        }
        if self.match_token_kind(TokenType::For) {
            return self.for_statement();
        }
        if self.match_token_kind(TokenType::If) {
            return self.if_statement();
        }

        if self.match_token_kind(TokenType::While) {
            return self.while_statement();
        }

        if self.match_token_kind(TokenType::LeftBrace) {
            return Ok(Stmt::Block {
                statements: self.block(),
            });
        }

        self.expression_statement()
    }

    fn if_statement(&mut self) -> Result<Stmt, ParseError> {
        self.consume(TokenType::LeftParen, "Expect '(' after 'if'.")?;
        let condition = self.expression()?;
        self.consume(TokenType::RightParen, "Expect ')' after if condition.")?;

        let then_branch = self.statement()?;
        let mut else_branch = None;

        if self.match_token_kind(TokenType::Else) {
            else_branch = Some(self.statement()?);
        }

        Ok(Stmt::If {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch: else_branch.map(|s| Box::new(s)),
        })
    }

    fn block(&mut self) -> Vec<Stmt> {
        let mut statements = Vec::<Stmt>::new();

        while !self.check(TokenType::RightBrace) && !self.is_at_end() {
            match self.declaration() {
                Ok(d) => {
                    statements.push(d);
                }
                Err(_) => todo!(),
            }
        }

        self.consume(TokenType::RightBrace, "Expect '}' after block.");
        statements
    }

    fn return_statement(&mut self) -> Result<Stmt, ParseError> {
        if !self.check(TokenType::RightBrace) && !self.check(TokenType::RightParen) {
            Ok(Stmt::Return {
                value: Some(Box::new(self.expression()?)),
            })
        } else {
            Ok(Stmt::Return { value: None })
        }
    }

    fn expression_statement(&mut self) -> Result<Stmt, ParseError> {
        let expr = self.expression()?;

        Ok(Stmt::Expression {
            expression: Box::new(expr),
        })
    }

    // Expression
    pub fn expression(&mut self) -> Result<Expr, ParseError> {
        self.assignment()
    }

    // Assignment
    pub fn assignment(&mut self) -> Result<Expr, ParseError> {
        // let left = self.equality()?;
        let left = self.or()?;

        // If the next token is `=`, then we recursively get the RHS.
        if self.match_token_kind(TokenType::Equal) {
            let equals_line = self.previous().line;
            let right = self.assignment()?;

            // If the LHS is indeed some variable, then just return the Expr::Assign, since
            // only variables are able to be assigned. E.g. a = 5
            match left {
                Expr::Variable { name } => Ok(Expr::Assign {
                    name,
                    value: Box::new(right),
                }),
                _ => Err(ParseError {
                    message: "Assignment Error.".to_string(),
                    line: equals_line,
                }),
            }
        } else {
            // If not a variable, it is not an assignment so just return the LHS, for
            // example, 5 = x is not valid since it is a literal.
            Ok(left)
        }
    }

    // Logics
    pub fn or(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.and()?;

        while self.match_token_kind(TokenType::PipePipe) {
            let operator = self.previous().clone();
            let right = self.and()?;

            expr = Expr::Logical {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            }
        }

        Ok(expr)
    }

    pub fn and(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.equality()?;

        while self.match_token_kind(TokenType::AmpAmp) {
            let operator = self.previous().clone();
            let right = self.equality()?;

            expr = Expr::Logical {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            }
        }
        Ok(expr)
    }

    // Equality
    pub fn equality(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.comparison()?;

        while self.match_token_kind(TokenType::BangEqual)
            || self.match_token_kind(TokenType::EqualEqual)
        {
            let operator = self.previous().clone();
            let right = self.comparison()?;

            expr = Expr::Binary {
                left: Box::new(expr),
                operator: operator.clone(),
                right: Box::new(right),
            }
        }

        Ok(expr)
    }

    // Comparison
    pub fn comparison(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.term()?;

        while self.match_token_kind(TokenType::Greater)
            || self.match_token_kind(TokenType::Less)
            || self.match_token_kind(TokenType::LessEqual)
            || self.match_token_kind(TokenType::GreaterEqual)
        {
            let operator = self.previous().clone();
            let right = self.term()?;

            expr = Expr::Binary {
                left: Box::new(expr),
                operator: operator.clone(),
                right: Box::new(right),
            }
        }
        Ok(expr)
    }

    // Term
    pub fn term(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.factor()?;

        while self.match_token_kind(TokenType::Plus) || self.match_token_kind(TokenType::Minus) {
            let operator = self.previous().clone();
            let right = self.factor()?;

            expr = Expr::Binary {
                left: Box::new(expr),
                operator: operator.clone(),
                right: Box::new(right),
            }
        }
        Ok(expr)
    }

    // Factor
    pub fn factor(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.unary()?;

        while self.match_token_kind(TokenType::Slash) || self.match_token_kind(TokenType::Star) {
            let operator = self.previous().clone();
            let right = self.unary()?;

            expr = Expr::Binary {
                left: Box::new(expr),
                operator: operator.clone(),
                right: Box::new(right),
            }
        }
        Ok(expr)
    }

    // Unary
    pub fn unary(&mut self) -> Result<Expr, ParseError> {
        if self.match_token_kind(TokenType::Bang) || self.match_token_kind(TokenType::Minus) {
            let operator = self.previous().clone();
            let right = self.unary()?;

            return Ok(Expr::Unary {
                operator: operator.clone(),
                right: Box::new(right),
            });
        }

        // self.primary()
        self.call()
    }

    // Method call .
    fn call(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.primary()?;

        while self.check(TokenType::Dot) {
            self.consume(TokenType::Dot, "")?;
            let name = self.consume(TokenType::Identifier, "")?.clone();

            if self.match_token_kind(TokenType::LeftParen) {
                let mut args = Vec::new();
                if !self.check(TokenType::RightParen) {
                    args.push(Box::new(self.expression()?));
                    while self.match_token_kind(TokenType::Comma) {
                        args.push(Box::new(self.expression()?));
                    }
                }
                self.consume(TokenType::RightParen, "")?;

                expr = Expr::Call {
                    receiver: Box::new(expr),
                    name: name.lexeme,
                    arguments: args,
                };
            } else {
                expr = Expr::Get {
                    object: Box::new(expr),
                    name,
                };
            }
        }

        Ok(expr)
    }

    // Primary
    pub fn primary(&mut self) -> Result<Expr, ParseError> {
        // if self.match_token_kind(TokenType::False) {
        //     return Expr::Literal {
        //         value: Literal::Number(0.0),
        //     };
        // }

        // if self.match_token_kind(TokenType::True) {
        //     return Expr::Literal {
        //         value: Literal::Number(1.0),
        //     };
        // }

        // if self.match_token_kind(TokenType::Null) {
        //     return Expr::Literal{...};
        // }

        if self.match_token_kind(TokenType::Number) || self.match_token_kind(TokenType::String) {
            return Ok(Expr::Literal {
                // Unsafe
                value: self.previous().literal.clone().unwrap(),
            });
        }

        if self.match_token_kind(TokenType::LeftParen) {
            let expr = self.expression()?;
            self.consume(TokenType::RightParen, "Expect ')' after expression.")?;

            return Ok(Expr::Grouping {
                expression: Box::new(expr),
            });
        }

        if self.match_token_kind(TokenType::Identifier) {
            return Ok(Expr::Variable {
                name: self.previous().clone(),
            });
        }

        Err(ParseError {
            message: "Expect expression.".to_string(),
            line: self.peek().line,
        })
    }

    fn while_statement(&mut self) -> Result<Stmt, ParseError> {
        self.consume(TokenType::LeftParen, "Expect '(' after 'while'.");

        let condition = self.expression()?;
        self.consume(TokenType::RightParen, "Expect ')' after condition.");

        let body = self.statement()?;

        Ok(Stmt::While {
            condition: Box::new(condition),
            body: Box::new(body),
        })
    }

    fn for_statement(&mut self) -> Result<Stmt, ParseError> {
        self.consume(TokenType::LeftParen, "Expect '(' after 'for'.");
        // Parse ID
        let loop_var = self
            .consume(TokenType::Identifier, "Expect loop variable name")?
            .clone();

        self.consume(TokenType::In, "");
        // Parse start expression literal
        let start = self.expression()?;

        self.consume(TokenType::DotDot, "");

        // Parse second end expression literal
        let end = self.expression()?;

        self.consume(TokenType::RightParen, "Expect ')' after 'for'.");
        let body = self.statement()?;

        // Desugaring and using existing nodes

        // Need to create the condition first (e.g. i < end)
        let condition = Expr::Binary {
            left: Box::new(Expr::Variable {
                name: loop_var.clone(),
            }),
            operator: Token::new(TokenType::Less, "<".to_string(), None, loop_var.line),
            right: Box::new(end),
        };

        let i_plus_one = Expr::Binary {
            left: Box::new(Expr::Variable {
                name: loop_var.clone(),
            }),
            operator: Token::new(TokenType::Plus, "+".to_string(), None, loop_var.line),
            right: Box::new(Expr::Literal {
                value: Literal::Number(1.0),
            }),
        };

        let increment = Expr::Assign {
            name: loop_var.clone(),
            value: Box::new(i_plus_one),
        };

        Ok(Stmt::Block {
            statements: vec![
                Stmt::Var {
                    name: loop_var.clone(),
                    initializer: Box::new(start),
                },
                Stmt::While {
                    condition: Box::new(condition),
                    body: Box::new(Stmt::Block {
                        statements: vec![
                            body,
                            Stmt::Expression {
                                expression: Box::new(increment),
                            },
                        ],
                    }),
                },
            ],
        })
    }

    fn class_declaration(&mut self) -> Result<Stmt, ParseError> {
        let name = self
            .consume(TokenType::Identifier, "Expect class name.")?
            .clone();
        self.consume(TokenType::LeftBrace, "Expect '{' before class body.")?;

        let mut methods = Vec::<Method>::new();
        let mut constructor: Option<Method> = None;

        while !self.check(TokenType::RightBrace) && !self.is_at_end() {
            // Constructor check first, else it is just a method
            if self.match_token_kind(TokenType::Construct) {
                constructor = Some(self.method()?);
            } else {
                methods.push(self.method()?);
            }
        }

        self.consume(TokenType::RightBrace, "Expect '}' after class body.")?;

        Ok(Stmt::Class {
            name: name.clone(),
            constructor,
            methods,
        })
    }

    fn method(&mut self) -> Result<Method, ParseError> {
        // class Point {
        //     construct new(x, y) {
        //         _x = x
        //         _y = y
        //     }

        //     getX() {
        //         _x
        //     }

        //     add(other) {
        //         _x + other.getX()
        //     }
        // }

        let name = self.consume(TokenType::Identifier, "")?.clone();
        self.consume(TokenType::LeftParen, "")?;

        let mut params = Vec::<Token>::new();

        while !self.check(TokenType::RightParen) {
            // Parse first param
            params.push(self.consume(TokenType::Identifier, "")?.clone());

            // Check if there are more params, the comma seperates them
            while self.match_token_kind(TokenType::Comma) {
                params.push(self.consume(TokenType::Identifier, "")?.clone());
            }
        }

        self.consume(TokenType::RightParen, "")?;
        //Param end

        // The curly and body
        self.consume(TokenType::LeftBrace, "")?;
        let body = self.block();

        Ok(Method { name, params, body })
    }
}

// Helpers
impl Parser {
    pub fn match_token_kind(&mut self, tk_kind: TokenType) -> bool {
        // Should this be done with an aray of tokentype?
        if self.check(tk_kind) {
            self.advance();
            return true;
        }
        false
    }

    pub fn check(&self, tk_kind: TokenType) -> bool {
        if self.is_at_end() {
            return false;
        }

        self.peek().kind == tk_kind
    }

    pub fn peek(&self) -> &Token {
        match self.tokens.get(self.current) {
            Some(tk) => tk,
            None => todo!(),
        }
    }

    pub fn is_at_end(&self) -> bool {
        self.peek().kind == TokenType::Eof
    }

    pub fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    pub fn previous(&self) -> &Token {
        //BUG Could underflow
        match self.tokens.get(self.current - 1) {
            Some(tk) => tk,
            None => todo!(),
        }
    }

    pub fn consume(&mut self, kind: TokenType, msg: &str) -> Result<&Token, ParseError> {
        if self.check(kind) {
            Ok(self.advance())
        } else {
            Err(ParseError {
                message: msg.to_string(),
                line: self.peek().line,
            })
        }
    }

    pub fn synchronize(&mut self) {
        self.advance();

        while !self.is_at_end() {
            if self.previous().kind == TokenType::Semicolon {
                return;
            }

            match self.peek().kind {
                TokenType::Class => return,
                TokenType::Var => return,
                TokenType::If => return,
                TokenType::While => return,
                TokenType::For => return,
                TokenType::Return => return,
                TokenType::Construct => return,
                TokenType::Static => return,
                TokenType::Import => return,
                _ => {}
            }
            self.advance();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::{Literal, Token, TokenType};

    // Helper to create a number token
    fn num(n: f64) -> Token {
        Token::new(
            TokenType::Number,
            n.to_string(),
            Some(Literal::Number(n)),
            1,
        )
    }

    // Helper to create operator tokens
    fn op(kind: TokenType, lexeme: &str) -> Token {
        Token::new(kind, lexeme.to_string(), None, 1)
    }

    // Helper to create EOF token
    fn eof() -> Token {
        Token::new(TokenType::Eof, "".to_string(), None, 1)
    }

    #[test]
    fn test_parse_single_number() {
        // 42
        let tokens = vec![num(42.0), eof()];
        let mut parser = Parser::new(tokens);
        let expr = parser.expression().unwrap();

        assert_eq!(format!("{}", expr), "42");
    }

    #[test]
    fn test_parse_binary_addition() {
        // 3 + 4
        let tokens = vec![num(3.0), op(TokenType::Plus, "+"), num(4.0), eof()];
        let mut parser = Parser::new(tokens);
        let expr = parser.expression().unwrap();

        assert_eq!(format!("{}", expr), "(+ 3 4)");
    }

    #[test]
    fn test_parse_binary_multiplication() {
        // 3 * 4
        let tokens = vec![num(3.0), op(TokenType::Star, "*"), num(4.0), eof()];
        let mut parser = Parser::new(tokens);
        let expr = parser.expression().unwrap();

        assert_eq!(format!("{}", expr), "(* 3 4)");
    }

    #[test]
    fn test_parse_precedence_mul_over_add() {
        // 3 + 4 * 2  should parse as 3 + (4 * 2)
        let tokens = vec![
            num(3.0),
            op(TokenType::Plus, "+"),
            num(4.0),
            op(TokenType::Star, "*"),
            num(2.0),
            eof(),
        ];
        let mut parser = Parser::new(tokens);
        let expr = parser.expression().unwrap();

        assert_eq!(format!("{}", expr), "(+ 3 (* 4 2))");
    }

    #[test]
    fn test_parse_precedence_div_over_sub() {
        // 10 - 6 / 2  should parse as 10 - (6 / 2)
        let tokens = vec![
            num(10.0),
            op(TokenType::Minus, "-"),
            num(6.0),
            op(TokenType::Slash, "/"),
            num(2.0),
            eof(),
        ];
        let mut parser = Parser::new(tokens);
        let expr = parser.expression().unwrap();

        assert_eq!(format!("{}", expr), "(- 10 (/ 6 2))");
    }

    #[test]
    fn test_parse_grouping_overrides_precedence() {
        // (3 + 4) * 2
        let tokens = vec![
            op(TokenType::LeftParen, "("),
            num(3.0),
            op(TokenType::Plus, "+"),
            num(4.0),
            op(TokenType::RightParen, ")"),
            op(TokenType::Star, "*"),
            num(2.0),
            eof(),
        ];
        let mut parser = Parser::new(tokens);
        let expr = parser.expression().unwrap();

        assert_eq!(format!("{}", expr), "(* (group (+ 3 4)) 2)");
    }

    #[test]
    fn test_parse_unary_negation() {
        // -5
        let tokens = vec![op(TokenType::Minus, "-"), num(5.0), eof()];
        let mut parser = Parser::new(tokens);
        let expr = parser.expression().unwrap();

        assert_eq!(format!("{}", expr), "(- 5)");
    }

    #[test]
    fn test_parse_unary_double_negation() {
        // --5
        let tokens = vec![
            op(TokenType::Minus, "-"),
            op(TokenType::Minus, "-"),
            num(5.0),
            eof(),
        ];
        let mut parser = Parser::new(tokens);
        let expr = parser.expression().unwrap();

        assert_eq!(format!("{}", expr), "(- (- 5))");
    }

    #[test]
    fn test_parse_left_associativity_subtraction() {
        // 1 - 2 - 3  should parse as (1 - 2) - 3
        let tokens = vec![
            num(1.0),
            op(TokenType::Minus, "-"),
            num(2.0),
            op(TokenType::Minus, "-"),
            num(3.0),
            eof(),
        ];
        let mut parser = Parser::new(tokens);
        let expr = parser.expression().unwrap();

        assert_eq!(format!("{}", expr), "(- (- 1 2) 3)");
    }

    #[test]
    fn test_parse_left_associativity_division() {
        // 8 / 4 / 2  should parse as (8 / 4) / 2
        let tokens = vec![
            num(8.0),
            op(TokenType::Slash, "/"),
            num(4.0),
            op(TokenType::Slash, "/"),
            num(2.0),
            eof(),
        ];
        let mut parser = Parser::new(tokens);
        let expr = parser.expression().unwrap();

        assert_eq!(format!("{}", expr), "(/ (/ 8 4) 2)");
    }

    #[test]
    fn test_parse_complex_expression() {
        // 3 + 4 * 2 - 1
        let tokens = vec![
            num(3.0),
            op(TokenType::Plus, "+"),
            num(4.0),
            op(TokenType::Star, "*"),
            num(2.0),
            op(TokenType::Minus, "-"),
            num(1.0),
            eof(),
        ];
        let mut parser = Parser::new(tokens);
        let expr = parser.expression().unwrap();

        assert_eq!(format!("{}", expr), "(- (+ 3 (* 4 2)) 1)");
    }

    #[test]
    fn test_parse_comparison() {
        // 3 < 4
        let tokens = vec![num(3.0), op(TokenType::Less, "<"), num(4.0), eof()];
        let mut parser = Parser::new(tokens);
        let expr = parser.expression().unwrap();

        assert_eq!(format!("{}", expr), "(< 3 4)");
    }

    #[test]
    fn test_parse_equality() {
        // 3 == 3
        let tokens = vec![num(3.0), op(TokenType::EqualEqual, "=="), num(3.0), eof()];
        let mut parser = Parser::new(tokens);
        let expr = parser.expression().unwrap();

        assert_eq!(format!("{}", expr), "(== 3 3)");
    }

    #[test]
    fn test_parse_nested_grouping() {
        // ((1 + 2))
        let tokens = vec![
            op(TokenType::LeftParen, "("),
            op(TokenType::LeftParen, "("),
            num(1.0),
            op(TokenType::Plus, "+"),
            num(2.0),
            op(TokenType::RightParen, ")"),
            op(TokenType::RightParen, ")"),
            eof(),
        ];
        let mut parser = Parser::new(tokens);
        let expr = parser.expression().unwrap();

        assert_eq!(format!("{}", expr), "(group (group (+ 1 2)))");
    }

    #[test]
    fn test_parse_empty_returns_error() {
        // Empty expression (just EOF)
        let tokens = vec![eof()];
        let mut parser = Parser::new(tokens);
        let result = parser.expression();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("Expect expression"));
    }

    #[test]
    fn test_parse_unclosed_paren_returns_error() {
        // (3 + 4  -- missing closing paren
        let tokens = vec![
            op(TokenType::LeftParen, "("),
            num(3.0),
            op(TokenType::Plus, "+"),
            num(4.0),
            eof(),
        ];
        let mut parser = Parser::new(tokens);
        let result = parser.expression();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("Expect ')' after expression"));
    }
}

// ==================== Integration Tests (Scanner + Parser) ====================
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::scanner::Scanner;

    // Helper to scan and parse source code, returning the AST string representation
    // Assumes source is a single expression statement and formats the inner expression
    fn parse_source(source: &str) -> Result<String, ParseError> {
        let mut scanner = Scanner::new(source);
        scanner.scan_tokens();

        if !scanner.errors.is_empty() {
            let (line, msg) = &scanner.errors[0];
            return Err(ParseError {
                message: msg.clone(),
                line: *line,
            });
        }

        let mut parser = Parser::new(scanner.tokens);
        let stmts = parser.parse()?;

        // Get first statement and extract the expression from it
        match stmts.first() {
            Some(Stmt::Expression { expression }) => Ok(format!("{}", expression)),
            _ => Ok("(no expression)".to_string()),
        }
    }

    #[test]
    fn test_integration_single_number() {
        let result = parse_source("42").unwrap();
        assert_eq!(result, "42");
    }

    #[test]
    fn test_integration_decimal_number() {
        let result = parse_source("3.14159").unwrap();
        assert!(result.starts_with("3.14"));
    }

    #[test]
    fn test_integration_simple_addition() {
        let result = parse_source("3 + 4").unwrap();
        assert_eq!(result, "(+ 3 4)");
    }

    #[test]
    fn test_integration_simple_subtraction() {
        let result = parse_source("10 - 5").unwrap();
        assert_eq!(result, "(- 10 5)");
    }

    #[test]
    fn test_integration_simple_multiplication() {
        let result = parse_source("6 * 7").unwrap();
        assert_eq!(result, "(* 6 7)");
    }

    #[test]
    fn test_integration_simple_division() {
        let result = parse_source("20 / 4").unwrap();
        assert_eq!(result, "(/ 20 4)");
    }

    #[test]
    fn test_integration_precedence_mul_over_add() {
        // 3 + 4 * 2 should parse as 3 + (4 * 2)
        let result = parse_source("3 + 4 * 2").unwrap();
        assert_eq!(result, "(+ 3 (* 4 2))");
    }

    #[test]
    fn test_integration_precedence_div_over_sub() {
        // 10 - 6 / 2 should parse as 10 - (6 / 2)
        let result = parse_source("10 - 6 / 2").unwrap();
        assert_eq!(result, "(- 10 (/ 6 2))");
    }

    #[test]
    fn test_integration_left_associativity_subtraction() {
        // 1 - 2 - 3 should parse as (1 - 2) - 3
        let result = parse_source("1 - 2 - 3").unwrap();
        assert_eq!(result, "(- (- 1 2) 3)");
    }

    #[test]
    fn test_integration_left_associativity_division() {
        // 8 / 4 / 2 should parse as (8 / 4) / 2
        let result = parse_source("8 / 4 / 2").unwrap();
        assert_eq!(result, "(/ (/ 8 4) 2)");
    }

    #[test]
    fn test_integration_grouping_overrides_precedence() {
        // (3 + 4) * 2 should respect the parentheses
        let result = parse_source("(3 + 4) * 2").unwrap();
        assert_eq!(result, "(* (group (+ 3 4)) 2)");
    }

    #[test]
    fn test_integration_nested_grouping() {
        let result = parse_source("((1 + 2))").unwrap();
        assert_eq!(result, "(group (group (+ 1 2)))");
    }

    #[test]
    fn test_integration_unary_negation() {
        let result = parse_source("-5").unwrap();
        assert_eq!(result, "(- 5)");
    }

    #[test]
    fn test_integration_double_negation() {
        let result = parse_source("--5").unwrap();
        assert_eq!(result, "(- (- 5))");
    }

    #[test]
    fn test_integration_unary_with_expression() {
        // -3 + 4 should parse as (-3) + 4
        let result = parse_source("-3 + 4").unwrap();
        assert_eq!(result, "(+ (- 3) 4)");
    }

    #[test]
    fn test_integration_comparison_less_than() {
        let result = parse_source("3 < 4").unwrap();
        assert_eq!(result, "(< 3 4)");
    }

    #[test]
    fn test_integration_comparison_greater_than() {
        let result = parse_source("5 > 2").unwrap();
        assert_eq!(result, "(> 5 2)");
    }

    #[test]
    fn test_integration_comparison_less_equal() {
        let result = parse_source("3 <= 4").unwrap();
        assert_eq!(result, "(<= 3 4)");
    }

    #[test]
    fn test_integration_comparison_greater_equal() {
        let result = parse_source("5 >= 2").unwrap();
        assert_eq!(result, "(>= 5 2)");
    }

    #[test]
    fn test_integration_equality_equal() {
        let result = parse_source("3 == 3").unwrap();
        assert_eq!(result, "(== 3 3)");
    }

    #[test]
    fn test_integration_equality_not_equal() {
        let result = parse_source("3 != 4").unwrap();
        assert_eq!(result, "(!= 3 4)");
    }

    #[test]
    fn test_integration_complex_arithmetic() {
        // 1 + 2 * 3 - 4 / 2 should parse correctly
        let result = parse_source("1 + 2 * 3 - 4 / 2").unwrap();
        assert_eq!(result, "(- (+ 1 (* 2 3)) (/ 4 2))");
    }

    #[test]
    fn test_integration_comparison_with_arithmetic() {
        // 3 + 4 > 2 * 3 should parse as (3 + 4) > (2 * 3)
        let result = parse_source("3 + 4 > 2 * 3").unwrap();
        assert_eq!(result, "(> (+ 3 4) (* 2 3))");
    }

    #[test]
    fn test_integration_string_literal() {
        let result = parse_source("\"hello\"").unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_integration_whitespace_ignored() {
        let result = parse_source("  3   +   4  ").unwrap();
        assert_eq!(result, "(+ 3 4)");
    }

    #[test]
    fn test_integration_newlines_handled() {
        let result = parse_source("3\n+\n4").unwrap();
        assert_eq!(result, "(+ 3 4)");
    }

    #[test]
    fn test_integration_comments_ignored() {
        let result = parse_source("3 + 4 // this is a comment").unwrap();
        assert_eq!(result, "(+ 3 4)");
    }

    #[test]
    fn test_integration_unclosed_paren_error() {
        let result = parse_source("(3 + 4");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("Expect ')' after expression"));
    }

    #[test]
    fn test_integration_empty_input_error() {
        let result = parse_source("");
        assert!(result.is_err());
    }

    #[test]
    fn test_integration_unexpected_char_error() {
        let result = parse_source("@");
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("Unexpected character"));
    }

    // ==================== File-based Tests ====================

    #[test]
    fn test_wren_script_complex_expressions() {
        // Read the Wren script file at compile time
        let source = include_str!("../tests/scripts/complex_expressions.wren");

        // Filter out comments and empty lines, parse each expression
        for line in source.lines() {
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }

            let result = parse_source(trimmed);
            assert!(
                result.is_ok(),
                "Failed to parse '{}': {:?}",
                trimmed,
                result.unwrap_err()
            );
        }
    }

    #[test]
    fn test_wren_script_complex_expressions_output() {
        // Verify specific expression produces correct AST
        let source = "3 + 4 * 2 - 10 / 5";
        let result = parse_source(source).unwrap();

        // Expected: (- (+ 3 (* 4 2)) (/ 10 5))
        // Because: * and / bind tighter than + and -, left associativity
        assert_eq!(result, "(- (+ 3 (* 4 2)) (/ 10 5))");
    }
}

// ==================== Class Parsing Tests ====================
#[cfg(test)]
mod class_tests {
    use super::*;
    use crate::ast::{Method, Stmt};
    use crate::scanner::Scanner;

    // Helper to parse source and return statements
    fn parse_statements(source: &str) -> Result<Vec<Stmt>, ParseError> {
        let mut scanner = Scanner::new(source);
        scanner.scan_tokens();

        if !scanner.errors.is_empty() {
            let (line, msg) = &scanner.errors[0];
            return Err(ParseError {
                message: msg.clone(),
                line: *line,
            });
        }

        let mut parser = Parser::new(scanner.tokens);
        parser.parse()
    }

    // ==================== Class Declaration Tests ====================

    #[test]
    fn test_empty_class() {
        let stmts = parse_statements("class Point {}").unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Stmt::Class {
                name,
                constructor,
                methods,
            } => {
                assert_eq!(name.lexeme, "Point");
                assert!(constructor.is_none());
                assert!(methods.is_empty());
            }
            _ => panic!("Expected Stmt::Class"),
        }
    }

    #[test]
    fn test_class_with_constructor() {
        let source = r#"
            class Point {
                construct new() {
                }
            }
        "#;
        let stmts = parse_statements(source).unwrap();
        assert_eq!(stmts.len(), 1);

        match &stmts[0] {
            Stmt::Class {
                name,
                constructor,
                methods,
            } => {
                assert_eq!(name.lexeme, "Point");
                assert!(constructor.is_some());
                let ctor = constructor.as_ref().unwrap();
                assert_eq!(ctor.name.lexeme, "new");
                assert!(ctor.params.is_empty());
                assert!(methods.is_empty());
            }
            _ => panic!("Expected Stmt::Class"),
        }
    }

    #[test]
    fn test_class_with_constructor_params() {
        let source = r#"
            class Point {
                construct new(x, y) {
                }
            }
        "#;
        let stmts = parse_statements(source).unwrap();

        match &stmts[0] {
            Stmt::Class { constructor, .. } => {
                let ctor = constructor.as_ref().unwrap();
                assert_eq!(ctor.name.lexeme, "new");
                assert_eq!(ctor.params.len(), 2);
                assert_eq!(ctor.params[0].lexeme, "x");
                assert_eq!(ctor.params[1].lexeme, "y");
            }
            _ => panic!("Expected Stmt::Class"),
        }
    }

    #[test]
    fn test_class_with_method() {
        let source = r#"
            class Point {
                getX() {
                }
            }
        "#;
        let stmts = parse_statements(source).unwrap();

        match &stmts[0] {
            Stmt::Class {
                name,
                constructor,
                methods,
            } => {
                assert_eq!(name.lexeme, "Point");
                assert!(constructor.is_none());
                assert_eq!(methods.len(), 1);
                assert_eq!(methods[0].name.lexeme, "getX");
                assert!(methods[0].params.is_empty());
            }
            _ => panic!("Expected Stmt::Class"),
        }
    }

    #[test]
    fn test_class_with_method_params() {
        let source = r#"
            class Point {
                add(other) {
                }
            }
        "#;
        let stmts = parse_statements(source).unwrap();

        match &stmts[0] {
            Stmt::Class { methods, .. } => {
                assert_eq!(methods.len(), 1);
                assert_eq!(methods[0].name.lexeme, "add");
                assert_eq!(methods[0].params.len(), 1);
                assert_eq!(methods[0].params[0].lexeme, "other");
            }
            _ => panic!("Expected Stmt::Class"),
        }
    }

    #[test]
    fn test_class_with_multiple_methods() {
        let source = r#"
            class Point {
                getX() {
                }
                getY() {
                }
                add(other) {
                }
            }
        "#;
        let stmts = parse_statements(source).unwrap();

        match &stmts[0] {
            Stmt::Class { methods, .. } => {
                assert_eq!(methods.len(), 3);
                assert_eq!(methods[0].name.lexeme, "getX");
                assert_eq!(methods[1].name.lexeme, "getY");
                assert_eq!(methods[2].name.lexeme, "add");
            }
            _ => panic!("Expected Stmt::Class"),
        }
    }

    #[test]
    fn test_class_with_constructor_and_methods() {
        let source = r#"
            class Point {
                construct new(x, y) {
                }
                getX() {
                }
                getY() {
                }
            }
        "#;
        let stmts = parse_statements(source).unwrap();

        match &stmts[0] {
            Stmt::Class {
                name,
                constructor,
                methods,
            } => {
                assert_eq!(name.lexeme, "Point");
                assert!(constructor.is_some());
                assert_eq!(constructor.as_ref().unwrap().name.lexeme, "new");
                assert_eq!(methods.len(), 2);
            }
            _ => panic!("Expected Stmt::Class"),
        }
    }

    #[test]
    fn test_class_method_with_body() {
        let source = r#"
            class Math {
                add(a, b) {
                    a + b
                }
            }
        "#;
        let stmts = parse_statements(source).unwrap();

        match &stmts[0] {
            Stmt::Class { methods, .. } => {
                assert_eq!(methods.len(), 1);
                assert_eq!(methods[0].name.lexeme, "add");
                assert!(!methods[0].body.is_empty());
            }
            _ => panic!("Expected Stmt::Class"),
        }
    }

    // ==================== Method Call Tests ====================

    #[test]
    fn test_simple_method_call() {
        let stmts = parse_statements("point.getX()").unwrap();

        match &stmts[0] {
            Stmt::Expression { expression } => {
                match expression.as_ref() {
                    Expr::Call {
                        receiver,
                        name,
                        arguments,
                    } => {
                        assert_eq!(name, "getX");
                        assert!(arguments.is_empty());
                        // receiver should be Variable "point"
                        match receiver.as_ref() {
                            Expr::Variable { name: var_name } => {
                                assert_eq!(var_name.lexeme, "point");
                            }
                            _ => panic!("Expected Variable as receiver"),
                        }
                    }
                    _ => panic!("Expected Expr::Call"),
                }
            }
            _ => panic!("Expected Stmt::Expression"),
        }
    }

    #[test]
    fn test_method_call_with_argument() {
        let stmts = parse_statements("point.add(other)").unwrap();

        match &stmts[0] {
            Stmt::Expression { expression } => match expression.as_ref() {
                Expr::Call {
                    name, arguments, ..
                } => {
                    assert_eq!(name, "add");
                    assert_eq!(arguments.len(), 1);
                }
                _ => panic!("Expected Expr::Call"),
            },
            _ => panic!("Expected Stmt::Expression"),
        }
    }

    #[test]
    fn test_method_call_with_multiple_arguments() {
        let stmts = parse_statements("obj.method(a, b, c)").unwrap();

        match &stmts[0] {
            Stmt::Expression { expression } => match expression.as_ref() {
                Expr::Call {
                    name, arguments, ..
                } => {
                    assert_eq!(name, "method");
                    assert_eq!(arguments.len(), 3);
                }
                _ => panic!("Expected Expr::Call"),
            },
            _ => panic!("Expected Stmt::Expression"),
        }
    }

    #[test]
    fn test_method_call_with_expression_argument() {
        let stmts = parse_statements("point.add(1 + 2)").unwrap();

        match &stmts[0] {
            Stmt::Expression { expression } => {
                match expression.as_ref() {
                    Expr::Call {
                        name, arguments, ..
                    } => {
                        assert_eq!(name, "add");
                        assert_eq!(arguments.len(), 1);
                        // Argument should be a binary expression
                        match arguments[0].as_ref() {
                            Expr::Binary { .. } => {}
                            _ => panic!("Expected Binary expression as argument"),
                        }
                    }
                    _ => panic!("Expected Expr::Call"),
                }
            }
            _ => panic!("Expected Stmt::Expression"),
        }
    }

    #[test]
    fn test_chained_method_calls() {
        let stmts = parse_statements("a.b().c()").unwrap();

        match &stmts[0] {
            Stmt::Expression { expression } => {
                match expression.as_ref() {
                    Expr::Call { receiver, name, .. } => {
                        assert_eq!(name, "c");
                        // receiver should be another Call (a.b())
                        match receiver.as_ref() {
                            Expr::Call {
                                name: inner_name, ..
                            } => {
                                assert_eq!(inner_name, "b");
                            }
                            _ => panic!("Expected nested Call"),
                        }
                    }
                    _ => panic!("Expected Expr::Call"),
                }
            }
            _ => panic!("Expected Stmt::Expression"),
        }
    }

    #[test]
    fn test_property_access() {
        let stmts = parse_statements("point.x").unwrap();

        match &stmts[0] {
            Stmt::Expression { expression } => match expression.as_ref() {
                Expr::Get { object, name } => {
                    assert_eq!(name.lexeme, "x");
                    match object.as_ref() {
                        Expr::Variable { name: var_name } => {
                            assert_eq!(var_name.lexeme, "point");
                        }
                        _ => panic!("Expected Variable as object"),
                    }
                }
                _ => panic!("Expected Expr::Get"),
            },
            _ => panic!("Expected Stmt::Expression"),
        }
    }

    #[test]
    fn test_chained_property_access() {
        let stmts = parse_statements("a.b.c").unwrap();

        match &stmts[0] {
            Stmt::Expression { expression } => {
                match expression.as_ref() {
                    Expr::Get { object, name } => {
                        assert_eq!(name.lexeme, "c");
                        // object should be Get (a.b)
                        match object.as_ref() {
                            Expr::Get {
                                name: inner_name, ..
                            } => {
                                assert_eq!(inner_name.lexeme, "b");
                            }
                            _ => panic!("Expected nested Get"),
                        }
                    }
                    _ => panic!("Expected Expr::Get"),
                }
            }
            _ => panic!("Expected Stmt::Expression"),
        }
    }

    #[test]
    fn test_method_call_on_property() {
        // a.b.method() - access property b, then call method
        let stmts = parse_statements("a.b.method()").unwrap();

        match &stmts[0] {
            Stmt::Expression { expression } => {
                match expression.as_ref() {
                    Expr::Call { receiver, name, .. } => {
                        assert_eq!(name, "method");
                        // receiver should be Get (a.b)
                        match receiver.as_ref() {
                            Expr::Get {
                                name: prop_name, ..
                            } => {
                                assert_eq!(prop_name.lexeme, "b");
                            }
                            _ => panic!("Expected Get as receiver"),
                        }
                    }
                    _ => panic!("Expected Expr::Call"),
                }
            }
            _ => panic!("Expected Stmt::Expression"),
        }
    }

    #[test]
    fn test_constructor_call_syntax() {
        // Point.new(1, 2) - this looks like a method call on Point
        let stmts = parse_statements("Point.new(1, 2)").unwrap();

        match &stmts[0] {
            Stmt::Expression { expression } => match expression.as_ref() {
                Expr::Call {
                    receiver,
                    name,
                    arguments,
                } => {
                    assert_eq!(name, "new");
                    assert_eq!(arguments.len(), 2);
                    match receiver.as_ref() {
                        Expr::Variable { name: var_name } => {
                            assert_eq!(var_name.lexeme, "Point");
                        }
                        _ => panic!("Expected Variable as receiver"),
                    }
                }
                _ => panic!("Expected Expr::Call"),
            },
            _ => panic!("Expected Stmt::Expression"),
        }
    }

    // ==================== Field Access Tests ====================

    #[test]
    fn test_underscore_field_as_variable() {
        // _x is parsed as a regular variable (handled in codegen)
        let stmts = parse_statements("_x").unwrap();

        match &stmts[0] {
            Stmt::Expression { expression } => match expression.as_ref() {
                Expr::Variable { name } => {
                    assert_eq!(name.lexeme, "_x");
                }
                _ => panic!("Expected Expr::Variable"),
            },
            _ => panic!("Expected Stmt::Expression"),
        }
    }

    #[test]
    fn test_underscore_field_assignment() {
        // _x = 5 is parsed as regular assignment
        let stmts = parse_statements("_x = 5").unwrap();

        match &stmts[0] {
            Stmt::Expression { expression } => match expression.as_ref() {
                Expr::Assign { name, .. } => {
                    assert_eq!(name.lexeme, "_x");
                }
                _ => panic!("Expected Expr::Assign"),
            },
            _ => panic!("Expected Stmt::Expression"),
        }
    }

    // ==================== Complex Scenarios ====================

    #[test]
    fn test_class_followed_by_usage() {
        let source = r#"
            class Point {
                construct new(x) {
                }
            }
            Point.new(5)
        "#;
        let stmts = parse_statements(source).unwrap();
        assert_eq!(stmts.len(), 2);

        // First statement is class
        match &stmts[0] {
            Stmt::Class { name, .. } => {
                assert_eq!(name.lexeme, "Point");
            }
            _ => panic!("Expected Stmt::Class"),
        }

        // Second statement is constructor call
        match &stmts[1] {
            Stmt::Expression { expression } => match expression.as_ref() {
                Expr::Call { name, .. } => {
                    assert_eq!(name, "new");
                }
                _ => panic!("Expected Expr::Call"),
            },
            _ => panic!("Expected Stmt::Expression"),
        }
    }

    #[test]
    fn test_method_call_in_expression() {
        // 1 + point.getValue()
        let stmts = parse_statements("1 + point.getValue()").unwrap();

        match &stmts[0] {
            Stmt::Expression { expression } => match expression.as_ref() {
                Expr::Binary { right, .. } => match right.as_ref() {
                    Expr::Call { name, .. } => {
                        assert_eq!(name, "getValue");
                    }
                    _ => panic!("Expected Call on right side"),
                },
                _ => panic!("Expected Binary"),
            },
            _ => panic!("Expected Stmt::Expression"),
        }
    }

    #[test]
    fn test_var_with_constructor_call() {
        let stmts = parse_statements("var p = Point.new(1, 2)").unwrap();

        match &stmts[0] {
            Stmt::Var { name, initializer } => {
                assert_eq!(name.lexeme, "p");
                match initializer.as_ref() {
                    Expr::Call {
                        name: method_name,
                        arguments,
                        ..
                    } => {
                        assert_eq!(method_name, "new");
                        assert_eq!(arguments.len(), 2);
                    }
                    _ => panic!("Expected Call as initializer"),
                }
            }
            _ => panic!("Expected Stmt::Var"),
        }
    }
}

use crate::{
    ast::{Expr, Stmt},
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

        self.primary()
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

use std::fmt::Display;

use crate::token::{Literal, Token};

#[derive(Debug)]
pub enum Expr {
    Binary {
        left: Box<Expr>,
        operator: Token,
        right: Box<Expr>,
    },
    Unary {
        operator: Token,
        right: Box<Expr>,
    },
    Literal {
        //TODO Confusing with the naming.
        value: Literal,
    },
    Grouping {
        expression: Box<Expr>,
    },
    Variable {
        name: Token,
    },
    Assign {
        name: Token,
        value: Box<Expr>,
    },
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Binary {
                left,
                operator,
                right,
            } => {
                write!(f, "({} {} {})", operator.lexeme, left, right)
            }
            Expr::Unary { operator, right } => {
                write!(f, "({} {})", operator.lexeme, right)
            }
            Expr::Literal { value } => {
                write!(f, "{}", value)
            }
            Expr::Grouping { expression } => {
                write!(f, "(group {})", expression)
            }
            Expr::Variable { name } => {
                write!(f, "(var {})", name)
            }
            Expr::Assign { name, value } => {
                write!(f, "(assign {} {})", name, value)
            }
        }
    }
}

// Statements
#[derive(Debug)]
pub enum Stmt {
    Expression { expression: Box<Expr> },
    Var { name: Token, initializer: Box<Expr> },
    Block { statements: Vec<Stmt> },
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::{Literal, Token, TokenType};

    #[test]
    fn test_ast_print_nested_expression() {
        // No parser yet, so just manually creating the ast:
        // (* (- 123) (group 45.67))
        let expr = Expr::Binary {
            left: Box::new(Expr::Unary {
                operator: Token::new(TokenType::Minus, "-".to_string(), None, 1),
                right: Box::new(Expr::Literal {
                    value: Literal::Number(123.0),
                }),
            }),
            operator: Token::new(TokenType::Star, "*".to_string(), None, 1),
            right: Box::new(Expr::Grouping {
                expression: Box::new(Expr::Literal {
                    value: Literal::Number(45.67),
                }),
            }),
        };

        let output = format!("{}", expr);
        assert_eq!(output, "(* (- 123) (group 45.67))");
    }
}

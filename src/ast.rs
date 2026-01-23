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
    Logical {
        left: Box<Expr>,
        operator: Token,
        right: Box<Expr>,
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
    // Class impl
    Call {
        receiver: Box<Expr>,
        name: String,
        arguments: Vec<Box<Expr>>,
    },
    Get {
        object: Box<Expr>,
        name: Token,
    },
    Set {
        object: Box<Expr>,
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
            Expr::Logical {
                left,
                operator,
                right,
            } => todo!(),
            Expr::Call {
                receiver,
                name,
                arguments,
            } => todo!(),
            Expr::Get { object, name } => todo!(),
            Expr::Set {
                object,
                name,
                value,
            } => todo!(),
        }
    }
}

// Statements
#[derive(Debug)]
pub enum Stmt {
    Expression {
        expression: Box<Expr>,
    },
    Var {
        name: Token,
        initializer: Box<Expr>,
    },
    While {
        condition: Box<Expr>,
        body: Box<Stmt>,
    },
    Block {
        statements: Vec<Stmt>,
    },
    If {
        condition: Box<Expr>,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },
    Class {
        name: Token,
        constructor: Option<Method>,
        methods: Vec<Method>,
    },
}

#[derive(Debug)]
pub struct Method {
    pub name: Token,
    pub params: Vec<Token>,
    pub body: Vec<Stmt>,
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

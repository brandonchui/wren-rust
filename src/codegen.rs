use std::any::Any;
use std::collections::HashMap;

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module;
use inkwell::values::{FloatValue, PointerValue};

use crate::ast::{Expr, Stmt};
use crate::token::TokenType;

pub struct CodeGen<'ctx> {
    pub context: &'ctx Context,
    pub module: Module<'ctx>,
    pub builder: Builder<'ctx>,
    // pub execution_engine: ExecutionEngine<'ctx>,
    pub variables: HashMap<String, PointerValue<'ctx>>,
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        CodeGen {
            context,
            module: context.create_module("main"),
            builder: context.create_builder(),
            variables: HashMap::<String, PointerValue<'ctx>>::new(),
            // execution_engine: todo!(),
        }
    }

    pub fn compile(&mut self, statements: &[Stmt]) {
        let f64_type = self.context.f64_type();
        let fn_type = f64_type.fn_type(&[], false);

        let function = self.module.add_function("main", fn_type, None);

        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        // let result = self.codegen_expr(expr);

        let mut last_value = None;
        for stmt in statements {
            last_value = self.codegen_stmt(stmt);
        }

        match last_value {
            Some(val) => {
                self.builder.build_return(Some(&val));
            }
            _ => (),
        }
    }

    pub fn codegen_expr(&mut self, expr: &Expr) -> FloatValue<'ctx> {
        match expr {
            Expr::Binary {
                left,
                operator,
                right,
            } => {
                let l = self.codegen_expr(left);
                let r = self.codegen_expr(right);

                match operator.kind {
                    TokenType::Plus => self.builder.build_float_add(l, r, "addtmp").unwrap(),
                    TokenType::Minus => self.builder.build_float_sub(l, r, "subtmp").unwrap(),
                    TokenType::Star => self.builder.build_float_mul(l, r, "multmp").unwrap(),
                    TokenType::Slash => self.builder.build_float_div(l, r, "divtmp").unwrap(),
                    _ => todo!(),
                }
            }
            Expr::Unary { operator, right } => {
                let val = self.codegen_expr(right);
                match operator.kind {
                    TokenType::Minus => self.builder.build_float_neg(val, "negtmp").unwrap(),
                    _ => todo!(),
                }
            }
            Expr::Literal { value } => match value {
                crate::token::Literal::Number(n) => self.context.f64_type().const_float(*n),
                crate::token::Literal::StringLit(_) => todo!(),
            },
            Expr::Grouping { expression } => {
                // Recurse
                self.codegen_expr(expression)
            }
            Expr::Variable { name } => {
                // Stores
                match self.variables.get(&name.lexeme) {
                    Some(p) => self
                        .builder
                        .build_load(self.context.f64_type(), *p, &name.lexeme)
                        .unwrap()
                        .into_float_value(),
                    None => todo!(),
                }
            }
        }
    }

    pub fn codegen_stmt(&mut self, stmt: &Stmt) -> Option<FloatValue<'ctx>> {
        match stmt {
            Stmt::Expression { expression } => Some(self.codegen_expr(expression)),
            Stmt::Var { name, initializer } => {
                let ptr = self
                    .builder
                    .build_alloca(self.context.f64_type(), &name.lexeme);

                match ptr {
                    Ok(p) => {
                        let var_expr = self.codegen_expr(initializer);

                        self.builder.build_store(p, var_expr);
                        self.variables.insert(name.lexeme.clone(), p);
                    }
                    Err(_) => todo!(),
                }

                None
            }
        }
    }

    // Debugging
    pub fn print_ir(&self) {
        self.module.print_to_stderr();
    }

    pub fn jit_run(&self) -> f64 {
        let engine = self
            .module
            .create_jit_execution_engine(inkwell::OptimizationLevel::None)
            .unwrap();

        unsafe {
            let func = engine
                .get_function::<unsafe extern "C" fn() -> f64>("main")
                .unwrap();
            func.call()
        }
    }
}

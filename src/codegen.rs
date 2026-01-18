use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module;
use inkwell::values::FloatValue;

use crate::ast::Expr;
use crate::token::TokenType;

pub struct CodeGen<'ctx> {
    pub context: &'ctx Context,
    pub module: Module<'ctx>,
    pub builder: Builder<'ctx>,
    // pub execution_engine: ExecutionEngine<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        CodeGen {
            context,
            module: context.create_module("main"),
            builder: context.create_builder(),
            // execution_engine: todo!(),
        }
    }

    pub fn compile(&self, expr: &Expr) {
        let f64_type = self.context.f64_type();
        let fn_type = f64_type.fn_type(&[], false);

        let function = self.module.add_function("main", fn_type, None);

        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let result = self.codegen_expr(expr);

        self.builder.build_return(Some(&result));
    }

    pub fn codegen_expr(&self, expr: &Expr) -> FloatValue<'ctx> {
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

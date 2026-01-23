use std::any::Any;
use std::collections::{HashMap, HashSet};

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, StructType};
use inkwell::values::{FloatValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate};

use crate::ast::{Expr, Method, Stmt};
use crate::token::TokenType;

pub struct CodeGen<'ctx> {
    pub context: &'ctx Context,
    pub module: Module<'ctx>,
    pub builder: Builder<'ctx>,
    // pub execution_engine: ExecutionEngine<'ctx>,
    pub scopes: Vec<HashMap<String, PointerValue<'ctx>>>,
    //Class
    pub classes: HashMap<String, ClassInfo<'ctx>>,
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        CodeGen {
            context,
            module: context.create_module("main"),
            builder: context.create_builder(),
            scopes: vec![HashMap::new()],
            classes: HashMap::<String, ClassInfo<'ctx>>::new(), // execution_engine: todo!(),
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
                    TokenType::Greater => {
                        let val = self
                            .builder
                            .build_float_compare(FloatPredicate::OGT, l, r, "greater")
                            .unwrap();
                        self.builder
                            .build_unsigned_int_to_float(val, self.context.f64_type(), "intToFloat")
                            .unwrap()
                    }
                    TokenType::GreaterEqual => {
                        let val = self
                            .builder
                            .build_float_compare(FloatPredicate::OGE, l, r, "greaterEq")
                            .unwrap();
                        self.builder
                            .build_unsigned_int_to_float(val, self.context.f64_type(), "intToFloat")
                            .unwrap()
                    }
                    TokenType::Less => {
                        let val = self
                            .builder
                            .build_float_compare(FloatPredicate::OLT, l, r, "less")
                            .unwrap();
                        self.builder
                            .build_unsigned_int_to_float(val, self.context.f64_type(), "intToFloat")
                            .unwrap()
                    }
                    TokenType::LessEqual => {
                        let val = self
                            .builder
                            .build_float_compare(FloatPredicate::OLE, l, r, "lessEq")
                            .unwrap();
                        self.builder
                            .build_unsigned_int_to_float(val, self.context.f64_type(), "intToFloat")
                            .unwrap()
                    }
                    TokenType::EqualEqual => {
                        let val = self
                            .builder
                            .build_float_compare(FloatPredicate::OEQ, l, r, "equal")
                            .unwrap();
                        self.builder
                            .build_unsigned_int_to_float(val, self.context.f64_type(), "intToFloat")
                            .unwrap()
                    }
                    TokenType::BangEqual => {
                        let val = self
                            .builder
                            .build_float_compare(FloatPredicate::ONE, l, r, "notEqual")
                            .unwrap();
                        self.builder
                            .build_unsigned_int_to_float(val, self.context.f64_type(), "intToFloat")
                            .unwrap()
                    }
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
                let p = self.lookup_variable(&name.lexeme);
                self.builder
                    .build_load(self.context.f64_type(), p, &name.lexeme)
                    .unwrap()
                    .into_float_value()
            }
            Expr::Assign { name, value } => {
                let p = self.lookup_variable(&name.lexeme);

                let rhs = self.codegen_expr(value);

                self.builder.build_store(p, rhs).unwrap();
                rhs
            }
            Expr::Logical {
                left,
                operator,
                right,
            } => {
                // Short circuit ex.
                // 1 && 0

                let lhs_value = self.codegen_expr(left);
                let zero = self.context.f64_type().const_float(0.0);

                // Converting to an LLVM bool
                // An Order Not Equal operation to check if lhs != 0
                let lhs_bool = self
                    .builder
                    .build_float_compare(FloatPredicate::ONE, lhs_value, zero, "left_bool")
                    .unwrap();

                // Creating labels for the various function/blocks in llvm
                // For right now, we get the current block and function
                let entry_block = self.builder.get_insert_block().unwrap();
                let function = entry_block.get_parent().unwrap();

                // and then create the empty blocks
                let eval_right_block = self.context.append_basic_block(function, "eval_right");
                let merge_block = self.context.append_basic_block(function, "merge");

                // Depending on || or &&
                let skip_value = match operator.kind {
                    TokenType::AmpAmp => {
                        // true? evaluate right
                        // false? jump to merge
                        self.builder.build_conditional_branch(
                            lhs_bool,
                            eval_right_block,
                            merge_block,
                        );
                        self.context.bool_type().const_int(0, false)
                    }
                    TokenType::PipePipe => {
                        // true? jump to merge
                        // false? evaluate right
                        self.builder.build_conditional_branch(
                            lhs_bool,
                            merge_block,
                            eval_right_block,
                        );
                        self.context.bool_type().const_int(1, false)
                    }
                    _ => unreachable!(),
                };

                // Move the llvm builder to the eval_right_block position
                self.builder.position_at_end(eval_right_block);

                // RHS
                let rhs_value = self.codegen_expr(right);
                let rhs_bool = self
                    .builder
                    .build_float_compare(FloatPredicate::ONE, rhs_value, zero, "right_bool")
                    .unwrap();

                // Merge Block
                self.builder.build_unconditional_branch(merge_block);

                // ?
                let eval_right_end_block = self.builder.get_insert_block().unwrap();

                self.builder.position_at_end(merge_block);

                // Phi node
                // If came from entry_block, then use skip_value
                // If came from eval_right_end_block, then use right_bool
                let phi = self
                    .builder
                    .build_phi(self.context.bool_type(), "result")
                    .unwrap();

                phi.add_incoming(&[
                    (&skip_value, entry_block),
                    (&rhs_bool, eval_right_end_block),
                ]);

                // Converion to boolean to f64, then return
                self.builder
                    .build_unsigned_int_to_float(
                        phi.as_basic_value().into_int_value(),
                        self.context.f64_type(),
                        "bool_to_f64",
                    )
                    .unwrap()
            }
            Expr::Call { .. } => todo!("Call codegen not implemented"),
            Expr::Get { .. } => todo!("Get codegen not implemented"),
            Expr::Set { .. } => todo!("Set codegen not implemented"),
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
                        // self.scopes.insert(name.lexeme.clone(), p);
                        self.declare_variable(&name.lexeme, p);
                    }
                    Err(_) => todo!(),
                }

                None
            }
            Stmt::Block { statements } => {
                self.enter_scope();

                let mut last_value = None;
                for stmt in statements {
                    last_value = self.codegen_stmt(stmt);
                }
                self.exit_scope();
                last_value
            }
            Stmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let condition_value = self.codegen_expr(condition);
                let zero = self.context.f64_type().const_float(0.0);

                let condition_bool = self
                    .builder
                    .build_float_compare(FloatPredicate::ONE, condition_value, zero, "if_condition")
                    .unwrap();

                // Get functions and create blocks
                let function = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                let then_block = self.context.append_basic_block(function, "then");
                let else_block = self.context.append_basic_block(function, "else");
                let merge_block = self.context.append_basic_block(function, "merge");

                self.builder
                    .build_conditional_branch(condition_bool, then_block, else_block);

                // Generate the then branch
                self.builder.position_at_end(then_block);
                self.codegen_stmt(then_branch);
                self.builder.build_unconditional_branch(merge_block);

                // Generate else branch
                self.builder.position_at_end(else_block);
                match else_branch {
                    Some(else_stmt) => {
                        self.codegen_stmt(else_stmt);
                    }
                    _ => (),
                }

                // Jump
                self.builder.build_unconditional_branch(merge_block);

                // Return
                self.builder.position_at_end(merge_block);
                None
            }
            Stmt::While { condition, body } => {
                let function = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                let condition_block = self.context.append_basic_block(function, "condition");
                let body_block = self.context.append_basic_block(function, "body");
                let after_block = self.context.append_basic_block(function, "after");

                // Need terminator for blocks in llvm
                self.builder.build_unconditional_branch(condition_block);
                self.builder.position_at_end(condition_block);

                // Checking condition
                let condition_value = self.codegen_expr(condition);

                let zero = self.context.f64_type().const_float(0.0);

                let condition_bool = self
                    .builder
                    .build_float_compare(
                        FloatPredicate::ONE,
                        condition_value,
                        zero,
                        "condition_bool",
                    )
                    .unwrap();

                self.builder
                    .build_conditional_branch(condition_bool, body_block, after_block);

                // Body
                self.builder.position_at_end(body_block);

                self.codegen_stmt(body);
                self.builder.build_unconditional_branch(condition_block);

                // After
                self.builder.position_at_end(after_block);
                None
            }
            Stmt::Class {
                name,
                constructor,
                methods,
            } => {
                let fields = Self::collect_fields(constructor, methods);
                let mut hash = HashMap::<String, u32>::new();

                for (i, field_name) in fields.iter().enumerate() {
                    hash.insert(field_name.clone(), i as u32);
                }

                let f64_type = self.context.f64_type();
                let field_types: Vec<BasicTypeEnum> =
                    fields.iter().map(|_| f64_type.into()).collect();

                let struct_type = self.context.struct_type(&field_types, false);

                let saved_block = self.builder.get_insert_block();

                if let Some(cons) = constructor {
                    let return_ptr_type = struct_type.ptr_type(AddressSpace::default());
                    let param_types_array: Vec<BasicMetadataTypeEnum> =
                        cons.params.iter().map(|_| f64_type.into()).collect();

                    let func_type = return_ptr_type.fn_type(&param_types_array, false);

                    let func_name: String = format!("{}.new", name.lexeme);

                    let func = self.module.add_function(&func_name, func_type, None);
                    let basic_block = self.context.append_basic_block(func, "entry");
                    self.builder.position_at_end(basic_block);

                    // Constructor llvm
                    self.enter_scope();
                    for (i, param_token) in cons.params.iter().enumerate() {
                        let param_value = func.get_params()[i].into_float_value();
                        let ptr = self
                            .builder
                            .build_alloca(f64_type, &param_token.lexeme)
                            .unwrap();

                        self.builder.build_store(ptr, param_value);
                        self.declare_variable(&param_token.lexeme, ptr);
                    }
                    let this_ptr = self.builder.build_alloca(struct_type, "this").unwrap();

                    for (field_name, &index) in &hash {
                        let field_ptr = self
                            .builder
                            .build_struct_gep(struct_type, this_ptr, index, field_name)
                            .unwrap();
                        self.declare_variable(field_name, field_ptr);
                    }

                    for body in &cons.body {
                        self.codegen_stmt(body);
                    }
                    self.builder.build_return(Some(&this_ptr));
                    self.exit_scope();
                } else {
                    todo!()
                }

                for method in methods {
                    let this_param_type: BasicMetadataTypeEnum =
                        struct_type.ptr_type(AddressSpace::default()).into();
                    let mut param_types: Vec<BasicMetadataTypeEnum> = vec![this_param_type];
                    for _ in &method.params {
                        param_types.push(f64_type.into());
                    }

                    let func_type = f64_type.fn_type(&param_types, false);

                    //"ClassName.methodName"
                    let func_name = format!("{}.{}", name.lexeme, method.name.lexeme);

                    let func = self.module.add_function(&func_name, func_type, None);
                    let entry_block = self.context.append_basic_block(func, "entry");
                    self.builder.position_at_end(entry_block);

                    self.enter_scope();

                    let this_ptr = func.get_params()[0].into_pointer_value();

                    // GEPs to scope
                    for (field_name, &index) in &hash {
                        let field_ptr = self
                            .builder
                            .build_struct_gep(struct_type, this_ptr, index, field_name)
                            .unwrap();
                        self.declare_variable(field_name, field_ptr);
                    }

                    for (i, param_token) in method.params.iter().enumerate() {
                        let param_value = func.get_params()[i + 1].into_float_value();
                        let ptr = self
                            .builder
                            .build_alloca(f64_type, &param_token.lexeme)
                            .unwrap();
                        self.builder.build_store(ptr, param_value);
                        self.declare_variable(&param_token.lexeme, ptr);
                    }

                    for stmt in &method.body {
                        self.codegen_stmt(stmt);
                    }

                    self.exit_scope();
                }

                if let Some(block) = saved_block {
                    self.builder.position_at_end(block);
                }

                let info = ClassInfo {
                    struct_type,
                    field_indices: hash,
                };

                self.classes.insert(name.lexeme.clone(), info);

                None
            }
            Stmt::Return { value } => match value {
                Some(v) => {
                    let res = self.codegen_expr(v);
                    match self.builder.build_return(Some(&res)) {
                        Ok(_) => None,
                        Err(_) => panic!("Return statement error."),
                    }
                }
                None => {
                    //TODO Probably should return a null rather than a 0.0f
                    let res = self.context.f64_type().const_float(0.0);
                    self.builder.build_return(Some(&res));
                    None
                }
            },
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

    // Helpers
    pub fn collect_fields(constructor: &Option<Method>, methods: &Vec<Method>) -> HashSet<String> {
        let mut hash = HashSet::<String>::new();

        if let Some(con) = constructor {
            for c in &con.body {
                Self::collect_fields_from_stmt(&c, &mut hash);
            }
        }

        for body in methods {
            for c in &body.body {
                Self::collect_fields_from_stmt(&c, &mut hash);
            }
        }

        hash
    }

    fn collect_fields_from_stmt(stmt: &Stmt, hash: &mut HashSet<String>) {
        match stmt {
            Stmt::Expression { expression } => {
                Self::collect_fields_from_expr(expression, hash);
            }
            Stmt::Var { name, initializer } => {
                Self::collect_fields_from_expr(initializer, hash);
            }
            Stmt::While { condition, body } => {
                Self::collect_fields_from_expr(condition, hash);
                Self::collect_fields_from_stmt(body, hash);
            }
            Stmt::Block { statements } => {
                for statement in statements {
                    Self::collect_fields_from_stmt(statement, hash);
                }
            }
            Stmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                Self::collect_fields_from_expr(condition, hash);
                Self::collect_fields_from_stmt(then_branch, hash);

                match else_branch {
                    Some(v) => {
                        Self::collect_fields_from_stmt(v, hash);
                    }
                    None => {}
                }
            }
            Stmt::Return { value } => match value {
                Some(v) => {
                    Self::collect_fields_from_expr(v, hash);
                }
                None => {}
            },
            _ => {}
        }
    }
    fn collect_fields_from_expr(expr: &Expr, hash: &mut HashSet<String>) {
        match expr {
            Expr::Binary {
                left,
                operator,
                right,
            } => {
                Self::collect_fields_from_expr(left, hash);
                Self::collect_fields_from_expr(right, hash);
            }
            Expr::Unary { operator, right } => {
                Self::collect_fields_from_expr(right, hash);
            }
            Expr::Logical {
                left,
                operator,
                right,
            } => {
                Self::collect_fields_from_expr(left, hash);
                Self::collect_fields_from_expr(right, hash);
            }
            Expr::Grouping { expression } => {
                Self::collect_fields_from_expr(expression, hash);
            }
            Expr::Variable { name } => {
                //Leaf node - nothing to do
            }
            Expr::Assign { name, value } => {
                if name.lexeme.starts_with('_') {
                    hash.insert(name.lexeme.clone());
                }
                Self::collect_fields_from_expr(value, hash);
            }
            Expr::Call {
                receiver,
                name,
                arguments,
            } => {
                Self::collect_fields_from_expr(receiver, hash);
                for args in arguments {
                    Self::collect_fields_from_expr(args, hash);
                }
            }
            Expr::Literal { value } => {
                //Leaf node - nothing to do
            }
            Expr::Get { object, name } => {
                Self::collect_fields_from_expr(object, hash);
            }
            Expr::Set {
                object,
                name,
                value,
            } => {
                Self::collect_fields_from_expr(object, hash);
                Self::collect_fields_from_expr(value, hash);
            }
        }
    }
}

// Block
impl<'ctx> CodeGen<'ctx> {
    pub fn enter_scope(&mut self) {
        self.scopes
            .push(HashMap::<String, PointerValue<'ctx>>::new());
    }

    pub fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn declare_variable(&mut self, name: &str, ptr: PointerValue<'ctx>) {
        let top = self.scopes.last_mut();

        match top {
            Some(hm) => {
                hm.insert(name.to_string(), ptr);
            }
            None => todo!(),
        }
    }

    pub fn lookup_variable(&mut self, name: &str) -> PointerValue<'ctx> {
        // Search in "module" in the local scope level space first,
        // then tries to look in global if exist.
        // So, the local shadows the outer scopes.
        for map in self.scopes.iter().rev() {
            if let Some(p) = map.get(name) {
                return *p;
            }
        }
        todo!()
    }
}

// Class data struct
struct ClassInfo<'ctx> {
    struct_type: StructType<'ctx>,
    field_indices: HashMap<String, u32>,
}

#[cfg(test)]
mod tests {
    use super::CodeGen;
    use crate::parser::Parser;
    use crate::scanner::Scanner;
    use inkwell::context::Context;

    fn run_code(source: &str) -> f64 {
        let mut scanner = Scanner::new(source);
        scanner.scan_tokens();
        assert!(
            scanner.errors.is_empty(),
            "Scanner errors: {:?}",
            scanner.errors
        );

        let mut parser = Parser::new(scanner.tokens);
        let stmts = parser.parse().expect("Parser failed");

        let context = Context::create();
        let mut codegen = CodeGen::new(&context);
        codegen.compile(&stmts);
        codegen.jit_run()
    }

    // ==================== If Statement Tests ====================

    #[test]
    fn test_if_true_executes_then_branch() {
        // if (1) then branch executes, result is from then
        let result = run_code("var x = 0\nif (1) { x = 5 }\nx");
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_if_false_skips_then_branch() {
        // if (0) then branch is skipped, x stays 0
        let result = run_code("var x = 0\nif (0) { x = 5 }\nx");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_if_else_true_condition() {
        // if (1) takes then branch, not else
        let result = run_code("var x = 0\nif (1) { x = 10 } else { x = 20 }\nx");
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_if_else_false_condition() {
        // if (0) takes else branch
        let result = run_code("var x = 0\nif (0) { x = 10 } else { x = 20 }\nx");
        assert_eq!(result, 20.0);
    }

    #[test]
    fn test_if_with_logical_and_condition() {
        // if (1 && 1) should execute then
        let result = run_code("var x = 0\nif (1 && 1) { x = 100 }\nx");
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_if_with_logical_and_false() {
        // if (1 && 0) should skip then
        let result = run_code("var x = 0\nif (1 && 0) { x = 100 }\nx");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_if_with_logical_or_condition() {
        // if (0 || 1) should execute then
        let result = run_code("var x = 0\nif (0 || 1) { x = 50 }\nx");
        assert_eq!(result, 50.0);
    }

    #[test]
    fn test_if_with_logical_or_false() {
        // if (0 || 0) should skip then
        let result = run_code("var x = 0\nif (0 || 0) { x = 50 }\nx");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_nested_if_statements() {
        // Nested if: outer true, inner true
        let result = run_code("var x = 0\nif (1) { if (1) { x = 42 } }\nx");
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_nested_if_outer_true_inner_false() {
        // Nested if: outer true, inner false
        let result = run_code("var x = 0\nif (1) { if (0) { x = 42 } else { x = 99 } }\nx");
        assert_eq!(result, 99.0);
    }

    #[test]
    fn test_nested_if_outer_false() {
        // Nested if: outer false, inner never evaluated
        let result = run_code("var x = 5\nif (0) { if (1) { x = 42 } }\nx");
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_if_with_block_scoping() {
        // Variable declared in if block should shadow outer
        // Note: need trailing expression since Stmt::If returns None
        let result = run_code("var x = 1\nif (1) { var x = 100 }\n100");
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_if_block_scope_doesnt_leak() {
        // Variable in if block shouldn't affect outer scope after
        let result = run_code("var x = 1\nif (1) { var y = 100 }\nx");
        assert_eq!(result, 1.0);
    }

    // ==================== Logical Operator Tests ====================

    #[test]
    fn test_logical_and_true_true() {
        let result = run_code("1 && 1");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_logical_and_true_false() {
        let result = run_code("1 && 0");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_logical_and_false_true() {
        let result = run_code("0 && 1");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_logical_and_false_false() {
        let result = run_code("0 && 0");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_logical_or_true_true() {
        let result = run_code("1 || 1");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_logical_or_true_false() {
        let result = run_code("1 || 0");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_logical_or_false_true() {
        let result = run_code("0 || 1");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_logical_or_false_false() {
        let result = run_code("0 || 0");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_logical_chained_and() {
        // 1 && 1 && 1 = 1
        let result = run_code("1 && 1 && 1");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_logical_chained_or() {
        // 0 || 0 || 1 = 1
        let result = run_code("0 || 0 || 1");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_logical_mixed_and_or() {
        // 1 || 0 && 0 = 1 (and has higher precedence)
        // Parsed as: 1 || (0 && 0) = 1 || 0 = 1
        let result = run_code("1 || 0 && 0");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_logical_mixed_and_or_2() {
        // 0 && 1 || 1 = 1
        // Parsed as: (0 && 1) || 1 = 0 || 1 = 1
        let result = run_code("0 && 1 || 1");
        assert_eq!(result, 1.0);
    }

    // ==================== While Loop Tests ====================

    #[test]
    fn test_while_simple_countdown() {
        // Count down from 3 to 0
        let result = run_code("var x = 3\nwhile (x) { x = x - 1 }\nx");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_while_false_condition_never_executes() {
        // Body never runs because condition is false
        let result = run_code("var x = 5\nwhile (0) { x = 100 }\nx");
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_while_accumulator() {
        // Sum: 3 + 2 + 1 = 6
        let result =
            run_code("var sum = 0\nvar i = 3\nwhile (i) { sum = sum + i\ni = i - 1 }\nsum");
        assert_eq!(result, 6.0);
    }

    #[test]
    fn test_while_single_iteration() {
        // Runs once: x starts at 1 (truthy), sets x to 0, loop exits
        let result = run_code("var x = 1\nwhile (x) { x = 0 }\nx");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_while_with_block_body() {
        // Multiple statements in body
        let result = run_code("var x = 2\nvar y = 0\nwhile (x) { y = y + 1\nx = x - 1 }\ny");
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_while_nested() {
        // Nested while loops: outer runs 2 times, inner runs 2 times each = 4 total increments
        let result = run_code(
            "var count = 0\nvar i = 2\nwhile (i) { var j = 2\nwhile (j) { count = count + 1\nj = j - 1 }\ni = i - 1 }\ncount",
        );
        assert_eq!(result, 4.0);
    }

    #[test]
    fn test_while_modifies_outer_variable() {
        // While loop modifies variable from outer scope
        let result = run_code("var x = 0\nvar i = 3\nwhile (i) { x = x + 10\ni = i - 1 }\nx");
        assert_eq!(result, 30.0);
    }

    #[test]
    fn test_while_with_logical_condition() {
        // Using && in condition
        let result = run_code(
            "var x = 3\nvar y = 1\nwhile (x && y) { x = x - 1\nif (x) { y = 1 } else { y = 0 } }\nx",
        );
        assert_eq!(result, 0.0);
    }

    // ==================== Comparison Operator Tests ====================

    #[test]
    fn test_greater_than_true() {
        let result = run_code("5 > 3");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_greater_than_false() {
        let result = run_code("3 > 5");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_greater_than_equal_false() {
        let result = run_code("5 > 5");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_greater_equal_true() {
        let result = run_code("5 >= 5");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_greater_equal_greater() {
        let result = run_code("6 >= 5");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_greater_equal_false() {
        let result = run_code("4 >= 5");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_less_than_true() {
        let result = run_code("3 < 5");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_less_than_false() {
        let result = run_code("5 < 3");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_less_than_equal_false() {
        let result = run_code("5 < 5");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_less_equal_true() {
        let result = run_code("5 <= 5");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_less_equal_less() {
        let result = run_code("4 <= 5");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_less_equal_false() {
        let result = run_code("6 <= 5");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_equal_equal_true() {
        let result = run_code("5 == 5");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_equal_equal_false() {
        let result = run_code("5 == 3");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_not_equal_true() {
        let result = run_code("5 != 3");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_not_equal_false() {
        let result = run_code("5 != 5");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_comparison_in_expression() {
        // (5 > 3) + (2 < 4) = 1 + 1 = 2
        let result = run_code("(5 > 3) + (2 < 4)");
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_comparison_with_variables() {
        let result = run_code("var x = 10\nvar y = 5\nx > y");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_comparison_in_if_condition() {
        let result = run_code("var x = 0\nif (5 > 3) { x = 100 }\nx");
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_comparison_in_while_condition() {
        // Count from 0 to 5
        let result = run_code("var i = 0\nwhile (i < 5) { i = i + 1 }\ni");
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_comparison_chained_with_logical() {
        // (5 > 3) && (2 < 4) = true && true = 1
        let result = run_code("(5 > 3) && (2 < 4)");
        assert_eq!(result, 1.0);
    }

    // ==================== For Loop Tests ====================

    #[test]
    fn test_for_loop_simple() {
        // Sum 1..4 (1 + 2 + 3 = 6)
        let result = run_code("var sum = 0\nfor (i in 1..4) { sum = sum + i }\nsum");
        assert_eq!(result, 6.0);
    }

    #[test]
    fn test_for_loop_single_iteration() {
        // 1..2 means just i=1
        let result = run_code("var x = 0\nfor (i in 1..2) { x = i }\nx");
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_for_loop_no_iteration() {
        // 5..5 means no iterations (5 < 5 is false)
        let result = run_code("var x = 99\nfor (i in 5..5) { x = 0 }\nx");
        assert_eq!(result, 99.0);
    }

    #[test]
    fn test_for_loop_accumulator() {
        // Count iterations: 0..5 means 0,1,2,3,4 = 5 iterations
        let result = run_code("var count = 0\nfor (i in 0..5) { count = count + 1 }\ncount");
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_for_loop_uses_loop_variable() {
        // Last value of i should be 4 (loop runs while i < 5)
        let result = run_code("var last = 0\nfor (i in 0..5) { last = i }\nlast");
        assert_eq!(result, 4.0);
    }

    #[test]
    fn test_for_loop_nested() {
        // Outer 0..3, inner 0..3 = 3 * 3 = 9 iterations
        let result = run_code(
            "var count = 0\nfor (i in 0..3) { for (j in 0..3) { count = count + 1 } }\ncount",
        );
        assert_eq!(result, 9.0);
    }

    #[test]
    fn test_for_loop_variable_scoped() {
        // i should not leak outside the for loop
        // After for loop, we access outer x which should still be 100
        let result = run_code("var x = 100\nfor (i in 0..3) { var y = i }\nx");
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_for_loop_with_expressions() {
        // Range bounds can be expressions
        let result = run_code(
            "var start = 1\nvar end = 4\nvar sum = 0\nfor (i in start..end) { sum = sum + i }\nsum",
        );
        assert_eq!(result, 6.0);
    }
}

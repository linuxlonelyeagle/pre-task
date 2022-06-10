
// Generated from toy.g4 by ANTLR 4.10.1

#pragma once


#include "antlr4-runtime.h"
#include "toyVisitor.h"


/**
 * This class provides an empty implementation of toyVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  toyBaseVisitor : public toyVisitor {
public:

  virtual std::any visitModule(toyParser::ModuleContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExpression(toyParser::ExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIdentifierexpr(toyParser::IdentifierexprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitReturnExpression(toyParser::ReturnExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTensorLiteral(toyParser::TensorLiteralContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLiteralList(toyParser::LiteralListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStructLiteral(toyParser::StructLiteralContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDecl(toyParser::DeclContext *ctx) override {
    return visitChildren(ctx);
  }
  
  virtual std::any visitType(toyParser::TypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFundefinition(toyParser::FundefinitionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPrototype(toyParser::PrototypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDecl_list(toyParser::Decl_listContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBlock(toyParser::BlockContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBlock_expr(toyParser::Block_exprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStructdefine(toyParser::StructdefineContext *ctx) override {
    return visitChildren(ctx);
  }


};


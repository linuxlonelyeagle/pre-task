
// Generated from toy.g4 by ANTLR 4.10.1

#pragma once


#include "antlr4-runtime.h"
#include "toyParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by toyParser.
 */
class  toyVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by toyParser.
   */
    virtual std::any visitModule(toyParser::ModuleContext *context) = 0;

    virtual std::any visitExpression(toyParser::ExpressionContext *context) = 0;

    virtual std::any visitIdentifierexpr(toyParser::IdentifierexprContext *context) = 0;

    virtual std::any visitReturnExpression(toyParser::ReturnExpressionContext *context) = 0;

    virtual std::any visitTensorLiteral(toyParser::TensorLiteralContext *context) = 0;

    virtual std::any visitLiteralList(toyParser::LiteralListContext *context) = 0;

    virtual std::any visitStructLiteral(toyParser::StructLiteralContext *context) = 0;

    virtual std::any visitDecl(toyParser::DeclContext *context) = 0;

    virtual std::any visitType(toyParser::TypeContext *context) = 0;

    virtual std::any visitFundefinition(toyParser::FundefinitionContext *context) = 0;

    virtual std::any visitPrototype(toyParser::PrototypeContext *context) = 0;

    virtual std::any visitDecl_list(toyParser::Decl_listContext *context) = 0;

    virtual std::any visitBlock(toyParser::BlockContext *context) = 0;

    virtual std::any visitBlock_expr(toyParser::Block_exprContext *context) = 0;

    virtual std::any visitStructdefine(toyParser::StructdefineContext *context) = 0;


};


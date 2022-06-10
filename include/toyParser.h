
// Generated from toy.g4 by ANTLR 4.10.1

#pragma once


#include "antlr4-runtime.h"




class  toyParser : public antlr4::Parser {
public:
  enum {
    T__0 = 1, Parenthese_open = 2, Parenthese_close = 3, Bracket_open = 4, 
    Bracket_close = 5, Sbracket_open = 6, Sbracket_close = 7, Return = 8, 
    Semicolon = 9, Var = 10, Def = 11, Identifier = 12, Number = 13, Equal = 14, 
    AngleBrackets_open = 15, AngleBrackets_close = 16, Comma = 17, WS = 18, 
    AddSub = 19, MulDiv = 20, Dian = 21, Comment = 22
  };

  enum {
    RuleModule = 0, RuleExpression = 1, RuleIdentifierexpr = 2, RuleReturnExpression = 3, 
    RuleTensorLiteral = 4, RuleLiteralList = 5, RuleStructLiteral = 6, RuleDecl = 7, 
    RuleType = 8, RuleFundefinition = 9, RulePrototype = 10, RuleDecl_list = 11, 
    RuleBlock = 12, RuleBlock_expr = 13, RuleStructdefine = 14
  };

  explicit toyParser(antlr4::TokenStream *input);

  toyParser(antlr4::TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options);

  ~toyParser() override;

  std::string getGrammarFileName() const override;

  const antlr4::atn::ATN& getATN() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;


  class ModuleContext;
  class ExpressionContext;
  class IdentifierexprContext;
  class ReturnExpressionContext;
  class TensorLiteralContext;
  class LiteralListContext;
  class StructLiteralContext;
  class DeclContext;
  class TypeContext;
  class FundefinitionContext;
  class PrototypeContext;
  class Decl_listContext;
  class BlockContext;
  class Block_exprContext;
  class StructdefineContext; 

  class  ModuleContext : public antlr4::ParserRuleContext {
  public:
    ModuleContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<StructdefineContext *> structdefine();
    StructdefineContext* structdefine(size_t i);
    std::vector<FundefinitionContext *> fundefinition();
    FundefinitionContext* fundefinition(size_t i);


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ModuleContext* module();

  class  ExpressionContext : public antlr4::ParserRuleContext {
  public:
    ExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Number();
    TensorLiteralContext *tensorLiteral();
    antlr4::tree::TerminalNode *Parenthese_open();
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    antlr4::tree::TerminalNode *Parenthese_close();
    StructLiteralContext *structLiteral();
    IdentifierexprContext *identifierexpr();
    antlr4::tree::TerminalNode *Dian();
    antlr4::tree::TerminalNode *MulDiv();
    antlr4::tree::TerminalNode *AddSub();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExpressionContext* expression();
  ExpressionContext* expression(int precedence);
  class  IdentifierexprContext : public antlr4::ParserRuleContext {
  public:
    IdentifierexprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *Parenthese_open();
    antlr4::tree::TerminalNode *Parenthese_close();
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    std::vector<antlr4::tree::TerminalNode *> Comma();
    antlr4::tree::TerminalNode* Comma(size_t i);


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  IdentifierexprContext* identifierexpr();

  class  ReturnExpressionContext : public antlr4::ParserRuleContext {
  public:
    ReturnExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Return();
    ExpressionContext *expression();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ReturnExpressionContext* returnExpression();

  class  TensorLiteralContext : public antlr4::ParserRuleContext {
  public:
    TensorLiteralContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Sbracket_open();
    antlr4::tree::TerminalNode *Sbracket_close();
    std::vector<TensorLiteralContext *> tensorLiteral();
    TensorLiteralContext* tensorLiteral(size_t i);
    std::vector<antlr4::tree::TerminalNode *> Comma();
    antlr4::tree::TerminalNode* Comma(size_t i);
    antlr4::tree::TerminalNode *Number();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TensorLiteralContext* tensorLiteral();

  class  LiteralListContext : public antlr4::ParserRuleContext {
  public:
    LiteralListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TensorLiteralContext *tensorLiteral();
    antlr4::tree::TerminalNode *Comma();
    LiteralListContext *literalList();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  LiteralListContext* literalList();

  class  StructLiteralContext : public antlr4::ParserRuleContext {
  public:
    StructLiteralContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Bracket_open();
    antlr4::tree::TerminalNode *Bracket_close();
    std::vector<LiteralListContext *> literalList();
    LiteralListContext* literalList(size_t i);


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  StructLiteralContext* structLiteral();

  class  DeclContext : public antlr4::ParserRuleContext {
  public:
    DeclContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Var();
    std::vector<antlr4::tree::TerminalNode *> Identifier();
    antlr4::tree::TerminalNode* Identifier(size_t i);
    TypeContext *type();
    antlr4::tree::TerminalNode *Equal();
    ExpressionContext *expression();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DeclContext* decl();

  class  TypeContext : public antlr4::ParserRuleContext {
  public:
    TypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *AngleBrackets_open();
    antlr4::tree::TerminalNode *AngleBrackets_close();
    std::vector<antlr4::tree::TerminalNode *> Number();
    antlr4::tree::TerminalNode* Number(size_t i);
    std::vector<antlr4::tree::TerminalNode *> Comma();
    antlr4::tree::TerminalNode* Comma(size_t i);


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TypeContext* type();

  class  FundefinitionContext : public antlr4::ParserRuleContext {
  public:
    FundefinitionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PrototypeContext *prototype();
    BlockContext *block();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  FundefinitionContext* fundefinition();

  class  PrototypeContext : public antlr4::ParserRuleContext {
  public:
    PrototypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Def();
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *Parenthese_open();
    Decl_listContext *decl_list();
    antlr4::tree::TerminalNode *Parenthese_close();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  PrototypeContext* prototype();

  class  Decl_listContext : public antlr4::ParserRuleContext {
  public:
    Decl_listContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *Comma();
    Decl_listContext *decl_list();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Decl_listContext* decl_list();

  class  BlockContext : public antlr4::ParserRuleContext {
  public:
    BlockContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Bracket_open();
    antlr4::tree::TerminalNode *Bracket_close();
    std::vector<Block_exprContext *> block_expr();
    Block_exprContext* block_expr(size_t i);
    std::vector<antlr4::tree::TerminalNode *> Semicolon();
    antlr4::tree::TerminalNode* Semicolon(size_t i);


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BlockContext* block();

  class  Block_exprContext : public antlr4::ParserRuleContext {
  public:
    Block_exprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    DeclContext *decl();
    ReturnExpressionContext *returnExpression();
    ExpressionContext *expression();


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Block_exprContext* block_expr();

  class  StructdefineContext : public antlr4::ParserRuleContext {
  public:
    StructdefineContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *Bracket_open();
    antlr4::tree::TerminalNode *Bracket_close();
    std::vector<DeclContext *> decl();
    DeclContext* decl(size_t i);
    std::vector<antlr4::tree::TerminalNode *> Semicolon();
    antlr4::tree::TerminalNode* Semicolon(size_t i);


    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  StructdefineContext* structdefine();


  bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex, size_t predicateIndex) override;

  bool expressionSempred(ExpressionContext *_localctx, size_t predicateIndex);

  // By default the static state used to implement the parser is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:
};


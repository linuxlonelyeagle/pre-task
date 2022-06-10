
// Generated from toy.g4 by ANTLR 4.10.1


#include "toyVisitor.h"

#include "toyParser.h"


using namespace antlrcpp;

using namespace antlr4;

namespace {

struct ToyParserStaticData final {
  ToyParserStaticData(std::vector<std::string> ruleNames,
                        std::vector<std::string> literalNames,
                        std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  ToyParserStaticData(const ToyParserStaticData&) = delete;
  ToyParserStaticData(ToyParserStaticData&&) = delete;
  ToyParserStaticData& operator=(const ToyParserStaticData&) = delete;
  ToyParserStaticData& operator=(ToyParserStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

std::once_flag toyParserOnceFlag;
ToyParserStaticData *toyParserStaticData = nullptr;

void toyParserInitialize() {
  assert(toyParserStaticData == nullptr);
  auto staticData = std::make_unique<ToyParserStaticData>(
    std::vector<std::string>{
      "module", "expression", "identifierexpr", "returnExpression", "tensorLiteral", 
      "literalList", "structLiteral", "decl", "type", "fundefinition", "prototype", 
      "decl_list", "block", "block_expr", "structdefine"
    },
    std::vector<std::string>{
      "", "'struct'", "'('", "')'", "'{'", "'}'", "'['", "']'", "'return'", 
      "';'", "'var'", "'def'", "", "", "'='", "'<'", "'>'", "','", "", "", 
      "", "'.'"
    },
    std::vector<std::string>{
      "", "", "Parenthese_open", "Parenthese_close", "Bracket_open", "Bracket_close", 
      "Sbracket_open", "Sbracket_close", "Return", "Semicolon", "Var", "Def", 
      "Identifier", "Number", "Equal", "AngleBrackets_open", "AngleBrackets_close", 
      "Comma", "WS", "AddSub", "MulDiv", "Dian", "Comment"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,22,193,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,2,
  	7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,2,14,7,
  	14,1,0,5,0,32,8,0,10,0,12,0,35,9,0,1,0,4,0,38,8,0,11,0,12,0,39,1,1,1,
  	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,51,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  	1,1,1,1,5,1,62,8,1,10,1,12,1,65,9,1,1,2,1,2,1,2,1,2,1,2,1,2,5,2,73,8,
  	2,10,2,12,2,76,9,2,3,2,78,8,2,1,2,3,2,81,8,2,1,3,1,3,1,3,3,3,86,8,3,1,
  	4,1,4,1,4,1,4,5,4,92,8,4,10,4,12,4,95,9,4,3,4,97,8,4,1,4,1,4,3,4,101,
  	8,4,1,5,1,5,1,5,1,5,1,5,3,5,108,8,5,1,6,1,6,4,6,112,8,6,11,6,12,6,113,
  	1,6,1,6,1,7,1,7,1,7,3,7,121,8,7,1,7,1,7,3,7,125,8,7,1,7,1,7,1,7,1,7,3,
  	7,131,8,7,3,7,133,8,7,1,8,1,8,1,8,1,8,5,8,139,8,8,10,8,12,8,142,9,8,3,
  	8,144,8,8,1,8,1,8,1,9,1,9,1,9,1,10,1,10,1,10,1,10,1,10,1,10,1,11,1,11,
  	1,11,1,11,1,11,3,11,162,8,11,1,12,1,12,1,12,1,12,5,12,168,8,12,10,12,
  	12,12,171,9,12,1,12,1,12,1,13,1,13,1,13,3,13,178,8,13,1,14,1,14,1,14,
  	1,14,1,14,1,14,5,14,186,8,14,10,14,12,14,189,9,14,1,14,1,14,1,14,0,1,
  	2,15,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,0,0,207,0,33,1,0,0,0,2,50,
  	1,0,0,0,4,80,1,0,0,0,6,85,1,0,0,0,8,100,1,0,0,0,10,107,1,0,0,0,12,109,
  	1,0,0,0,14,132,1,0,0,0,16,134,1,0,0,0,18,147,1,0,0,0,20,150,1,0,0,0,22,
  	161,1,0,0,0,24,163,1,0,0,0,26,177,1,0,0,0,28,179,1,0,0,0,30,32,3,28,14,
  	0,31,30,1,0,0,0,32,35,1,0,0,0,33,31,1,0,0,0,33,34,1,0,0,0,34,37,1,0,0,
  	0,35,33,1,0,0,0,36,38,3,18,9,0,37,36,1,0,0,0,38,39,1,0,0,0,39,37,1,0,
  	0,0,39,40,1,0,0,0,40,1,1,0,0,0,41,42,6,1,-1,0,42,51,5,13,0,0,43,51,3,
  	8,4,0,44,45,5,2,0,0,45,46,3,2,1,0,46,47,5,3,0,0,47,51,1,0,0,0,48,51,3,
  	12,6,0,49,51,3,4,2,0,50,41,1,0,0,0,50,43,1,0,0,0,50,44,1,0,0,0,50,48,
  	1,0,0,0,50,49,1,0,0,0,51,63,1,0,0,0,52,53,10,8,0,0,53,54,5,21,0,0,54,
  	62,3,2,1,9,55,56,10,7,0,0,56,57,5,20,0,0,57,62,3,2,1,8,58,59,10,6,0,0,
  	59,60,5,19,0,0,60,62,3,2,1,7,61,52,1,0,0,0,61,55,1,0,0,0,61,58,1,0,0,
  	0,62,65,1,0,0,0,63,61,1,0,0,0,63,64,1,0,0,0,64,3,1,0,0,0,65,63,1,0,0,
  	0,66,81,5,12,0,0,67,68,5,12,0,0,68,77,5,2,0,0,69,74,3,2,1,0,70,71,5,17,
  	0,0,71,73,3,2,1,0,72,70,1,0,0,0,73,76,1,0,0,0,74,72,1,0,0,0,74,75,1,0,
  	0,0,75,78,1,0,0,0,76,74,1,0,0,0,77,69,1,0,0,0,77,78,1,0,0,0,78,79,1,0,
  	0,0,79,81,5,3,0,0,80,66,1,0,0,0,80,67,1,0,0,0,81,5,1,0,0,0,82,86,5,8,
  	0,0,83,84,5,8,0,0,84,86,3,2,1,0,85,82,1,0,0,0,85,83,1,0,0,0,86,7,1,0,
  	0,0,87,96,5,6,0,0,88,93,3,8,4,0,89,90,5,17,0,0,90,92,3,8,4,0,91,89,1,
  	0,0,0,92,95,1,0,0,0,93,91,1,0,0,0,93,94,1,0,0,0,94,97,1,0,0,0,95,93,1,
  	0,0,0,96,88,1,0,0,0,96,97,1,0,0,0,97,98,1,0,0,0,98,101,5,7,0,0,99,101,
  	5,13,0,0,100,87,1,0,0,0,100,99,1,0,0,0,101,9,1,0,0,0,102,108,3,8,4,0,
  	103,104,3,8,4,0,104,105,5,17,0,0,105,106,3,10,5,0,106,108,1,0,0,0,107,
  	102,1,0,0,0,107,103,1,0,0,0,108,11,1,0,0,0,109,111,5,4,0,0,110,112,3,
  	10,5,0,111,110,1,0,0,0,112,113,1,0,0,0,113,111,1,0,0,0,113,114,1,0,0,
  	0,114,115,1,0,0,0,115,116,5,5,0,0,116,13,1,0,0,0,117,118,5,10,0,0,118,
  	120,5,12,0,0,119,121,3,16,8,0,120,119,1,0,0,0,120,121,1,0,0,0,121,124,
  	1,0,0,0,122,123,5,14,0,0,123,125,3,2,1,0,124,122,1,0,0,0,124,125,1,0,
  	0,0,125,133,1,0,0,0,126,127,5,12,0,0,127,130,5,12,0,0,128,129,5,14,0,
  	0,129,131,3,2,1,0,130,128,1,0,0,0,130,131,1,0,0,0,131,133,1,0,0,0,132,
  	117,1,0,0,0,132,126,1,0,0,0,133,15,1,0,0,0,134,143,5,15,0,0,135,140,5,
  	13,0,0,136,137,5,17,0,0,137,139,5,13,0,0,138,136,1,0,0,0,139,142,1,0,
  	0,0,140,138,1,0,0,0,140,141,1,0,0,0,141,144,1,0,0,0,142,140,1,0,0,0,143,
  	135,1,0,0,0,143,144,1,0,0,0,144,145,1,0,0,0,145,146,5,16,0,0,146,17,1,
  	0,0,0,147,148,3,20,10,0,148,149,3,24,12,0,149,19,1,0,0,0,150,151,5,11,
  	0,0,151,152,5,12,0,0,152,153,5,2,0,0,153,154,3,22,11,0,154,155,5,3,0,
  	0,155,21,1,0,0,0,156,162,5,12,0,0,157,158,5,12,0,0,158,159,5,17,0,0,159,
  	162,3,22,11,0,160,162,1,0,0,0,161,156,1,0,0,0,161,157,1,0,0,0,161,160,
  	1,0,0,0,162,23,1,0,0,0,163,169,5,4,0,0,164,165,3,26,13,0,165,166,5,9,
  	0,0,166,168,1,0,0,0,167,164,1,0,0,0,168,171,1,0,0,0,169,167,1,0,0,0,169,
  	170,1,0,0,0,170,172,1,0,0,0,171,169,1,0,0,0,172,173,5,5,0,0,173,25,1,
  	0,0,0,174,178,3,14,7,0,175,178,3,6,3,0,176,178,3,2,1,0,177,174,1,0,0,
  	0,177,175,1,0,0,0,177,176,1,0,0,0,178,27,1,0,0,0,179,180,5,1,0,0,180,
  	181,5,12,0,0,181,187,5,4,0,0,182,183,3,14,7,0,183,184,5,9,0,0,184,186,
  	1,0,0,0,185,182,1,0,0,0,186,189,1,0,0,0,187,185,1,0,0,0,187,188,1,0,0,
  	0,188,190,1,0,0,0,189,187,1,0,0,0,190,191,5,5,0,0,191,29,1,0,0,0,24,33,
  	39,50,61,63,74,77,80,85,93,96,100,107,113,120,124,130,132,140,143,161,
  	169,177,187
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  toyParserStaticData = staticData.release();
}

}

toyParser::toyParser(TokenStream *input) : toyParser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

toyParser::toyParser(TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options) : Parser(input) {
  toyParser::initialize();
  _interpreter = new atn::ParserATNSimulator(this, *toyParserStaticData->atn, toyParserStaticData->decisionToDFA, toyParserStaticData->sharedContextCache, options);
}

toyParser::~toyParser() {
  delete _interpreter;
}

const atn::ATN& toyParser::getATN() const {
  return *toyParserStaticData->atn;
}

std::string toyParser::getGrammarFileName() const {
  return "toy.g4";
}

const std::vector<std::string>& toyParser::getRuleNames() const {
  return toyParserStaticData->ruleNames;
}

const dfa::Vocabulary& toyParser::getVocabulary() const {
  return toyParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView toyParser::getSerializedATN() const {
  return toyParserStaticData->serializedATN;
}


//----------------- ModuleContext ------------------------------------------------------------------

toyParser::ModuleContext::ModuleContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<toyParser::StructdefineContext *> toyParser::ModuleContext::structdefine() {
  return getRuleContexts<toyParser::StructdefineContext>();
}

toyParser::StructdefineContext* toyParser::ModuleContext::structdefine(size_t i) {
  return getRuleContext<toyParser::StructdefineContext>(i);
}

std::vector<toyParser::FundefinitionContext *> toyParser::ModuleContext::fundefinition() {
  return getRuleContexts<toyParser::FundefinitionContext>();
}

toyParser::FundefinitionContext* toyParser::ModuleContext::fundefinition(size_t i) {
  return getRuleContext<toyParser::FundefinitionContext>(i);
}


size_t toyParser::ModuleContext::getRuleIndex() const {
  return toyParser::RuleModule;
}


std::any toyParser::ModuleContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitModule(this);
  else
    return visitor->visitChildren(this);
}

toyParser::ModuleContext* toyParser::module() {
  ModuleContext *_localctx = _tracker.createInstance<ModuleContext>(_ctx, getState());
  enterRule(_localctx, 0, toyParser::RuleModule);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(33);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == toyParser::T__0) {
      setState(30);
      structdefine();
      setState(35);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(37); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(36);
      fundefinition();
      setState(39); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == toyParser::Def);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionContext ------------------------------------------------------------------

toyParser::ExpressionContext::ExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* toyParser::ExpressionContext::Number() {
  return getToken(toyParser::Number, 0);
}

toyParser::TensorLiteralContext* toyParser::ExpressionContext::tensorLiteral() {
  return getRuleContext<toyParser::TensorLiteralContext>(0);
}

tree::TerminalNode* toyParser::ExpressionContext::Parenthese_open() {
  return getToken(toyParser::Parenthese_open, 0);
}

std::vector<toyParser::ExpressionContext *> toyParser::ExpressionContext::expression() {
  return getRuleContexts<toyParser::ExpressionContext>();
}

toyParser::ExpressionContext* toyParser::ExpressionContext::expression(size_t i) {
  return getRuleContext<toyParser::ExpressionContext>(i);
}

tree::TerminalNode* toyParser::ExpressionContext::Parenthese_close() {
  return getToken(toyParser::Parenthese_close, 0);
}

toyParser::StructLiteralContext* toyParser::ExpressionContext::structLiteral() {
  return getRuleContext<toyParser::StructLiteralContext>(0);
}

toyParser::IdentifierexprContext* toyParser::ExpressionContext::identifierexpr() {
  return getRuleContext<toyParser::IdentifierexprContext>(0);
}

tree::TerminalNode* toyParser::ExpressionContext::Dian() {
  return getToken(toyParser::Dian, 0);
}

tree::TerminalNode* toyParser::ExpressionContext::MulDiv() {
  return getToken(toyParser::MulDiv, 0);
}

tree::TerminalNode* toyParser::ExpressionContext::AddSub() {
  return getToken(toyParser::AddSub, 0);
}


size_t toyParser::ExpressionContext::getRuleIndex() const {
  return toyParser::RuleExpression;
}


std::any toyParser::ExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitExpression(this);
  else
    return visitor->visitChildren(this);
}


toyParser::ExpressionContext* toyParser::expression() {
   return expression(0);
}

toyParser::ExpressionContext* toyParser::expression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  toyParser::ExpressionContext *_localctx = _tracker.createInstance<ExpressionContext>(_ctx, parentState);
  toyParser::ExpressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 2;
  enterRecursionRule(_localctx, 2, toyParser::RuleExpression, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(50);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx)) {
    case 1: {
      setState(42);
      match(toyParser::Number);
      break;
    }

    case 2: {
      setState(43);
      tensorLiteral();
      break;
    }

    case 3: {
      setState(44);
      match(toyParser::Parenthese_open);
      setState(45);
      expression(0);
      setState(46);
      match(toyParser::Parenthese_close);
      break;
    }

    case 4: {
      setState(48);
      structLiteral();
      break;
    }

    case 5: {
      setState(49);
      identifierexpr();
      break;
    }

    default:
      break;
    }
    _ctx->stop = _input->LT(-1);
    setState(63);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 4, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(61);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 3, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<ExpressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExpression);
          setState(52);

          if (!(precpred(_ctx, 8))) throw FailedPredicateException(this, "precpred(_ctx, 8)");

          setState(53);
          match(toyParser::Dian);
          setState(54);
          expression(9);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<ExpressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExpression);
          setState(55);

          if (!(precpred(_ctx, 7))) throw FailedPredicateException(this, "precpred(_ctx, 7)");

          setState(56);
          match(toyParser::MulDiv);
          setState(57);
          expression(8);
          break;
        }

        case 3: {
          _localctx = _tracker.createInstance<ExpressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExpression);
          setState(58);

          if (!(precpred(_ctx, 6))) throw FailedPredicateException(this, "precpred(_ctx, 6)");

          setState(59);
          match(toyParser::AddSub);
          setState(60);
          expression(7);
          break;
        }

        default:
          break;
        } 
      }
      setState(65);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 4, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- IdentifierexprContext ------------------------------------------------------------------

toyParser::IdentifierexprContext::IdentifierexprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* toyParser::IdentifierexprContext::Identifier() {
  return getToken(toyParser::Identifier, 0);
}

tree::TerminalNode* toyParser::IdentifierexprContext::Parenthese_open() {
  return getToken(toyParser::Parenthese_open, 0);
}

tree::TerminalNode* toyParser::IdentifierexprContext::Parenthese_close() {
  return getToken(toyParser::Parenthese_close, 0);
}

std::vector<toyParser::ExpressionContext *> toyParser::IdentifierexprContext::expression() {
  return getRuleContexts<toyParser::ExpressionContext>();
}

toyParser::ExpressionContext* toyParser::IdentifierexprContext::expression(size_t i) {
  return getRuleContext<toyParser::ExpressionContext>(i);
}

std::vector<tree::TerminalNode *> toyParser::IdentifierexprContext::Comma() {
  return getTokens(toyParser::Comma);
}

tree::TerminalNode* toyParser::IdentifierexprContext::Comma(size_t i) {
  return getToken(toyParser::Comma, i);
}


size_t toyParser::IdentifierexprContext::getRuleIndex() const {
  return toyParser::RuleIdentifierexpr;
}


std::any toyParser::IdentifierexprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitIdentifierexpr(this);
  else
    return visitor->visitChildren(this);
}

toyParser::IdentifierexprContext* toyParser::identifierexpr() {
  IdentifierexprContext *_localctx = _tracker.createInstance<IdentifierexprContext>(_ctx, getState());
  enterRule(_localctx, 4, toyParser::RuleIdentifierexpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(80);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(66);
      match(toyParser::Identifier);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(67);
      match(toyParser::Identifier);
      setState(68);
      match(toyParser::Parenthese_open);
      setState(77);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << toyParser::Parenthese_open)
        | (1ULL << toyParser::Bracket_open)
        | (1ULL << toyParser::Sbracket_open)
        | (1ULL << toyParser::Identifier)
        | (1ULL << toyParser::Number))) != 0)) {
        setState(69);
        expression(0);
        setState(74);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == toyParser::Comma) {
          setState(70);
          match(toyParser::Comma);
          setState(71);
          expression(0);
          setState(76);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
      }
      setState(79);
      match(toyParser::Parenthese_close);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ReturnExpressionContext ------------------------------------------------------------------

toyParser::ReturnExpressionContext::ReturnExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* toyParser::ReturnExpressionContext::Return() {
  return getToken(toyParser::Return, 0);
}

toyParser::ExpressionContext* toyParser::ReturnExpressionContext::expression() {
  return getRuleContext<toyParser::ExpressionContext>(0);
}


size_t toyParser::ReturnExpressionContext::getRuleIndex() const {
  return toyParser::RuleReturnExpression;
}


std::any toyParser::ReturnExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitReturnExpression(this);
  else
    return visitor->visitChildren(this);
}

toyParser::ReturnExpressionContext* toyParser::returnExpression() {
  ReturnExpressionContext *_localctx = _tracker.createInstance<ReturnExpressionContext>(_ctx, getState());
  enterRule(_localctx, 6, toyParser::RuleReturnExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(85);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(82);
      match(toyParser::Return);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(83);
      match(toyParser::Return);
      setState(84);
      expression(0);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TensorLiteralContext ------------------------------------------------------------------

toyParser::TensorLiteralContext::TensorLiteralContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* toyParser::TensorLiteralContext::Sbracket_open() {
  return getToken(toyParser::Sbracket_open, 0);
}

tree::TerminalNode* toyParser::TensorLiteralContext::Sbracket_close() {
  return getToken(toyParser::Sbracket_close, 0);
}

std::vector<toyParser::TensorLiteralContext *> toyParser::TensorLiteralContext::tensorLiteral() {
  return getRuleContexts<toyParser::TensorLiteralContext>();
}

toyParser::TensorLiteralContext* toyParser::TensorLiteralContext::tensorLiteral(size_t i) {
  return getRuleContext<toyParser::TensorLiteralContext>(i);
}

std::vector<tree::TerminalNode *> toyParser::TensorLiteralContext::Comma() {
  return getTokens(toyParser::Comma);
}

tree::TerminalNode* toyParser::TensorLiteralContext::Comma(size_t i) {
  return getToken(toyParser::Comma, i);
}

tree::TerminalNode* toyParser::TensorLiteralContext::Number() {
  return getToken(toyParser::Number, 0);
}


size_t toyParser::TensorLiteralContext::getRuleIndex() const {
  return toyParser::RuleTensorLiteral;
}


std::any toyParser::TensorLiteralContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitTensorLiteral(this);
  else
    return visitor->visitChildren(this);
}

toyParser::TensorLiteralContext* toyParser::tensorLiteral() {
  TensorLiteralContext *_localctx = _tracker.createInstance<TensorLiteralContext>(_ctx, getState());
  enterRule(_localctx, 8, toyParser::RuleTensorLiteral);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(100);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case toyParser::Sbracket_open: {
        enterOuterAlt(_localctx, 1);
        setState(87);
        match(toyParser::Sbracket_open);
        setState(96);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == toyParser::Sbracket_open

        || _la == toyParser::Number) {
          setState(88);
          tensorLiteral();
          setState(93);
          _errHandler->sync(this);
          _la = _input->LA(1);
          while (_la == toyParser::Comma) {
            setState(89);
            match(toyParser::Comma);
            setState(90);
            tensorLiteral();
            setState(95);
            _errHandler->sync(this);
            _la = _input->LA(1);
          }
        }
        setState(98);
        match(toyParser::Sbracket_close);
        break;
      }

      case toyParser::Number: {
        enterOuterAlt(_localctx, 2);
        setState(99);
        match(toyParser::Number);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LiteralListContext ------------------------------------------------------------------

toyParser::LiteralListContext::LiteralListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

toyParser::TensorLiteralContext* toyParser::LiteralListContext::tensorLiteral() {
  return getRuleContext<toyParser::TensorLiteralContext>(0);
}

tree::TerminalNode* toyParser::LiteralListContext::Comma() {
  return getToken(toyParser::Comma, 0);
}

toyParser::LiteralListContext* toyParser::LiteralListContext::literalList() {
  return getRuleContext<toyParser::LiteralListContext>(0);
}


size_t toyParser::LiteralListContext::getRuleIndex() const {
  return toyParser::RuleLiteralList;
}


std::any toyParser::LiteralListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitLiteralList(this);
  else
    return visitor->visitChildren(this);
}

toyParser::LiteralListContext* toyParser::literalList() {
  LiteralListContext *_localctx = _tracker.createInstance<LiteralListContext>(_ctx, getState());
  enterRule(_localctx, 10, toyParser::RuleLiteralList);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(107);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(102);
      tensorLiteral();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(103);
      tensorLiteral();
      setState(104);
      match(toyParser::Comma);
      setState(105);
      literalList();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructLiteralContext ------------------------------------------------------------------

toyParser::StructLiteralContext::StructLiteralContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* toyParser::StructLiteralContext::Bracket_open() {
  return getToken(toyParser::Bracket_open, 0);
}

tree::TerminalNode* toyParser::StructLiteralContext::Bracket_close() {
  return getToken(toyParser::Bracket_close, 0);
}

std::vector<toyParser::LiteralListContext *> toyParser::StructLiteralContext::literalList() {
  return getRuleContexts<toyParser::LiteralListContext>();
}

toyParser::LiteralListContext* toyParser::StructLiteralContext::literalList(size_t i) {
  return getRuleContext<toyParser::LiteralListContext>(i);
}


size_t toyParser::StructLiteralContext::getRuleIndex() const {
  return toyParser::RuleStructLiteral;
}


std::any toyParser::StructLiteralContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitStructLiteral(this);
  else
    return visitor->visitChildren(this);
}

toyParser::StructLiteralContext* toyParser::structLiteral() {
  StructLiteralContext *_localctx = _tracker.createInstance<StructLiteralContext>(_ctx, getState());
  enterRule(_localctx, 12, toyParser::RuleStructLiteral);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(109);
    match(toyParser::Bracket_open);
    setState(111); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(110);
      literalList();
      setState(113); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == toyParser::Sbracket_open

    || _la == toyParser::Number);
    setState(115);
    match(toyParser::Bracket_close);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DeclContext ------------------------------------------------------------------

toyParser::DeclContext::DeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* toyParser::DeclContext::Var() {
  return getToken(toyParser::Var, 0);
}

std::vector<tree::TerminalNode *> toyParser::DeclContext::Identifier() {
  return getTokens(toyParser::Identifier);
}

tree::TerminalNode* toyParser::DeclContext::Identifier(size_t i) {
  return getToken(toyParser::Identifier, i);
}

toyParser::TypeContext* toyParser::DeclContext::type() {
  return getRuleContext<toyParser::TypeContext>(0);
}

tree::TerminalNode* toyParser::DeclContext::Equal() {
  return getToken(toyParser::Equal, 0);
}

toyParser::ExpressionContext* toyParser::DeclContext::expression() {
  return getRuleContext<toyParser::ExpressionContext>(0);
}


size_t toyParser::DeclContext::getRuleIndex() const {
  return toyParser::RuleDecl;
}


std::any toyParser::DeclContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitDecl(this);
  else
    return visitor->visitChildren(this);
}

toyParser::DeclContext* toyParser::decl() {
  DeclContext *_localctx = _tracker.createInstance<DeclContext>(_ctx, getState());
  enterRule(_localctx, 14, toyParser::RuleDecl);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(132);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case toyParser::Var: {
        enterOuterAlt(_localctx, 1);
        setState(117);
        match(toyParser::Var);
        setState(118);
        match(toyParser::Identifier);
        setState(120);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == toyParser::AngleBrackets_open) {
          setState(119);
          type();
        }
        setState(124);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == toyParser::Equal) {
          setState(122);
          match(toyParser::Equal);
          setState(123);
          expression(0);
        }
        break;
      }

      case toyParser::Identifier: {
        enterOuterAlt(_localctx, 2);
        setState(126);
        match(toyParser::Identifier);
        setState(127);
        match(toyParser::Identifier);
        setState(130);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == toyParser::Equal) {
          setState(128);
          match(toyParser::Equal);
          setState(129);
          expression(0);
        }
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypeContext ------------------------------------------------------------------

toyParser::TypeContext::TypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* toyParser::TypeContext::AngleBrackets_open() {
  return getToken(toyParser::AngleBrackets_open, 0);
}

tree::TerminalNode* toyParser::TypeContext::AngleBrackets_close() {
  return getToken(toyParser::AngleBrackets_close, 0);
}

std::vector<tree::TerminalNode *> toyParser::TypeContext::Number() {
  return getTokens(toyParser::Number);
}

tree::TerminalNode* toyParser::TypeContext::Number(size_t i) {
  return getToken(toyParser::Number, i);
}

std::vector<tree::TerminalNode *> toyParser::TypeContext::Comma() {
  return getTokens(toyParser::Comma);
}

tree::TerminalNode* toyParser::TypeContext::Comma(size_t i) {
  return getToken(toyParser::Comma, i);
}


size_t toyParser::TypeContext::getRuleIndex() const {
  return toyParser::RuleType;
}


std::any toyParser::TypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitType(this);
  else
    return visitor->visitChildren(this);
}

toyParser::TypeContext* toyParser::type() {
  TypeContext *_localctx = _tracker.createInstance<TypeContext>(_ctx, getState());
  enterRule(_localctx, 16, toyParser::RuleType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(134);
    match(toyParser::AngleBrackets_open);
    setState(143);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == toyParser::Number) {
      setState(135);
      match(toyParser::Number);
      setState(140);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == toyParser::Comma) {
        setState(136);
        match(toyParser::Comma);
        setState(137);
        match(toyParser::Number);
        setState(142);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
    }
    setState(145);
    match(toyParser::AngleBrackets_close);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FundefinitionContext ------------------------------------------------------------------

toyParser::FundefinitionContext::FundefinitionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

toyParser::PrototypeContext* toyParser::FundefinitionContext::prototype() {
  return getRuleContext<toyParser::PrototypeContext>(0);
}

toyParser::BlockContext* toyParser::FundefinitionContext::block() {
  return getRuleContext<toyParser::BlockContext>(0);
}


size_t toyParser::FundefinitionContext::getRuleIndex() const {
  return toyParser::RuleFundefinition;
}


std::any toyParser::FundefinitionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitFundefinition(this);
  else
    return visitor->visitChildren(this);
}

toyParser::FundefinitionContext* toyParser::fundefinition() {
  FundefinitionContext *_localctx = _tracker.createInstance<FundefinitionContext>(_ctx, getState());
  enterRule(_localctx, 18, toyParser::RuleFundefinition);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(147);
    prototype();
    setState(148);
    block();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PrototypeContext ------------------------------------------------------------------

toyParser::PrototypeContext::PrototypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* toyParser::PrototypeContext::Def() {
  return getToken(toyParser::Def, 0);
}

tree::TerminalNode* toyParser::PrototypeContext::Identifier() {
  return getToken(toyParser::Identifier, 0);
}

tree::TerminalNode* toyParser::PrototypeContext::Parenthese_open() {
  return getToken(toyParser::Parenthese_open, 0);
}

toyParser::Decl_listContext* toyParser::PrototypeContext::decl_list() {
  return getRuleContext<toyParser::Decl_listContext>(0);
}

tree::TerminalNode* toyParser::PrototypeContext::Parenthese_close() {
  return getToken(toyParser::Parenthese_close, 0);
}


size_t toyParser::PrototypeContext::getRuleIndex() const {
  return toyParser::RulePrototype;
}


std::any toyParser::PrototypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitPrototype(this);
  else
    return visitor->visitChildren(this);
}

toyParser::PrototypeContext* toyParser::prototype() {
  PrototypeContext *_localctx = _tracker.createInstance<PrototypeContext>(_ctx, getState());
  enterRule(_localctx, 20, toyParser::RulePrototype);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(150);
    match(toyParser::Def);
    setState(151);
    match(toyParser::Identifier);
    setState(152);
    match(toyParser::Parenthese_open);
    setState(153);
    decl_list();
    setState(154);
    match(toyParser::Parenthese_close);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Decl_listContext ------------------------------------------------------------------

toyParser::Decl_listContext::Decl_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* toyParser::Decl_listContext::Identifier() {
  return getToken(toyParser::Identifier, 0);
}

tree::TerminalNode* toyParser::Decl_listContext::Comma() {
  return getToken(toyParser::Comma, 0);
}

toyParser::Decl_listContext* toyParser::Decl_listContext::decl_list() {
  return getRuleContext<toyParser::Decl_listContext>(0);
}


size_t toyParser::Decl_listContext::getRuleIndex() const {
  return toyParser::RuleDecl_list;
}


std::any toyParser::Decl_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitDecl_list(this);
  else
    return visitor->visitChildren(this);
}

toyParser::Decl_listContext* toyParser::decl_list() {
  Decl_listContext *_localctx = _tracker.createInstance<Decl_listContext>(_ctx, getState());
  enterRule(_localctx, 22, toyParser::RuleDecl_list);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(161);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 20, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(156);
      match(toyParser::Identifier);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(157);
      match(toyParser::Identifier);
      setState(158);
      match(toyParser::Comma);
      setState(159);
      decl_list();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);

      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BlockContext ------------------------------------------------------------------

toyParser::BlockContext::BlockContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* toyParser::BlockContext::Bracket_open() {
  return getToken(toyParser::Bracket_open, 0);
}

tree::TerminalNode* toyParser::BlockContext::Bracket_close() {
  return getToken(toyParser::Bracket_close, 0);
}

std::vector<toyParser::Block_exprContext *> toyParser::BlockContext::block_expr() {
  return getRuleContexts<toyParser::Block_exprContext>();
}

toyParser::Block_exprContext* toyParser::BlockContext::block_expr(size_t i) {
  return getRuleContext<toyParser::Block_exprContext>(i);
}

std::vector<tree::TerminalNode *> toyParser::BlockContext::Semicolon() {
  return getTokens(toyParser::Semicolon);
}

tree::TerminalNode* toyParser::BlockContext::Semicolon(size_t i) {
  return getToken(toyParser::Semicolon, i);
}


size_t toyParser::BlockContext::getRuleIndex() const {
  return toyParser::RuleBlock;
}


std::any toyParser::BlockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitBlock(this);
  else
    return visitor->visitChildren(this);
}

toyParser::BlockContext* toyParser::block() {
  BlockContext *_localctx = _tracker.createInstance<BlockContext>(_ctx, getState());
  enterRule(_localctx, 24, toyParser::RuleBlock);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(163);
    match(toyParser::Bracket_open);
    setState(169);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << toyParser::Parenthese_open)
      | (1ULL << toyParser::Bracket_open)
      | (1ULL << toyParser::Sbracket_open)
      | (1ULL << toyParser::Return)
      | (1ULL << toyParser::Var)
      | (1ULL << toyParser::Identifier)
      | (1ULL << toyParser::Number))) != 0)) {
      setState(164);
      block_expr();
      setState(165);
      match(toyParser::Semicolon);
      setState(171);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(172);
    match(toyParser::Bracket_close);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Block_exprContext ------------------------------------------------------------------

toyParser::Block_exprContext::Block_exprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

toyParser::DeclContext* toyParser::Block_exprContext::decl() {
  return getRuleContext<toyParser::DeclContext>(0);
}

toyParser::ReturnExpressionContext* toyParser::Block_exprContext::returnExpression() {
  return getRuleContext<toyParser::ReturnExpressionContext>(0);
}

toyParser::ExpressionContext* toyParser::Block_exprContext::expression() {
  return getRuleContext<toyParser::ExpressionContext>(0);
}


size_t toyParser::Block_exprContext::getRuleIndex() const {
  return toyParser::RuleBlock_expr;
}


std::any toyParser::Block_exprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitBlock_expr(this);
  else
    return visitor->visitChildren(this);
}

toyParser::Block_exprContext* toyParser::block_expr() {
  Block_exprContext *_localctx = _tracker.createInstance<Block_exprContext>(_ctx, getState());
  enterRule(_localctx, 26, toyParser::RuleBlock_expr);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(177);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 22, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(174);
      decl();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(175);
      returnExpression();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(176);
      expression(0);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StructdefineContext ------------------------------------------------------------------

toyParser::StructdefineContext::StructdefineContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* toyParser::StructdefineContext::Identifier() {
  return getToken(toyParser::Identifier, 0);
}

tree::TerminalNode* toyParser::StructdefineContext::Bracket_open() {
  return getToken(toyParser::Bracket_open, 0);
}

tree::TerminalNode* toyParser::StructdefineContext::Bracket_close() {
  return getToken(toyParser::Bracket_close, 0);
}

std::vector<toyParser::DeclContext *> toyParser::StructdefineContext::decl() {
  return getRuleContexts<toyParser::DeclContext>();
}

toyParser::DeclContext* toyParser::StructdefineContext::decl(size_t i) {
  return getRuleContext<toyParser::DeclContext>(i);
}

std::vector<tree::TerminalNode *> toyParser::StructdefineContext::Semicolon() {
  return getTokens(toyParser::Semicolon);
}

tree::TerminalNode* toyParser::StructdefineContext::Semicolon(size_t i) {
  return getToken(toyParser::Semicolon, i);
}


size_t toyParser::StructdefineContext::getRuleIndex() const {
  return toyParser::RuleStructdefine;
}


std::any toyParser::StructdefineContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<toyVisitor*>(visitor))
    return parserVisitor->visitStructdefine(this);
  else
    return visitor->visitChildren(this);
}

toyParser::StructdefineContext* toyParser::structdefine() {
  StructdefineContext *_localctx = _tracker.createInstance<StructdefineContext>(_ctx, getState());
  enterRule(_localctx, 28, toyParser::RuleStructdefine);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(179);
    match(toyParser::T__0);
    setState(180);
    match(toyParser::Identifier);
    setState(181);
    match(toyParser::Bracket_open);
    setState(187);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == toyParser::Var

    || _la == toyParser::Identifier) {
      setState(182);
      decl();
      setState(183);
      match(toyParser::Semicolon);
      setState(189);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(190);
    match(toyParser::Bracket_close);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool toyParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 1: return expressionSempred(antlrcpp::downCast<ExpressionContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool toyParser::expressionSempred(ExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 8);
    case 1: return precpred(_ctx, 7);
    case 2: return precpred(_ctx, 6);

  default:
    break;
  }
  return true;
}

void toyParser::initialize() {
  std::call_once(toyParserOnceFlag, toyParserInitialize);
}

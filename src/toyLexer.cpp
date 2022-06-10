
// Generated from toy.g4 by ANTLR 4.10.1


#include "toyLexer.h"


using namespace antlr4;



using namespace antlr4;

namespace {

struct ToyLexerStaticData final {
  ToyLexerStaticData(std::vector<std::string> ruleNames,
                          std::vector<std::string> channelNames,
                          std::vector<std::string> modeNames,
                          std::vector<std::string> literalNames,
                          std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), channelNames(std::move(channelNames)),
        modeNames(std::move(modeNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  ToyLexerStaticData(const ToyLexerStaticData&) = delete;
  ToyLexerStaticData(ToyLexerStaticData&&) = delete;
  ToyLexerStaticData& operator=(const ToyLexerStaticData&) = delete;
  ToyLexerStaticData& operator=(ToyLexerStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> channelNames;
  const std::vector<std::string> modeNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

std::once_flag toylexerLexerOnceFlag;
ToyLexerStaticData *toylexerLexerStaticData = nullptr;

void toylexerLexerInitialize() {
  assert(toylexerLexerStaticData == nullptr);
  auto staticData = std::make_unique<ToyLexerStaticData>(
    std::vector<std::string>{
      "T__0", "Parenthese_open", "Parenthese_close", "Bracket_open", "Bracket_close", 
      "Sbracket_open", "Sbracket_close", "Return", "Semicolon", "Var", "Def", 
      "Identifier", "Number", "Equal", "AngleBrackets_open", "AngleBrackets_close", 
      "Comma", "WS", "AddSub", "MulDiv", "Dian", "Comment"
    },
    std::vector<std::string>{
      "DEFAULT_TOKEN_CHANNEL", "HIDDEN"
    },
    std::vector<std::string>{
      "DEFAULT_MODE"
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
  	4,0,22,124,6,-1,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
  	6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,2,14,
  	7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,7,20,2,21,
  	7,21,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,2,1,2,1,3,1,3,1,4,1,4,1,5,
  	1,5,1,6,1,6,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,8,1,8,1,9,1,9,1,9,1,9,1,10,
  	1,10,1,10,1,10,1,11,1,11,5,11,84,8,11,10,11,12,11,87,9,11,1,12,1,12,5,
  	12,91,8,12,10,12,12,12,94,9,12,1,13,1,13,1,14,1,14,1,15,1,15,1,16,1,16,
  	1,17,1,17,1,17,1,17,1,18,1,18,1,19,1,19,1,20,1,20,1,21,1,21,5,21,116,
  	8,21,10,21,12,21,119,9,21,1,21,1,21,1,21,1,21,1,117,0,22,1,1,3,2,5,3,
  	7,4,9,5,11,6,13,7,15,8,17,9,19,10,21,11,23,12,25,13,27,14,29,15,31,16,
  	33,17,35,18,37,19,39,20,41,21,43,22,1,0,6,2,0,65,90,97,122,4,0,48,57,
  	65,90,95,95,97,122,1,0,48,57,3,0,9,10,13,13,32,32,2,0,43,43,45,45,2,0,
  	42,42,47,47,126,0,1,1,0,0,0,0,3,1,0,0,0,0,5,1,0,0,0,0,7,1,0,0,0,0,9,1,
  	0,0,0,0,11,1,0,0,0,0,13,1,0,0,0,0,15,1,0,0,0,0,17,1,0,0,0,0,19,1,0,0,
  	0,0,21,1,0,0,0,0,23,1,0,0,0,0,25,1,0,0,0,0,27,1,0,0,0,0,29,1,0,0,0,0,
  	31,1,0,0,0,0,33,1,0,0,0,0,35,1,0,0,0,0,37,1,0,0,0,0,39,1,0,0,0,0,41,1,
  	0,0,0,0,43,1,0,0,0,1,45,1,0,0,0,3,52,1,0,0,0,5,54,1,0,0,0,7,56,1,0,0,
  	0,9,58,1,0,0,0,11,60,1,0,0,0,13,62,1,0,0,0,15,64,1,0,0,0,17,71,1,0,0,
  	0,19,73,1,0,0,0,21,77,1,0,0,0,23,81,1,0,0,0,25,88,1,0,0,0,27,95,1,0,0,
  	0,29,97,1,0,0,0,31,99,1,0,0,0,33,101,1,0,0,0,35,103,1,0,0,0,37,107,1,
  	0,0,0,39,109,1,0,0,0,41,111,1,0,0,0,43,113,1,0,0,0,45,46,5,115,0,0,46,
  	47,5,116,0,0,47,48,5,114,0,0,48,49,5,117,0,0,49,50,5,99,0,0,50,51,5,116,
  	0,0,51,2,1,0,0,0,52,53,5,40,0,0,53,4,1,0,0,0,54,55,5,41,0,0,55,6,1,0,
  	0,0,56,57,5,123,0,0,57,8,1,0,0,0,58,59,5,125,0,0,59,10,1,0,0,0,60,61,
  	5,91,0,0,61,12,1,0,0,0,62,63,5,93,0,0,63,14,1,0,0,0,64,65,5,114,0,0,65,
  	66,5,101,0,0,66,67,5,116,0,0,67,68,5,117,0,0,68,69,5,114,0,0,69,70,5,
  	110,0,0,70,16,1,0,0,0,71,72,5,59,0,0,72,18,1,0,0,0,73,74,5,118,0,0,74,
  	75,5,97,0,0,75,76,5,114,0,0,76,20,1,0,0,0,77,78,5,100,0,0,78,79,5,101,
  	0,0,79,80,5,102,0,0,80,22,1,0,0,0,81,85,7,0,0,0,82,84,7,1,0,0,83,82,1,
  	0,0,0,84,87,1,0,0,0,85,83,1,0,0,0,85,86,1,0,0,0,86,24,1,0,0,0,87,85,1,
  	0,0,0,88,92,7,2,0,0,89,91,7,2,0,0,90,89,1,0,0,0,91,94,1,0,0,0,92,90,1,
  	0,0,0,92,93,1,0,0,0,93,26,1,0,0,0,94,92,1,0,0,0,95,96,5,61,0,0,96,28,
  	1,0,0,0,97,98,5,60,0,0,98,30,1,0,0,0,99,100,5,62,0,0,100,32,1,0,0,0,101,
  	102,5,44,0,0,102,34,1,0,0,0,103,104,7,3,0,0,104,105,1,0,0,0,105,106,6,
  	17,0,0,106,36,1,0,0,0,107,108,7,4,0,0,108,38,1,0,0,0,109,110,7,5,0,0,
  	110,40,1,0,0,0,111,112,5,46,0,0,112,42,1,0,0,0,113,117,5,35,0,0,114,116,
  	9,0,0,0,115,114,1,0,0,0,116,119,1,0,0,0,117,118,1,0,0,0,117,115,1,0,0,
  	0,118,120,1,0,0,0,119,117,1,0,0,0,120,121,5,10,0,0,121,122,1,0,0,0,122,
  	123,6,21,0,0,123,44,1,0,0,0,4,0,85,92,117,1,6,0,0
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  toylexerLexerStaticData = staticData.release();
}

}

toyLexer::toyLexer(CharStream *input) : Lexer(input) {
  toyLexer::initialize();
  _interpreter = new atn::LexerATNSimulator(this, *toylexerLexerStaticData->atn, toylexerLexerStaticData->decisionToDFA, toylexerLexerStaticData->sharedContextCache);
}

toyLexer::~toyLexer() {
  delete _interpreter;
}

std::string toyLexer::getGrammarFileName() const {
  return "toy.g4";
}

const std::vector<std::string>& toyLexer::getRuleNames() const {
  return toylexerLexerStaticData->ruleNames;
}

const std::vector<std::string>& toyLexer::getChannelNames() const {
  return toylexerLexerStaticData->channelNames;
}

const std::vector<std::string>& toyLexer::getModeNames() const {
  return toylexerLexerStaticData->modeNames;
}

const dfa::Vocabulary& toyLexer::getVocabulary() const {
  return toylexerLexerStaticData->vocabulary;
}

antlr4::atn::SerializedATNView toyLexer::getSerializedATN() const {
  return toylexerLexerStaticData->serializedATN;
}

const atn::ATN& toyLexer::getATN() const {
  return *toylexerLexerStaticData->atn;
}




void toyLexer::initialize() {
  std::call_once(toylexerLexerOnceFlag, toylexerLexerInitialize);
}


// Generated from toy.g4 by ANTLR 4.10.1

#pragma once


#include "antlr4-runtime.h"




class  toyLexer : public antlr4::Lexer {
public:
  enum {
    T__0 = 1, Parenthese_open = 2, Parenthese_close = 3, Bracket_open = 4, 
    Bracket_close = 5, Sbracket_open = 6, Sbracket_close = 7, Return = 8, 
    Semicolon = 9, Var = 10, Def = 11, Identifier = 12, Number = 13, Equal = 14, 
    AngleBrackets_open = 15, AngleBrackets_close = 16, Comma = 17, WS = 18, 
    AddSub = 19, MulDiv = 20, Dian = 21, Comment = 22
  };

  explicit toyLexer(antlr4::CharStream *input);

  ~toyLexer() override;


  std::string getGrammarFileName() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const std::vector<std::string>& getChannelNames() const override;

  const std::vector<std::string>& getModeNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  const antlr4::atn::ATN& getATN() const override;

  // By default the static state used to implement the lexer is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:

  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

};


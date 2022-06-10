#ifndef MLIR_VISITOR_H
#define MLIR_VISITOR_H

#include "toyBaseVisitor.h"
#include "toyLexer.h"
#include "toyParser.h"
#include "Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"
#include <numeric>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <memory>



mlir::MLIRContext context;


class mlirvisitor : public toyBaseVisitor {
  
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  mlir::OpBuilder builder;

  mlir::LogicalResult declear(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var)) 
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success(); 
  }

  mlir::Location loc(int line, int row) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, row);
  }

  mlir::Type getType(llvm::ArrayRef<int64_t> shape ) {
    if (shape.empty()) 
      return mlir::UnrankedTensorType::get(builder.getF64Type());
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  virtual std::any visitFundefinition(toyParser::FundefinitionContext *ctx) override {
    llvm::ScopedHashTableScope <llvm::StringRef, mlir::Value> varScope(symbolTable);
    builder.setInsertionPointToEnd(theModule.getBody());
    visit(ctx->prototype());
    visit(ctx->block());
    return 0;
  }

  virtual std::any visitPrototype(toyParser::PrototypeContext *ctx) override {
    mlir::Location location = loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    auto varNumber = 0;
    if (ctx->decl_list()) {
      auto list = ctx->decl_list();
      while (list->Identifier()) {
        varNumber++;
        if (list->decl_list()) 
          list = list->decl_list();
        else 
          break;
      }
    }

    llvm::SmallVector<mlir::Type, 4> argTypes(varNumber, mlir::UnrankedTensorType::get(builder.getF64Type()));
    auto funType = builder.getFunctionType(argTypes, llvm::None);
    auto func = builder.create<mlir::toy::FuncOp>(location,ctx->Identifier()->toString(), funType);
    mlir::Block &entryblock = func.front();
    builder.setInsertionPointToStart(&entryblock);
    return 0;
  }

  
  virtual std::any visitExpression(toyParser::ExpressionContext *ctx) override {
    // tensor
    mlir::Value value;
    if (ctx->tensorLiteral()) {
      return tensor(ctx->tensorLiteral());
    } else if (ctx->identifierexpr()) {
      return visit(ctx->identifierexpr());
      
    }
    return value;
  }

  std::any tensor(toyParser::TensorLiteralContext *ctx);

  virtual std::any visitDecl(toyParser::DeclContext *ctx) override {
    mlir::Value value = std::any_cast<mlir::Value>(visit(ctx->expression()));
    // reshape
    if (ctx->type()) {
      std::vector<int64_t> v0;
      auto v1 = ctx->type()->Number();
      for (auto i : v1) {
        auto j = atoi(i->toString().c_str());
        v0.push_back(j);
      }
      mlir::Location location = loc(ctx->Identifier(0)->getSymbol()->getLine(), ctx->Identifier(0)->getSymbol()->getCharPositionInLine());
      value = builder.create<mlir::toy::ReshapeOp>(location, getType(v0), value);  
    }
    auto var = std::make_shared<std::string>(ctx->Identifier(0)->toString());

    llvm::StringRef varName(var->c_str(),ctx->Identifier(0)->toString().size());
    mlir::failed(declear(varName, value)); 
    return 0;
  }

  virtual std::any visitIdentifierexpr(toyParser::IdentifierexprContext *ctx) override {
    mlir::Value value;
    // call 
    if (ctx->Parenthese_open()) {
     
      auto location = loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
      llvm::SmallVector<mlir::Value, 4> oprands;
      int num = 0;
      
      for (auto i : ctx->expression()) {
        mlir::Value arg = std::any_cast<mlir::Value>(visit(i));      
        oprands.push_back(arg);
      }
      if (ctx->Identifier()->toString() == "print") {
        auto arg = oprands[0];
        builder.create<mlir::toy::PrintOp>(location, arg);
        return 0;
      }
      
      value =  builder.create<mlir::toy::GenericCallOp>(location, ctx->Identifier()->toString(), oprands);
      return value;
    } else {   // variable
      value = symbolTable.lookup(ctx->Identifier()->toString());
      return value;
    } 
  }
  
  virtual std::any visitReturnExpression(toyParser::ReturnExpressionContext *ctx) override {
    auto location = loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    mlir::Value expr = nullptr;
    if (ctx->expression()) {
      expr = std::any_cast<mlir::Value>(ctx->expression());
    }
    builder.create<mlir::toy::ReturnOp>(location, expr ? llvm::makeArrayRef(expr) : llvm::ArrayRef<mlir::Value>());
    return 0;
  }


  public:
  mlirvisitor() : builder(&context) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
   }
  
  std::string filename;
  mlir::ModuleOp theModule;

};

std::any  mlirvisitor::tensor(toyParser::TensorLiteralContext *ctx) {
    // get dims
    bool flag = false;
    std::vector<int64_t> dims;
    std::vector<double> data;
    dims.push_back(ctx->Comma().size()+1);
    if (ctx->tensorLiteral(0)->tensorLiteral(0)) {
      flag = true;
      dims.push_back(ctx->tensorLiteral(0)->Comma().size()+1);
    }
    // get data 
    auto list = ctx;
    if (flag)
    for (auto i : ctx->tensorLiteral()) {
      for (auto j : i->tensorLiteral()) {
        data.push_back(std::atof(j->Number()->toString().c_str()));
      }
    }
    else if (!flag) {
      for (auto i : ctx->tensorLiteral()) {
        data.push_back(std::atof(i->Number()->toString().c_str()));
      }
    }
    mlir::Type elementType = builder.getF64Type();
    auto type = getType(dims);
    auto dataType = mlir::RankedTensorType::get(dims, elementType);
    auto dataAttribute = mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));
    auto loaction = loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    mlir::Value value = builder.create<mlir::toy::ConstantOp>(loaction, type, dataAttribute);
    return value;
  }



#endif
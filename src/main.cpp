#include <iostream>
#include <fstream>
#include "toyLexer.h"
#include "toyParser.h"
#include "Dialect.h"
#include "mlirvisitor.h"
#include "Passes.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"


int main(int argc, char* argv[]) {
  if (argc ==2) {
    std::fstream in(argv[1]);
    antlr4::ANTLRInputStream input(in);
    toyLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    toyParser parser(&tokens);
    antlr4::tree::ParseTree* tree = parser.module();
    mlirvisitor visitor = mlirvisitor();
    visitor.filename = argv[1];
    context.getOrLoadDialect<mlir::toy::ToyDialect>();
    mlir::OwningOpRef<mlir::ModuleOp> module;
    
    visitor.visit(tree);
    mlir::PassManager pm(&context);
    mlir::applyPassManagerCLOptions(pm);
    pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
    pm.addPass(mlir::toy::createLowerToAffinePass());
    pm.addPass(mlir::toy::createLowerToLLVMPass());
    pm.run((visitor.theModule));
   visitor.theModule.dump();
  }
  
  return 0;
}



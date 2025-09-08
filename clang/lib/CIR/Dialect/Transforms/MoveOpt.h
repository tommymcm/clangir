//===- MoveOpt.h - optimize copies to moves -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_MOVEOPT_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_MOVEOPT_H

#include "LiveObjectAnalysis.h"
#include "PassDetail.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

using namespace cir;
using namespace mlir;

// Move optimization pass class
struct MoveOptPass : public MoveOptBase<MoveOptPass> {
protected:
  clang::ASTContext *astCtx = nullptr;
  mlir::ModuleOp theModule;
  mlir::AliasAnalysis *aliasAnalysis;

public:
  MoveOptPass() = default;

  bool isCopy(cir::CallOp call);
  bool isLastUse(cir::CallOp call, PointsToAnalysis &pta,
                 LiveObjectAnalysis &liveObjects);
  static cir::ASTFunctionDeclInterface
  getMoveDecl(cir::ASTCXXMethodDeclInterface copyDecl);
  cir::FuncOp getMoveFuncFor(cir::CallOp call);
  bool isApplicable(cir::CallOp call, PointsToAnalysis &pta,
                    LiveObjectAnalysis &liveObjects);
  void transform(cir::CallOp call);
  void runOnOperation() override final;
  void setASTContext(clang::ASTContext *astCtx);
};

// Factory function declarations
std::unique_ptr<Pass> mlir::createMoveOptPass();
std::unique_ptr<Pass> mlir::createMoveOptPass(clang::ASTContext *astCtx);

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_MOVEOPT_H

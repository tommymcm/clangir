//===- MoveOpt.cpp - optimize copies to moves -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MoveOpt.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"

// MoveOptPass method implementations
bool MoveOptPass::isCopy(cir::CallOp call) {
  // Get the method decl of the callee.
  auto calleeFunc = call.getDirectCallee(theModule);
  auto methodDecl = mlir::dyn_cast_if_present<cir::ASTCXXMethodDeclInterface>(
      calleeFunc.getAstAttr());
  if (!methodDecl)
    return false;

  // Check if this is the copy constructor.
  if (auto ctorDecl =
          mlir::dyn_cast<cir::ASTCXXConstructorDeclInterface>(methodDecl)) {
    return ctorDecl.isCopyConstructor();
  }

  // Check if this is the copy assignment operator.
  return methodDecl.isCopyAssignmentOperator();
}

bool MoveOptPass::isLastUse(cir::CallOp call, PointsToAnalysis &pta,
                            LiveObjectAnalysis &liveObjects) {
  // Get the pointer being moved.
  auto &ptrUse = call->getOpOperand(1);

  // Unpack the pointer use.
  auto ptr = ptrUse.get();

  // Get the set of objects that could be used.
  const auto &pointsTo = pta.valPointsTo[ptr];

  if (pointsTo.contains(pta.unknown())) {
    call->emitRemark("move opt: copied object may be unknown");
    return false;
  }

  // Have any of the pointees escaped out?
  const auto &escaped = pta.memPointsTo[pta.unknown()];
  for (auto obj : pointsTo) {
    if (escaped.contains(obj)) {
      call->emitRemark("move opt: copied object may have escaped");
      return false;
    }
  }

  // Query the memory analysis.
  if (!liveObjects.isLastUse(ptrUse)) {
    call->emitRemark("move opt: copied object is alive after use");
    return false;
  }

  return true;
}

cir::ASTFunctionDeclInterface
MoveOptPass::getMoveDecl(cir::ASTCXXMethodDeclInterface copyDecl) {
  auto recordDecl = copyDecl.getParent();
  if (!recordDecl)
    return {};

  // Find the matching move constructor.
  if (isa<ASTCXXConstructorDeclInterface>(copyDecl))
    return recordDecl.getMoveConstructor();

  // Find the matching move assignment operator.
  return recordDecl.getMoveAssignmentOperator();
}

cir::FuncOp MoveOptPass::getMoveFuncFor(cir::CallOp call) {
  // Get the method callee.
  auto copyFunc = call.getDirectCallee(theModule);
  if (!copyFunc)
    return {};

  auto copyDecl = mlir::dyn_cast_if_present<cir::ASTCXXMethodDeclInterface>(
      copyFunc.getAstAttr());
  if (!copyDecl)
    return {};

  // Convert the copy into a move within the record..
  auto moveDecl = getMoveDecl(copyDecl);
  if (!moveDecl)
    return {};

  // Get the move function from the module, if it exists.
  auto moveFuncName = moveDecl.getMangledName();
  mlir::Operation *moveFuncGlobal =
      mlir::SymbolTable::lookupSymbolIn(theModule, moveFuncName);

  auto moveFunc = mlir::dyn_cast_if_present<cir::FuncOp>(moveFuncGlobal);
  if (!moveFunc)
    return {};

  return moveFunc;
}

bool MoveOptPass::isApplicable(cir::CallOp call, PointsToAnalysis &pta,
                               LiveObjectAnalysis &liveObjects) {
  return isCopy(call) && getMoveFuncFor(call) &&
         isLastUse(call, pta, liveObjects);
}

void MoveOptPass::transform(cir::CallOp call) {
  // Get the move function for this call.
  auto moveFunc = getMoveFuncFor(call);
  assert(moveFunc && "Missing move function");

  // Replace the copy with a move.
  call.setCallee(moveFunc.getName());

  call.emitRemark("move opt: transformed copy into move");
}

void MoveOptPass::runOnOperation() {
  assert(astCtx && "Missing ASTContext, please construct with the right ctor");

  mlir::Operation *op = getOperation();

  // Record the current module as we go.
  if (mlir::isa<mlir::ModuleOp>(op))
    theModule = mlir::cast<mlir::ModuleOp>(op);

  // Run the live object analysis.
  auto pta = getAnalysis<PointsToAnalysis>();
  auto liveObjects = getAnalysis<LiveObjectAnalysis>();

  // Collect all move constructors.
  llvm::SmallVector<cir::CallOp> copies;
  op->walk([&](cir::CallOp call) {
    if (isApplicable(call, pta, liveObjects))
      copies.push_back(call);
  });

  if (copies.empty())
    return;

  // Transform each of the moves, if applicable.
  for (auto copy : copies)
    transform(copy);
}

void MoveOptPass::setASTContext(clang::ASTContext *astCtx) {
  this->astCtx = astCtx;
}

std::unique_ptr<Pass> mlir::createMoveOptPass() {
  auto pass = std::make_unique<MoveOptPass>();
  return std::move(pass);
}

std::unique_ptr<Pass> mlir::createMoveOptPass(clang::ASTContext *astCtx) {
  auto pass = std::make_unique<MoveOptPass>();
  pass->setASTContext(astCtx);
  return std::move(pass);
}

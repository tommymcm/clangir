//===- PointsToAnalysis.cpp - Points-to analysis for CIR -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <vector>

#include "PassDetail.h"
#include "PointsToAnalysis.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/Passes.h"

PointsToAnalysis::PointsToAnalysis(mlir::Operation *op) : op(op) {
  runOnOperation();
}

void PointsToAnalysis::runOnOperation() {
  // Initialize the unknown mlir::Value/slot.
  valPointsTo[unknown()].insert(unknown());
  valPointsTo[unknown()].insert(unknown());

  // Initialize the points to problem
  op->walk([&](mlir::Operation *op) { initialize(op); });

  // Solve the points to problem
  solve();
}

// Unknown pointer is represented by the NULL value.
mlir::Value PointsToAnalysis::unknown() { return {}; }

void PointsToAnalysis::initialize(mlir::Operation *op) {
  // TODO: Update this to use mlir::MemorySlot and MemorySlotOpInterface.

  if (auto alloca = mlir::dyn_cast<cir::AllocaOp>(op)) {
    auto ptr = alloca.getResult();
    // The result of the alloca points to the object allocated by the operation,
    // we represent these with the same mlir::Value.
    valPointsTo[ptr].insert(ptr);
    workset.insert(ptr);
    return;
  }

  if (auto global = mlir::dyn_cast<cir::GetGlobalOp>(op)) {
    auto ptr = global.getResult();
    // Overly conservative, we don't known anything about global state.
    valPointsTo[ptr].insert(unknown());
    memPointsTo[ptr].insert(unknown());
    workset.insert(ptr);
    return;
  }

  if (auto funcOp = mlir::dyn_cast<cir::FuncOp>(op)) {
    // Create memory locations for each argument.
    // NOTE: We will be conservative here and assume that any argument can
    // smuggle out a pointer.
    auto *body = funcOp.getCallableRegion();
    if (!body)
      return;

    for (auto arg : body->getArguments()) {
      valPointsTo[arg].insert(
          unknown()); // overly conservative! could check for restrict
      memPointsTo[arg].insert(unknown());
      workset.insert(arg);
    }

    return;
  }

  if (auto effect = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
    for (mlir::Value res : effect->getResults()) {
      if (mlir::hasEffect<mlir::MemoryEffects::Allocate>(op, res)) {
        valPointsTo[res].insert(res);
        // TODO: should we initialize the memory pointed to? e.g., for a
        // realloc?
        workset.insert(res);
      }
    }
    return;
  }
}

void PointsToAnalysis::handleStore(mlir::Value ptr, mlir::Value val) {
  for (const auto &obj : valPointsTo[ptr])
    if (memPointsTo[obj].set_union(valPointsTo[val]))
      workset.insert(obj);
}

void PointsToAnalysis::handleLoad(mlir::Value val, mlir::Value ptr) {
  for (const auto &obj : valPointsTo[ptr])
    if (valPointsTo[val].set_union(memPointsTo[obj]))
      workset.insert(val);
}

void PointsToAnalysis::handleMove(mlir::Value dst, mlir::Value ptr) {
  // FIXME(cir): In a flow-insensitive analysis, a move looks like a copy.
  // When this becomes flow-sensitive, the moved object will point to nothing
  // following a move.
  for (const auto &dstObj : valPointsTo[dst])
    for (const auto &obj : valPointsTo[ptr])
      if (memPointsTo[dstObj].set_union(memPointsTo[obj]))
        workset.insert(dstObj);
}

void PointsToAnalysis::handleCopy(mlir::Value dst, mlir::Value ptr) {
  for (const auto &dstObj : valPointsTo[dst])
    for (const auto &obj : valPointsTo[ptr])
      if (memPointsTo[dstObj].set_union(memPointsTo[obj]))
        workset.insert(dstObj);
}

void PointsToAnalysis::handleDataFlow(mlir::Value res, mlir::Value ptr) {
  if (valPointsTo[res].set_union(valPointsTo[ptr]))
    workset.insert(res);
}

bool PointsToAnalysis::handleSpecialCall(cir::CallOp call) {
  auto calleeFunc =
      call.getDirectCallee(call->getParentOfType<mlir::ModuleOp>());
  if (!calleeFunc)
    return false;

  auto ast = calleeFunc.getAstAttr();
  if (!ast)
    return false;

  if (ast && mlir::isa<cir::ASTCXXMethodDeclInterface>(ast)) {
    mlir::Value thisPtr = call.getOperand(0);
    mlir::Value argPtr = call.getOperand(1);

    auto methodDecl = mlir::cast<ASTCXXMethodDeclInterface>(ast);
    if (methodDecl.isMoveAssignmentOperator()) {
      handleMove(thisPtr, argPtr);
      return true;
    }
    if (methodDecl.isCopyAssignmentOperator()) {
      handleCopy(thisPtr, argPtr);
      return true;
    }

    if (auto ctorDecl =
            mlir::dyn_cast<cir::ASTCXXConstructorDeclInterface>(ast)) {
      if (ctorDecl.isMoveConstructor()) {
        handleMove(thisPtr, argPtr);
        return true;
      }
      if (ctorDecl.isCopyConstructor()) {
        handleCopy(thisPtr, argPtr);
        return true;
      }
      if (ctorDecl.isDefaultConstructor()) {
        return true;
      }
    }
  }

  return false;
}

void PointsToAnalysis::handleCall(cir::CallOp call) {

  // Special handling for copy/move.
  if (handleSpecialCall(call))
    return;

  // Check the call side effects.
  bool mayRead, mayWrite;
  switch (call.getSideEffect()) {
  case cir::SideEffect::All:
    // The callee can have any side effects.
    mayRead = true;
    mayWrite = true;
    break;
  case cir::SideEffect::Pure:
    // The callee may read data from memory, but it cannot write data to memory.
    mayRead = true;
    mayWrite = false;
    break;
  case cir::SideEffect::Const:
    // The callee may not read or write data from memory.
    return;
  }

  // Assume that any arg passed in may be stored to unknown, or have
  // unknown stored to it.
  auto argEnd = call.arg_operand_end();
  for (auto argIt = call.arg_operand_begin(); argIt != argEnd; ++argIt) {
    mlir::Value arg = *argIt;

    // The pointer may have an unknown address stored to it.
    if (mayWrite)
      handleStore(arg, unknown()); // *arg = unknown

    // The pointer may be stored to an unknown address.
    // FIXME(cir): If this function is argmemonly, then don't do this!
    bool mayWriteInaccessibleMem = mayWrite;
    if (mayWriteInaccessibleMem)
      handleStore(unknown(), arg); // *unknown = arg

    // The pointer or its pointees may be returned from the call.
    for (auto result : call.getResults()) {
      handleDataFlow(result, arg); // return arg
      if (mayRead)
        handleLoad(result, arg); // return *arg
    }
  }

  // If the function may write, then saturate the transitive closure of the
  // points-to graph rooted at the arguments.
  if (mayWrite) {
    // SetVector maintains insertion order, so we can iterate across it
    // front-to-back, pushing new unique items to the back as we go, while not
    // enqueuing items that have already been seen.
    llvm::SetVector<mlir::Value> closure(call.arg_operand_begin(),
                                         call.arg_operand_end());
    for (size_t index = 0; index < closure.size(); ++index) {
      mlir::Value arg = closure[index];

      for (auto obj : valPointsTo[arg])
        closure.insert(obj);

      if (mayRead)
        for (mlir::Value obj : memPointsTo[arg])
          closure.insert(obj);
    }

    // Any object in the closure may have their memory point to any other.
    auto objEnd = closure.end();
    for (auto objIt = closure.begin(); objIt != objEnd; ++objIt) {
      mlir::Value obj = *objIt;
      for (auto objJt = closure.begin(); objJt != objEnd; ++objJt)
        if (memPointsTo[obj].set_union(memPointsTo[*objJt]))
          workset.insert(obj);
    }
  }
}

void PointsToAnalysis::handleUse(mlir::OpOperand &use) {
  // Unpack the use.
  mlir::Value ptr = use.get();
  mlir::Operation *user = use.getOwner();

  // Check for dataflow and memory effects for general operations.
  bool mayReadPtr = mlir::hasEffect<mlir::MemoryEffects::Read>(user, ptr);
  bool mayWritePtr = mlir::hasEffect<mlir::MemoryEffects::Write>(user, ptr);

  // Operations with results may propagate the pointer to their result(s)
  for (auto result : user->getResults()) {
    handleDataFlow(result, ptr);

    // Read effects may propagate pointees to result(s)
    if (mayReadPtr)
      handleLoad(result, ptr);
  }

  // Write effects may store pointers to other addresses.
  for (mlir::Value arg : user->getOperands()) {
    bool mayReadArg = mlir::hasEffect<mlir::MemoryEffects::Read>(user);
    bool mayWriteArg = mlir::hasEffect<mlir::MemoryEffects::Write>(user);

    // We may write the pointer to the argument.
    if (mayWriteArg) {
      handleStore(arg, ptr);

      // If the pointer is also read, its pointees may be stored.
      if (mayReadPtr)
        handleCopy(arg, ptr);
    }

    // We may write the argument to the pointer.
    if (mayWritePtr) {
      handleStore(ptr, arg);

      // If the arg is also read, its pointees may be stored.
      if (mayReadArg)
        handleCopy(ptr, arg);
    }
  }
}

void PointsToAnalysis::solve() {
  // Iterate until we've exhausted the workset.
  while (!workset.empty()) {
    // Pop an item off the workset.
    mlir::Value ptr = workset.back();
    workset.pop_back();

    if (!ptr)
      continue;

    for (mlir::OpOperand &use : ptr.getUses()) {
      mlir::Operation *user = use.getOwner();

      if (auto store = mlir::dyn_cast<cir::StoreOp>(user)) {
        handleStore(store.getAddr(), store.getValue());
        continue;
      }

      if (auto load = mlir::dyn_cast<cir::LoadOp>(user)) {
        handleLoad(load.getResult(), load.getAddr());
        continue;
      }

      if (auto copy = mlir::dyn_cast<cir::CopyOp>(user)) {
        handleCopy(copy.getDst(), copy.getSrc());
        continue;
      }

      if (mlir::isa<cir::PtrStrideOp, cir::GetElementOp, cir::GetMemberOp>(
              user)) {
        handleDataFlow(user->getResult(0), ptr);
        continue;
      }

      if (auto yield = mlir::dyn_cast<cir::YieldOp>(user)) {
        // Propagate to the parent result.
        auto *yieldTo = yield->getParentRegion()->getParentOp();
        auto out = yieldTo->getResult(use.getOperandNumber());
        handleDataFlow(out, ptr);
        continue;
      }

      if (mlir::isa<cir::TernaryOp>(user)) {
        // Do nothing.
        continue;
      }

      if (auto call = mlir::dyn_cast<cir::CallOp>(user)) {
        handleCall(call);
        continue;
      }

      handleUse(use);
    }
  }
}

static void insertSorted(std::vector<std::string> &vec, std::string &&val) {
  vec.insert(std::lower_bound(vec.begin(), vec.end(), val), val);
}

static void dumpSet(PointsToAnalysis &pta, mlir::InFlightDiagnostic &diag,
                    mlir::Value ptr) {
  // Sort the points-to set so it will be easier to read and test.
  std::vector<std::string> pointsTo;
  for (auto obj : pta.valPointsTo[ptr]) {
    if (obj == pta.unknown()) {
      insertSorted(pointsTo, ":unknown:");
      continue;
    }

    if (auto *op = obj.getDefiningOp()) {
      if (auto alloca = mlir::dyn_cast<cir::AllocaOp>(op))
        insertSorted(pointsTo, alloca.getName().str());
      continue;
    }

    insertSorted(pointsTo, ":external:");
  }

  // Emit the sorted points-to set to the diagnostic.
  diag << "{ ";
  bool first = true;
  for (const auto &name : pointsTo) {
    if (first)
      first = false;
    else
      diag << ", ";
    diag << name;
  }
  diag << " }";
}

static bool isCopy(mlir::Operation *op, mlir::Value &dst, mlir::Value &src) {
  if (auto copyOp = mlir::dyn_cast<cir::CopyOp>(op)) {
    dst = copyOp.getDst();
    src = copyOp.getSrc();
    return true;
  }
  if (auto memCpyOp = mlir::dyn_cast<cir::MemCpyOp>(op)) {
    dst = memCpyOp.getDst();
    src = memCpyOp.getSrc();
    return true;
  }
  if (auto memCpyInlineOp = mlir::dyn_cast<cir::MemCpyInlineOp>(op)) {
    dst = memCpyInlineOp.getDst();
    src = memCpyInlineOp.getSrc();
    return true;
  }

  auto callOp = mlir::dyn_cast<cir::CallOp>(op);
  if (!callOp)
    return false;

  auto calleeFunc =
      callOp.getDirectCallee(callOp->getParentOfType<mlir::ModuleOp>());
  if (!calleeFunc)
    return false;

  auto cxxSpecialMember = calleeFunc.getCxxSpecialMember();
  if (!cxxSpecialMember)
    return false;

  if (auto cxxCtor = mlir::dyn_cast<cir::CXXCtorAttr>(*cxxSpecialMember)) {
    if (cxxCtor.getCtorKind() != cir::CtorKind::Copy)
      return false;
    dst = callOp.getOperand(0);
    src = callOp.getOperand(1);
    return true;
  }

  if (auto cxxAssign = mlir::dyn_cast<cir::CXXAssignAttr>(*cxxSpecialMember)) {
    if (cxxAssign.getAssignKind() != cir::AssignKind::Copy)
      return false;
    dst = op->getOperand(0);
    src = op->getOperand(1);
    return true;
  }

  return false;
}

static bool isMove(mlir::Operation *op, mlir::Value &dst, mlir::Value &src) {
  if (auto memMoveOp = mlir::dyn_cast<cir::MemMoveOp>(op)) {
    dst = memMoveOp.getDst();
    src = memMoveOp.getSrc();
    return true;
  }

  auto callOp = mlir::dyn_cast<cir::CallOp>(op);
  if (!callOp)
    return false;

  auto calleeFunc =
      callOp.getDirectCallee(callOp->getParentOfType<mlir::ModuleOp>());
  if (!calleeFunc)
    return false;

  auto cxxSpecialMember = calleeFunc.getCxxSpecialMember();
  if (!cxxSpecialMember)
    return false;

  if (auto cxxCtor = mlir::dyn_cast<cir::CXXCtorAttr>(*cxxSpecialMember)) {
    if (cxxCtor.getCtorKind() != cir::CtorKind::Move)
      return false;
    dst = callOp.getOperand(0);
    src = callOp.getOperand(1);
    return true;
  }

  if (auto cxxAssign = mlir::dyn_cast<cir::CXXAssignAttr>(*cxxSpecialMember)) {
    if (cxxAssign.getAssignKind() != cir::AssignKind::Move)
      return false;
    dst = callOp.getOperand(0);
    src = callOp.getOperand(1);
    return true;
  }

  return false;
}

static void dump(PointsToAnalysis &pta, cir::FuncOp func) {
  func->walk([&](mlir::Operation *op) {
    // Handle memory load/store operations.
    if (auto load = mlir::dyn_cast<cir::LoadOp>(op)) {
      mlir::InFlightDiagnostic diag = op->emitRemark("load ");
      dumpSet(pta, diag, load.getAddr());
      return;
    }
    if (auto store = mlir::dyn_cast<cir::StoreOp>(op)) {
      mlir::InFlightDiagnostic diag = op->emitRemark("store ");
      dumpSet(pta, diag, store.getAddr());
      return;
    }

    // Handle copy/move operations.
    mlir::Value dst, src;
    if (isCopy(op, dst, src)) {
      mlir::InFlightDiagnostic diag = op->emitRemark("copy from ");
      dumpSet(pta, diag, src);
      diag << " to ";
      dumpSet(pta, diag, dst);
      return;
    }
    if (isMove(op, dst, src)) {
      mlir::InFlightDiagnostic diag = op->emitRemark("move from ");
      dumpSet(pta, diag, src);
      diag << " to ";
      dumpSet(pta, diag, dst);
      return;
    }

    // Handle memory effects.
    if (auto effect = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
      // Iterate over each operand, reporting memory effects.
      for (mlir::Value val : op->getOperands()) {
        if (mlir::hasEffect<mlir::MemoryEffects::Read>(op, val)) {
          mlir::InFlightDiagnostic diag = op->emitRemark("read from ");
          dumpSet(pta, diag, val);
        }
        if (mlir::hasEffect<mlir::MemoryEffects::Write>(op, val)) {
          mlir::InFlightDiagnostic diag = op->emitRemark("write to ");
          dumpSet(pta, diag, val);
        }
      }
    }
  });
}

void PointsToDiagnosticPass::runOnOperation() {
  PointsToAnalysis &pta = getAnalysis<PointsToAnalysis>();
  getOperation()->walk([&](cir::FuncOp func) { dump(pta, func); });
}

std::unique_ptr<mlir::Pass> mlir::createPointsToDiagnosticPass() {
  auto pass = std::make_unique<PointsToDiagnosticPass>();
  return std::move(pass);
}

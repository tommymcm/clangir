//===- LiveObjectAnalysis.cpp - Live object analysis ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LiveObjectAnalysis.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/raw_ostream.h"

using namespace cir;
using namespace mlir;

LiveObjectAnalysis::LiveObjectAnalysis(mlir::Operation *op,
                                       mlir::AnalysisManager &am)
    : pta(am.getCachedAnalysis<PointsToAnalysis>().value_or(
          am.getAnalysis<PointsToAnalysis>())) {
  // Analyze the operation.
  analyze(op);
}

bool LiveObjectAnalysis::isLastUse(mlir::OpOperand &use) {
  mlir::Value ptr = use.get();
  mlir::Operation *user = use.getOwner();

  // Analyze the points where this object is live.
  const auto &pointsTo = pta.valPointsTo[ptr];

  // Are all of the pointees dead after this use?
  auto liveOut = out[user];
  for (auto obj : pointsTo)
    if (liveOut.contains(obj))
      return false;

  return true;
}

const llvm::SmallSetVector<mlir::Value, 8> &
LiveObjectAnalysis::getIn(mlir::Operation *op) {
  return in[op];
}

const llvm::SmallSetVector<mlir::Value, 8> &
LiveObjectAnalysis::getOut(mlir::Operation *op) {
  return out[op];
}

static bool usesPtr(mlir::OpOperand &use) {
  mlir::Value ptr = use.get();
  mlir::Operation *user = use.getOwner();
  if (mlir::hasEffect<mlir::MemoryEffects::Read>(user, ptr) ||
      mlir::hasEffect<mlir::MemoryEffects::Write>(user, ptr))
    return true;
  else if (auto call = mlir::dyn_cast<cir::CallOp>(user))
    switch (call.getSideEffect()) {
    case cir::SideEffect::All:
    case cir::SideEffect::Pure:
      return true;
    case cir::SideEffect::Const:
      return false;
    }
  else if (mlir::isa<cir::ReturnOp>(user))
    return true;

  return false;
}

void LiveObjectAnalysis::initialize(mlir::Value obj) {
  // The definition of the object kills it.
  auto *def = obj.getDefiningOp();
  kill[def].insert(obj);
  workset.insert(def);

  // Find all pointers that point to this object.
  for (const auto &[ptr, set] : pta.valPointsTo) {
    if (!set.contains(obj))
      continue;

    // Find all uses of this pointer.
    for (auto &use : ptr.getUses()) {
      auto *user = use.getOwner();
      if (usesPtr(use)) {
        // Add the read to the GEN set and enqueue it.
        gen[user].insert(obj);
        workset.insert(user);
      }
    }
  }

  // If the object escapes the function, mark it as live at all exit points.
  if (pta.escaped(obj)) {
    auto funcOp = def->getParentOfType<cir::FuncOp>();
    funcOp->walk([&](cir::ReturnOp retOp) {
      out[retOp].insert(obj);
      workset.insert(retOp);
    });
  }
}

static mlir::Operation *getEntryOp(mlir::Block *block) {
  if (!block || block->empty())
    return nullptr;
  mlir::Operation &entryOp = block->front();
  return &entryOp;
}

static mlir::Operation *getEntryOp(mlir::Region *region) {
  if (!region || region->empty())
    return nullptr;
  mlir::Block &entryBlock = region->front();
  return getEntryOp(&entryBlock);
}

void LiveObjectAnalysis::unionEntries(
    ObjectSet &opOut, mlir::RegionBranchOpInterface regionBranch) {
  // Fetch the successors of this program point.
  llvm::SmallVector<mlir::RegionSuccessor> successors;
  regionBranch.getSuccessorRegions(mlir::RegionBranchPoint::parent(),
                                   successors);

  for (const mlir::RegionSuccessor &succ : successors) {
    // Union with the first op's IN set.
    mlir::Operation *entryOp = getEntryOp(succ.getSuccessor());
    if (!entryOp)
      continue;
    opOut.set_union(in[entryOp]);
  }
}

void LiveObjectAnalysis::unionSuccessors(ObjectSet &opOut,
                                         mlir::Region *region) {

  // Get the parent operation of this region..
  mlir::Operation *parentOp = region->getParentOp();

  // If the parent is not a region branch operation, we have no successors to
  // union with.
  auto regionBranch = mlir::dyn_cast<mlir::RegionBranchOpInterface>(parentOp);
  if (!regionBranch)
    return;

  // Fetch the successors of this region.
  llvm::SmallVector<mlir::RegionSuccessor> successors;
  regionBranch.getSuccessorRegions(mlir::RegionBranchPoint(region), successors);

  // Handle each region successor.
  for (const mlir::RegionSuccessor &succ : successors) {
    // Union with the OUT set of the parent op.
    if (succ.isParent())
      opOut.set_union(out[parentOp]);

    // Otherwise, union with the first op's IN set, if it exists.
    if (mlir::Operation *succOp = getEntryOp(succ.getSuccessor()))
      opOut.set_union(in[succOp]);
  }

  return;
}

void LiveObjectAnalysis::unionSuccessors(ObjectSet &opOut, mlir::Block *block) {
  // If the block has no successors, it terminates the region.
  if (block->hasNoSuccessors()) {
    unionSuccessors(opOut, block->getParent());
  }

  // Otherwise, union with the immediate successors of this block.
  for (auto *succBlock : block->getSuccessors()) {
    if (mlir::Operation *succOp = getEntryOp(succBlock))
      opOut.set_union(in[succOp]);
  }
}

void LiveObjectAnalysis::unionSuccessors(ObjectSet &opOut,
                                         mlir::Operation *op) {
  // Skip non-operations.
  if (!op)
    return;

  // If this is a goto, find and union with the label.
  if (auto gotoOp = mlir::dyn_cast<cir::GotoOp>(op)) {
    mlir::Operation *funcOp = gotoOp->getParentOfType<cir::FuncOp>();
    cir::LabelOp foundLabelOp;
    funcOp->walk([&](cir::LabelOp labelOp) {
      if (labelOp.getLabelAttr() == gotoOp.getLabelAttr())
        foundLabelOp = labelOp;
    });
    if (!foundLabelOp)
      llvm_unreachable("NYI, failed to find label for goto");
    opOut.set_union(in[foundLabelOp]);
    return;
  }

  // If the operation is a region branch, or a loop, union with all of its entry
  // operations.
  if (auto regionBranch = mlir::dyn_cast<mlir::RegionBranchOpInterface>(op))
    unionEntries(opOut, regionBranch);

  // If the operation has a successor operation, union with it and return.
  if (mlir::Operation *succOp = op->getNextNode()) {
    opOut.set_union(in[succOp]);
    return;
  }

  // Otherwise, we are the block terminator.
  auto *block = op->getBlock();
  unionSuccessors(opOut, block);
}

void LiveObjectAnalysis::enqueueExits(mlir::Region *region) {
  auto *parentOp = region->getParentOp();
  region->walk([&](mlir::Operation *op) {
    if (mlir::isa<cir::ContinueOp, cir::ConditionOp>(op)) {
      if (op->getParentOfType<cir::LoopOpInterface>() == parentOp)
        workset.insert(op);
      return;
    }

    if (mlir::isa<cir::YieldOp>(op)) {
      if (op->getParentOp() == parentOp)
        workset.insert(op);
      return;
    }
  });
}

void LiveObjectAnalysis::enqueuePredecessors(mlir::Region *region) {

  auto *parent = region->getParentOp();

  // Find all predecessors for the given region.
  // FIXME(cir): This could be refactored into a `getPredecessorRegions` method
  // in `cir::LoopOpInterface`.
  if (auto whileLoop = mlir::dyn_cast<cir::WhileOp>(parent)) {
    // :op
    // while(:cond) // preds = { :op, :body }
    // { :body }    // preds = { :cond }
    mlir::Region *cond = &whileLoop.getCond();
    mlir::Region *body = &whileLoop.getBody();
    if (region == cond) {
      enqueueExits(body);
      workset.insert(parent);
    } else if (region == body) {
      enqueueExits(cond);
    }
    return;
  }

  if (auto doWhileLoop = mlir::dyn_cast<cir::DoWhileOp>(parent)) {
    // :op
    // do { :body } // preds = { :op, :cond }
    // while(:cond) // preds = { :body }
    mlir::Region *body = &doWhileLoop.getBody();
    mlir::Region *cond = &doWhileLoop.getCond();
    if (region == body) {
      enqueueExits(cond);
      workset.insert(parent);
    } else if (region == cond) {
      enqueueExits(body);
    }
    return;
  }

  if (auto forLoop = mlir::dyn_cast<cir::ForOp>(parent)) { // :op
    // for (:op ;
    //      :cond ; // preds = { :op, :step }
    //      :step ) // preds = { :body }
    // { :body }    // preds = { :cond }
    mlir::Region *cond = &forLoop.getCond();
    mlir::Region *step = &forLoop.getStep();
    mlir::Region *body = &forLoop.getBody();
    if (region == cond) {
      enqueueExits(step);
      workset.insert(parent);
    } else if (region == step) {
      enqueueExits(body);
    } else if (region == body) {
      enqueueExits(cond);
    }
    return;
  }

  // Otherwise, the parent is the only predecessor.
  workset.insert(parent);
}

void LiveObjectAnalysis::enqueuePredecessors(mlir::Block *block) {
  // If the block has no predecessors, it is the entry block for the region.
  // Look at the parent region to determine its predecessors.
  if (block->hasNoPredecessors()) {
    enqueuePredecessors(block->getParent());
    return;
  }

  // Otherwise, enqueue the direct block predecessors.
  for (auto *pred : block->getPredecessors()) {
    mlir::Operation *terminator = pred->getTerminator();
    if (!terminator)
      continue;
    workset.insert(terminator);
  }
}

void LiveObjectAnalysis::enqueueExits(mlir::Operation *op) {

  // If the previous operation is a loop, find all cir.break and cir.condition
  // operations for the loop.
  if (auto loopOp = mlir::dyn_cast<cir::LoopOpInterface>(op)) {
    loopOp->walk([&](mlir::Operation *op) {
      // Enqueue break operations whose target is this loop.
      if (auto breakOp = mlir::dyn_cast<cir::BreakOp>(op)) {
        auto breakTarget = breakOp.getBreakTarget();
        if (breakTarget == loopOp)
          workset.insert(op);
        return;
      }

      // Enqueue condition operations whose innermost parent loop is this
      // loop.
      if (mlir::isa<cir::ConditionOp>(op)) {
        auto innerLoopOp = op->getParentOfType<cir::LoopOpInterface>();
        if (innerLoopOp == loopOp)
          workset.insert(op);
        return;
      }
    });
    return;
  }

  // If the previous operation is a switch, find its cir.break operations.
  if (auto switchOp = mlir::dyn_cast<cir::SwitchOp>(op)) {
    switchOp->walk([&](cir::BreakOp breakOp) {
      // Enqueue break operations whose target is this switch.
      mlir::Operation *breakTarget = breakOp.getBreakTarget();
      if (breakTarget == switchOp)
        workset.insert(breakOp);
    });
    return;
  }

  // Enqueue all immediate yield ops.
  op->walk([&](cir::YieldOp yieldOp) {
    if (yieldOp->getParentOp() == op)
      workset.insert(yieldOp);
  });
}

void LiveObjectAnalysis::enqueuePredecessors(mlir::Operation *op) {

  // If the operation is a label, find all gotos that target it.
  if (auto labelOp = mlir::dyn_cast<cir::LabelOp>(op)) {
    // Find all gotos that target this label.
    mlir::StringAttr label = labelOp.getLabelAttr();
    cir::FuncOp func = labelOp->getParentOfType<cir::FuncOp>();
    func->walk([&](cir::GotoOp gotoOp) {
      if (gotoOp.getLabelAttr() == label)
        workset.insert(gotoOp);
    });

    // We may fall-through to the label, so don't return early.
  }

  // Get the previous operation in the block, if it exists.
  auto *prevOp = op->getPrevNode();

  // If the previous operation does not exist, get the predecessor of its
  // block.
  if (!prevOp) {
    // Otherwise, enqueue the block's predecessors.
    enqueuePredecessors(op->getBlock());
    return;
  }

  // If the previous operation is a region branch, enqueue all of its exit
  // operations.
  if (mlir::isa<mlir::RegionBranchOpInterface>(prevOp))
    enqueueExits(prevOp);

  // Enqueue the previous operation.
  workset.insert(prevOp);
}

void LiveObjectAnalysis::compute() {
  // Iterate until we've exhausted the workset.
  while (!workset.empty()) {
    mlir::Operation *op = workset.back();
    workset.pop_back();

    // Populate the OUT set with the IN set of each successor.
    ObjectSet &opOut = out[op];
    unionSuccessors(opOut, op);

    // Save the old IN set for comparison with the result of the transfer
    // function.
    ObjectSet &opIn = in[op];
    ObjectSet oldIn = opIn;

    // OUT = (IN + GEN) \ KILL
    opIn.set_union(opOut);
    if (gen.contains(op))
      opIn.set_union(gen[op]);
    if (kill.contains(op))
      opIn.set_subtract(kill[op]);

    // If the IN set didn't change, we're done with this operation.
    if (opIn == oldIn)
      continue;

    // Otherwise, enqueue the predecessors.
    enqueuePredecessors(op);
  }
}

void LiveObjectAnalysis::analyze(mlir::Operation *op) {
  // Populate the GEN and KILL sets with all allocations.
  op->walk([&](mlir::MemoryEffectOpInterface effect) {
    for (auto res : effect->getResults())
      if (mlir::hasEffect<mlir::MemoryEffects::Allocate>(effect, res))
        initialize(res);
  });

  // Compute the live-in and live-out sets.
  compute();
}

static void dump(LiveObjectAnalysis &liveObjects, mlir::Operation *op) {

  // Compute the difference between the in and out sets.
  const auto &in = liveObjects.getIn(op);
  const auto &out = liveObjects.getOut(op);

  auto diff = in;
  diff.set_subtract(out);

  // Emit a remark for each object where this is their last use.
  for (mlir::Value obj : diff) {
    // Fetch the allocation name, if it exists.
    auto alloca = dyn_cast<cir::AllocaOp>(obj.getDefiningOp());
    if (!alloca || alloca.getName().empty())
      continue;
    op->emitRemark("last use of ") << alloca.getName();
  }
}

void LiveObjectDiagnosticPass::runOnOperation() {
  cir::FuncOp func = cast<cir::FuncOp>(getOperation());
  LiveObjectAnalysis &liveObjects = getAnalysis<LiveObjectAnalysis>();
  func->walk([&](mlir::Operation *op) { dump(liveObjects, op); });
}

std::unique_ptr<mlir::Pass> mlir::createLiveObjectDiagnosticPass() {
  auto pass = std::make_unique<LiveObjectDiagnosticPass>();
  return std::move(pass);
}

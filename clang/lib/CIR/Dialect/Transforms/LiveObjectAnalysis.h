//===- LiveObjectAnalysis.h - Live object analysis -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_LIVEOBJECTANALYSIS_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_LIVEOBJECTANALYSIS_H

#include "PassDetail.h"
#include "PointsToAnalysis.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"

using ObjectSet = llvm::SmallSetVector<mlir::Value, 8>;

struct LiveObjectAnalysis {
protected:
  // Compute a points-to analysis.
  PointsToAnalysis &pta;

  // Track set of live objects.
  llvm::DenseMap<mlir::Operation *, ObjectSet> gen, kill, in, out;

  // Working set for solver.
  llvm::SetVector<mlir::Operation *> workset;

public:
  LiveObjectAnalysis(mlir::Operation *op, mlir::AnalysisManager &am);

  bool isLastUse(mlir::OpOperand &use);

  const ObjectSet &getIn(mlir::Operation *op);
  const ObjectSet &getOut(mlir::Operation *op);

protected:
  void unionEntries(ObjectSet &opOut,
                    mlir::RegionBranchOpInterface regionBranch);

  void unionSuccessors(ObjectSet &opOut, mlir::Region *region);
  void unionSuccessors(ObjectSet &opOut, mlir::Block *block);
  void unionSuccessors(ObjectSet &opOut, mlir::Operation *op);

  void enqueueExits(mlir::Region *region);
  void enqueueExits(mlir::Operation *op);

  void enqueuePredecessors(mlir::Region *region);
  void enqueuePredecessors(mlir::Block *block);
  void enqueuePredecessors(mlir::Operation *op);

  void initialize(mlir::Value obj);
  void compute();
  void analyze(mlir::Operation *op);
};

struct LiveObjectDiagnosticPass
    : public mlir::LiveObjectDiagnosticBase<LiveObjectDiagnosticPass> {

  bool canScheduleOn(mlir::RegisteredOperationName opInfo) const override {
    return opInfo.hasInterface<mlir::FunctionOpInterface>();
  }

  void runOnOperation() override final;
};

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_LIVEOBJECTANALYSIS_H

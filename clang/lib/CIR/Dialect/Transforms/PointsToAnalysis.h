//===- PointsToAnalysis.h - Points-to analysis for CIR -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_POINTSTOANALYSIS_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_POINTSTOANALYSIS_H

#include "PassDetail.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

using namespace cir;

// Points-to analysis class
struct PointsToAnalysis {
public:
  using PointsToSet = typename llvm::SmallSetVector<mlir::Value, 8>;

protected:
  // Working set for solver.
  llvm::SetVector<mlir::Value> workset;

  // Base operation.
  mlir::Operation *op;

public:
  // MLIR Analysis interface.
  PointsToAnalysis(mlir::Operation *op);
  void runOnOperation();

  // Track the slot(s) pointed to by a given value.
  llvm::DenseMap<mlir::Value, PointsToSet> valPointsTo;

  // Track the slot(s) pointed to by a given slot.
  llvm::DenseMap<mlir::Value, PointsToSet> memPointsTo;

  // Unknown pointer is represented by the NULL value.
  mlir::Value unknown();

protected:
  // Data flow initialization.
  void initialize(mlir::Operation *op);

  // Data flow solver.
  void handleStore(mlir::Value ptr, mlir::Value val);
  void handleLoad(mlir::Value val, mlir::Value ptr);
  void handleMove(mlir::Value dst, mlir::Value ptr);
  void handleCopy(mlir::Value dst, mlir::Value ptr);
  void handleDataFlow(mlir::Value res, mlir::Value ptr);
  bool handleSpecialCall(cir::CallOp call);
  void handleCall(cir::CallOp call);
  void handleUse(mlir::OpOperand &use);
  void solve();
};

struct PointsToDiagnosticPass
    : public mlir::PointsToDiagnosticBase<PointsToDiagnosticPass> {
  void runOnOperation() override final;
};

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_POINTSTOANALYSIS_H

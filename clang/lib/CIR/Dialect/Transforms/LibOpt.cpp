//===- LibOpt.cpp - Optimize CIR raised C/C++ library idioms --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Module.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"

#include "StdHelpers.h"

using cir::CIRBaseBuilderTy;
using namespace mlir;
using namespace cir;

namespace {

struct LibOptPass : public LibOptBase<LibOptPass> {
  LibOptPass() = default;
  void runOnOperation() override;
  void xformStdFindIntoMemchr(StdFindOp findOp);
  void xformStrLenIntoMemchr(StrLenOp strLen);

  // Handle pass options
  struct Options {
    enum : unsigned {
      None = 0,
      RemarkTransforms = 1,
      RemarkAll = 1 << 1,
    };
    unsigned val = None;
    bool isOptionsParsed = false;

    void parseOptions(ArrayRef<llvm::StringRef> remarks) {
      if (isOptionsParsed)
        return;

      for (auto &remark : remarks) {
        val |= StringSwitch<unsigned>(remark)
                   .Case("transforms", RemarkTransforms)
                   .Case("all", RemarkAll)
                   .Default(None);
      }
      isOptionsParsed = true;
    }

    void parseOptions(LibOptPass &pass) {
      llvm::SmallVector<llvm::StringRef, 4> remarks;

      for (auto &r : pass.remarksList)
        remarks.push_back(r);

      parseOptions(remarks);
    }

    bool emitRemarkAll() { return val & RemarkAll; }
    bool emitRemarkTransforms() {
      return emitRemarkAll() || val & RemarkTransforms;
    }
  } opts;

  ///
  /// AST related
  /// -----------
  clang::ASTContext *astCtx;
  void setASTContext(clang::ASTContext *c) { astCtx = c; }

  /// Tracks current module.
  ModuleOp theModule;
};
} // namespace

static bool isSequentialContainer(mlir::Type t) {
  // TODO: other sequential ones, vector, dequeue, list, forward_list.
  return isStdArrayType(t);
}

static bool getIntegralNTTPAt(RecordType t, size_t pos, unsigned &size) {
  auto *d =
      dyn_cast<clang::ClassTemplateSpecializationDecl>(t.getAst().getRawDecl());
  if (!d)
    return false;

  auto &templArgs = d->getTemplateArgs();
  if (pos >= templArgs.size())
    return false;

  auto arraySizeTemplateArg = templArgs[pos];
  if (arraySizeTemplateArg.getKind() != clang::TemplateArgument::Integral)
    return false;

  size = arraySizeTemplateArg.getAsIntegral().getSExtValue();
  return true;
}

static bool containerHasStaticSize(RecordType t, unsigned &size) {
  // TODO: add others.
  if (!isStdArrayType(t))
    return false;

  // Get "size" from std::array<T, size>
  unsigned sizeNTTPPos = 1;
  return getIntegralNTTPAt(t, sizeNTTPPos, size);
}

void LibOptPass::xformStdFindIntoMemchr(StdFindOp findOp) {
  // template <class T>
  //  requires (sizeof(T) == 1 && is_integral_v<T>)
  // T* find(T* first, T* last, T value) {
  //   if (auto result = __builtin_memchr(first, value, last - first))
  //     return result;
  //   return last;
  // }

  auto first = findOp.getOperand(0);
  auto last = findOp.getOperand(1);
  auto value = findOp->getOperand(2);
  if (!mlir::isa<PointerType>(first.getType()) || !mlir::isa<PointerType>(last.getType()))
    return;

  // Transformation:
  // - 1st arg: the data pointer
  //   - Assert the Iterator is a pointer to primitive type.
  //   - Check IterBeginOp is char sized. TODO: add other types that map to
  //   char size.
  auto iterResTy = mlir::dyn_cast<PointerType>(findOp.getType());
  assert(iterResTy && "expected pointer type for iterator");
  auto underlyingDataTy = mlir::dyn_cast<IntType>(iterResTy.getPointee());
  if (!underlyingDataTy || underlyingDataTy.getWidth() != 8)
    return;

  // - 2nd arg: the pattern
  //   - Check it's a pointer type.
  //   - Load the pattern from memory
  //   - cast it to `int`.
  auto patternAddrTy = mlir::dyn_cast<PointerType>(value.getType());
  if (!patternAddrTy || patternAddrTy.getPointee() != underlyingDataTy)
    return;

  // - 3rd arg: the size
  //   - Create and pass a cir.const with NTTP value

  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(findOp.getOperation());
  auto memchrOp0 =
      builder.createBitcast(first.getLoc(), first, builder.getVoidPtrTy());

  // FIXME: get datalayout based "int" instead of fixed size 4.
  auto loadPattern =
      builder.create<LoadOp>(value.getLoc(), underlyingDataTy, value);
  auto memchrOp1 = builder.createIntCast(
      loadPattern, IntType::get(builder.getContext(), 32, true));

  const auto uInt64Ty = IntType::get(builder.getContext(), 64, false);

  // Build memchr op:
  //  void *memchr(const void *s, int c, size_t n);
  auto memChr = [&] {
    if (auto iterBegin = first.getDefiningOp<IterBeginOp>();
        iterBegin && last.getDefiningOp<IterEndOp>()) {
      // Both operands have the same type, use iterBegin.

      // Look at this pointer to retrieve container information.
      auto thisPtr =
          mlir::cast<PointerType>(iterBegin.getOperand().getType()).getPointee();
      auto containerTy = mlir::dyn_cast<RecordType>(thisPtr);

      unsigned staticSize = 0;
      if (containerTy && isSequentialContainer(containerTy) &&
          containerHasStaticSize(containerTy, staticSize)) {
        return builder.create<MemChrOp>(
            findOp.getLoc(), memchrOp0, memchrOp1,
            builder.create<ConstantOp>(
                findOp.getLoc(), cir::IntAttr::get(uInt64Ty, staticSize)));
      }
    }
    return builder.create<MemChrOp>(
        findOp.getLoc(), memchrOp0, memchrOp1,
        builder.create<PtrDiffOp>(findOp.getLoc(), uInt64Ty, last, first));
  }();

  auto MemChrResult =
      builder.createBitcast(findOp.getLoc(), memChr.getResult(), iterResTy);

  // if (result)
  //   return result;
  // else
  // return last;
  auto NullPtr = builder.getNullPtr(first.getType(), findOp.getLoc());
  auto CmpResult = builder.create<CmpOp>(
      findOp.getLoc(), BoolType::get(builder.getContext()), CmpOpKind::eq,
      NullPtr.getRes(), MemChrResult);

  auto result = builder.create<TernaryOp>(
      findOp.getLoc(), CmpResult.getResult(),
      [&](mlir::OpBuilder &ob, mlir::Location Loc) {
        ob.create<YieldOp>(Loc, last);
      },
      [&](mlir::OpBuilder &ob, mlir::Location Loc) {
        ob.create<YieldOp>(Loc, MemChrResult);
      });

  findOp.replaceAllUsesWith(result);
  findOp.erase();
}

cir::CmpOpKind comparison(cir::CmpOp op, bool flip) {
  auto kind = op.getKind();
  if (!flip)
    return kind;

  // Invert the comparison.
  switch (kind) {
  case cir::CmpOpKind::lt:
    return cir::CmpOpKind::ge;
  case cir::CmpOpKind::gt:
    return cir::CmpOpKind::le;
  case cir::CmpOpKind::le:
    return cir::CmpOpKind::gt;
  case cir::CmpOpKind::ge:
    return cir::CmpOpKind::lt;
  case cir::CmpOpKind::eq:
  case cir::CmpOpKind::ne:
    return kind;
  }

  return kind;
}

static bool mayWrite(mlir::Operation *op) {
  if (mlir::hasEffect<mlir::MemoryEffects::Write>(op))
    return true;

  if (auto call = dyn_cast<cir::CIRCallOpInterface>(op)) {
    switch (call.getSideEffect()) {
    case cir::SideEffect::All:
      return true;
    case cir::SideEffect::Pure:
    case cir::SideEffect::Const:
      return false;
    }
  }

  return false;
}

static bool mayRead(mlir::Operation *op) {
  if (mlir::hasEffect<mlir::MemoryEffects::Read>(op))
    return true;

  if (auto call = dyn_cast<cir::CIRCallOpInterface>(op)) {
    switch (call.getSideEffect()) {
    case cir::SideEffect::All:
    case cir::SideEffect::Pure:
    case cir::SideEffect::Const:
      return true;
    }
  }

  return false;
}

// Safely moves 'op' before 'to' while preserving program semantics.
// This function ensures the move is safe by:
// 1. Checking basic preconditions (same block, not moving before itself)
// 2. Analyzing memory effects to prevent reordering operations that could
//    change program behavior (e.g., read-after-write, write-after-read)
// 3. Identifying and moving dependent operations together to maintain
//    def-use relationships
// 4. Detecting dependency cycles (cannot move before an operation we depend on)
// Returns true if the move was successful, false if it would be unsafe.
static bool moveBeforeSafely(mlir::Operation *op, mlir::Operation *to) {
  // We cannot move an operation before itself.
  if (op == to)
    return false;

  // We do not support moving between blocks.
  if (to->getBlock() != op->getBlock())
    return false;

  // Operation is already before the target.
  if (op->isBeforeInBlock(to))
    return true;

  // Collect the side effects of this operation.
  bool opMayRead = mayRead(op), opMayWrite = mayWrite(op);

  // Walk the block in the direction we are moving, collect any operations that
  // must move along with us.
  llvm::SmallPtrSet<mlir::Value, 4> operands(op->operand_begin(),
                                             op->operand_end());
  llvm::SmallVector<mlir::Operation *, 4> opsToMove = {op};
  auto rit = std::next(mlir::Block::reverse_iterator(op)),
       rie = std::next(mlir::Block::reverse_iterator(to));
  for (; rit != rie; ++rit) {
    auto *cur = &*rit;

    // If the current operation is a region branch, we conservatively won't move
    // across it.
    if (isa<mlir::RegionBranchOpInterface>(cur))
      break;

    // If the operation(s) being moved may read, and the current operation may
    // write, stop!
    auto curMayWrite = mayWrite(cur);
    if (opMayRead && curMayWrite)
      break;

    // If the operation(s) being moved may write, and the current operation may
    // read, stop!
    auto curMayRead = mayRead(cur);
    if (opMayWrite && mayRead(cur))
      break;

    // If this operation produces an operand of the operation(s) being moved,
    // add it to stack of ops to move.
    for (auto result : cur->getResults()) {
      if (operands.contains(result)) {
        // We cannot move before an operation that we depend on.
        if (cur == to)
          return false;

        opsToMove.push_back(cur);
        opMayRead |= curMayRead;
        opMayWrite |= curMayWrite;
        break;
      }
    }
  }
  if (rit != rie)
    return false;

  // Move the ops before the point in reverse order.
  for (auto *toMove : llvm::reverse(opsToMove))
    toMove->moveBefore(to);

  return true;
}

// Determines whether to add one to the max length based on the comparison type
static bool shouldAddOneToMax(cir::CmpOp cmp, bool flip) {
  switch (comparison(cmp, flip)) {
  case cir::CmpOpKind::lt: // strlen <  n  ==>  memchr(n) < n
  case cir::CmpOpKind::ge: // strlen >= n  ==>  memchr(n) >= n
    return false;
  case cir::CmpOpKind::gt: // strlen >  n  ==>  memchr(n+1) > n
  case cir::CmpOpKind::le: // strlen <= n  ==>  memchr(n+1) <= n
  case cir::CmpOpKind::eq: // strlen == n  ==>  memchr(n+1) == n
  case cir::CmpOpKind::ne: // strlen != n  ==>  memchr(n+1) != n
    return true;
  }
  return false;
}

// Handles the special case where max is zero by converting strlen(str) to
// cast(*str)
static bool handleZeroMax(mlir::Value max, CIRBaseBuilderTy &builder,
                          mlir::Location loc, StrLenOp strLen, cir::CmpOp cmp,
                          bool flip) {
  // Check if max is a constant
  auto maxDef = max.getDefiningOp();
  auto maxConst = mlir::dyn_cast_or_null<cir::ConstantOp>(maxDef);
  if (!maxConst)
    return false; // Not a constant, can't be zero constant case

  auto maxAttr = mlir::dyn_cast<cir::IntAttr>(maxConst.getValue());

  // Check if the max value is zero
  if (!mlir::isa<cir::ZeroAttr>(maxConst.getValue()) &&
      (!maxAttr || maxAttr.getValue() != 0)) {
    return false; // Not a zero max case
  }

  // For zero max with specific comparison types, convert strlen(str) ==>
  // cast(*str)
  switch (comparison(cmp, flip)) {
  default:
    return false; // Not a case we handle
  case cir::CmpOpKind::gt:
  case cir::CmpOpKind::eq:
  case cir::CmpOpKind::ne:
    mlir::Value str = strLen.getOperand();
    auto strTy = mlir::cast<cir::PointerType>(str.getType());
    mlir::Type charTy = strTy.getPointee();
    auto firstChar = cir::LoadOp::create(builder, loc, charTy, str);
    auto firstInt = cir::CastOp::create(builder, loc, max.getType(),
                                        cir::CastKind::integral, firstChar);
    strLen.replaceAllUsesWith(firstInt.getResult());
    strLen.erase();
    return true; // Handled the zero max case
  }
}

// Creates the adjusted max value, handling constants and non-constants
// differently
static mlir::Value createAdjustedMax(mlir::Value max, bool addOne,
                                     CIRBaseBuilderTy &builder,
                                     mlir::Location loc) {
  auto maxDef = max.getDefiningOp();

  if (auto maxConst = mlir::dyn_cast<cir::ConstantOp>(maxDef)) {
    // If the max def is a constant, create a new constant op.
    auto maxAttr = mlir::dyn_cast<cir::IntAttr>(maxConst.getValue());
    if (!maxAttr)
      return {};

    if (addOne)
      maxAttr = cir::IntAttr::get(maxAttr.getType(), ++maxAttr.getValue());

    auto newMax = cir::ConstantOp::create(builder, loc, maxAttr);
    return newMax.getResult();

  } else if (addOne) {
    // Add one to the max value, if needed.
    // FIXME(cir): we should probably ensure that max is not INT_MAX
    auto oneAttr = cir::IntAttr::get(max.getType(), 1);
    auto one = cir::ConstantOp::create(builder, loc, oneAttr);
    auto addOp = cir::BinOp::create(builder, loc, max.getType(),
                                    cir::BinOpKind::Add, max, one);
    addOp.setNoUnsignedWrap(true);
    return addOp.getResult();
  }

  return max;
}

void LibOptPass::xformStrLenIntoMemchr(StrLenOp strLen) {
  mlir::Value len = strLen.getResult();
  auto strTy = mlir::cast<cir::PointerType>(strLen.getOperand().getType());
  mlir::Type charTy = strTy.getPointee();

  // Find the single cmp use of the length, if it exists.
  mlir::OpOperand *operand;
  cir::CmpOp cmp;
  while (len.hasOneUse()) {
    operand = &*len.use_begin();
    auto *user = operand->getOwner();
    cmp = mlir::dyn_cast<cir::CmpOp>(user);
    if (auto cast = mlir::dyn_cast<cir::CastOp>(user))
      len = cast.getResult();
    else
      break;
  }

  // If we couldn't find a single cmp, N/A.
  if (!cmp)
    return;

  // Determine the max length and whether to add one.
  bool flip = operand->getOperandNumber() != 0;
  mlir::Value max = cmp.getOperand(flip ? 0 : 1);
  bool addOne = shouldAddOneToMax(cmp, flip);

  // Ensure the max length dominates the call to strLen, or is a constant, or
  // can be safely moved before it.
  auto maxDef = max.getDefiningOp();
  if (maxDef && strLen && !isa<cir::ConstantOp>(maxDef)) {
    // Move the definition before the StrLenOp, if possible.
    if (!moveBeforeSafely(maxDef, strLen))
      return;
  }

  auto loc = strLen.getLoc();
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPoint(strLen);

  // Handle the special case where max is zero
  if (handleZeroMax(max, builder, loc, strLen, cmp, flip))
    return; // Zero max case was handled, early return

  // Create the adjusted max value, handling constants and non-constants.
  max = createAdjustedMax(max, addOne, builder, loc);
  // If we failed to adjust the max value, early return.
  if (!max)
    return;

  // Convert the string to a void*
  mlir::Type lenTy = strLen.getResult().getType();
  mlir::Value str = strLen.getOperand();
  mlir::Type voidPtrTy = builder.getVoidPtrTy(strTy.getAddrSpace());
  auto voidPtr =
      cir::CastOp::create(builder, loc, voidPtrTy, cir::CastKind::bitcast, str);

  // Get the null character.
  mlir::Value nullChar = builder.getNullValue(charTy, loc);
  mlir::Type intTy = builder.getSIntNTy(32);
  auto castedNullChar = cir::CastOp::create(builder, loc, intTy,
                                            cir::CastKind::integral, nullChar);

  // Convert the max value to a size_t, if needed.
  if (max.getType() != lenTy)
    max = builder.create<cir::CastOp>(loc, lenTy, cir::CastKind::integral, max);

  // Build the MemChrOp
  auto memChr =
      cir::MemChrOp::create(builder, loc, voidPtr, castedNullChar, max);
  auto memChrStr = cir::CastOp::create(
      builder, loc, strTy, cir::CastKind::bitcast, memChr.getResult());
  auto ptrDiff = cir::PtrDiffOp::create(builder, loc, lenTy, memChrStr, str);

  strLen.replaceAllUsesWith(ptrDiff.getResult());
  strLen.erase();
}

void LibOptPass::runOnOperation() {
  assert(astCtx && "Missing ASTContext, please construct with the right ctor");
  opts.parseOptions(*this);
  auto *op = getOperation();
  if (isa<::mlir::ModuleOp>(op))
    theModule = cast<::mlir::ModuleOp>(op);

  llvm::SmallVector<StdFindOp> stdFindToTransform;
  op->walk([&](StdFindOp findOp) { stdFindToTransform.push_back(findOp); });

  llvm::SmallVector<StrLenOp> strLenToTransform;
  op->walk([&](StrLenOp strLen) { strLenToTransform.push_back(strLen); });

  for (auto c : stdFindToTransform)
    xformStdFindIntoMemchr(c);

  for (auto strLen : strLenToTransform)
    xformStrLenIntoMemchr(strLen);
}

std::unique_ptr<Pass> mlir::createLibOptPass() {
  return std::make_unique<LibOptPass>();
}

std::unique_ptr<Pass> mlir::createLibOptPass(clang::ASTContext *astCtx) {
  auto pass = std::make_unique<LibOptPass>();
  pass->setASTContext(astCtx);
  return std::move(pass);
}

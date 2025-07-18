//===- AArch64.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/CIR/Target/AArch64.h"
#include "ABIInfoImpl.h"
#include "LowerFunctionInfo.h"
#include "LowerTypes.h"
#include "TargetInfo.h"
#include "TargetLoweringInfo.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using AArch64ABIKind = cir::AArch64ABIKind;
using ABIArgInfo = cir::ABIArgInfo;
using MissingFeatures = cir::MissingFeatures;

namespace cir {

//===----------------------------------------------------------------------===//
// AArch64 ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class AArch64ABIInfo : public ABIInfo {
  AArch64ABIKind Kind;

public:
  AArch64ABIInfo(LowerTypes &CGT, AArch64ABIKind Kind)
      : ABIInfo(CGT), Kind(Kind) {}

private:
  AArch64ABIKind getABIKind() const { return Kind; }
  bool isDarwinPCS() const { return Kind == AArch64ABIKind::DarwinPCS; }

  ABIArgInfo classifyReturnType(mlir::Type RetTy, bool IsVariadic) const;
  ABIArgInfo classifyArgumentType(mlir::Type RetTy, bool IsVariadic,
                                  unsigned CallingConvention) const;

  void computeInfo(LowerFunctionInfo &FI) const override {
    if (!cir::classifyReturnType(getCXXABI(), FI, *this))
      FI.getReturnInfo() =
          classifyReturnType(FI.getReturnType(), FI.isVariadic());

    for (auto &it : FI.arguments())
      it.info = classifyArgumentType(it.type, FI.isVariadic(),
                                     FI.getCallingConvention());
  }
};

class AArch64TargetLoweringInfo : public TargetLoweringInfo {
public:
  AArch64TargetLoweringInfo(LowerTypes &LT, AArch64ABIKind Kind)
      : TargetLoweringInfo(std::make_unique<AArch64ABIInfo>(LT, Kind)) {
    cir_cconv_assert(!MissingFeatures::swift());
  }

  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      cir::AddressSpace addrSpace) const override {
    switch (addrSpace) {
    case cir::AddressSpace::OffloadPrivate:
    case cir::AddressSpace::OffloadLocal:
    case cir::AddressSpace::OffloadGlobal:
    case cir::AddressSpace::OffloadConstant:
    case cir::AddressSpace::OffloadGeneric:
      return 0;
    default:
      cir_cconv_unreachable("Unknown CIR address space for this target");
    }
  }
};

} // namespace

ABIArgInfo AArch64ABIInfo::classifyReturnType(mlir::Type RetTy,
                                              bool IsVariadic) const {
  if (mlir::isa<VoidType>(RetTy))
    return ABIArgInfo::getIgnore();

  if (const auto _ = mlir::dyn_cast<VectorType>(RetTy)) {
    cir_cconv_assert_or_abort(!cir::MissingFeatures::vectorType(), "NYI");
  }

  // Large vector types should be returned via memory.
  if (mlir::isa<VectorType>(RetTy) && getContext().getTypeSize(RetTy) > 128)
    cir_cconv_unreachable("NYI");

  if (!isAggregateTypeForABI(RetTy)) {
    // NOTE(cir): Skip enum handling.

    if (MissingFeatures::fixedSizeIntType())
      cir_cconv_unreachable("NYI");

    return (isPromotableIntegerTypeForABI(RetTy) && isDarwinPCS()
                ? ABIArgInfo::getExtend(RetTy)
                : ABIArgInfo::getDirect());
  }

  uint64_t Size = getContext().getTypeSize(RetTy);
  cir_cconv_assert(!cir::MissingFeatures::emitEmptyRecordCheck());
  cir_cconv_assert(
      !cir::MissingFeatures::supportisHomogeneousAggregateQueryForAArch64());

  // Aggregates <= 16 bytes are returned directly in registers or on the stack.
  if (Size <= 128) {
    if (Size <= 64 && !getDataLayout().isBigEndian()) {
      // Composite types are returned in lower bits of a 64-bit register for LE,
      // and in higher bits for BE. However, integer types are always returned
      // in lower bits for both LE and BE, and they are not rounded up to
      // 64-bits. We can skip rounding up of composite types for LE, but not for
      // BE, otherwise composite types will be indistinguishable from integer
      // types.
      return ABIArgInfo::getDirect(
          cir::IntType::get(LT.getMLIRContext(), Size, false));
    }

    unsigned Alignment = getContext().getTypeAlign(RetTy);
    Size = llvm::alignTo(Size, 64); // round up to multiple of 8 bytes

    // We use a pair of i64 for 16-byte aggregate with 8-byte alignment.
    // For aggregates with 16-byte alignment, we use i128.
    if (Alignment < 128 && Size == 128) {
      mlir::Type baseTy = cir::IntType::get(LT.getMLIRContext(), 64, false);
      return ABIArgInfo::getDirect(cir::ArrayType::get(baseTy, Size / 64));
    }

    return ABIArgInfo::getDirect(
        IntType::get(LT.getMLIRContext(), Size, false));
  }

  return getNaturalAlignIndirect(RetTy);
}

ABIArgInfo
AArch64ABIInfo::classifyArgumentType(mlir::Type Ty, bool IsVariadic,
                                     unsigned CallingConvention) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  // TODO(cir): check for illegal vector types.
  if (MissingFeatures::vectorType())
    cir_cconv_unreachable("NYI");

  if (!isAggregateTypeForABI(Ty)) {
    // NOTE(cir): Enum is IntType in CIR. Skip enum handling here.

    if (MissingFeatures::fixedSizeIntType())
      cir_cconv_unreachable("NYI");

    return (isPromotableIntegerTypeForABI(Ty) && isDarwinPCS()
                ? ABIArgInfo::getExtend(Ty)
                : ABIArgInfo::getDirect());
  }

  uint64_t Size = getContext().getTypeSize(Ty);

  // Aggregates <= 16 bytes are passed directly in registers or on the stack.
  if (Size <= 128) {
    unsigned Alignment;
    if (Kind == AArch64ABIKind::AAPCS) {
      Alignment = getContext().getTypeAlign(Ty);
      Alignment = Alignment < 128 ? 64 : 128;
    } else {
      Alignment = std::max(
          getContext().getTypeAlign(Ty),
          (unsigned)getTarget().getPointerWidth(clang::LangAS::Default));
    }
    Size = llvm::alignTo(Size, Alignment);

    // We use a pair of i64 for 16-byte aggregate with 8-byte alignment.
    // For aggregates with 16-byte alignment, we use i128.
    mlir::Type baseTy =
        cir::IntType::get(LT.getMLIRContext(), Alignment, false);
    auto argTy = Size == Alignment
                     ? baseTy
                     : cir::ArrayType::get(baseTy, Size / Alignment);
    return ABIArgInfo::getDirect(argTy);
  }

  return getNaturalAlignIndirect(Ty, /*ByVal=*/false);
}

std::unique_ptr<TargetLoweringInfo>
createAArch64TargetLoweringInfo(LowerModule &CGM, AArch64ABIKind Kind) {
  return std::make_unique<AArch64TargetLoweringInfo>(CGM.getTypes(), Kind);
}

} // namespace cir

#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "tritonamdgpu-dot-slice-and-interleave"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUDOTSLICEANDINTERLEAVE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

/// Move op and its predecessors within the same block up to be after
/// lastInsertedOp, preserving dominance and def-use order.
static void moveOpAndPredecessorsUpSameBlock(Operation *op,
                                             Operation *lastInsertedOp) {
  assert(lastInsertedOp != nullptr && "lastInsertedOp cannot be null");
  assert(op->getBlock() == lastInsertedOp->getBlock() && "op and lastInsertedOp must be in the same block");

  Operation *checkedOp = lastInsertedOp;
  if (lastInsertedOp->isBeforeInBlock(op)) {
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = [&checkedOp](Operation *sliceOp) {
      return sliceOp->getBlock() == checkedOp->getBlock() &&
             checkedOp->isBeforeInBlock(sliceOp);
    };
    (void)getBackwardSlice(op, &backwardSlice, opt);
    for (auto predOp : backwardSlice)
      predOp->moveAfter(lastInsertedOp);
    op->moveAfter(lastInsertedOp);
  } else {
    auto hasUnsafeUser = [&checkedOp](Operation *user) {
      return user != checkedOp && user->getBlock() == checkedOp->getBlock() &&
             user->isBeforeInBlock(checkedOp);
    };
    if (llvm::any_of(op->getUsers(), hasUnsafeUser))
      return;
    op->moveAfter(lastInsertedOp);
  }
}

/// Generate local slices for a value defined by ttg.local_load. Returns failure
/// if the value is not a local_load or slicing would violate constraints.
static LogicalResult genLocalSlices(OpBuilder &builder, Value v,
                                    Attribute dotEncoding, unsigned opIdx,
                                    unsigned numSlices, int64_t sliceWidth,
                                    SmallVector<Operation *> &subviews,
                                    SmallVector<Operation *> &slices) {
  LDBG("genLocalSlices: opIdx=" << opIdx << ", numSlices=" << numSlices
                                << ", sliceWidth=" << sliceWidth);

  auto localLoad = v.getDefiningOp<ttg::LocalLoadOp>();
  if (!localLoad) {
    LDBG("genLocalSlices: FAILED - value is not defined by ttg.local_load");
    if (auto defOp = v.getDefiningOp())
      defOp->emitWarning("genLocalSlices: value is not defined by ttg.local_load");
    return failure();
  }

  Value memDesc = localLoad.getSrc();
  auto type = dyn_cast<ttg::MemDescType>(memDesc.getType());
  if (!type) {
    LDBG("genLocalSlices: FAILED - memDesc type is not ttg::MemDescType");
    localLoad->emitWarning("genLocalSlices: memDesc type is not ttg::MemDescType");
    return failure();
  }

  SmallVector<int64_t> shape = llvm::to_vector(type.getShape());
  Type elementType = type.getElementType();
  int64_t kIdx = opIdx == 0 ? 1 : 0;
  shape[kIdx] = sliceWidth;

  LDBG("genLocalSlices: original shape=[" << shape[0] << "," << shape[1]
                                          << "], kIdx=" << kIdx);

  // Each slice cannot be smaller than a reasonable minimum to avoid creating
  // degenerate fragments. Use 16 as a conservative default (MFMA min width).
  if (sliceWidth < 16) {
    LDBG("genLocalSlices: FAILED - sliceWidth ("
         << sliceWidth << ") < 16, would create degenerate fragments");
    localLoad->emitWarning("genLocalSlices: sliceWidth too small, would create degenerate fragments");
    return failure();
  }

  auto dotOperandEnc = ttg::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, dotEncoding,
      /*kWidth=*/
      cast<ttg::DotOperandEncodingAttr>(
          cast<RankedTensorType>(localLoad.getResult().getType()).getEncoding())
          .getKWidth());

  auto subviewDescType = ttg::MemDescType::get(
      shape, elementType, type.getEncoding(), type.getMemorySpace(),
      type.getMutableMemory(), type.getAllocShape());

  for (unsigned i = 0; i < numSlices; i++) {
    SmallVector<int32_t> logicalOffsets;
    SmallVector<int64_t> offsets = {0, 0};
    offsets[kIdx] = i;
    for (int64_t off : offsets)
      logicalOffsets.push_back(static_cast<int32_t>(off * sliceWidth));

    LDBG("genLocalSlices: creating slice "
         << i << "/" << numSlices << " with offsets=[" << logicalOffsets[0]
         << "," << logicalOffsets[1] << "]");

    Value newSmem = builder.create<ttg::MemDescSubsliceOp>(
        v.getLoc(), subviewDescType, memDesc, logicalOffsets);
    Value prefetchSlice = builder.create<ttg::LocalLoadOp>(
        v.getLoc(), RankedTensorType::get(shape, elementType, dotOperandEnc),
        newSmem);
    subviews.push_back(newSmem.getDefiningOp());
    slices.push_back(prefetchSlice.getDefiningOp());
  }

  LDBG("genLocalSlices: SUCCESS - created " << numSlices << " slices");
  return success();
}

/// Create a sliced ttg.local_load from a base ttg.local_load operation.
/// This creates a subview of the shared memory descriptor and loads from it
/// with the specified shape, offsets, and dot operand encoding.
static Value createLocalLoadSlice(OpBuilder &builder, Location loc,
                                  ttg::LocalLoadOp baseLoad,
                                  ArrayRef<int64_t> sliceShape,
                                  ArrayRef<int64_t> offsets,
                                  Attribute dotOperandEnc) {
  Value memDesc = baseLoad.getSrc();
  auto memTy = cast<ttg::MemDescType>(memDesc.getType());
  auto subviewDescType = ttg::MemDescType::get(
      sliceShape, memTy.getElementType(), memTy.getEncoding(),
      memTy.getMemorySpace(), memTy.getMutableMemory(), memTy.getAllocShape());
  SmallVector<int32_t> logicalOffsets;
  logicalOffsets.reserve(offsets.size());
  for (int64_t off : offsets)
    logicalOffsets.push_back(static_cast<int32_t>(off));
  Value newSmem = builder.create<ttg::MemDescSubsliceOp>(
      loc, subviewDescType, memDesc, logicalOffsets);
  auto tensorTy = RankedTensorType::get(llvm::to_vector(sliceShape),
                                        memTy.getElementType(), dotOperandEnc);
  return builder.create<ttg::LocalLoadOp>(loc, tensorTy, newSmem);
}



/// Slice a tt.dot operation into smaller tiles across M, N, and K dimensions.
///
/// This function splits matrix multiplication operations to improve ILP of mma
/// and memory access patterns. The transformation decomposes a single large
/// tt.dot into multiple smaller dots that operate on sub-matrices (tiles) of
/// the original operands.
///
/// **Tiling Strategy:**
/// - M dimension: Split output rows into `mTile`-sized chunks
/// - N dimension: Split output columns into `nTile`-sized chunks  
/// - K dimension: Split reduction dimension into `kTile`-sized chunks
///
/// **Generated Pattern:**
/// For an original dot C = A * B where:
/// - A: [M, K], B: [K, N], C: [M, N]
/// - Tiles: mTile × nTile × kTile
///
/// The transformation generates:
/// ```
/// for mi in 0..M/mTile:
///   for ni in 0..N/nTile:
///     acc = extract_slice(C, [mi*mTile, ni*nTile])
///     for ki in 0..K/kTile:
///       a_slice = local_load(memdesc_subslice(A_memdesc, [mi*mTile, ki*kTile]))
///       b_slice = local_load(memdesc_subslice(B_memdesc, [ki*kTile, ni*nTile]))
///       acc = tt.dot(a_slice, b_slice, acc)
///     tiles.append(acc)
/// result = amdgpu.concat(tiles) // if multiple tiles
/// ```
///
/// **Encoding Preservation:**
/// - Maintains original dot operand encodings (DotOperandEncodingAttr)
/// - Preserves kWidth from original operands for hardware compatibility
/// - Output tiles inherit parent encoding for correct concatenation
///
/// **Memory Access Pattern:**
/// - Uses ttg.memdesc_subslice for efficient shared memory sub-views
/// - Leverages ttg.local_load for optimized MFMA/WMMA operand loading
/// - Accumulation pattern enables register reuse across K-slices
///
/// \param builder MLIR OpBuilder for creating new operations
/// \param dot Original tt.dot operation to slice
/// \param mTile Target tile size for M dimension (0 = no M slicing)
/// \param nTile Target tile size for N dimension (0 = no N slicing) 
/// \param kTile Target tile size for K dimension (0 = no K slicing)
/// \param result [out] The resulting value after tiling and concatenation
///
/// \returns success() if slicing was performed, failure() if:
/// - Operands are not from ttg.local_load operations
/// - Dimensions are not evenly divisible by tile sizes
/// - Tile sizes are invalid (negative or too large)
///
/// \note This function requires that both operands of the dot operation
///       originate from ttg.local_load operations operating on shared memory
///       descriptors, as it needs to create sub-slices of the memory regions.
static LogicalResult sliceDotMNK(OpBuilder &builder, tt::DotOp dot,
                                 int64_t mTile, int64_t nTile, int64_t kTile,
                                 Value &result) {
  Location loc = dot.getLoc();
  auto dTy = cast<RankedTensorType>(dot.getType());
  auto dShape = dTy.getShape();
  int64_t M = dShape[0];
  int64_t N = dShape[1];

  auto aTy = cast<RankedTensorType>(dot.getA().getType());
  auto aShape = aTy.getShape(); // [M, K]
  auto bTy = cast<RankedTensorType>(dot.getB().getType());
  auto bShape = bTy.getShape(); // [K, N]
  int64_t K = aShape[1];

  LDBG("sliceDotMNK: input dot shape=["
       << M << "x" << N << "], A=[" << aShape[0] << "x" << aShape[1] << "], B=["
       << bShape[0] << "x" << bShape[1] << "]");
  LDBG("sliceDotMNK: requested tiles mTile=" << mTile << ", nTile=" << nTile
                                             << ", kTile=" << kTile);

  // Extract warpsPerCTA from the encoding to determine minimum tile constraints
  Attribute encoding = dTy.getEncoding();
  SmallVector<unsigned> warpsPerCTA;
  int64_t instrM, instrN;
  if (auto mfmaEnc = dyn_cast<ttg::AMDMfmaEncodingAttr>(encoding)) {
    warpsPerCTA = llvm::to_vector(mfmaEnc.getWarpsPerCTA());
    instrM = mfmaEnc.getMDim();
    instrN = mfmaEnc.getNDim();
  } else {
    LDBG("sliceDotMNK: FAILED - unsupported encoding type");
    dot->emitWarning("sliceDotMNK: unsupported encoding type");
    return failure();
  }

  // Calculate minimum tile sizes based on warpsPerCTA constraints
  // Each warp handles instrShape, so total CTA handles instrShape * warpsPerCTA
  int64_t minMTile = instrM * warpsPerCTA[0];  // instrShape[0] * warpsPerCTA[0]
  int64_t minNTile = instrN * warpsPerCTA[1];  // instrShape[1] * warpsPerCTA[1]

  LDBG("sliceDotMNK: warpsPerCTA=[" << warpsPerCTA[0] << "," << warpsPerCTA[1] 
       << "], minTiles=[" << minMTile << "," << minNTile << "]");

  // Adjust requested tile sizes to be multiples of minimum tile sizes
  if (mTile <= 0) {
    mTile = M;
  } else {
    mTile = std::max(mTile, minMTile);
    mTile = (mTile / minMTile) * minMTile; // Round to multiple of minMTile
  }
  
  if (nTile <= 0) {
    nTile = N;
  } else {
    nTile = std::max(nTile, minNTile);
    nTile = (nTile / minNTile) * minNTile; // Round to multiple of minNTile
  }
  
  if (kTile <= 0)
    kTile = K;

  LDBG("sliceDotMNK: adjusted tiles mTile=" << mTile << ", nTile=" << nTile
                                             << ", kTile=" << kTile);

  if (M % mTile != 0 || N % nTile != 0 || K % kTile != 0) {
    LDBG("sliceDotMNK: FAILED - dimension not divisible by tile size: M%"
         << mTile << "=" << (M % mTile) << ", N%" << nTile << "=" << (N % nTile)
         << ", K%" << kTile << "=" << (K % kTile));
    dot->emitWarning("sliceDotMNK: dimension not divisible by tile size");
    return failure();
  }

  auto aLoad = dot.getA().getDefiningOp<ttg::LocalLoadOp>();
  auto bLoad = dot.getB().getDefiningOp<ttg::LocalLoadOp>();
  if (!aLoad || !bLoad) {
    LDBG(
        "sliceDotMNK: FAILED - dot operands are not from ttg.local_load (aLoad="
        << (aLoad ? "yes" : "no") << ", bLoad=" << (bLoad ? "yes" : "no")
        << ")");
    dot->emitWarning("sliceDotMNK: dot operands are not from ttg.local_load");
    return failure();
  }

  Attribute parentEnc = dTy.getEncoding();
  auto aEnc = ttg::DotOperandEncodingAttr::get(
      builder.getContext(), 0, parentEnc,
      /*kWidth=*/
      cast<ttg::DotOperandEncodingAttr>(aTy.getEncoding()).getKWidth());
  auto bEnc = ttg::DotOperandEncodingAttr::get(
      builder.getContext(), 1, parentEnc,
      /*kWidth=*/
      cast<ttg::DotOperandEncodingAttr>(bTy.getEncoding()).getKWidth());

  SmallVector<Value> tiles;
  int64_t mParts = M / mTile;
  int64_t nParts = N / nTile;
  int64_t kParts = K / kTile;

  LDBG("sliceDotMNK: creating " << mParts << "x" << nParts
                                << " tiles, each with " << kParts
                                << " K-slices");

  for (int64_t mi = 0; mi < mParts; ++mi) {
    for (int64_t ni = 0; ni < nParts; ++ni) {
      // Initial accumulator is C slice
      int64_t mOff = mi * mTile;
      int64_t nOff = ni * nTile;

      LDBG("sliceDotMNK: processing tile ("
           << mi << "," << ni << ") at offset [" << mOff << "," << nOff << "]");

      Value acc = builder.create<mlir::triton::amdgpu::ExtractSliceOp>(
          loc,
          RankedTensorType::get({mTile, nTile}, dTy.getElementType(),
                                dTy.getEncoding()),
          dot.getC(),
          ArrayRef<int64_t>({mOff, nOff}));

      for (int64_t ki = 0; ki < kParts; ++ki) {
        int64_t kOff = ki * kTile;

        LDBG("sliceDotMNK: K-slice " << ki << "/" << kParts
                                     << " at kOff=" << kOff);

        // A slice: [mTile, kTile] at [mOff, kOff]
        Value aSlice = createLocalLoadSlice(builder, loc, aLoad, {mTile, kTile},
                                            {mOff, kOff}, aEnc);
        // B slice: [kTile, nTile] at [kOff, nOff]
        Value bSlice = createLocalLoadSlice(builder, loc, bLoad, {kTile, nTile},
                                            {kOff, nOff}, bEnc);

        // Create a new dot with the correct tiled result type
        auto accType = cast<RankedTensorType>(acc.getType());
        acc = builder.create<tt::DotOp>(loc, accType, aSlice, bSlice, acc);
      }
      tiles.push_back(acc);
    }
  }

  if (tiles.size() == 1) {
    LDBG("sliceDotMNK: single tile result, no concatenation needed");
    result = tiles.front();
    return success();
  }

  LDBG("sliceDotMNK: concatenating " << tiles.size()
                                     << " tiles into final result");

  // Concatenate tiles row-major into full [M,N]
  SmallVector<Type> srcTypes;
  srcTypes.reserve(tiles.size());
  for (size_t i = 0; i < tiles.size(); ++i)
    srcTypes.push_back(tiles[i].getType());
  result = builder.create<mlir::triton::amdgpu::ConcatOp>(loc, dTy, tiles);

  LDBG("sliceDotMNK: SUCCESS - created tiled dot with "
       << mParts << "x" << nParts << " output tiles");
  return success();
}

class TritonAMDGPUDotSliceAndInterleave
    : public impl::TritonAMDGPUDotSliceAndInterleaveBase<
          TritonAMDGPUDotSliceAndInterleave> {
public:
  using Base::Base;

  void runOnOperation() override {
    auto funcOp = getOperation();

    LDBG("=== DotSliceAndInterleave Pass Start ===");
    LDBG("Processing function: " << funcOp->getName());
    LDBG("targetSliceMNK size: " << targetSliceMNK.size());
    for (size_t i = 0; i < targetSliceMNK.size(); ++i) {
      LDBG("targetSliceMNK[" << i << "] = " << targetSliceMNK[i]);
    }
    LDBG("loadsPerGroup: " << loadsPerGroup
                           << ", dotsPerGroup: " << dotsPerGroup);

    // Parse K from targetSliceMNK if provided
    int64_t targetK = 0;
    if (!targetSliceMNK.empty() && (int)targetSliceMNK.size() >= 3)
      targetK = targetSliceMNK[2];
    if (targetK <= 0 && loadsPerGroup <= 0 && dotsPerGroup <= 0) {
      LDBG("=== Pass EARLY EXIT - no slicing requested (targetK=" << targetK
                                                                  << ") ===");
      return; // No-op by default
    }

    // Handle dots not enclosed in loops (best-effort in-place slicing).
    SmallVector<tt::DotOp> standaloneDots;
    funcOp->walk([&](tt::DotOp dot) {
      standaloneDots.push_back(dot);
    });

    LDBG("Found " << standaloneDots.size()
                  << " standalone tt.dot operations");

    for (tt::DotOp dot : standaloneDots) {
      if (!dot.getA().getDefiningOp<ttg::LocalLoadOp>() ||
          !dot.getB().getDefiningOp<ttg::LocalLoadOp>()) {
        LDBG("Skipping standalone dot - operands not from ttg.local_load");
        continue;
      }
      if (targetK <= 0) {
        LDBG("Skipping standalone dot - targetK <= 0");
        continue;
      }

      LDBG("Processing standalone dot...");
      OpBuilder builder(dot);
      int64_t mTile =
          (!targetSliceMNK.empty() && (int)targetSliceMNK.size() >= 1)
              ? targetSliceMNK[0]
              : 0;
      int64_t nTile =
          (!targetSliceMNK.empty() && (int)targetSliceMNK.size() >= 2)
              ? targetSliceMNK[1]
              : 0;
      Value out;
      if (failed(sliceDotMNK(builder, dot, mTile, nTile, targetK, out))) {
        LDBG("Failed to slice standalone dot: " << dot);
        dot->emitWarning("Failed to slice standalone dot");
        continue;
      }
      LDBG("Successfully sliced standalone dot, replacing original");
      dot->replaceAllUsesWith(ValueRange{out});
      dot->erase();
    }

    LDBG("=== DotSliceAndInterleave Pass Complete ===");
  }
};

} // namespace

} // namespace mlir

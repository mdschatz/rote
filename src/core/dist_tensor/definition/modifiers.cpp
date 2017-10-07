/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote {

// TODO: Check if this should retain order of object
template <typename T>
void DistTensorBase<T>::Align(const std::vector<Unsigned> &modeAlignments) {
#ifndef RELEASE
  CallStackEntry entry("DistTensor::Align");
#endif
  Unsigned i;
  Empty();
  modeAlignments_ = modeAlignments;
  for (i = 0; i < modeAlignments.size(); i++)
    constrainedModeAlignments_[i] = true;
  SetShifts();
}

template <typename T>
void DistTensorBase<T>::AlignMode(Mode mode, Unsigned modeAlignment) {
#ifndef RELEASE
  CallStackEntry entry("DistTensor::AlignMode");
  const Unsigned order = Order();
  if (mode < 0 || mode >= order)
    LogicError("0 <= mode < object order must be true");
#endif
  EmptyData();
  modeAlignments_[mode] = modeAlignment;
  constrainedModeAlignments_[mode] = true;
  SetShifts();
}

// NOTE: This needs to be generalized
template <typename T>
void DistTensorBase<T>::AlignWith(const DistTensorBase<T> &A) {
  Unsigned i;
  Unsigned order = A.Order();
  const rote::Grid &grid = A.Grid();
  SetGrid(grid);

  for (i = 0; i < order; i++) {
    modeAlignments_[i] = A.modeAlignments_[i] % ModeStride(i);
    constrainedModeAlignments_[i] = true;
  }
  SetShifts();
}

template <typename T>
void DistTensorBase<T>::SetDistribution(const TensorDistribution &tenDist) {
  dist_ = tenDist;
}

template <typename T>
void DistTensorBase<T>::AlignModeWith(Mode mode, const DistTensorBase<T> &A) {
  AlignModeWith(mode, A, mode);
}

template <typename T>
void DistTensorBase<T>::AlignModeWith(Mode mode, const DistTensorBase<T> &A,
                                      Mode modeA) {
#ifndef RELEASE
  CallStackEntry cse("DistTensor::AlignModeWith");
#endif
  ModeArray modes(1);
  modes[0] = mode;
  ModeArray modesA(1);
  modesA[0] = modeA;
  AlignModesWith(modes, A, modesA);
}

template <typename T>
void DistTensorBase<T>::AlignModesWith(const ModeArray &modes,
                                       const DistTensorBase<T> &A,
                                       const ModeArray &modesA) {
#ifndef RELEASE
  CallStackEntry cse("DistTensor::AlignModesWith");
#endif
  Unsigned i;
  for (i = 0; i < modes.size(); i++) {
    Mode mode = modes[i];
    Mode modeA = modesA[i];
    modeAlignments_[mode] = A.modeAlignments_[modeA] % ModeStride(mode);
    constrainedModeAlignments_[mode] = true;
    SetModeShift(mode);
  }
}

template <typename T>
void DistTensorBase<T>::Attach(const ObjShape &shape,
                               const std::vector<Unsigned> &modeAlignments,
                               T *buffer, const std::vector<Unsigned> &strides,
                               const rote::Grid &g) {
#ifndef RELEASE
  CallStackEntry entry("DistTensor::Attach");
#endif
  Empty();

  grid_ = &g;
  shape_ = shape;
  modeAlignments_ = modeAlignments;
  viewType_ = VIEW;
  SetShifts();
  if (Participating()) {
    ObjShape localShape = Lengths(shape, ModeShifts(), ModeStrides());
    tensor_.Attach(PermuteVector(localShape, localPerm_), buffer, strides);
  }
}

template <typename T>
void DistTensorBase<T>::LockedAttach(
    const ObjShape &shape, const std::vector<Unsigned> &modeAlignments,
    const T *buffer, const std::vector<Unsigned> &strides,
    const rote::Grid &g) {
#ifndef RELEASE
  CallStackEntry entry("DistTensor::LockedAttach");
#endif
  grid_ = &g;
  shape_ = shape;
  modeAlignments_ = modeAlignments;
  SetShifts();
  if (Participating()) {
    ObjShape localShape = Lengths(shape, ModeShifts(), ModeStrides());
    tensor_.LockedAttach(localShape, buffer, strides);
  }
}

template <typename T>
void DistTensorBase<T>::LockedAttach(
    const ObjShape &shape, const std::vector<Unsigned> &modeAlignments,
    const T *buffer, const Permutation &perm,
    const std::vector<Unsigned> &strides, const rote::Grid &g) {
#ifndef RELEASE
  CallStackEntry entry("DistTensor::LockedAttach");
#endif
  grid_ = &g;
  shape_ = shape;
  modeAlignments_ = modeAlignments;
  localPerm_ = perm;
  SetShifts();
  if (Participating()) {
    // Account for local permutation
    ObjShape localShape =
        PermuteVector(Lengths(shape, ModeShifts(), ModeStrides()), localPerm_);
    tensor_.LockedAttach(localShape, buffer, strides);
  }
}

template <typename T>
void DistTensorBase<T>::ResizeTo(const DistTensorBase<T> &A) {
#ifndef RELEASE
  CallStackEntry entry("DistTensor::ResizeTo");
  AssertNotLocked();
#endif
  ResizeTo(A.Shape());
}

// TODO: FIX Participating
template <typename T> void DistTensorBase<T>::ResizeTo(const ObjShape &shape) {
#ifndef RELEASE
  CallStackEntry entry("DistTensor::ResizeTo");
  AssertNotLocked();
#endif
  if (AnyElemwiseNotEqual(shape_, shape)) {
    shape_ = shape;
    SetShifts();
  }
  if (Participating() && viewType_ == OWNER) {
    // Account for local permutation
    tensor_.ResizeTo(PermuteVector(
        Lengths(shape, modeShifts_, gridView_.ParticipatingShape()),
        localPerm_));
  }
}

template <typename T>
void DistTensorBase<T>::ResizeTo(const ObjShape &shape,
                                 const std::vector<Unsigned> &strides) {
#ifndef RELEASE
  CallStackEntry entry("DistTensor::ResizeTo");
  AssertNotLocked();
#endif
  shape_ = shape;
  if (Participating()) {
    tensor_.ResizeTo(Lengths(shape, ModeShifts(), ModeStrides()), strides);
  }
}

template <typename T> void DistTensorBase<T>::Set(const Location &loc, T u) {
#ifndef RELEASE
  CallStackEntry entry("DistTensor::Set");
  AssertValidEntry(loc);
#endif
  if (!Participating())
    return;
  const Location owningProc = DetermineOwner(loc);
  const GridView gv = GetGridView();

  if (!AnyElemwiseNotEqual(gv.ParticipatingLoc(), owningProc)) {
    const Location localLoc = Global2LocalIndex(loc);
    SetLocal(PermuteVector(localLoc, localPerm_), u);
  }
}

template <typename T> void DistTensorBase<T>::Update(const Location &loc, T u) {
#ifndef RELEASE
  CallStackEntry entry("DistTensor::Update");
  AssertValidEntry(loc);
#endif
  const GridView gv = GetGridView();
  const Location owningProc = DetermineOwner(loc);
  if (!AnyElemwiseNotEqual(gv.ParticipatingLoc(), owningProc)) {
    const Location localLoc = Global2LocalIndex(loc);
    UpdateLocal(localLoc, u);
  }
}

//
// Functions which explicitly work in the complex plane
//

template <typename T>
void DistTensorBase<T>::SetRealPart(const Location &loc, BASE(T) u) {
  NOT_USED(loc);
  NOT_USED(u);
}

template <typename T>
void DistTensorBase<T>::SetImagPart(const Location &loc, BASE(T) u) {
  NOT_USED(loc);
  NOT_USED(u);
}

template <typename T>
void DistTensorBase<T>::UpdateRealPart(const Location &loc, BASE(T) u) {
  NOT_USED(loc);
  NOT_USED(u);
}

template <typename T>
void DistTensorBase<T>::UpdateImagPart(const Location &loc, BASE(T) u) {
  NOT_USED(loc);
  NOT_USED(u);
}

template <typename T>
void DistTensorBase<T>::SetRealPartOfDiagonal(const DistTensor<BASE(T)> &d,
                                              Int offset) {
  NOT_USED(d);
  NOT_USED(offset);
}

template <typename T>
void DistTensorBase<T>::SetImagPartOfDiagonal(const DistTensor<BASE(T)> &d,
                                              Int offset) {
  NOT_USED(d);
  NOT_USED(offset);
}

template <typename T>
void DistTensorBase<T>::SetAlignmentsAndResize(
    const std::vector<Unsigned> &modeAligns, const ObjShape &shape) {
#ifndef RELEASE
  CallStackEntry cse("DistTensor::SetAlignmentsAndResize");
#endif
  if (!Viewing()) {
    modeAlignments_ = modeAligns;
    SetShifts();
  }
  ResizeTo(shape);
}

template <typename T>
void DistTensorBase<T>::ForceAlignmentsAndResize(
    const std::vector<Unsigned> &modeAligns, const ObjShape &shape) {
#ifndef RELEASE
  CallStackEntry cse("DistTensor::ForceAlignmentsAndResize");
#endif
  SetAlignmentsAndResize(modeAligns, shape);
  if (AnyElemwiseNotEqual(modeAlignments_, modeAligns))
    LogicError("Could not set alignments");
}

template <typename T>
void DistTensorBase<T>::SetModeAlignmentAndResize(Mode mode, Unsigned mo
deAlign,
                                                  const ObjShape &shape) {
#ifndef RELEASE
  CallStackEntry cse("DistTensor::SetModeAlignmentAndResize");
  const Unsigned order = Order();
  if (mode < 0 || mode >= order)
    LogicError("0 <= mode < object order must be true");
#endif
  if (!Viewing() && !ConstrainedModeAlignment(mode)) {
    modeAlignments_[mode] = modeAlign;
    SetModeShift(mode);
  }
  ResizeTo(shape);
}

template <typename T>
void DistTensorBase<T>::ForceModeAlignmentAndResize(Mode mode,
                                                    Unsigned modeAlign,
                                                    const ObjShape &shape) {
#ifndef RELEASE
  CallStackEntry cse("DistTensor::ForceColAlignmentAndResize");
  const Unsigned order = Order();
  if (mode < 0 || mode >= order)
    LogicError("0 <= mode < object order must be true");
#endif
  SetModeAlignmentAndResize(mode, modeAlign, shape);
  if (modeAlignments_[mode] != modeAlign)
    LogicError("Could not set mode alignment");
}

template <typename T> void DistTensorBase<T>::FreeAlignments() {
  Unsigned i;
  const Unsigned order = Order();
  for (i = 0; i < order; i++)
    constrainedModeAlignments_[i] = false;
}

// TODO: Figure out how to extend this
template <typename T> void DistTensorBase<T>::MakeConsistent() {
  /*
  #ifndef RELEASE
      CallStackEntry cse("DistTensor::MakeConsistent");
  #endif
      const rote::Grid& g = Grid();
      const Int root = g.VCToViewingMap(0);
      Int message[7];
      if( g.ViewingRank() == root )
      {
          message[0] = viewType_;
          message[1] = height_;
          message[2] = width_;
          message[3] = constrainedColAlignment_;
          message[4] = constrainedRowAlignment_;
          message[5] = colAlignment_;
          message[6] = rowAlignment_;
      }
      mpi::Broadcast( message, 7, root, g.ViewingComm() );
      const ViewType newViewType = static_cast<ViewType>(message[0]);
      const Int newHeight = message[1];
      const Int newWidth = message[2];
      const bool newConstrainedCol = message[3];
      const bool newConstrainedRow = message[4];
      const Int newColAlignment = message[5];
      const Int newRowAlignment = message[6];
      if( !Participating() )
      {
          viewType_ = newViewType;
          height_ = newHeight;
          width_ = newWidth;
          constrainedColAlignment_ = newConstrainedCol;
          constrainedRowAlignment_ = newConstrainedRow;
          colAlignment_ = newColAlignment;
          rowAlignment_ = newRowAlignment;
          colShift_ = 0;
          rowShift_ = 0;
      }
  #ifndef RELEASE
      else
      {
          if( viewType_ != newViewType )
              LogicError("Inconsistent ViewType");
          if( height_ != newHeight )
              LogicError("Inconsistent height");
          if( width_ != newWidth )
              LogicError("Inconsistent width");
          if( constrainedColAlignment_ != newConstrainedCol ||
              colAlignment_ != newColAlignment )
              LogicError("Inconsistent column constraint");
          if( constrainedRowAlignment_ != newConstrainedRow ||
              rowAlignment_ != newRowAlignment )
              LogicError("Inconsistent row constraint");
      }
  #endif
  */
}

template <typename T>
void DistTensorBase<T>::SetLocalRealPart(const Location &loc, BASE(T) alpha) {
  tensor_.SetRealPart(loc, alpha);
}

template <typename T>
void DistTensorBase<T>::SetLocalImagPart(const Location &loc, BASE(T) alpha) {
  tensor_.SetImagPart(loc, alpha);
}

template <typename T>
void DistTensorBase<T>::UpdateLocalRealPart(const Location &loc,
                                            BASE(T) alpha) {
  tensor_.UpdateRealPart(loc, alpha);
}

template <typename T>
void DistTensorBase<T>::UpdateLocalImagPart(const Location &loc,
                                            BASE(T) alpha) {
  tensor_.UpdateImagPart(loc, alpha);
}

// TODO: Figure out participating logic
template <typename T> void DistTensorBase<T>::SetShifts() {

  const Unsigned order = Order();
  if (Participating()) {
    for (Unsigned i = 0; i < order; i++)
      modeShifts_[i] = Shift(ModeRank(i), modeAlignments_[i], ModeStride(i));
  } else {
    for (Unsigned i = 0; i < order; i++)
      modeShifts_[i] = 0;
  }
}

template <typename T> void DistTensorBase<T>::SetModeShift(Mode mode) {
  const Unsigned order = Order();
  if (mode < 0 || mode >= order)
    LogicError("0 < mode <= object order must be true");

  if (Participating())
    modeShifts_[mode] =
        Shift(ModeRank(mode), modeAlignments_[mode], ModeStride(mode));
  else
    modeShifts_[mode] = 0;
}

template <typename T> void DistTensorBase<T>::SetGrid(const rote::Grid &grid) {
  Empty();
  grid_ = &grid;
  SetShifts();
}

// TODO: Figure out how to clear grid and gridView
template <typename T> void DistTensorBase<T>::Empty() {
  std::fill(shape_.begin(), shape_.end(), 0);

  std::fill(modeAlignments_.begin(), modeAlignments_.end(), 0);
  // NOTE: C++ complains if I fill with 'false' for the boolean vector
  std::fill(constrainedModeAlignments_.begin(),
            constrainedModeAlignments_.end(), false);
  std::fill(modeShifts_.begin(), modeShifts_.end(), 0);

  tensor_.Empty();

  viewType_ = OWNER;
}

// TODO: Figure out if this is fully correct
template <typename T> void DistTensorBase<T>::EmptyData() {
  std::fill(shape_.begin(), shape_.end(), 0);

  tensor_.Empty();
  viewType_ = OWNER;
}

template <typename T>
void DistTensorBase<T>::SetLocal(const Location &loc, T alpha) {
  tensor_.Set(loc, alpha);
}

template <typename T>
void DistTensorBase<T>::UpdateLocal(const Location &loc, T alpha) {
  tensor_.Update(loc, alpha);
}

template <typename T>
void DistTensorBase<T>::SetLocalPermutation(const Permutation &perm) {
#ifndef RELEASE
  CallStackEntry cse("DistTensor::SetLocalPermutation");
#endif
  Permutation permOldToNew = localPerm_.PermutationTo(perm);
  tensor_.ResizeTo(PermuteVector(tensor_.Shape(), permOldToNew));
  localPerm_ = perm;
}

template <typename T> void DistTensorBase<T>::SetDefaultPermutation() {
#ifndef RELEASE
  CallStackEntry cse("DistTensor::SetDefaultPermutation");
#endif
  Permutation defaultPerm(Order());
  localPerm_ = defaultPerm;
}

#define FULL(T) template class DistTensorBase<T>;

FULL(Int)
#ifndef DISABLE_FLOAT
FULL(float)
#endif
FULL(double)

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>)
#endif
FULL(std::complex<double>)
#endif

} // namespace rote

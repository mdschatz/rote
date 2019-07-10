#pragma once
#ifndef ROTE_CONV_MACROS
#define ROTE_CONV_MACROS

#define LOCKEDPART2(A, partA, B, partB, bs, ...)                                      \
  const rote::Grid& A##_g = A.Grid();                                                 \
  const rote::Grid& B##_g = B.Grid();                                                 \
  TensorDistribution A_dist = A.TensorDist();                                         \
  TensorDistribution B_dist = B.TensorDist();                                         \
  DistTensor<T> A##_T(A_dist, A##_g), A##_B(A_dist, A##_g), A##_0(A_dist, A##_g), A##_1(A_dist, A##_g), A##_2(A_dist, A##_g); \
  DistTensor<T> B##_T(B_dist, B##_g), B##_B(B_dist, B##_g), B##_0(B_dist, A##_g), B##_1(B_dist, B##_g), B##_2(B_dist, B##_g); \
  LockedPartitionDown(A, A##_T, A##_B, partA, 0);                                     \
  LockedPartitionDown(B, B##_T, B##_B, partB, 0);                                     \
  while (A##_T.Dimension(partA) < A.Dimension(partA)) {                               \
    LockedRepartitionDown(A##_T, A##_0, A##_1, A##_B, A##_2, partA, bs);              \
    LockedRepartitionDown(B##_T, B##_0, B##_1, B##_B, B##_2, partB, bs);              \
    __VA_ARGS__                                                                       \
    SlideLockedPartitionDown(A##_T, A##_0, A##_1, A##_B, A##_2, partA);               \
    SlideLockedPartitionDown(B##_T, B##_0, B##_1, B##_B, B##_2, partB);               \
  }

#define LOCKEDPART(A, partA, bs, ...)                                                 \
  const rote::Grid& A##_g = A.Grid();                                                 \
  TensorDistribution A_dist = A.TensorDist();                                         \
  DistTensor<T> A##_T(A_dist, A##_g), A##_B(A_dist, A##_g), A##_0(A_dist, A##_g), A##_1(A_dist, A##_g), A##_2(A_dist, A##_g); \
  LockedPartitionDown(A, A##_T, A##_B, partA, 0);                                           \
  while (A##_T.Dimension(partA) < A.Dimension(partA)) {                               \
    LockedRepartitionDown(A##_T, A##_0, A##_1, A##_B, A##_2, partA, bs);                    \
    __VA_ARGS__                                                                       \
    SlideLockedPartitionDown(A##_T, A##_0, A##_1, A##_B, A##_2, partA);                     \
  }

#define PART2(A, partA, B, partB, bs, ...)                                            \
  const rote::Grid& A##_g = A.Grid();                                                 \
  const rote::Grid& B##_g = B.Grid();                                                 \
  TensorDistribution A_dist = A.TensorDist();                                         \
  TensorDistribution B_dist = B.TensorDist();                                         \
  DistTensor<T> A##_T(A_dist, A##_g), A##_B(A_dist, A##_g), A##_0(A_dist, A##_g), A##_1(A_dist, A##_g), A##_2(A_dist, A##_g); \
  DistTensor<T> B##_T(B_dist, B##_g), B##_B(B_dist, B##_g), B##_0(B_dist, A##_g), B##_1(B_dist, B##_g), B##_2(B_dist, B##_g); \
  PartitionDown(A, A##_T, A##_B, partA, 0);                                           \
  PartitionDown(B, B##_T, B##_B, partB, 0);                                           \
  while (A##_T.Dimension(partA) < A.Dimension(partA)) {                               \
    RepartitionDown(A##_T, A##_0, A##_1, A##_B, A##_2, partA, bs);                    \
    RepartitionDown(B##_T, B##_0, B##_1, B##_B, B##_2, partB, bs);                    \
    __VA_ARGS__                                                                       \
    SlidePartitionDown(A##_T, A##_0, A##_1, A##_B, A##_2, partA);                     \
    SlidePartitionDown(B##_T, B##_0, B##_1, B##_B, B##_2, partB);                     \
  }

#define PARTLOCKEDPART(A, partA, B, partB, bs, ...)                                   \
  const rote::Grid& A##_g = A.Grid();                                                 \
  const rote::Grid& B##_g = B.Grid();                                                 \
  TensorDistribution A_dist = A.TensorDist();                                         \
  TensorDistribution B_dist = B.TensorDist();                                         \
  DistTensor<T> A##_T(A_dist, A##_g), A##_B(A_dist, A##_g), A##_0(A_dist, A##_g), A##_1(A_dist, A##_g), A##_2(A_dist, A##_g); \
  DistTensor<T> B##_T(B_dist, B##_g), B##_B(B_dist, B##_g), B##_0(B_dist, A##_g), B##_1(B_dist, B##_g), B##_2(B_dist, B##_g); \
  PartitionDown(A, A##_T, A##_B, partA, 0);                                           \
  LockedPartitionDown(B, B##_T, B##_B, partB, 0);                                     \
  while (A##_T.Dimension(partA) < A.Dimension(partA)) {                               \
    RepartitionDown(A##_T, A##_0, A##_1, A##_B, A##_2, partA, bs);                    \
    LockedRepartitionDown(B##_T, B##_0, B##_1, B##_B, B##_2, partB, bs);              \
    __VA_ARGS__                                                                       \
    SlidePartitionDown(A##_T, A##_0, A##_1, A##_B, A##_2, partA);                     \
    SlideLockedPartitionDown(B##_T, B##_0, B##_1, B##_B, B##_2, partB);               \
  }

#define PARTLOCKEDHALOPART(A, partA, B, partB, halo, bs, ...)                                   \
  const rote::Grid& A##_g = A.Grid();                                                 \
  const rote::Grid& B##_g = B.Grid();                                                 \
  TensorDistribution A_dist = A.TensorDist();                                         \
  TensorDistribution B_dist = B.TensorDist();                                         \
  DistTensor<T> A##_T(A_dist, A##_g), A##_B(A_dist, A##_g), A##_0(A_dist, A##_g), A##_1(A_dist, A##_g), A##_2(A_dist, A##_g); \
  DistTensor<T> B##_T(B_dist, B##_g), B##_B(B_dist, B##_g), B##_0(B_dist, A##_g), B##_1(B_dist, B##_g), B##_2(B_dist, B##_g); \
  PartitionDown(A, A##_T, A##_B, partA, 0);                                           \
  LockedPartitionDown(B, B##_T, B##_B, partB, 0);                                     \
  while (A##_T.Dimension(partA) < A.Dimension(partA)) {                               \
    RepartitionDown(A##_T, A##_0, A##_1, A##_B, A##_2, partA, bs);                    \
    LockedRepartitionDown(B##_T, B##_0, B##_1, B##_B, B##_2, partB, bs);              \
    __VA_ARGS__                                                                       \
    SlidePartitionDown(A##_T, A##_0, A##_1, A##_B, A##_2, partA);                     \
    SlideLockedPartitionDown(B##_T, B##_0, B##_1, B##_B, B##_2, partB);               \
  }

#endif // ifndef ROTE_CONV_MACROS

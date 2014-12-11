#include "tensormental.hpp"
using namespace tmen;

void TestPackToGemm(const ObjShape& shapeA, const std::vector<Unsigned>& stridesA, const ObjShape& shapeB, const std::vector<Unsigned>& stridesB, const std::vector<Unsigned>& packStrides) {
    PrintVector(shapeA, "shapeA");
    PrintVector(stridesA, "stridesA");
    Tensor<double> A(shapeA, stridesA, 1, 0); //1 and 0 are dummies so that my constructor is not overloaded
    Tensor<double> B(shapeB, stridesB, 1, 0);
    MakeUniform(A);

    const Unsigned order = A.Order();

    PackData data;
    data.loopShape = B.Shape();
    data.dstBufStrides = B.Strides();
    data.srcBufStrides = packStrides;

    const std::vector<Unsigned> zeros(order, 0);
    const std::vector<Unsigned> ones(order, 1);
    data.loopStarts = zeros;
    data.loopIncs = ones;

    Print(A, "A");
    PackCommHelper(data, order - 1, A.LockedBuffer(), B.Buffer());
    Print(B, "B");
}

int main( int argc, char* argv[] ) {
    Unsigned I = 8;
    Unsigned J = 16;
    Unsigned K = 32;

     Unsigned mr = 2;
     Unsigned kc = 4;
     Unsigned mc = 4;
    // Unsigned nr = 4;
    // Unsigned nc = 8;

//    Unsigned mr = 4;
//    Unsigned mc = 256;
//    Unsigned kc = 384;
//    Unsigned nr = 8;
//    Unsigned nc = 4096;
    // Unsigned k = ;


    Unsigned order = 5;

    // Source Tensor for Data
    ObjShape tenShapeA(order);
    tenShapeA[0] = I;
    tenShapeA[1] = J;
    tenShapeA[2] = K;
    tenShapeA[3] = 1;
    tenShapeA[4] = 1;

    // "Field-Major" order matrix
    ObjShape tenShapeB(order);
    tenShapeB[0] = mr;
    tenShapeB[1] = kc;
    tenShapeB[2] = mc/mr;
    tenShapeB[3] = (I*J)/mc;
    tenShapeB[4] = K/kc;

    //Make sure the strides take into account the previous mode information

    // This is column major ordered in mode 0
    std::vector<Unsigned> stridesA(order);
    stridesA[0] = 1;
    stridesA[1] = I;
    stridesA[2] = I*J;
    stridesA[3] = I*J*K;
    stridesA[4] = I*J*K;

    // A "Field-Major" stride given proper n and k
    std::vector<Unsigned> stridesB(order);
    stridesB[0] = 1;
    stridesB[1] = mr;
    stridesB[2] = mr * kc;
    stridesB[3] = mc * kc;
    stridesB[4] = (I*J) * kc;

    std::vector<Unsigned> packStrides(order);
    packStrides[0] = 1;
    packStrides[1] = I*J;
    packStrides[2] = mr;
    packStrides[3] = mc;
    packStrides[4] = I*J*kc;

    TestPackToGemm(tenShapeA, stridesA, tenShapeB, stridesB, packStrides);

}

/* c++ lambdatwist.cpp -I lambdatwist-p3p -O3 -shared -o _lambdatwist.so */
/* g++ lambdatwist.cpp -I lambdatwist-p3p -O3 -shared -std=c++14 -o _lambdatwist.so */
#include <cmath>
#include <utils/cvl/matrix.h>
#include "p4p.h"
#include <lambdatwist/lambdatwist.p3p.h>

using namespace cvl;

extern "C" int p3p(double objectPoints[9], double imagePoints[6], double rmats[36], double tvecs[12]) {
    Vector3<double> y1{imagePoints[0], imagePoints[1], 1};
    Vector3<double> y2{imagePoints[2], imagePoints[3], 1};
    Vector3<double> y3{imagePoints[4], imagePoints[5], 1};
    Vector3<double> x1{&objectPoints[0], true};
    Vector3<double> x2{&objectPoints[3], true};
    Vector3<double> x3{&objectPoints[6], true};
    Vector<cvl::Matrix<double,3,3>,4> Rs;
    Vector<Vector3<double>,4> Ts;
    int result = p3p_lambdatwist(y1, y2, y3, x1, x2, x3, Rs, Ts);
    for(int i=0; i<4; i++) {
        memcpy(&rmats[9 * i], Rs[i].data(), 9 * sizeof(double));
        memcpy(&tvecs[3 * i], Ts[i].data(), 3 * sizeof(double));
    }
    return result;
}

// extern "C" int p4p(double objectPoints[12], double imagePoints[8], double rmats[36], double tvecs[12]) {
//     std::vector<Vector3D> xs = {
//         Vector3D(objectPoints[0],objectPoints[1],objectPoints[2]),
//         Vector3D(objectPoints[3],objectPoints[4],objectPoints[5]),
//         Vector3D(objectPoints[6],objectPoints[7],objectPoints[8]),
//         Vector3D(objectPoints[9],objectPoints[10],objectPoints[11])
//     };

//     std::vector<Vector2D> yns = {
//         Vector2D(imagePoints[0], imagePoints[1]),
//         Vector2D(imagePoints[2], imagePoints[3]),
//         Vector2D(imagePoints[4], imagePoints[5]),
//         Vector2D(imagePoints[6], imagePoints[7])
//     };

//     // Indices representing the correspondence between 3D and 2D points
//     Vector4<uint> indexes(0, 1, 2, 3);

//     // Call the p4p function
//     PoseD result = p4p(xs, yns, indexes);

//     // // Display the result
//     // std::cout << "Rotation Matrix:\n" << result.rotation() << "\n";
//     // std::cout << "Translation Vector:\n" << result.translation() << "\n";

//     return 1;
// }

extern "C" int p4p(double objectPoints[12], double imagePoints[8], double rmats[36], double tvecs[12]) {
    Vector3<double> y1{imagePoints[0], imagePoints[1], 1};
    Vector3<double> y2{imagePoints[2], imagePoints[3], 1};
    Vector3<double> y3{imagePoints[4], imagePoints[5], 1};
    Vector2<double> y4{imagePoints[6], imagePoints[7]};
    Vector3<double> x1{&objectPoints[0], true};
    Vector3<double> x2{&objectPoints[3], true};
    Vector3<double> x3{&objectPoints[6], true};
    Vector3<double> x4{&objectPoints[9], true};
    Vector<cvl::Matrix<double,3,3>,4> Rs;
    Vector<Vector3<double>,4> Ts;
    int result = p3p_lambdatwist(y1, y2, y3, x1, x2, x3, Rs, Ts);

    // pick the minimum, whatever it is
    PoseD P=PoseD(); // identity
    double e0=std::numeric_limits<double>::max();


    for(int i=0; i<result; ++i)
    {
        // the lambdatwist rotations have a problem some Rs not quite beeing rotations... ???
        // this is extremely uncommon, except when you have very particular kinds of noise,
        // this gets caught by the benchmark somehow?
        // it never occurs for the correct solution, only ever for possible alternatives.
        // investigate numerical problem later...
        Vector4<double> q=getRotationQuaternion(Rs[i]);
        q.normalize();
        PoseD tmp(q,Ts[i]);
        if(!tmp.isnormal()) continue;

        Vector3d xr=tmp*x4;
        if(xr[2]<0) continue;
        double e=(xr.dehom()-y4).squaredNorm();
        if (std::isnan(e)) continue;
        if (e<e0){
            P=tmp;
            e0=e;
        }
    }

    cvl::Matrix<double,3,3> Ri=getRotationMatrix(P.q);
    Vector3<double> Ti = P.t;

    for(int i=0; i<1; i++) {
        memcpy(&rmats[9 * i], &Ri[i], 9 * sizeof(double));
        memcpy(&tvecs[3 * i], &Ti[i], 3 * sizeof(double));
    }
    return result;
}
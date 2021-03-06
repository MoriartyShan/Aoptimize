//
// Created by moriarty on 10/17/20.
//

#ifndef BOARDDETECT_POINTSAXISDISTANCE_H
#define BOARDDETECT_POINTSAXISDISTANCE_H
#include <opencv2/opencv.hpp>
#include <ceres/rotation.h>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include "common.h"
#include "../Common/camera.h"

class PointAxisDistance {
private:
  const Pixel2f _point1;
  const Pixel2f _point2;
  const Matrix33 _cameraK;
  const scalar _distance;
  const int _axis;//0-x, 1-z
public:
  static ceres::CostFunction* Create(
      const Pixel2f& point1,
      const Pixel2f& point2,
      const Matrix33& cameraK,
      const scalar distance,
      const int axis) {
    return (new ceres::AutoDiffCostFunction<PointAxisDistance, 2, 3, 1>(
        new PointAxisDistance(point1, point2, cameraK, distance, axis)));
  }

  PointAxisDistance(
      const Pixel2f& point1,
      const Pixel2f& point2,
      const Matrix33& cameraK,
      const scalar distance,
      const int axis) :
      _point1(point1), _point2(point2), _cameraK(cameraK),
      _distance(distance), _axis(axis) {};

  template <typename T>
  bool operator()(const T* const Cam_r_Car, const T* const height, T* residuals) const {
    T Cam_R_Car[9], cameraK[9], m_matrix[9];
    T point12d[2] = {(T)_point1.x, (T)_point1.y}, point22d[2] = {(T)_point2.x, (T)_point2.y};
    ceres::MatrixAdapter<T, 3, 1> m_Matrix(m_matrix), Cam_R_Car_M(Cam_R_Car);

    ceres::AngleAxisToRotationMatrix(Cam_r_Car, Cam_R_Car_M);
    InsertMatrixToPointer(_cameraK, cameraK);
    MatrixMulti(cameraK, Cam_R_Car, m_matrix);

    T p13d[3], p23d[3];
    Camera::GetPoint3d(m_matrix, point12d, *height, p13d);
    Camera::GetPoint3d(m_matrix, point22d, *height, p23d);

    if (_axis == 0l) {
      residuals[0] = (T)100 * (ceres::abs(p13d[0] - p23d[0]) - (T)_distance);
      residuals[1] = (T)0;
    } else if (_axis == 1l) {
      residuals[0] = ceres::abs(p13d[2] - p23d[2]) - (T)_distance;
      residuals[1] = (T)0;
    } else if (_axis == 2l) {
      residuals[0] = (p13d[0] - p23d[0])*(T)100;
      residuals[1] = (T)100 * (p13d[0] - p23d[0]) / (p13d[2] - p23d[2]);
    }else {
      std::cout << __FILE__ << ":" << __LINE__ << ":" << " What a axis";
      exit(0);
    }

    return true;
  };
};

#endif //BOARDDETECT_POINTSAXISDISTANCE_H

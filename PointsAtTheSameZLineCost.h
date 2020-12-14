//
// Created by moriarty on 10/17/20.
//

#ifndef BOARDDETECT_POINTSATTHESAMEZLINECOST_H
#define BOARDDETECT_POINTSATTHESAMEZLINECOST_H
// a line that all points' z value is different
#include "common.h"
#include "../Common/camera.h"

#include <vector>
#include <opencv2/opencv.hpp>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
class AxisLine {
private:
  std::vector<Pixel2f> _points;
  Matrix33 _cameraK;
  const int _axis; // -1: after calculate position, all x is the same; 1: after calculate position, all z is the same
public:
  static ceres::CostFunction* Create(
      const Matrix33& cameraK,
      const std::vector<Pixel2f> &points,
      const int axis) {
    return (new ceres::AutoDiffCostFunction<AxisLine, 1, 3, 1>(
        new AxisLine(cameraK, points, axis)));
  }

  AxisLine(
      const Matrix33& cameraK,
      const std::vector<Pixel2f> &points,
      const int axis) :
      _cameraK(cameraK), _points(points), _axis(axis) {
  };

  template <typename T>
  bool operator()(const T* const aim, T* residuals) const {
    return operator()(aim, aim + 3, residuals);
  };

  template <typename T>
  bool operator()(const T* const Cam_r_Car, const T* const height,  T* residuals) const {
    std::vector<std::array<T, 2>> psz(_points.size());
    T Cam_R_Car[9], cameraK[9], m_matrix[9];
    std::array<T, 2> average = { (T)0 };;
    ceres::MatrixAdapter<T, 3, 1> m_Matrix(m_matrix), Cam_R_Car_M(Cam_R_Car);

    ceres::AngleAxisToRotationMatrix(Cam_r_Car, Cam_R_Car_M);
    InsertMatrixToPointer(_cameraK, cameraK);
    MatrixMulti(cameraK, Cam_R_Car, m_matrix);

    CHECK(_axis != 0);
    const int main_idx = _axis < 0 ? 0 : 1;

    for (int i = 0; i < _points.size(); i++) {
      T uv[2] = {(T)_points[i].x, (T)_points[i].y};
      T p3d[3];
      Camera::GetPoint3d(m_matrix, uv, *height, p3d);
      psz[i][1] = p3d[2];
      psz[i][0] = p3d[0];
      average[0] += psz[i][0];
      average[1] += psz[i][1];
    }
    average[0] /= (T)_points.size();
    average[1] /= (T)_points.size();
    auto dev = GetDeviation(psz, average);
    residuals[0] = ceres::abs(ceres::sqrt(dev[main_idx]) / average[main_idx]);
    return true;
  };
};
#endif //BOARDDETECT_POINTSATTHESAMEZLINECOST_H

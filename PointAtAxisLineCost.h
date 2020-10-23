//
// Created by moriarty on 10/17/20.
//

#ifndef BOARDDETECT_POINTATAXISLINECOST_H
#define BOARDDETECT_POINTATAXISLINECOST_H
#include <opencv2/opencv.hpp>
#include <ceres/rotation.h>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include "common.h"
#include "../Common/camera.h"

class PointAtAxisLine {
private:
  const cv::Point2d _point;
  const cv::Matx33d _cameraK;
  const int _axis;//0-x, 1-z
public:
  static ceres::CostFunction* Create(
      const cv::Point2d& point, const cv::Matx33d& cameraK, const int axis) {
    return (new ceres::AutoDiffCostFunction<PointAtAxisLine, 2, 3, 1, 1>(
        new PointAtAxisLine(point, cameraK, axis)));
  }
  static ceres::CostFunction* CreateCompact(
      const cv::Point2d& point, const cv::Matx33d& cameraK, const int axis) {
    return (new ceres::AutoDiffCostFunction<PointAtAxisLine, 2, 4, 1>(
        new PointAtAxisLine(point, cameraK, axis)));
  }

  PointAtAxisLine(
      const cv::Point2d& point, const cv::Matx33d& cameraK, const int axis) :
      _point(point), _cameraK(cameraK), _axis(axis) {
//    LOG(ERROR) << "points = " << point;
  };
  template <typename T>
  bool operator()(const T* const aim, const T* const value, T* residuals) const {
    return operator()(aim, aim + 3, value, residuals);
  }
  template <typename T>
  bool operator()(const T* const Cam_r_Car, const T* const height, const T* const value, T* residuals) const {
    T Cam_R_Car[9], cameraK[9], m_matrix[9];
    T point2d[2] = {(T)_point.x, (T)_point.y};
    ceres::MatrixAdapter<T, 3, 1> m_Matrix(m_matrix), Cam_R_Car_M(Cam_R_Car);

    ceres::AngleAxisToRotationMatrix(Cam_r_Car, Cam_R_Car_M);
    InsertMatrixToPointer(_cameraK, cameraK);
    MatrixMulti(cameraK, Cam_R_Car, m_matrix);

    T p3d[3];
    Camera::GetPoint3d(m_matrix, point2d, *height, p3d);
    T r;
    if (_axis == 0) {
      r = p3d[0] - *value;
    } else if (_axis == 1){
      r = (p3d[2] - *value);
    } else {
      std::cout << __FILE__ << ":" << __LINE__ << ":" << " What a axis";
      exit(0);
    }
    residuals[0] = r;

    residuals[1] = (T)0;
    if (p3d[2] < (T)0) {
      residuals[1] = -p3d[2];
    }
//    LOG(ERROR) << "output residual " << p3d[0] << "," << p3d[1] << "," << p3d[2] << "," << r << ","
//    << *value << "," << residuals[0] << "," << residuals[1]; // << "," << -p3d[2] << "," << ceres::exp(-p3d[2]);

    return true;
  };
};



#endif //BOARDDETECT_POINTATAXISLINECOST_H

//
// Created by moriarty on 11/18/20.
//

#ifndef BOARDDETECT_POINTDISTANCECOST_H
#define BOARDDETECT_POINTDISTANCECOST_H
#include <ceres/rotation.h>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

class PointDistance {
private:
  const float _distance_2;
  const cv::Point2d _point1;
  const cv::Point2d _point2;
  const cv::Matx33d _cameraK;
public:
  static ceres::CostFunction* Create(
      const float distance,
      const cv::Point2d& point1,
      const cv::Point2d& point2,
      const cv::Matx33d& cameraK) {
    return (new ceres::AutoDiffCostFunction<PointDistance, 2, 3, 1>(
        new PointDistance(distance, point1, point2, cameraK)));
  }

  PointDistance(
      const float distance,
      const cv::Point2d& point1,
      const cv::Point2d& point2,
      const cv::Matx33d& cameraK) :
      _distance_2(distance * distance), _point1(point1), _point2(point2), _cameraK(cameraK) {
    //    LOG(ERROR) << "points = " << point;
  };

  template <typename T>
  bool operator()(const T* const Cam_r_Car, const T* const height, T* residuals) const {
    T Cam_R_Car[9], cameraK[9], m_matrix[9];
    T point2d_1[2] = {(T)_point1.x, (T)_point1.y};
    T point2d_2[2] = {(T)_point2.x, (T)_point2.y};
    ceres::MatrixAdapter<T, 3, 1> m_Matrix(m_matrix), Cam_R_Car_M(Cam_R_Car);

    ceres::AngleAxisToRotationMatrix(Cam_r_Car, Cam_R_Car_M);
    InsertMatrixToPointer(_cameraK, cameraK);
    MatrixMulti(cameraK, Cam_R_Car, m_matrix);

    T p3d[6];
    Camera::GetPoint3d(m_matrix, point2d_1, *height, p3d);
    Camera::GetPoint3d(m_matrix, point2d_2, *height, p3d + 3);

    T d[3] = { p3d[0] - p3d[3],  p3d[1] - p3d[4], p3d[2] - p3d[5] };
    residuals[0] = ceres::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) - (T)std::sqrt(_distance_2);

    residuals[1] = (T)0;
    if (p3d[2] < (T)0) {
      residuals[1] += p3d[2];
    }
    if (p3d[5] < (T)0) {
      residuals[1] += p3d[5];
    }
    residuals[0] *= (T)0.0008;
    residuals[1] *= (T)0.0008;
    return true;
  };
#if 0
  template <>
  bool operator()<double>(const double* const Cam_r_Car, const double* const height, double* residuals) const {
    using T = double;
    T Cam_R_Car[9], cameraK[9], m_matrix[9];
    T point2d_1[2] = { (T)_point1.x, (T)_point1.y };
    T point2d_2[2] = { (T)_point2.x, (T)_point2.y };
    ceres::MatrixAdapter<T, 3, 1> m_Matrix(m_matrix), Cam_R_Car_M(Cam_R_Car);

    ceres::AngleAxisToRotationMatrix(Cam_r_Car, Cam_R_Car_M);
    InsertMatrixToPointer(_cameraK, cameraK);
    MatrixMulti(cameraK, Cam_R_Car, m_matrix);

    T p3d[6];
    Camera::GetPoint3d(m_matrix, point2d_1, *height, p3d);
    Camera::GetPoint3d(m_matrix, point2d_2, *height, p3d + 3);

    T d[3] = { p3d[0] - p3d[3],  p3d[1] - p3d[4], p3d[2] - p3d[5] };
    residuals[0] = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - (T)_distance_2;

    LOG(ERROR) << "[" << p3d[0] << "," << p3d[1] << "," << p3d[2] << "],[" << p3d[3]
               << "," << p3d[4] << "," << p3d[5] << "],[" << d[0] << "," << d[1] << "," << d[2] << "]" << residuals[0];
    residuals[1] = (T)0;
    if (p3d[2] < (T)0) {
      residuals[1] += p3d[2];
    }
    if (p3d[5] < (T)0) {
      residuals[1] += p3d[5];
    }

    return true;
  };
#endif
};


#endif //BOARDDETECT_POINTDISTANCECOST_H

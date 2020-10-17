//
// Created by moriarty on 10/17/20.
//

#ifndef BOARDDETECT_POINTSATTHESAMEZLINECOST_H
#define BOARDDETECT_POINTSATTHESAMEZLINECOST_H
// a line that all points' z value is different
#include <vector>
#include <opencv2/opencv.hpp>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "common.h"
#include "../Common/camera.h"

class ZLine {
private:
  std::vector<cv::Point2d> points;
  cv::Matx33d _cameraK;
public:
  static ceres::CostFunction* Create(const cv::Matx33d& cameraK, const std::vector<cv::Point2d> &_points) {
    return (new ceres::AutoDiffCostFunction<ZLine, 1, 4>(
        new ZLine(cameraK, _points)));
  }

  ZLine(const cv::Matx33d& cameraK, const std::vector<cv::Point2d> &_points) {
    _cameraK = cameraK;
    points = _points;
  };

  template <typename T>
  bool operator()(const T* const aim, T* residuals) const {
    std::vector<std::array<T, 2>> psz(points.size());
    T Cam_R_Car[9], cameraK[9], m_matrix[9];
    std::array<T, 2> average = { (T)0 };;
    ceres::MatrixAdapter<T, 3, 1> m_Matrix(m_matrix), Cam_R_Car_M(Cam_R_Car);

    ceres::AngleAxisToRotationMatrix(aim, Cam_R_Car_M);
    InsertMatrixToPointer(_cameraK, cameraK);
    MatrixMulti(cameraK, Cam_R_Car, m_matrix);

    for (int i = 0; i < points.size(); i++) {
      T uv[2] = {(T)points[i].x, (T)points[i].y};
      T p3d[3];
      Camera::GetPoint3d(m_matrix, uv, aim[3], p3d);
      psz[i][1] = p3d[2];
      psz[i][0] = p3d[0];
      average[0] += psz[i][0];
      average[1] += psz[i][1];
    }
    average[0] /= (T)points.size();
    average[1] /= (T)points.size();
    auto dev = GetDeviation(psz, average);
    residuals[0] = ceres::abs(ceres::sqrt(dev[0]) / average[0]);
    return true;
  };
};
#endif //BOARDDETECT_POINTSATTHESAMEZLINECOST_H

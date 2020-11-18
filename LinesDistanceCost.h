//
// Created by moriarty on 10/17/20.
//

#ifndef BOARDDETECT_LINESDISTANCECOST_H
#define BOARDDETECT_LINESDISTANCECOST_H
#include "common.h"

#include <ceres/rotation.h>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>

class ZLineDistance {
private:
  std::vector<cv::Point2d> _line1;
  std::vector<cv::Point2d> _line2;
  cv::Matx33d _cameraK;
  double _distance;
public:
  static ceres::CostFunction* Create(
      const cv::Matx33d& cameraK,
      const double distance,
      const std::vector<cv::Point2d>& line1,
      const std::vector<cv::Point2d>& line2) {
    return (new ceres::AutoDiffCostFunction<ZLineDistance, 1, 3, 1>(
        new ZLineDistance(cameraK, distance, line1, line2)));
  }

  ZLineDistance(
      const cv::Matx33d& cameraK,
      const double distance,
      const std::vector<cv::Point2d>& line1,
      const std::vector<cv::Point2d>& line2) {
    _cameraK = cameraK;
    _distance = distance;
    _line1 = line1;
    _line2 = line2;
  };

  template <typename T>
  bool operator()(const T* const cam_r_car, const T* const height, T* residuals) const {
    std::vector<std::array<T, 2>> line1(_line1.size());
    std::vector<std::array<T, 2>> line2(_line2.size());
    T Cam_R_Car[9], cameraK[9], m_matrix[9];
    ceres::MatrixAdapter<T, 3, 1> m_Matrix(m_matrix), Cam_R_Car_Matrix(Cam_R_Car);

    ceres::AngleAxisToRotationMatrix(cam_r_car, Cam_R_Car_Matrix);
    InsertMatrixToPointer(_cameraK, cameraK);
    MatrixMulti(cameraK, Cam_R_Car, m_matrix);

    std::array<T, 2> average = { (T)0 };

    for (int i = 0; i < _line1.size(); i++) {
      T uv[2] = { (T)_line1[i].x, (T)_line1[i].y };
      T p3d[3];

      Camera::GetPoint3d(m_matrix, uv, *height, p3d);
      line1[i][0] = p3d[0];
      line1[i][1] = p3d[2];
      average[0] += line1[i][0];
    }

    for (int i = 0; i < _line2.size(); i++) {
      T uv[2] = { (T)_line2[i].x, (T)_line2[i].y };
      T p3d[3];
      Camera::GetPoint3d<T>(m_matrix, uv, *height, p3d);
      line2[i][0] = p3d[0];
      line2[i][1] = p3d[2];

      average[1] += line2[i][0];
    }

    T max = (T)0, min = (T)100;
    for (int i = 0; i < line1.size(); i++) {
      for (int j = 0; j < line2.size(); j++) {
        T dist = (line1[i][0] - line2[j][0]);
        //LOG(ERROR) << dist;
        if (dist < (T)0) {
          dist = -dist;
        }
        if (max < dist) {
          max = dist;
        }
        if (min > dist) {
          min = dist;
        }
      }
    }
    max = max - (T)_distance;
    min = min - (T)_distance;
    residuals[0] = max * max + min * min;
    return true;
  };
};

#endif //BOARDDETECT_LINESDISTANCECOST_H

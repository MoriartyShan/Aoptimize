//
// Created by moriarty on 2020/10/15.
//

#include "../Common/common.h"
#include "../Common/camera.h"
#include <ceres/rotation.h>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

class RotationAngleAndtranslationCost2D {
private:
  const Pixel2f _point1, _point2;
public:
  static ceres::CostFunction* Create(
      const Pixel2f& point1, const Pixel2f& point2) {
    return (new ceres::AutoDiffCostFunction<RotationAngleAndtranslationCost2D, 2, 1, 2>(
        new RotationAngleAndtranslationCost2D(point1, point2)));
  }

  RotationAngleAndtranslationCost2D(
      const Pixel2f& point1, const Pixel2f& point2) :
      _point1(point1), _point2(point2) {};

  template <typename T>
  bool operator()(const T* const angle, const T* const translation, T* residuals) const {
    T sinv = ceres::sin(*angle);
    T cosv = ceres::cos(*angle);
    T x1 = (T)_point1.x, y1 = (T)_point1.y;
    T estimate_x2 = x1 * cosv + y1 * sinv + translation[0];
    T estimate_y2 = y1 * cosv - x1 * sinv + translation[1];
    residuals[0] = estimate_x2 - (T)_point2.x;
    residuals[1] = estimate_y2 - (T)_point2.y;
    return true;
  };
};



class Extrinsics {
public:
  double _angle21;//anti-clockwise rotate
  cv::Vec2d _t21;

  static Pixel2f transform_point(
      const Matrix22 &R21, const cv::Vec2d &t21,
      const Pixel2f &point) {
    Pixel2f out = R21 * point;
    return Pixel2f(out.x + t21(0), out.y + t21(1));
  }

  Matrix22 RotateMatrix() const {
    scalar sinv = std::sin(_angle21), cosv = std::cos(_angle21);
    return Matrix22(cosv, sinv, -sinv, cosv);
  }

  Pixel2f transform_point(const Pixel2f &p) {

    return transform_point(RotateMatrix(), _t21, p);
  }
  //init it with two points
  void init(
      const std::vector<Pixel2f>& points_1, const std::vector<Pixel2f>& points_2) {
    CHECK(points_1.size() == points_2.size())
        << "Points size should be equal:" << points_1.size() << "," << points_2.size();
    CHECK(points_1.size() >= 2)
        << "points size should be larger than 2(" << points_1.size() << ")";
    Pixel2f v1 = points_1[0] - points_1[1], v2 = points_2[0] - points_2[1];
    Pixel2f nv1 = v1 / cv::norm(v1), nv2 = v2 / cv::norm(v2);
    scalar dot = nv1.dot(nv2);
    scalar cross = nv1.x * nv2.y - nv1.y * nv1.x;
    scalar angle = std::acos(dot);
    if (cross > 0) {
      _angle21 = -angle;
    } else {
      _angle21 = angle;
    }
    Matrix22 R21 = RotateMatrix();
    Vec2 t21 = points_2[0] - R21 * points_1[0];
    _t21 = t21;
    LOG(ERROR) << v1 << "," << v2;
    LOG(ERROR) << "dot = " << dot;
    LOG(ERROR) << "angle= " << _angle21 << ", t = " << _t21;
    LOG(ERROR) << points_1[0] << "=>" << transform_point(points_1[0]) << ","
               << points_2[0];
  }
};


int main() {
  auto camera_1 = Camera::create("/home/moriarty/WindowsD/Projects/WDataSets/HKvisionCalib/extrinsics/1023/extrinsic.yaml");
  auto camera_2 = Camera::create("/home/moriarty/WindowsD/Projects/WDataSets/HKvisionCalib/extrinsics/1022/extrinsic.yaml");
  std::vector<Pixel2f> points_1 = {Pixel2f(543, 199),
                                       Pixel2f(550, 188),
                                       Pixel2f(136, 220),
                                       Pixel2f(214, 196),
                                       Pixel2f(529, 223),
                                       Pixel2f(404, 200),
                                       Pixel2f(385, 209),
                                       Pixel2f(355, 226),
                                       Pixel2f(326, 241),
                                       Pixel2f(275, 270)};

  std::vector<Pixel2f> points_2 = {Pixel2f(555, 436),
                                       Pixel2f(892, 438),
                                       Pixel2f(496, 266),
                                       Pixel2f(720, 271),
                                       Pixel2f(138, 415),
                                       Pixel2f(544, 340),
                                       Pixel2f(404, 339),
                                       Pixel2f(237, 337),
                                       Pixel2f(140, 336),
                                       Pixel2f(37, 334)};

  std::vector<Pixel2f> undist_points_1, undist_points_2;
  std::vector<Pixel2f> points_3d_2d_1, points_3d_2d_2;
  camera_1->UndistortPoints(points_1, undist_points_1);
  camera_2->UndistortPoints(points_2, undist_points_2);

  const int size = points_1.size();


  points_3d_2d_1.reserve(size);
  points_3d_2d_2.reserve(size);
  for (int i = 0; i < size; i++) {
    Point3f point;
    point = camera_1->GetPoint3d(undist_points_1[i]);
    points_3d_2d_1.emplace_back(point.x, point.z);
    point = camera_2->GetPoint3d(undist_points_2[i]);
    points_3d_2d_2.emplace_back(point.x, point.z);
  }

  Extrinsics extrinsics;
  extrinsics.init(points_3d_2d_1, points_3d_2d_2);
  ceres::Problem problem;
  const Matrix22 R21 = extrinsics.RotateMatrix();
  const cv::Vec2d t21 = extrinsics._t21;
  for (int i = 0; i < size; i++) {
    auto p = Extrinsics::transform_point(R21, t21, points_3d_2d_1[i]);
    LOG(ERROR) << "point[" << i << "]:" << points_3d_2d_1[i] << "=>" << p << "-"
               << points_3d_2d_2[i] << "=" << p - points_3d_2d_2[i];
    auto cost_func = RotationAngleAndtranslationCost2D::Create(
        points_3d_2d_1[i], points_3d_2d_2[i]);
    problem.AddResidualBlock(
        cost_func, nullptr, &extrinsics._angle21, extrinsics._t21.val);
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
//  options.logging_type = ceres::SILENT;
  //options.max_num_iterations = 1000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  LOG(ERROR) << summary.BriefReport();

  const Matrix22 R21_new = extrinsics.RotateMatrix();
  const cv::Vec2d t21_new = extrinsics._t21;
  for (int i = 0; i < size; i++) {
    auto p = Extrinsics::transform_point(R21_new, t21_new, points_3d_2d_1[i]);
    LOG(ERROR) << "point[" << i << "]:" << points_3d_2d_1[i] << "=>" << p << "-"
               << points_3d_2d_2[i] << "=" << p - points_3d_2d_2[i];
  }


  LOG(ERROR) << "angle = " << extrinsics._angle21 << " t = " << extrinsics._t21;

  return 0;
}
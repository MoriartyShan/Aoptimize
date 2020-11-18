#include <ceres/rotation.h>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <glog/logging.h>

#include "common.h"
#include "../Common/camera.h"

class Distance {
private:
  const float _distance_2;
  const cv::Point2d _point1;
  const cv::Point2d _point2;
  const cv::Matx33d _cameraK;
public:
  static ceres::CostFunction* Create(
    const float distance,
    const cv::Point2d& point1, const cv::Point2d& point2, const cv::Matx33d& cameraK) {
    return (new ceres::AutoDiffCostFunction<Distance, 2, 3, 1>(
      new Distance(distance, point1, point2, cameraK)));
  }

  Distance(
    const float distance,
    const cv::Point2d& point1, const cv::Point2d& point2, const cv::Matx33d& cameraK) :
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
    residuals[0] = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - (T)_distance_2;

    residuals[1] = (T)0;
    if (p3d[2] < (T)0) {
      residuals[1] += p3d[2];
    }
    if (p3d[5] < (T)0) {
      residuals[1] += p3d[5];
    }
/*
    residuals[1] = (T)0;
    if (p3d[2] < (T)0) {
      residuals[1] = -p3d[2];
    }*/
    //    LOG(ERROR) << "output residual " << p3d[0] << "," << p3d[1] << "," << p3d[2] << "," << r << ","
    //    << *value << "," << residuals[0] << "," << residuals[1]; // << "," << -p3d[2] << "," << ceres::exp(-p3d[2]);

    return true;
  };
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
    /*
        residuals[1] = (T)0;
        if (p3d[2] < (T)0) {
          residuals[1] = -p3d[2];
        }*/
        //    LOG(ERROR) << "output residual " << p3d[0] << "," << p3d[1] << "," << p3d[2] << "," << r << ","
        //    << *value << "," << residuals[0] << "," << residuals[1]; // << "," << -p3d[2] << "," << ceres::exp(-p3d[2]);

    return true;
  };
};


bool is_good(const std::vector<cv::Point3d> &p3d) {
  if (p3d[0].y < 3) {
    return false;
  }
  for (auto &p : p3d) {
    if (p.z < 0) {
      return false;
    }
  }

  if (p3d[0].x > 0) {
    return false;
  }

  if (p3d[1].x > 0) {
    return false;
  }

  if (p3d[2].x < 0) {
    return false;
  }

  if (p3d[3].x < 0) {
    return false;
  }
  return true;

}

void shrink_to_pi(cv::Vec3d &rvec) {
  double mod = cv::norm(rvec);
  rvec /= mod;
  while (mod > M_PI) {
    mod -= (2 * M_PI);
  }
  rvec *= mod;
  return;
}

void shrink_to_pi(double *vec) {
  cv::Vec3d rvec(vec[0], vec[1], vec[2]);
  shrink_to_pi(rvec);
  vec[0] = rvec(0);
  vec[1] = rvec(1);
  vec[2] = rvec(2);
  return;
}

bool optimize() {
  std::vector<cv::Point2d> points2d;
  const std::string path = "D:\\Projects\\WDataSets\\HKvisionCalib\\extrinsics\\test_1522\\";
  auto camera = Camera::create(path + "camera.yaml");

  std::vector<float> dist = { 38.4, 15.8, 39.178, 16 };
  cv::Matx33d cameraK = camera->Intrinsic();
  points2d.emplace_back(0, 719);
  points2d.emplace_back(470, 241);
  points2d.emplace_back(745, 235);
  points2d.emplace_back(1230, 669);
  ceres::Problem problem;
  cv::Vec4d r(rand(), rand(), rand(), 5);
  shrink_to_pi(r.val);
  LOG(ERROR) << "init v = " << r;
  //0.259139, -0.0341831, -0.00638832
  double *r_ptr = r.val, *h_ptr = r_ptr + 3;
  std::vector<Distance> cost_functions;

  for (size_t i = 0; i < points2d.size(); i++) {
    if (i == (points2d.size() - 1)) {
      cost_functions.emplace_back(dist[i], points2d[i], points2d[0], cameraK);
      auto cost_func = Distance::Create(dist[i], points2d[i], points2d[0], cameraK);
      problem.AddResidualBlock(cost_func, nullptr, r_ptr, h_ptr);
    }
    else {
      cost_functions.emplace_back(dist[i], points2d[i], points2d[i + 1], cameraK);
      auto cost_func = Distance::Create(dist[i], points2d[i], points2d[i + 1], cameraK);
      problem.AddResidualBlock(cost_func, nullptr, r_ptr, h_ptr);
    }

  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  //options.logging_type = log_type;
  //options.max_num_iterations = 1000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  LOG(ERROR) << summary.BriefReport();

  LOG(ERROR) << r;

  camera->set_Extrinsic(r_ptr);
  

  std::vector<cv::Point3d> points3d;
  for (size_t i = 0; i < points2d.size(); i++) {
    points3d.emplace_back(camera->GetPoint3d(points2d[i]));
  }

  //double residual[2];
  //for (auto &f : cost_functions) {
  //  f(r_ptr, h_ptr, residual);
  //}

  if (!is_good(points3d)) {
    return false;
  }
  for (size_t i = 0; i < points3d.size(); i++) {
    if (i != points3d.size() - 1) {
      LOG(ERROR) << points2d[i] << points3d[i] << cv::norm(points3d[i] - points3d[i + 1]) << "," << dist[i];
    } else {
      LOG(ERROR) << points2d[i] << points3d[i] << cv::norm(points3d[i] - points3d[0]) << "," << dist[i];
    }
    
  }
  
  camera->WriteToYaml(path + "extrinsic.yaml");
  LOG(ERROR) << "finished";
  return true;
}


int main() {
  srand(time(0));
  while (!optimize()) {
    LOG(ERROR) << "Keep optimize";
  }

  getchar();
  return 0;

}
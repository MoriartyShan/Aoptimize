//
// Created by moriarty on 2020/4/27.
//

#include <ceres/rotation.h>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include <array>
#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#define XP 3
#define YP 1
#define ZP 3

class Zerror {
private:
  std::vector<std::array<double, 3>> points;
public:
  static ceres::CostFunction* Create(std::vector<std::array<double, 3>> &_points) {
    return (new ceres::AutoDiffCostFunction<Zerror, 1, 3>(
      new Zerror(_points)));
  }

  Zerror(std::vector<std::array<double, 3>> &_points) {
    for (int i = 0; i < _points.size(); i++) {
      points.emplace_back(_points[i]);
    }
  };
  template <typename T>
  bool operator()(const T* const rvec, T* residuals) const {
    std::vector<T> ps(points.size());
    *residuals = (T)0;
    for (int i = 0; i < points.size(); i++) {
      T p[3] = {(T)points[i][0], (T)points[i][1], (T)points[i][2]};
      T np[3];
      ceres::AngleAxisRotatePoint(rvec, p, np);
      ps[i] = np[2];
    }

    for (int i = 0; i < ps.size(); i++) {
      for (int j = i + 1; j < ps.size(); j++) {
        T d = ps[i] - ps[j];
        if (d < (T)0) {
          d = -d;
        }
        residuals[0] += d;
      }
    }
    residuals[0] *= (T)ZP;
    return true;
  };
};

class Yerror {
private:
  std::vector<std::array<double, 3>> points;
public:
  static ceres::CostFunction* Create(std::vector<std::array<double, 3>> &_points) {
    return (new ceres::AutoDiffCostFunction<Yerror, 1, 3>(
      new Yerror(_points)));
  }

  Yerror(std::vector<std::array<double, 3>> &_points) {
    for (int i = 0; i < _points.size(); i++) {
      points.emplace_back(_points[i]);
    }
  };
  template <typename T>
  bool operator()(const T* const rvec, T* residuals) const {
    std::vector<T> ps(points.size());
    *residuals = (T)0;
    for (int i = 0; i < points.size(); i++) {
      T p[3] = {(T)points[i][0], (T)points[i][1], (T)points[i][2]};
      T np[3];
      ceres::AngleAxisRotatePoint(rvec, p, np);
      ps[i] = np[1];
    }
    for (int i = 0; i < ps.size(); i++) {
      for (int j = i + 1; j < ps.size(); j++) {
        T d = ps[i] - ps[j];
        if (d < (T)0) {
          d = -d;
        }
        residuals[0] += d;
      }
    }
    residuals[0] *= (T)YP;
    return true;
  };
};


class Xerror {
private:
  std::vector<std::array<double, 3>> points;
public:
  static ceres::CostFunction* Create(std::vector<std::array<double, 3>> &_points) {
    return (new ceres::AutoDiffCostFunction<Xerror, 1, 3>(
      new Xerror(_points)));
  }

  Xerror(std::vector<std::array<double, 3>> &_points) {
    for (int i = 0; i < _points.size(); i++) {
      points.emplace_back(_points[i]);
    }
  };
  template <typename T>
  bool operator()(const T* const rvec, T* residuals) const {
    std::vector<T> ps(points.size());
    *residuals = (T)0;
    for (int i = 0; i < points.size(); i++) {
      T p[3] = { (T)points[i][0], (T)points[i][1], (T)points[i][2] };
      T np[3];
      ceres::AngleAxisRotatePoint(rvec, p, np);
      ps[i] = np[0];
    }
    for (int i = 0; i < ps.size(); i++) {
      for (int j = i + 1; j < ps.size(); j++) {
        T d = ps[i] - ps[j];
        if (d < (T)0) {
          d = -d;
        }
        residuals[0] += d;
      }
    }
    residuals[0] *= (T)XP;
    return true;
  };
};




int main() {
  std::vector<std::array<double, 3>> ypoints, xpoints, zpoints;

  ceres::Problem problem;
  double r_vec[3] = {1, 0, 0};

  ypoints.emplace_back(std::array<double, 3>({-1.44952, 0.453195, 10.2696}));
  ypoints.emplace_back(std::array<double, 3>({-1.37075, 0.991858, 4.34676}));
  ypoints.emplace_back(std::array<double, 3>({1.82523, 0.977567, 4.43107}));
  ypoints.emplace_back(std::array<double, 3>({1.73377, 0.417538, 10.9695}));
  auto ycostfunc = Yerror::Create(ypoints);
  problem.AddResidualBlock(ycostfunc, new ceres::TrivialLoss(), r_vec);

  xpoints.emplace_back(std::array<double, 3>({-1.44952, 0.453195, 10.2696}));
  xpoints.emplace_back(std::array<double, 3>({-1.37075, 0.991858, 4.34676}));
  auto xcostfunc1 = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc1, new ceres::TrivialLoss(), r_vec);

  xpoints.clear();
  xpoints.emplace_back(std::array<double, 3>({1.82523, 0.977567, 4.43107}));
  xpoints.emplace_back(std::array<double, 3>({1.73377, 0.417538, 10.9695}));
  auto xcostfunc2 = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc2, new ceres::TrivialLoss(), r_vec);

  zpoints.emplace_back(std::array<double, 3>({-1.44952, 0.453195, 10.2696}));
  zpoints.emplace_back(std::array<double, 3>({1.73377, 0.417538, 10.9695}));
  auto zcostfunc1 = Zerror::Create(zpoints);
  problem.AddResidualBlock(zcostfunc1, new ceres::TrivialLoss(), r_vec);

  zpoints.clear();
  zpoints.emplace_back(std::array<double, 3>({-1.37075, 0.991858, 4.34676}));
  zpoints.emplace_back(std::array<double, 3>({1.82523, 0.977567, 4.43107}));
  auto zcostfunc2 = Zerror::Create(zpoints);
  problem.AddResidualBlock(zcostfunc2, new ceres::TrivialLoss(), r_vec);

  std::cout << "resudual number = " << problem.NumResidualBlocks() << "," << problem.NumResiduals() << std::endl;

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  LOG(ERROR) << "FindRotation:" << summary.BriefReport();
  LOG(ERROR) << "vector = " << r_vec[0] << "," << r_vec[1] << "," << r_vec[2];
  cv::Vec3d r(r_vec[0], r_vec[1], r_vec[2]);
  cv::Mat R;
  cv::Rodrigues(r, R);
  LOG(ERROR) << "Get R:\n" << R;

  
  for (int i = 0; i < ypoints.size(); i++) {
    cv::Matx31d p(ypoints[i][0], ypoints[i][1], ypoints[i][2]);
    LOG(ERROR) << p.t() << "-" << (R * p).t();
  }

  system("pause");
  return 0;
}
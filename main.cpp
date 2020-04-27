//
// Created by moriarty on 2020/4/27.
//

#include <ceres/rotation.h>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include <array>
#include <vector>
#include <Eigen/Core>

class Zerror {
private:
  std::vector<std::array<double, 3>> points;
public:
  static ceres::CostFunction* Create(std::vector<std::array<double, 3>> &_points) {
    return (new ceres::AutoDiffCostFunction<Zerror, 1, 3>(
      new Zerror(_points)));
  }

  Zerror(std::vector<std::array<double, 3>> &_points) {
    points = _points;
  };
  template <typename T>
  bool operator()(const T* const rvec, T* residuals) const {
    std::vector<std::array<T, 3>> ps(points.size());
    for (int i = 0; i < points.size(); i++) {
      T p[3] = {(T)points[i][0], (T)points[i][1], (T)points[i][2]};
      ceres::AngleAxisRotatePoint(rvec, p, ps[i].data());
    }
    for (int i = 1; i < ps.size(); i++) {
      T d = ps[i][2] - ps[0][2];
      *residuals += (d * d);
    }
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
    points = _points;
  };
  template <typename T>
  bool operator()(const T* const rvec, T* residuals) const {
    std::vector<std::array<T, 3>> ps(points.size());
    for (int i = 0; i < points.size(); i++) {
      T p[3] = {(T)points[i][0], (T)points[i][1], (T)points[i][2]};
      ceres::AngleAxisRotatePoint(rvec, p, ps[i].data());
    }
    for (int i = 1; i < ps.size(); i++) {
      T d = ps[i][1] - ps[0][1];
      *residuals += (d * d);
    }
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
    points = _points;
  };
  template <typename T>
  bool operator()(const T* const rvec, T* residuals) const {
    std::vector<std::array<T, 3>> ps(points.size());
    for (int i = 0; i < points.size(); i++) {
      T p[3] = {(T)points[i][0], (T)points[i][1], (T)points[i][2]};
      ceres::AngleAxisRotatePoint(rvec, p, ps[i].data());
    }
    for (int i = 1; i < ps.size(); i++) {
      T d = ps[i][0] - ps[0][0];
      *residuals += (d * d);
    }
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
  problem.AddResidualBlock(ycostfunc, nullptr, r_vec);

  xpoints.emplace_back(std::array<double, 3>({-1.44952, 0.453195, 10.2696}));
  xpoints.emplace_back(std::array<double, 3>({-1.37075, 0.991858, 4.34676}));
  auto xcostfunc1 = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc1, nullptr, r_vec);

  xpoints.clear();
  xpoints.emplace_back(std::array<double, 3>({1.82523, 0.977567, 4.43107}));
  xpoints.emplace_back(std::array<double, 3>({1.73377, 0.417538, 10.9695}));
  auto xcostfunc2 = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc2, nullptr, r_vec);

  zpoints.emplace_back(std::array<double, 3>({-1.44952, 0.453195, 10.2696}));
  zpoints.emplace_back(std::array<double, 3>({1.73377, 0.417538, 10.9695}));
  auto zcostfunc1 = Xerror::Create(zpoints);
  problem.AddResidualBlock(zcostfunc1, nullptr, r_vec);

  zpoints.clear();
  zpoints.emplace_back(std::array<double, 3>({-1.37075, 0.991858, 4.34676}));
  zpoints.emplace_back(std::array<double, 3>({1.82523, 0.977567, 4.43107}));
  auto zcostfunc2 = Xerror::Create(zpoints);
  problem.AddResidualBlock(zcostfunc2, nullptr, r_vec);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  LOG(ERROR) << "FindRotation:" << summary.BriefReport();
  LOG(ERROR) << "vector = " << r_vec[0] << "," << r_vec[1] << "," << r_vec[2];

  return 0;
}
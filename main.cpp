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

#define XP 5
#define YP 1
#define ZP 1

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
  std::vector<std::array<double, 3>> ypoints, xpoints, zpoints, all;

  ceres::Problem problem;
  double r_vec[3] = {1, 0, 0};
#if 1
  zpoints.emplace_back(std::array<double, 3>({ -2.22685, -3.00641, 14.3505 }));
  zpoints.emplace_back(std::array<double, 3>({ -2.34249, -1.73752, 14.9136 })); 
  zpoints.emplace_back(std::array<double, 3>({ 0.647695, -1.68376, 14.7771 }));
  zpoints.emplace_back(std::array<double, 3>({ 0.644386, -2.99821, 14.5515 }));
  zpoints.emplace_back(std::array<double, 3>({ 4.14688, -3.09886, 15.8304 }));
  zpoints.emplace_back(std::array<double, 3>({ 3.94199, -1.63197, 14.7113 }));
  zpoints.emplace_back(std::array<double, 3>({ 6.71896, -1.6101, 14.5443 }));
  zpoints.emplace_back(std::array<double, 3>({ 6.90355, -3.01945, 15.3327 }));
  auto zcostfunc1 = Zerror::Create(zpoints);
  problem.AddResidualBlock(zcostfunc1, new ceres::TrivialLoss(), r_vec);

  xpoints.emplace_back(std::array<double, 3>({ -2.22685, -3.00641, 14.3505 }));
  xpoints.emplace_back(std::array<double, 3>({ -2.34249, -1.73752, 14.9136 }));
  auto xcostfunc1 = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc1, new ceres::TrivialLoss(), r_vec);

  xpoints.clear();
  xpoints.emplace_back(std::array<double, 3>({ 0.647695, -1.68376, 14.7771 }));
  xpoints.emplace_back(std::array<double, 3>({ 0.644386, -2.99821, 14.5515 }));
  xcostfunc1 = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc1, new ceres::TrivialLoss(), r_vec);

  xpoints.clear();
  xpoints.emplace_back(std::array<double, 3>({ 4.14688, -3.09886, 15.8304 }));
  xpoints.emplace_back(std::array<double, 3>({ 3.94199, -1.63197, 14.7113 }));
  xcostfunc1 = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc1, new ceres::TrivialLoss(), r_vec);

  xpoints.clear();
  xpoints.emplace_back(std::array<double, 3>({ 6.71896, -1.6101, 14.5443 }));
  xpoints.emplace_back(std::array<double, 3>({ 6.90355, -3.01945, 15.3327 }));
  xcostfunc1 = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc1, new ceres::TrivialLoss(), r_vec);


  ypoints.emplace_back(std::array<double, 3>({ -2.34249, -1.73752, 14.9136 }));
  ypoints.emplace_back(std::array<double, 3>({ 0.647695, -1.68376, 14.7771 }));
  ypoints.emplace_back(std::array<double, 3>({ 3.94199, -1.63197, 14.7113 }));
  ypoints.emplace_back(std::array<double, 3>({ 6.71896, -1.6101, 14.5443 }));
  auto ycostfunc = Yerror::Create(ypoints);
  problem.AddResidualBlock(ycostfunc, new ceres::TrivialLoss(), r_vec);

  ypoints.clear();
  ypoints.emplace_back(std::array<double, 3>({ -2.22685, -3.00641, 14.3505 }));
  ypoints.emplace_back(std::array<double, 3>({ 0.644386, -2.99821, 14.5515 }));
  ypoints.emplace_back(std::array<double, 3>({ 4.14688, -3.09886, 15.8304 }));
  ypoints.emplace_back(std::array<double, 3>({ 6.90355, -3.01945, 15.3327 }));
  ycostfunc = Yerror::Create(ypoints);
  problem.AddResidualBlock(ycostfunc, new ceres::TrivialLoss(), r_vec);

  ///split two test
  ypoints.clear();
  ypoints.emplace_back(std::array<double, 3>({ 0.986761, 3.0441, 9.12497 }));
  ypoints.emplace_back(std::array<double, 3>({ 1.1901, 2.36408, 5.58576 }));
  ypoints.emplace_back(std::array<double, 3>({ 1.37157, 2.3098, 5.51177 }));
  ypoints.emplace_back(std::array<double, 3>({ 1.24182, 3.14053, 9.45639 }));
  ypoints.emplace_back(std::array<double, 3>({ -2.49757, 4.16068, 13.5848 }));
  ypoints.emplace_back(std::array<double, 3>({ -2.25602, 2.95318, 7.29003 }));
  ycostfunc = Yerror::Create(ypoints);
  problem.AddResidualBlock(ycostfunc, new ceres::TrivialLoss(), r_vec);

  xpoints.clear();
  xpoints.emplace_back(std::array<double, 3>({ 0.986761, 3.0441, 9.12497 }));
  xpoints.emplace_back(std::array<double, 3>({ 1.1901, 2.36408, 5.58576 }));
  xcostfunc1 = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc1, new ceres::TrivialLoss(), r_vec);

  xpoints.clear();
  xpoints.emplace_back(std::array<double, 3>({ 1.37157, 2.3098, 5.51177 }));
  xpoints.emplace_back(std::array<double, 3>({ 1.24182, 3.14053, 9.45639 }));
  auto xcostfunc2 = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc2, new ceres::TrivialLoss(), r_vec);

  xpoints.clear();
  xpoints.emplace_back(std::array<double, 3>({ -2.49757, 4.16068, 13.5848 }));
  xpoints.emplace_back(std::array<double, 3>({ -2.25602, 2.95318, 7.29003 }));
  auto xcostfunc3 = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc3, new ceres::TrivialLoss(), r_vec);

  zpoints.clear();
  zpoints.emplace_back(std::array<double, 3>({ 1.1901, 2.36408, 5.58576 }));
  zpoints.emplace_back(std::array<double, 3>({ 1.37157, 2.3098, 5.51177 }));
  zcostfunc1 = Zerror::Create(zpoints);
  problem.AddResidualBlock(zcostfunc1, new ceres::TrivialLoss(), r_vec);

  zpoints.clear();
  zpoints.emplace_back(std::array<double, 3>({ 0.986761, 3.0441, 9.12497 }));
  zpoints.emplace_back(std::array<double, 3>({ 1.24182, 3.14053, 9.45639 }));
  auto zcostfunc2 = Zerror::Create(zpoints);
  problem.AddResidualBlock(zcostfunc2, new ceres::TrivialLoss(), r_vec);
#endif
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


  all.emplace_back(std::array<double, 3>({ -2.22685, -3.00641, 14.3505 }));
  all.emplace_back(std::array<double, 3>({ -2.34249, -1.73752, 14.9136 }));
  all.emplace_back(std::array<double, 3>({ 0.647695, -1.68376, 14.7771 }));
  all.emplace_back(std::array<double, 3>({ 0.644386, -2.99821, 14.5515 }));
  all.emplace_back(std::array<double, 3>({ 4.14688, -3.09886, 15.8304 }));
  all.emplace_back(std::array<double, 3>({ 3.94199, -1.63197, 14.7113 }));
  all.emplace_back(std::array<double, 3>({ 6.71896, -1.6101, 14.5443 }));
  all.emplace_back(std::array<double, 3>({ 6.90355, -3.01945, 15.3327 }));
  all.emplace_back(std::array<double, 3>({ 0.986761, 3.0441, 9.12497 }));
  all.emplace_back(std::array<double, 3>({ 1.1901, 2.36408, 5.58576 }));
  all.emplace_back(std::array<double, 3>({ 1.37157, 2.3098, 5.51177 }));
  all.emplace_back(std::array<double, 3>({ 1.24182, 3.14053, 9.45639 }));
  all.emplace_back(std::array<double, 3>({ -2.49757, 4.16068, 13.5848 }));
  all.emplace_back(std::array<double, 3>({ -2.25602, 2.95318, 7.29003 }));
  all.emplace_back(std::array<double, 3>({ 1.01403, 2.10141, 5.04859 }));
  for (int i = 0; i < all.size(); i++) {
    cv::Matx31d p(all[i][0], all[i][1], all[i][2]);
    LOG(ERROR) << p.t() << "-" << (R * p).t();
  }

  system("pause");
  return 0;
}
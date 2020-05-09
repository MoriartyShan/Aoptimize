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

#define XP 7
#define YP 1
#define ZP 7

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
  ypoints.emplace_back(std::array<double, 3>({ 1.56694, 1.23576, 8.09722 }));
  ypoints.emplace_back(std::array<double, 3>({ 1.61549, 1.17857, 4.253 }));
  ypoints.emplace_back(std::array<double, 3>({ 1.83658, 1.16982, 4.30788 }));
  ypoints.emplace_back(std::array<double, 3>({ 1.63993, 1.12446, 7.39286 }));
  ypoints.emplace_back(std::array<double, 3>({ -1.6854, 1.51227, 11.4332 }));
  ypoints.emplace_back(std::array<double, 3>({ -1.47764, 1.41993, 5.23161 }));
  ypoints.emplace_back(std::array<double, 3>({ -1.36112, 1.46405, 5.39199 }));
  ypoints.emplace_back(std::array<double, 3>({ -1.46795, 1.43467, 11.004 }));
  auto ycostfunc = Yerror::Create(ypoints);
  problem.AddResidualBlock(ycostfunc, nullptr, r_vec);

  xpoints.emplace_back(std::array<double, 3>({ 1.56694, 1.23576, 8.09722 }));
  xpoints.emplace_back(std::array<double, 3>({ 1.61549, 1.17857, 4.253 }));
  auto xcostfunc = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc, nullptr, r_vec);

  xpoints.clear();
  xpoints.emplace_back(std::array<double, 3>({ 1.83658, 1.16982, 4.30788 }));
  xpoints.emplace_back(std::array<double, 3>({ 1.63993, 1.12446, 7.39286 }));
  xcostfunc = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc, nullptr, r_vec);

  xpoints.clear();
  xpoints.emplace_back(std::array<double, 3>({ -1.6854, 1.51227, 11.4332 }));
  xpoints.emplace_back(std::array<double, 3>({ -1.47764, 1.41993, 5.23161 }));
  xcostfunc = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc, nullptr, r_vec);

  xpoints.clear();
  xpoints.emplace_back(std::array<double, 3>({ -1.36112, 1.46405, 5.39199 }));
  xpoints.emplace_back(std::array<double, 3>({ -1.46795, 1.43467, 11.004 }));
  xcostfunc = Xerror::Create(xpoints);
  problem.AddResidualBlock(xcostfunc, nullptr, r_vec);

  zpoints.emplace_back(std::array<double, 3>({ 1.56694, 1.23576, 8.09722 }));
  zpoints.emplace_back(std::array<double, 3>({ 1.63993, 1.12446, 7.39286 }));
  auto zcostfunc = Zerror::Create(zpoints);
  problem.AddResidualBlock(zcostfunc, nullptr, r_vec);

  zpoints.clear();
  zpoints.emplace_back(std::array<double, 3>({ 1.61549, 1.17857, 4.253 }));
  zpoints.emplace_back(std::array<double, 3>({ 1.83658, 1.16982, 4.30788 }));
  zcostfunc = Zerror::Create(zpoints);
  problem.AddResidualBlock(zcostfunc, nullptr, r_vec);

  zpoints.clear();
  zpoints.emplace_back(std::array<double, 3>({ -1.6854, 1.51227, 11.4332 }));
  zpoints.emplace_back(std::array<double, 3>({ -1.46795, 1.43467, 11.004 }));
  zcostfunc = Zerror::Create(zpoints);
  problem.AddResidualBlock(zcostfunc, nullptr, r_vec);

  zpoints.clear();
  zpoints.emplace_back(std::array<double, 3>({ -1.47764, 1.41993, 5.23161 }));
  zpoints.emplace_back(std::array<double, 3>({ -1.36112, 1.46405, 5.39199 }));
  zcostfunc = Zerror::Create(zpoints);
  problem.AddResidualBlock(zcostfunc, nullptr, r_vec);

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


  all = ypoints;
  double y_norm = 0;
  for (int i = 0; i < all.size(); i++) {
    cv::Matx31d p(all[i][0], all[i][1], all[i][2]);
    cv::Mat cp = (R * p).t();
    LOG(ERROR) << p.t() << "-" << cp;
    y_norm += cp.at<double>(0, 1);
  }
  LOG(ERROR) << "y = " << y_norm / all.size();
  system("pause");
  return 0;
}
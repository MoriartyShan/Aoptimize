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
#define ZP 2

class Zerror {
private:
  std::vector<std::array<double, 3>> points;
public:
  static ceres::CostFunction* Create(const std::vector<std::array<double, 3>> &_points) {
    return (new ceres::AutoDiffCostFunction<Zerror, 1, 3>(
      new Zerror(_points)));
  }

  Zerror(const std::vector<std::array<double, 3>> &_points) {
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
  static ceres::CostFunction* Create(const std::vector<std::array<double, 3>> &_points) {
    return (new ceres::AutoDiffCostFunction<Yerror, 1, 3>(
      new Yerror(_points)));
  }

  Yerror(const std::vector<std::array<double, 3>> &_points) {
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
  static ceres::CostFunction* Create(const std::vector<std::array<double, 3>> &_points) {
    return (new ceres::AutoDiffCostFunction<Xerror, 1, 3>(
      new Xerror(_points)));
  }

  Xerror(const std::vector<std::array<double, 3>> &_points) {
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


void add_problems(
    ceres::Problem& problem, double *r_vec,
    const std::vector<cv::Point3d>& points,
    std::vector<std::vector<int>>* places, int id) {
  std::vector<std::array<double, 3>> use;
  for (int i = 0; i < places[id].size(); i++) {
    use.resize(places[id][i].size());
    for (int j = 0; j < places[id][i].size(); j++) {
      use[j][0] = points[places[id][i][j]].x;
      use[j][1] = points[places[id][i][j]].y;
      use[j][2] = points[places[id][i][j]].z;
    }
    ceres::CostFunction* costfunc;
    if (id == 0) {
      costfunc = Xerror::Create(use);
    } else if (1 == id) {
      costfunc = Yerror::Create(use);
    } else if (2 == id) {
      costfunc = Zerror::Create(use);
    } else {
      LOG(FATAL) << "wrong input id = " << id;
    }
    problem.AddResidualBlock(costfunc, nullptr, r_vec);
  }
}


int main() {
  std::vector<std::array<double, 3>> ypoints, xpoints, zpoints, all;

  ceres::Problem problem;
  double r_vec[3] = {1, 0, 0};
  std::vector<cv::Point3d> points;
  
  std::vector<std::vector<int>> places[3];
  std::string points_file("D:\\Projects\\BoardDetect\\Extrinsic\\res\\extrinsic_use\\points.yaml");
  {
    cv::FileStorage config(points_file, cv::FileStorage::READ);
    CHECK(config.isOpened());
    cv::read(config["points3d"], points);
    config["x"] >> places[0];
    config["y"] >> places[1];
    config["z"] >> places[2];
    config.release();
    LOG(ERROR) << "x = " << places[0].size();
    LOG(ERROR) << "y = " << places[1].size();
    LOG(ERROR) << "z = " << places[2].size();
    LOG(ERROR) << "points size = " << points.size();
  }

  for (int i = 0; i < 3; i++) {
    add_problems(problem, r_vec, points, places, i);
  }

  std::cout << "resudual number = " << problem.NumResidualBlocks() << "," << problem.NumResiduals() << std::endl;

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  LOG(ERROR) << "FindRotation:" << summary.BriefReport();
  LOG(ERROR) << "vector = " << r_vec[0] << "," << r_vec[1] << "," << r_vec[2];
  cv::Vec3d r(r_vec[0], r_vec[1], r_vec[2]), rinv;
  cv::Matx33d R, Rinv;
  cv::Rodrigues(r, R);
  LOG(ERROR) << "Get R:\n" << R;
  Rinv = R.inv();
  cv::Rodrigues(Rinv, rinv);
  LOG(ERROR) << "Get Rinv:\n" << Rinv << "\n" << rinv.t();

  double y_norm = 0;
  for (int i = 0; i < points.size(); i++) {
    auto cp = (R * points[i]);
    LOG(ERROR) << "[" << i << "]=" << points[i] << "-" << cp;
    if (i < 6) {
      y_norm += cp.y;
    }
    
  }
  LOG(ERROR) << "y = " << y_norm / 6;
  system("pause");
  return 0;
}
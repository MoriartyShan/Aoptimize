#include <ceres/rotation.h>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include "../Common/camera.h"


cv::Matx31d get_point3d(const cv::Matx33d &m_Matrix, const cv::Point2d& p2d, const double height) {
  using T = double;
  T uv[2] = { (T)p2d.x, (T)p2d.y };
  T a = m_Matrix(1, 0) - m_Matrix(2, 0) * uv[1];
  T b = m_Matrix(1, 2) - m_Matrix(2, 2) * uv[1];
  T c = m_Matrix(2, 1) * uv[1] - m_Matrix(1, 1) * height;

  T m = m_Matrix(0, 0) - m_Matrix(2, 0) * uv[0];
  T n = m_Matrix(0, 2) - m_Matrix(2, 2) * uv[0];
  T p = m_Matrix(2, 1) * uv[0] - m_Matrix(0, 1) * height;

  //MLOG() << "[a, b, c, m, n, p] = [" << a << "," << b << "," << c << "," << m << "," << n << "," << p << "]";

  T z = (c / a - (p / m)) / (b / a - (n / m));
  T x = (c / b - p / n) / (a / b - m / n);
  T depth = m_Matrix(2, 0) * x + m_Matrix(2, 2) * z + m_Matrix(2, 1) * height;

  return cv::Matx31d(x, z, depth);
}

//only for 3x3
template<typename T>
void MatrixMulti(const T *left, const T *right, T* result) {
  const int rown = 3, coln = 3;
  for (int i = 0; i < rown; i++) {
    for (int j = 0; j < coln; j++) {
      int idx = i * coln + j;
      result[idx] = (T)0;
      for (int m = 0; m < coln; m++) {
        result[idx] += left[i * coln + m] * right[m * coln + j];
      }
    }
  }
}

template<typename T> 
std::array<T, 2> GetDeviation(const std::vector<std::array<T, 2>> &vectors, const std::array<T, 2>& ave) {
  std::array<T, 2> res = { (T)0 };
  for (auto &v : vectors) {
    T d = v[0] - ave[0];
    res[0] += (d * d);

    d = v[1] - ave[1];
    res[1] += (d * d);
  }
  res[0] /= (T)vectors.size();
  res[1] /= (T)vectors.size();
  return res;
}

template<typename T>
void InsertMatrixToPointer(const cv::Matx33d &m, T *elem) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      elem[i * 3 + j] = (T)m(i, j);
    }
  }
  return;
}

// a line that all points' z value is different
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
    ceres::MatrixAdapter<T, 3, 1> m_Matrix(m_matrix);
   
    ceres::AngleAxisToRotationMatrix(aim, Cam_R_Car);
    InsertMatrixToPointer(_cameraK, cameraK);
    MatrixMulti(cameraK, Cam_R_Car, m_matrix);

    for (int i = 0; i < points.size(); i++) {
      T uv[2] = {(T)points[i].x, (T)points[i].y};
      T a = m_Matrix(1, 0) - m_Matrix(2, 0) * uv[1];
      T b = m_Matrix(1, 2) - m_Matrix(2, 2) * uv[1];
      T c = m_Matrix(2, 1) * uv[1] - m_Matrix(1, 1) * aim[3];

      T m = m_Matrix(0, 0) - m_Matrix(2, 0) * uv[0];
      T n = m_Matrix(0, 2) - m_Matrix(2, 2) * uv[0];
      T p = m_Matrix(2, 1) * uv[0] - m_Matrix(0, 1) * aim[3];

      T c_a = c / a, b_a = b / a;

      psz[i][1] = (c_a - (p / m)) / (b_a - (n / m));
      psz[i][0] = c_a - b_a * psz[i][1];

      average[0] += psz[i][0];
      average[1] += psz[i][1];
    }
    average[0] /= (T)points.size();
    average[1] /= (T)points.size();

    auto dev = GetDeviation(psz, average);

    residuals[0] = dev[0];
    //residuals[1] = dev[1];
    //LOG(ERROR) << residuals[0] << "-" << residuals[1];
    return true;
  };
};

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
    return (new ceres::AutoDiffCostFunction<ZLineDistance, 2, 4>(
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
  bool operator()(const T* const aim, T* residuals) const {
    std::vector<std::array<T, 2>> line1(_line1.size());
    std::vector<std::array<T, 2>> line2(_line2.size());
    T Cam_R_Car[9], cameraK[9], m_matrix[9];
    ceres::MatrixAdapter<T, 3, 1> m_Matrix(m_matrix);

    ceres::AngleAxisToRotationMatrix(aim, Cam_R_Car);
    InsertMatrixToPointer(_cameraK, cameraK);
    MatrixMulti(cameraK, Cam_R_Car, m_matrix);

    std::array<T, 2> average = { (T)0 };

    for (int i = 0; i < _line1.size(); i++) {
      T uv[2] = { (T)_line1[i].x, (T)_line1[i].y };
      T a = m_Matrix(1, 0) - m_Matrix(2, 0) * uv[1];
      T b = m_Matrix(1, 2) - m_Matrix(2, 2) * uv[1];
      T c = m_Matrix(2, 1) * uv[1] - m_Matrix(1, 1) * aim[3];

      T m = m_Matrix(0, 0) - m_Matrix(2, 0) * uv[0];
      T n = m_Matrix(0, 2) - m_Matrix(2, 2) * uv[0];
      T p = m_Matrix(2, 1) * uv[0] - m_Matrix(0, 1) * aim[3];

      T c_a = c / a, b_a = b / a;

      line1[i][1] = (c_a - (p / m)) / (b_a - (n / m));
      line1[i][0] = c_a - b_a * line1[i][1];
      average[0] += line1[i][0];
      //LOG(ERROR) << line1[i][1];
    }

    for (int i = 0; i < _line2.size(); i++) {
      T uv[2] = { (T)_line2[i].x, (T)_line2[i].y };
      T a = m_Matrix(1, 0) - m_Matrix(2, 0) * uv[1];
      T b = m_Matrix(1, 2) - m_Matrix(2, 2) * uv[1];
      T c = m_Matrix(2, 1) * uv[1] - m_Matrix(1, 1) * aim[3];

      T m = m_Matrix(0, 0) - m_Matrix(2, 0) * uv[0];
      T n = m_Matrix(0, 2) - m_Matrix(2, 2) * uv[0];
      T p = m_Matrix(2, 1) * uv[0] - m_Matrix(0, 1) * aim[3];

      T c_a = c / a, b_a = b / a;

      line2[i][1] = (c_a - (p / m)) / (b_a - (n / m)); //z
      line2[i][0] = c_a - b_a * line2[i][1]; //x
      average[1] += line2[i][0];
      //LOG(ERROR) << line2[i][1];
    }

    residuals[0] = (T)0;

    for (int i = 0; i < line1.size(); i++) {
      for (int j = 0; j < line2.size(); j++) {
        T dist = (line1[i][0] - line2[j][0]);
        //LOG(ERROR) << dist;
        if (dist < (T)0) {
          dist = dist + (T)_distance;
        } else {
          dist = (T)_distance - dist;
        }
        if (dist < (T)0) {
          dist = -dist;
        }
        //LOG(ERROR) << dist;
        residuals[0] += (dist * dist);
      }
    }
    residuals[0] /= (T)(line1.size() * line2.size());

    T dist = average[0] - average[1];
    residuals[1] = dist * dist;

    LOG(ERROR) << residuals[0] << "\n" << residuals[1];
    return true;
  };
};

void Ax_B(double a, double b, double c,
  double m, double n, double p) {
  double x = (c / b - p / n) / (a / b - m / n);
  double y = (c / a - (p / m)) / (b / a - (n / m));
  LOG(ERROR) << "res = " << x << "," << y;
}


int main() {
  std::string points_file("D:\\Projects\\BoardDetect\\resources\\extrinsic_use/points2.yaml");
  std::string camera_file("D:\\Projects\\BoardDetect\\resources\\hardwares\\C.yaml");
  std::string image_file("D:\\Projects\\Documents\\20200507\\find_lane\\2005071058421104.jpg");
  auto camera_ptr = Camera::create(camera_file);
#if 0
  cv::Mat img = cv::imread(image_file, cv::IMREAD_UNCHANGED), imgo;

  LOG(ERROR) << "cameraK = " << camera_ptr->Intrinsic() << " dist = " << camera_ptr->Distortion();

  cv::undistort(img, imgo, camera_ptr->Intrinsic(), camera_ptr->Distortion());
  cv::imwrite("undistort.jpg", imgo);
  return 0;
#endif
  std::vector<cv::Point2d> points[4], undistort_points[4];
  
  {
    cv::FileStorage config(points_file, cv::FileStorage::READ);
    CHECK(config.isOpened());
    cv::read(config["first"], points[0]);
    cv::read(config["second"], points[1]);
    cv::read(config["third"], points[2]);
    cv::read(config["fourth"], points[3]);
    
    LOG(ERROR) << "first = " << points[0].size();
    LOG(ERROR) << "second = " << points[1].size();
    LOG(ERROR) << "third = " << points[2].size();
    LOG(ERROR) << "fourth = " << points[03].size();
  }

  for (int i = 0; i < 4; i++) {
    std::sort(points[i].begin(), points[i].end(), [](const cv::Point2d &l, const cv::Point2d &r) {
      if (l.y < r.y || (l.y == r.y && l.x < r.x)) return true;
      else return false;
    });

    cv::undistortPoints(points[i], undistort_points[i], camera_ptr->Intrinsic(), \
      camera_ptr->Distortion(), cv::noArray(), camera_ptr->Intrinsic());
  }
  double res[4] = { -0.0151731, -0.0314901, -0.0323597,1.54488 };
  ceres::Problem problem;

  auto cost_func = ZLine::Create(camera_ptr->Intrinsic(), undistort_points[0]);
  problem.AddResidualBlock(cost_func, nullptr, res);

  cost_func = ZLine::Create(camera_ptr->Intrinsic(), undistort_points[1]);
  problem.AddResidualBlock(cost_func, nullptr, res);

  cost_func = ZLine::Create(camera_ptr->Intrinsic(), undistort_points[2]);
  problem.AddResidualBlock(cost_func, nullptr, res);

  cost_func = ZLine::Create(camera_ptr->Intrinsic(), undistort_points[3]);
  problem.AddResidualBlock(cost_func, nullptr, res);

  /////////
  const double lane_interval = 3.7;
  cost_func = ZLineDistance::Create(camera_ptr->Intrinsic(), lane_interval, undistort_points[0], undistort_points[1]);
  problem.AddResidualBlock(cost_func, nullptr, res);

  cost_func = ZLineDistance::Create(camera_ptr->Intrinsic(), lane_interval, undistort_points[1], undistort_points[2]);
  problem.AddResidualBlock(cost_func, nullptr, res);

  cost_func = ZLineDistance::Create(camera_ptr->Intrinsic(), lane_interval, undistort_points[2], undistort_points[3]);
  problem.AddResidualBlock(cost_func, nullptr, res);

  LOG(ERROR) << "resudual number = " << problem.NumResidualBlocks() << "," << problem.NumResiduals() << std::endl;

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  LOG(ERROR) << "FindRotation:" << summary.BriefReport();
  LOG(ERROR) << "vector = " << res[0] << "," << res[1] << "," << res[2] << "," << res[3];
  cv::Vec3d r(res[0], res[1], res[2]);
  cv::Matx33d R;
  cv::Rodrigues(r, R);
  LOG(ERROR) << "rotation = \n" << R;

  cv::Matx33d m_Matrix = camera_ptr->Intrinsic() * R;
  std::vector<std::array<double, 2>> p3ds[4];
  std::array<double, 2> average[4] = { {0} };
  for (int i = 0; i < 4; i++) {
    std::stringstream ss;
    for (auto &p : undistort_points[i]) {
      cv::Matx31d pm =
        get_point3d(m_Matrix, p, res[3]);
      ss << pm.t() << std::endl;
      p3ds[i].emplace_back(std::array<double, 2>({ pm(0), pm(1) }));
      average[i][0] += pm(0);
      average[i][1] += pm(1);
      //p3ds[i].back() = ;
    }
    average[i][0] /= undistort_points[i].size();
    average[i][1] /= undistort_points[i].size();

    auto dev = GetDeviation(p3ds[i], average[i]);

    LOG(ERROR) << std::endl << ss.str() << "\ndev = " << dev[0];
  }

  auto GetCost = [&lane_interval](
      const std::vector<std::array<double, 2>> &a, 
      const std::vector<std::array<double, 2>> &b) {
    double dist = 0;
    for (auto &ap : a) {
      for (auto & bp : b) {
        double d = ap[0] - bp[0];
        if (d < 0) {
          d = -d;
        }
        d = d - lane_interval;
        dist += (d * d);
      }
    }
    dist /= (a.size() * b.size());
    return dist;
  };

  LOG(ERROR) << "dist = " << GetCost(p3ds[0], p3ds[1]) << "," << GetCost(p3ds[2], p3ds[1]) << "," << GetCost(p3ds[2], p3ds[3]);

  cv::Vec3d r_vec;
  cv::Rodrigues(camera_ptr->Car_R_Cam().inv(), r_vec);
  LOG(ERROR) << "rvec better = " << r_vec;
  m_Matrix = camera_ptr->Intrinsic() * camera_ptr->Car_R_Cam().inv();
  p3ds[4];
  std::array<double, 2> aaverage[4] = { {0} };
  for (int i = 0; i < 4; i++) {
    std::stringstream ss;
    p3ds[i].clear();
    for (auto &p : undistort_points[i]) {
      cv::Matx31d pm =
        get_point3d(m_Matrix, p, 1.5);
      ss << pm.t() << std::endl;
      p3ds[i].emplace_back(std::array<double, 2>({ pm(0), pm(1) }));
      aaverage[i][0] += pm(0);
      aaverage[i][1] += pm(1);
      //p3ds[i].back() = ;
    }
    aaverage[i][0] /= undistort_points[i].size();
    aaverage[i][1] /= undistort_points[i].size();

    auto dev = GetDeviation(p3ds[i], aaverage[i]);

    LOG(ERROR) << std::endl << ss.str() << "\ndev = " << dev[0];
  }
  LOG(ERROR) << "dist = " << GetCost(p3ds[0], p3ds[1]) << "," << GetCost(p3ds[2], p3ds[1]) << "," << GetCost(p3ds[2], p3ds[3]);

#if 0
  double left[9] = { 1, 2.3, 1, 3, 12, 3.4, 3, 1, 0.4 };
  double right[9] = { 1, 32.3, 1, 23, 1.2, 14, 4.3, 14, 3.4 };
  double result[9] = { 0 }, res2[9];

  cv::Mat_<double> leftM(3, 3, left);
  cv::Mat_<double> rightM(3, 3, right);
  cv::Mat_<double> resultM(3, 3, result);
  cv::Mat_<double> result2M(3, 3, res2);
  resultM = leftM * rightM;

  MatrixMulti(left, right, res2);


  LOG(ERROR) << "test = \n" << resultM << "\n------------------\n" << result2M;
  LOG(ERROR) << "diff=\n" << resultM - result2M;
  LOG(ERROR) << "test " << (uint64_t)result << "  " << (uint64_t)resultM.data;
  LOG(ERROR) << "test " << (uint64_t)res2 << "  " << (uint64_t)result2M.data;


  Ax_B(-40.9238, -221.994, -1263.28, 1044.89, 171.676, -35.3262);
  cv::Matx22d A(
    -40.9238, -221.994,
    1044.89, 171.676
  );
  cv::Matx21d B(-1263.28, -35.3262), res;
  cv::solve(A, B, res);
  LOG(ERROR) << "res = " << res.t();
#endif
  system("pause");
  
  return 0;
}
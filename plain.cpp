#include "../Common/camera.h"
#include "LinesDistanceCost.h"
#include "PointsAtTheSameZLineCost.h"
#include "PointDistanceCost.h"
#include "data.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "cv_plot/plot.hpp"
#include <gflags/gflags.h>
#include <fstream>

DEFINE_int32(rand_seed, -1, "> 1 as seed, 1 is special,"
  " <= 0 for random");
DEFINE_string(data, "", "path to camera.yaml, points.yaml,"
                        " image.png/image.bmp, undist_image.png/undist_image.bmp");
DEFINE_bool(undist_image, false, "put camera.yaml and image.png/bmp in "
                                 "--data, it will write a undistored image in "
                                 "--data, whose name is undist_image.png");
DEFINE_bool(has_init, false, "if set true, it start "
  "from the value write in points.yaml");

#define PERFECT_THRESHOLD 0.002
#define MAX_ITERATE_TIME 1000

std::string toString(scalar *matrix) {
  std::stringstream matrix_string;
  matrix_string.precision(8);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      matrix_string << matrix[i * 3 + j] << ",";
    }
    matrix_string << "\n";
  }
  return matrix_string.str();
}

enum RESAULT_LEVEL {
  PERFECT = 0,
  GOOD = 1,
  BAD = 2
};

//make a rotate vector length is inside 2 * Pi
void shrink_to_pi(Vec3 &rvec) {
  scalar mod = cv::norm(rvec);
  rvec /= mod;
  while (mod > M_PI) {
    mod -= (2 * M_PI);
  }
  rvec *= mod;
  return;
}

void shrink_to_pi(scalar *vec) {
  Vec3 rvec(vec[0], vec[1], vec[2]);
  shrink_to_pi(rvec);
  vec[0] = rvec(0);
  vec[1] = rvec(1);
  vec[2] = rvec(2);
  return;
}

RESAULT_LEVEL isResultGood(
    const Params* input_params
    ) {
  const Camera::CameraPtr camera_ptr = input_params->camera_ptr;
  const scalar *res = input_params->res.data();
  const scalar height_threshold = 0.5;
  const scalar perfect_threshold = PERFECT_THRESHOLD;
  const scalar angle_threshold = 0.4330;
  const scalar tr_threshold = 0.9 * 3;

  if (input_params->res[3] < height_threshold) {
    //judge if height is too small
    LOG(ERROR) << "result height too small = "
           << "\n[" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
           << input_params->final_cost << "," << input_params->cur_min_cost;
    return RESAULT_LEVEL::BAD;
  }

  if (input_params->final_cost < perfect_threshold) {
    LOG(ERROR) << "get a perfect result = "
      << "\n[" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
      << input_params->final_cost << "," << input_params->cur_min_cost;
    return RESAULT_LEVEL::PERFECT;
  }

  const auto &cameraK = camera_ptr->Intrinsic();
  scalar residuals;
  std::vector<AxisLine> residual_cal;
  residual_cal.reserve(
      input_params->x_lines.size() + input_params->z_lines.size());
  for (int i = 0; i < input_params->x_lines.size(); i++) {
    residual_cal.emplace_back(cameraK, input_params->x_lines[i], -1);
  }
  for (int i = 0; i < input_params->z_lines.size(); i++) {
    residual_cal.emplace_back(cameraK, input_params->z_lines[i], 1);
  }

  for (int i = 0; i < residual_cal.size(); i++) {
    residual_cal[i](res, &residuals);
    if (residuals > 1) {
      LOG(ERROR) << "bad result because of residual["
             << i << "]" << residuals << "-"
             << "\n[" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
             << input_params->final_cost << "," << input_params->cur_min_cost;
      return RESAULT_LEVEL::BAD;
    }
  }

  Vec3 rv(res[0], res[1], res[2]);
  Matrix33 rotation;
  shrink_to_pi(rv);

  if (cv::norm(rv) > angle_threshold) {
    //(0.25, 0.25, 0.25), whose rotation matrix is 0.9-0.9-0.9 tr
    LOG(ERROR) << "too much rotation"
           << "\n[" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
           << input_params->final_cost << "," << input_params->cur_min_cost;
    return RESAULT_LEVEL::BAD;
  }

  cv::Rodrigues(rv, rotation);

  scalar tr = cv::trace(rotation);
  if (tr < tr_threshold) {
    LOG(ERROR) << "Too small tr " << tr << "\n" << rotation
           << "\n[" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
           << input_params->final_cost << "," << input_params->cur_min_cost;
    return RESAULT_LEVEL::BAD;
  }

  Matrix33 m_Matrix = camera_ptr->Intrinsic() * rotation;

  auto check_points_and_lines = [&m_Matrix, &res, &input_params](
      const std::vector<std::vector<Pixel2f>> &lines,
      const std::vector<scalar> &std_distance,
      const int axis) {
    //@axis = -1, std_distance is x distance
    //@axis = 1, std_distance is z distance
    const int cmp_idx = (axis < 0) ? 0 : 2;
    const int line_num = lines.size();
    std::vector<std::vector<std::array<scalar, 3>>> points3d(line_num);

    for (int i = 0; i < line_num; i++) {
      const int point_num = lines[i].size();
      points3d[i].resize(point_num);
      for (int j = 0; j < point_num; j++) {
        scalar uv[2] = { lines[i][j].x, lines[i][j].y };
        Camera::GetPoint3d(m_Matrix.val, uv, res[3], points3d[i][j].data());
        if (points3d[i][j][2] < 0) {
          LOG(ERROR) << "all z value should be positive [" << points3d[i][j][0] << ","
                 << points3d[i][j][1] << "," << points3d[i][j][2] << "]"
                 << "\n[" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
                 << input_params->final_cost << "," << input_params->cur_min_cost;
          return RESAULT_LEVEL::BAD;
        }
      }

      if (i > 0) {
        if (std_distance[i - 1] > 0 &&
            (points3d[i][0][cmp_idx] - points3d[i - 1][0][cmp_idx] - std_distance[i - 1] > 0.5)) {
          LOG(ERROR) << "two line distance is too much " << i << "=" << points3d[i][0][0]
                 << "," << points3d[i - 1][0][0] << ","
                 << points3d[i][0][0] - points3d[i - 1][0][0]
                 << "\n[" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
                 << input_params->final_cost << "," << input_params->cur_min_cost;
          return RESAULT_LEVEL::BAD;
        }
      }
    }
    return RESAULT_LEVEL::GOOD;
  };

  if (check_points_and_lines(input_params->x_lines, input_params->x_width, -1)
      == RESAULT_LEVEL::BAD) {
    return RESAULT_LEVEL::BAD;
  }
  if (check_points_and_lines(input_params->z_lines, input_params->z_width, 1)
      == RESAULT_LEVEL::BAD) {
    return RESAULT_LEVEL::BAD;
  }

  LOG(ERROR) << "get a good start [" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
         << input_params->final_cost << "," << input_params->cur_min_cost;

  return RESAULT_LEVEL::GOOD;
}

bool Optimize(
    Params &input_params,
    const ceres::LoggingType log_type = ceres::LoggingType::SILENT,
    const bool fix_height = false) {
  auto &camera_ptr = input_params.camera_ptr;
  const auto &cameraK = camera_ptr->Intrinsic();

  ceres::Problem problem;
  std::vector<double> current(4, 0);
  double *r_ptr = current.data(), *h_ptr = r_ptr + 3;

  input_params.update_res(current, false);
  {
    auto &x_lines = input_params.x_lines;
    const int size = x_lines.size();
    for (int i = 0; i < size; i++) {
      CHECK(x_lines[i].size() > 0);
      auto cost_func = AxisLine::Create(cameraK, x_lines[i], -1);
      problem.AddResidualBlock(cost_func, nullptr, r_ptr, h_ptr);
      if ((!fix_height) && (i > 0) && (input_params.x_width[i - 1] > 0)) {
//        LOG(FATAL) << "set std dist first";
        cost_func = ZLineDistance::Create(
            cameraK, input_params.x_width[i - 1],
            x_lines[i - 1], x_lines[i]);
        problem.AddResidualBlock(cost_func, nullptr, r_ptr, h_ptr);
      }
    }
  }

  {
    auto &z_lines = input_params.z_lines;
    const int size = z_lines.size();
    for (int i = 0; i < size; i++) {
      CHECK(z_lines[i].size() > 0);

      auto cost_func = AxisLine::Create(cameraK, z_lines[i], 1);
      problem.AddResidualBlock(cost_func, nullptr, r_ptr, h_ptr);
      if ((!fix_height) && (i > 0) && (input_params.z_width[i - 1] > 0)) {
        LOG(FATAL) << "set std dist first and z lines";
      }
    }
  }

  for (auto &p : input_params.points_distance_pairs) {
    auto cost_func = PointDistance::Create(p._distance, p._point1, p._point2, cameraK);
    problem.AddResidualBlock(cost_func, nullptr, r_ptr, h_ptr);
  }

  if (fix_height) {
    problem.SetParameterBlockConstant(h_ptr);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.logging_type = log_type;
  //options.max_num_iterations = 1000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (log_type != ceres::SILENT) {
    LOG(ERROR) << summary.BriefReport();
  }

  if (summary.termination_type == ceres::CONVERGENCE) {
    input_params.update_res(current, true);
    shrink_to_pi(input_params.res.data());
    input_params.final_cost = summary.final_cost;

    if (input_params.final_cost < input_params.cur_min_cost
        && RESAULT_LEVEL::GOOD >= isResultGood(&input_params)) {
      LOG(ERROR) << "final_cost " << input_params.final_cost;
      //only if this is a better result than current
      input_params.cur_min_cost = input_params.final_cost;
      input_params.best = input_params.res;
      return true;
    }
    return false;
  }
  return false;
}

void plot_points(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const cv::Rect2d &rect,
    const cv::Scalar &line_color,
    const double point_width,
    cv::Mat &img) {
//  double min[2] = {x[0], y[0]}, max[2] = {x[0], y[0]};
  const int size = x.size();
  CHECK(size == y.size()) << size << "," << y.size();

  LOG(ERROR) << "point size = " << size;

  cv::Point2d max(rect.x + rect.width, rect.y + rect.height), min(rect.x, rect.y);

  cv::Ptr<cv::plot::Plot2d> plot_ptr = cv::plot::createPlot2d(x, y);
//  cv::Mat img;
//  LOG(ERROR) << plot_ptr.get();
  plot_ptr->setNeedPlotLine(false);
  plot_ptr->setPlotSize(1600, 800);
  plot_ptr->setMaxX(max.x);
  plot_ptr->setMinX(min.x);
  plot_ptr->setMaxY(max.y);
  plot_ptr->setMinY(min.y);
  plot_ptr->setPlotLineWidth(point_width);
//  const auto background_color = CV_RGB(255, 255, 255);
//  const auto preground_color = CV_RGB(0, 0, 0);
  plot_ptr->setPlotLineColor(line_color);
//  plot_ptr->setPlotBackgroundColor(background_color);
////  plot_ptr->setPlotAxisColor()
//  plot_ptr->setPlotGridColor(preground_color);
//  plot_ptr->setPlotTextColor(preground_color);
//  plot_ptr->setPlotLineColor(CV_RGB(0, 0, 255));
//  LOG(ERROR) << plot_ptr.get();
  plot_ptr->render(img, false);
}

cv::Point2d get_max_min(const std::vector<double>& values) {
  if (values.size() == 0) {
    return cv::Point2d(NAN, NAN);
  }
  cv::Point2d _max_min(values[0], values[0]);
  for (auto &v : values) {
    if (_max_min.x < v) {
      _max_min.x = v;
    }
    if (_max_min.y > v) {
      _max_min.y = v;
    }
  }
  return _max_min;
}

void show_points(const Params &parameters, std::string& result) {
  Vec3 r(parameters.res[0], parameters.res[1], parameters.res[2]);
  Matrix33 R;
  cv::Rodrigues(r, R);

  std::stringstream ss;
  ss << "best result = [" << parameters.res[0] << "," << parameters.res[1]
             << "," << parameters.res[2] << "," << parameters.res[3] << "] " << parameters.cur_min_cost << std::endl;

  ss << "rotation = \n" << R << std::endl;
  ss << "car_R_cam = \n" << R.inv() << std::endl;

  Matrix33 m_Matrix_ceres = parameters.camera_ptr->Intrinsic() * R;
  result = ss.str();
  ss.str("");
  std::vector<double> x, y;
  std::vector<scalar> average(parameters.x_lines.size() + parameters.z_lines.size(), 0);

  for (int i = 0; i < parameters.x_lines.size(); i++) {
    const int start_idx = x.size();
    x.reserve(parameters.x_lines.size() + x.size());
    y.reserve(parameters.x_lines.size() + y.size());
    ss << "xline [" << i << "]\n";
    for (int j = 0; j < parameters.x_lines[i].size(); j++) {
      scalar uv[2] = { parameters.x_lines[i][j].x, parameters.x_lines[i][j].y };
      scalar p3d[3];
      average[i] += p3d[0];
      Camera::GetPoint3d(m_Matrix_ceres.val, uv, parameters.res[3], p3d);
      if (j == 0 || (j == parameters.x_lines[i].size() - 1)) {
        ss << "\t[" << j << "]-[" << p3d[0] << "," << p3d[1] << "," << p3d[2] << "]-[" << uv[0] << "," << uv[1] << "]\n";
      }
      x.emplace_back(p3d[0]);
      y.emplace_back(p3d[2]);
    }

    average[i] /= (scalar)parameters.x_lines[i].size();
//    LOG(ERROR) << "ave"
    ss << "\tdiffs:" << x.back() - x[start_idx] << "," << y.back() - y[start_idx] << std::endl;
    result += ss.str();
    result += ("\txline place:" + std::to_string(average[i]) + ",");
    if (i > 0) {
      scalar width = average[i] - average[i - 1];
      result += (" width = " + std::to_string(width) + "\n");
    } else {
      result += "\n";
    }
    ss.str("");
  }

  for (int i = 0; i < parameters.z_lines.size(); i++) {
    const int ave_idx = i + parameters.x_lines.size();
    const int start_idx = x.size();
    x.reserve(parameters.z_lines.size() + x.size());
    y.reserve(parameters.z_lines.size() + y.size());
    ss << "zline [" << i << "]\n";
    for (int j = 0; j < parameters.z_lines[i].size(); j++) {
      scalar uv[2] = { parameters.z_lines[i][j].x, parameters.z_lines[i][j].y };
      scalar p3d[3];
      average[ave_idx] += p3d[2];
      Camera::GetPoint3d(m_Matrix_ceres.val, uv, parameters.res[3], p3d);
      if (j == 0 || (j == parameters.z_lines[i].size() - 1)) {
        ss << "\t[" << j << "]-[" << p3d[0] << "," << p3d[1] << "," << p3d[2] << "]-[" << uv[0] << "," << uv[1] << "]\n";
      }
      x.emplace_back(p3d[0]);
      y.emplace_back(p3d[2]);
    }

    average[ave_idx] /= (scalar)parameters.z_lines[i].size();
    ss << "\tdiffs:" << x.back() - x[start_idx] << "," << y.back() - y[start_idx] << std::endl;
    result += ss.str();
    result += ("\tzline place:" + std::to_string(average[ave_idx]) + ",");
    if (i > 0) {
      scalar width = average[ave_idx] - average[ave_idx - 1];
      result += (" width = " + std::to_string(width) + "\n");
    } else {
      result += "\n";
    }
    ss.str("");
  }

  std::vector<double> px, py;
  px.reserve(parameters.points_distance_pairs.size() * 2);
  py.reserve(px.capacity());
  x.reserve(x.size() + px.capacity());
  y.reserve(x.size() + px.capacity());
//  LOG(ERROR) << "p capacity = " << px.capacity() << "," << parameters.points_distance_pairs.size() * 2;
  ss << "points distance = [\n";
  for (auto &p : parameters.points_distance_pairs) {
    scalar uv[2] = { p._point1.x, p._point1.y };
    scalar p3d[3];
    Camera::GetPoint3d(m_Matrix_ceres.val, uv, parameters.res[3], p3d);
    x.emplace_back(p3d[0]);
    y.emplace_back(p3d[2]);
    px.emplace_back(p3d[0]);
    py.emplace_back(p3d[2]);
//    LOG(ERROR) << "px = " << uv[0] << "," << uv[1] << "," << p3d[0] << "," << p3d[2];
    uv[0] = p._point2.x;
    uv[1] = p._point2.y;
    Camera::GetPoint3d(m_Matrix_ceres.val, uv, parameters.res[3], p3d);
    x.emplace_back(p3d[0]);
    y.emplace_back(p3d[2]);
    ss << "\t(" << px.back() << "," << py.back() << ")-(" << x.back() << "," << y.back() << ")-"
       <<  cv::norm(cv::Vec2d(x.back() - px.back(), y.back() - py.back())) << "," << p._distance << "]\n";
    px.emplace_back(p3d[0]);
    py.emplace_back(p3d[2]);

//    LOG(ERROR) << "px = " << uv[0] << "," << uv[1] << "," << p3d[0] << "," << p3d[2];
  }
  ss << "]\n";
  result += ss.str();
  cv::Mat plot_Result;
  cv::Rect2d rect;
  cv::Point2d _max_min = get_max_min(x);
  CHECK((!isnan(_max_min.x)) && (!isnan(_max_min.y))) << "max and min is NAN";
  rect.x = _max_min.y;
  rect.width = _max_min.x - _max_min.y;
  _max_min = get_max_min(y);
  rect.y = _max_min.y;
  rect.height = _max_min.x - _max_min.y;
//  LOG(ERROR) << "rect = " << rect;
  plot_points(x, y, rect, CV_RGB(255, 0, 0), 1, plot_Result);
  plot_points(px, py, rect, CV_RGB(0, 255, 0), 5, plot_Result);
  cv::flip(plot_Result, plot_Result, 0);
  cv::imshow("test", plot_Result);
  cv::imwrite(FLAGS_data + "/plot_image.png", plot_Result);
  return;
}

RESAULT_LEVEL optimize_from_zeros(Params &parameter) {
  parameter.set_res_zeros();
  Optimize(parameter);
  parameter.res = parameter.best;
  parameter.final_cost = parameter.cur_min_cost;
  return isResultGood(&parameter);
}

RESAULT_LEVEL optimize_from_set(Params &parameter) {
  bool has_good = false;

  if (Optimize(parameter)) {
    has_good = true;
  }

  if (has_good) {
    parameter.res = parameter.best;
    parameter.final_cost = parameter.cur_min_cost;
    return isResultGood(&parameter);
  }
  return RESAULT_LEVEL::BAD;

}

RESAULT_LEVEL optimize_from_random(Params &parameter) {
    parameter.set_res_random();
    Optimize(parameter);
    return isResultGood(&parameter);
}

//return true if it get a perfect result
//return false if not
bool optimize_from_good(Params &parameters) {
  std::vector<scalar> cur_best;
  scalar cur_min_cost;
  RESAULT_LEVEL current_result_level = RESAULT_LEVEL::BAD;
  int c = 0;
  do {
    cur_best = parameters.best;
    cur_min_cost = parameters.cur_min_cost;
    current_result_level = optimize_from_set(parameters);
    if ((current_result_level == RESAULT_LEVEL::BAD)
        || ((current_result_level == RESAULT_LEVEL::GOOD) && parameters.isEqualtoBest(cur_best, cur_min_cost))) {
      //bad result or the same iterate
      LOG(ERROR) << "stop this good start";
      return false;
    } else {
      LOG(ERROR) << "count " << c++;
    }
  } while (current_result_level == RESAULT_LEVEL::GOOD);
  return true;
}

int main(int argc, char **argv) {
  google::SetVersionString("1.0.0");
  google::SetUsageMessage(std::string(argv[0]) + " [OPTION]");
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  std::string camera_file = FLAGS_data + "/camera.yaml";

  unsigned int rand_seed = FLAGS_rand_seed;
  if (FLAGS_rand_seed <= 0) {
    rand_seed = time(0);
  }
  srand(rand_seed);
  LOG(ERROR) << "rand seed = " << rand_seed;

  auto camera_ptr = Camera::create(camera_file);
  PointsInputOutput io_helper(FLAGS_data);

  if (FLAGS_undist_image) {
    io_helper.draw_undist_images(camera_ptr);
    return 0;
  }

  Params parameters;
  parameters.init(camera_ptr);

  io_helper.read(parameters);
  io_helper.insert_points(parameters);
//  io_helper.write(FLAGS_data + "/out.yaml", parameters);

  RESAULT_LEVEL current_result_level = RESAULT_LEVEL::BAD;
  if (FLAGS_has_init) {
    current_result_level = optimize_from_set(parameters);
  }

  if (current_result_level != RESAULT_LEVEL::PERFECT) {
    //always try zeros if current_result_level is not perfect
    current_result_level = optimize_from_zeros(parameters);
  }

  int count = 0;
  while (current_result_level != RESAULT_LEVEL::PERFECT) {
    if (current_result_level == RESAULT_LEVEL::GOOD) {
      if (optimize_from_good(parameters)) {
        current_result_level = RESAULT_LEVEL::PERFECT;
        break;
      }
    }
    current_result_level = optimize_from_random(parameters);
    if (count++ > MAX_ITERATE_TIME) {
      LOG(ERROR) << "currently minimal residual is " << parameters.cur_min_cost
                 << ", out put it(" << count << ")";
      parameters.res = parameters.best;
      break;
    } else if (count % 100 == 50) {
      LOG(ERROR) << "currently minimal residual is " << parameters.cur_min_cost
                 << ", iterate time = " << count;
    }
  }

  camera_ptr->set_Extrinsic(parameters.res.data());
  camera_ptr->WriteToYaml(FLAGS_data + "/extrinsic.yaml");
  std::string string_result;
  show_points(parameters, string_result);
  LOG(ERROR) << "\n" << string_result;
  for (int i = 0; i < string_result.size(); i+=100) {
    std::cout << string_result.substr(i, 100);
  }
  cv::waitKey(0);

  KEEP_CMD_WINDOW();

  return 0;
}
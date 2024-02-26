#include <tuple>
#include <mutex>
#include <thread>
#include <queue>
#include <array>
#include <vector>
#include <atomic>
#include <chrono>
#include <iostream>
#include <functional>
#include <condition_variable>

#include <assert.h>
#include <string.h>
//#include <io.h>

#include "CubeEyeSink.h"
#include "CubeEyeCamera.h"
#include "CubeEyeBasicFrame.h"
#include <opencv2/opencv.hpp>
using namespace cv;

static class ReceivedDepthFrameSink : public meere::sensor::sink
, public meere::sensor::prepared_listener
{
public:
	int cameraId;  // 카메라 ID 추가
	cv::VideoWriter videoWriter;
	ReceivedDepthFrameSink(int id) : cameraId(id) {}  // 생성자 추가

	virtual std::string name() const {
		return std::string("ReceivedDepthFrameSink");
	}

	virtual void onCubeEyeCameraState(const meere::sensor::ptr_source source, meere::sensor::State state) {
		printf("%s:%d source(%s) state = %d\n", __FUNCTION__, __LINE__, source->uri().c_str(), state);

		if (meere::sensor::State::Running == state) {
			mReadFrameThreadStart = true;
			mReadFrameThread = std::thread(ReceivedDepthFrameSink::ReadFrameProc, this);
		}
		else if (meere::sensor::State::Stopped == state) {
			mReadFrameThreadStart = false;
			if (mReadFrameThread.joinable()) {
				mReadFrameThread.join();
			}
		}
	}

	virtual void onCubeEyeCameraError(const meere::sensor::ptr_source source, meere::sensor::Error error) {
		printf("%s:%d source(%s) error = %d\n", __FUNCTION__, __LINE__, source->uri().c_str(), error);
	}

	virtual void onCubeEyeFrameList(const meere::sensor::ptr_source source, const meere::sensor::sptr_frame_list& frames) {
		if (mReadFrameThreadStart) {
			static constexpr size_t _MAX_FRAMELIST_SIZE = 4;
			if (_MAX_FRAMELIST_SIZE > mFrameListQueue.size()) {
				auto _copied_frame_list = meere::sensor::copy_frame_list(frames);
				if (_copied_frame_list) {
					mFrameListQueue.push(std::move(_copied_frame_list));
				}
			}
		}
	}

public:
	virtual void onCubeEyeCameraPrepared(const meere::sensor::ptr_camera camera) {
		printf("%s:%d source(%s)\n", __FUNCTION__, __LINE__, camera->source()->uri().c_str());
	}

protected:
	static void ReadFrameProc(ReceivedDepthFrameSink* thiz) {
		thiz->videoWriter.open("camera_" + std::to_string(thiz->cameraId) + ".avi",
			cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
			cv::Size(640, 480), true);
		while (thiz->mReadFrameThreadStart) {
			if (thiz->mFrameListQueue.empty()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				continue;
			}

			auto _frames = std::move(thiz->mFrameListQueue.front());
			thiz->mFrameListQueue.pop();

			if (_frames) {
				static int _frame_cnt = 0;
				// if (2 > ++_frame_cnt) {
				// 	continue;
				// }
				_frame_cnt = 0;

				for (auto it : (*_frames)) {
					printf("frame : %d, "
						"frameWidth = %d "
						"frameHeight = %d "
						"frameDataType = %d "
						"timestamp = %lu \n",
						it->frameType(),
						it->frameWidth(),
						it->frameHeight(),
						it->frameDataType(),
						it->timestamp());

					int _frame_index = 0;

					// Depth frame
					if (it->frameType() == meere::sensor::CubeEyeFrame::FrameType_Depth) {
						// 16bits data type
						if (it->frameDataType() == meere::sensor::CubeEyeData::DataType_16U) {
							// casting 16bits basic frame 
							auto _sptr_basic_frame = meere::sensor::frame_cast_basic16u(it);
							auto _sptr_frame_data = _sptr_basic_frame->frameData();	// depth data array

							std::vector<uint16_t> depth_list(_sptr_basic_frame->frameHeight() * _sptr_basic_frame->frameWidth());
							for (int y = 0; y < _sptr_basic_frame->frameHeight(); y++) {
								for (int x = 0; x < _sptr_basic_frame->frameWidth(); x++) {
									int _frame_index = y * _sptr_basic_frame->frameWidth() + x;
									depth_list[_frame_index] = (*_sptr_frame_data)[_frame_index];
								}
							}
							// cv::Mat 객체 생성
							//cv::Mat depthImage(_sptr_basic_frame->frameHeight(), _sptr_basic_frame->frameWidth(), CV_16U, depth_list.data());
							cv::Mat undistortedDepthImage = undistortDepthImage(depth_list); //undistorted
							//cv::Mat resizedundistortedDepthImage = resizeImage(undistortedDepthImage);
							cv::Mat displayImage;
							undistortedDepthImage.convertTo(displayImage, CV_8UC1, 255.0 / 5000);
							cv::applyColorMap(displayImage, displayImage, cv::COLORMAP_JET);
							//cv::imshow("Depth Image", displayImage);

							auto now = std::chrono::high_resolution_clock::now();
							auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
							auto epoch = now_ms.time_since_epoch();
							auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
							long duration = value.count();

							std::string current_time_ms = std::to_string(duration);

							int fontFace = cv::FONT_HERSHEY_SIMPLEX;
							double fontScale = 0.5;
							int thickness = 1;
							cv::Point textOrg(500, 470); // Position for the text, adjust as needed

							cv::putText(displayImage, current_time_ms, textOrg, fontFace, fontScale,
								cv::Scalar::all(255), thickness, 8);

							std::string windowName = "Depth Image " + std::to_string(thiz->cameraId);
							cv::imshow(windowName, displayImage);  // 윈도우 이름 변경
							if (thiz->videoWriter.isOpened()) {
								thiz->videoWriter.write(displayImage);
							}
							cv::waitKey(1); // GUI 이벤트 처리
						}
					}
				}
			}
		}
		if (thiz->videoWriter.isOpened()) {
			thiz->videoWriter.release();
		}
	}

public:
	ReceivedDepthFrameSink() = default;
	virtual ~ReceivedDepthFrameSink() = default;

protected: // edit
	//undistort image
	static cv::Mat undistortDepthImage(const std::vector<uint16_t>& depthData) {
		int height = 480;
		int width = 640;
		cv::Mat depthImage(height, width, CV_16U, const_cast<uint16_t*>(depthData.data()));

		cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 560.54, 0, 320.831, 0, 560.556, 235.293, 0, 0, 1);
		cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.277451, -0.631118, -0.00177032, 0.000668223, 0);

		cv::Mat undistortedImage;
		cv::undistort(depthImage, undistortedImage, cameraMatrix, distCoeffs);

		return undistortedImage;
	}
	static cv::Mat resizeImage(const cv::Mat& inputImage) {
		cv::Mat resizedImage;
		cv::Size newSize(196, 196); // 새 이미지 크기: 256x256

		// 이미지 리사이즈
		cv::resize(inputImage, resizedImage, newSize);

		return resizedImage;
	}

protected:
	bool mReadFrameThreadStart;
	std::thread mReadFrameThread;
	std::queue<meere::sensor::sptr_frame_list> mFrameListQueue;
} _ReceivedDepthFrameSink1(1), _ReceivedDepthFrameSink2(2);

class WebcamHandler {
public:
	cv::VideoCapture cap;
	cv::VideoWriter videoWriter;

	WebcamHandler() {
		// Initialize webcam stream (0 for the default webcam)
		int cameraIndex = selectCamera();
		cap.open(cameraIndex);

		// Initialize VideoWriter for the webcam
		videoWriter.open("webcam_output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(640, 480), true);
	}

	int selectCamera() {
		std::cout << "Available cameras:\n";
		int maxCameras = 10;
		for (int i = 0; i < maxCameras; ++i) {
			cv::VideoCapture testCap(i);
			if (testCap.isOpened()) {
				std::cout << "Camera Index " << i << " available\n";
				testCap.release();
			}
		}

		std::cout << "Enter the camera index to use: ";
		int index;
		std::cin >> index;
		return index;
	}


	void processWebcamFrames() {
		cv::Mat frame;
		auto start = std::chrono::high_resolution_clock::now();

		while (true) {
			auto now = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();

			if (duration >= 10) {
				break; // 10초 후에 촬영 중단
			}

			if (!cap.read(frame)) {
				break;  // 프레임이 없으면 루프 종료
			}

			auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
			auto epoch = now_ms.time_since_epoch();
			auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
			long duration2 = value.count();

			std::string current_time_ms = std::to_string(duration2);;

			int fontFace = cv::FONT_HERSHEY_SIMPLEX;
			double fontScale = 0.5;
			int thickness = 1;
			cv::Point textOrg(500, 470); // Position for the text, adjust as needed

			cv::putText(frame, current_time_ms, textOrg, fontFace, fontScale,
				cv::Scalar::all(255), thickness, 8);

			// 프레임 표시
			cv::imshow("Webcam", frame);

			// 파일에 프레임 쓰기
			if (videoWriter.isOpened()) {
				videoWriter.write(frame);
			}

			if (cv::waitKey(1) >= 0) {
				break;  // 키 입력시 루프 종료
			}
		}

		// VideoWriter 객체 해제
		if (videoWriter.isOpened()) {
			videoWriter.release();
		}
	}
};

void initializeAndRunCamera(meere::sensor::sptr_camera& camera, ReceivedDepthFrameSink* sink) {
	if (!camera) {
		std::cerr << "Camera is null." << std::endl;
		return;
	}

	camera->addSink(sink);

	meere::sensor::result _rt = camera->prepare();
	assert(meere::sensor::success == _rt);
	if (meere::sensor::success != _rt) {
		std::cerr << "Camera prepare failed." << std::endl;
		return;
	}

	int _wantedFrame = meere::sensor::CubeEyeFrame::FrameType_Depth;
	_rt = camera->run(_wantedFrame);
	assert(meere::sensor::success == _rt);
	if (meere::sensor::success != _rt) {
		std::cerr << "Camera run failed." << std::endl;
		return;
	}

	std::this_thread::sleep_for(std::chrono::milliseconds(10000));  // 10초간 실행

	camera->stop();
	camera->release();
	meere::sensor::destroy_camera(camera);
	camera.reset();
}


int main(int argc, char* argv[])
{
	// search ToF camera source
	// camera is always 1
	meere::sensor::sptr_source_list _source_list = meere::sensor::search_camera_source();

	if (_source_list->size() < 2) {
        std::cout << "Two devices are required!" << std::endl;
        return -1;
    }

	meere::sensor::add_prepared_listener(&_ReceivedDepthFrameSink1);
	meere::sensor::add_prepared_listener(&_ReceivedDepthFrameSink2);

	meere::sensor::result _rt;

	// create ToF camera
	meere::sensor::sptr_camera _camera1 = meere::sensor::create_camera(_source_list->at(0));
	meere::sensor::sptr_camera _camera2 = meere::sensor::create_camera(_source_list->at(1));

	WebcamHandler webcamHandler;
	std::thread webcamThread(&WebcamHandler::processWebcamFrames, &webcamHandler);
    std::thread camera1Thread(initializeAndRunCamera, std::ref(_camera1), &_ReceivedDepthFrameSink1);
    std::thread camera2Thread(initializeAndRunCamera, std::ref(_camera2), &_ReceivedDepthFrameSink2);

    // 스레드가 종료될 때까지 대기
	webcamThread.join();
    camera1Thread.join();
    camera2Thread.join();

    return 0;
}
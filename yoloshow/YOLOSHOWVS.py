# -*- coding: utf-8 -*-
from utils import glo
import json
import os
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFileDialog, QMainWindow
from PySide6.QtCore import QTimer, Qt
from PySide6 import QtCore, QtGui
from ui.YOLOSHOWUIVS import Ui_MainWindow
from yoloshow.YOLOSHOWBASE import YOLOSHOWBASE, MODEL_THREAD_CLASSES
from yoloshow.YOLOThreadPool import YOLOThreadPool
GLOBAL_WINDOW_STATE = True
WIDTH_LEFT_BOX_STANDARD = 80
WIDTH_LEFT_BOX_EXTENDED = 200
WIDTH_LOGO = 60
UI_FILE_PATH = "ui/YOLOSHOWUIVS.ui"

# YOLOSHOW窗口类 动态加载UI文件 和 Ui_mainWindow
class YOLOSHOWVS(QMainWindow, YOLOSHOWBASE):
    def __init__(self):
        super().__init__()
        self.current_workpath = os.getcwd()
        self.inputPath = None
        self.result_statistic = None
        self.detect_result = None
        # --- 加载UI --- #
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground, True)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无头窗口
        # --- 加载UI --- #
        # 初始化侧边栏
        self.initSiderWidget()
        # --- 最大化 最小化 关闭 --- #
        self.ui.maximizeButton.clicked.connect(self.maxorRestore)
        self.ui.minimizeButton.clicked.connect(self.showMinimized)
        self.ui.closeButton.clicked.connect(self.close)
        self.ui.topbox.doubleClickFrame.connect(self.maxorRestore)
        # --- 最大化 最小化 关闭 --- #
        # --- 播放 暂停 停止 --- #
        self.playIcon = QtGui.QIcon()
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/play.png"), QtGui.QIcon.Normal,
                                QtGui.QIcon.Off)
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/pause.png"), QtGui.QIcon.Active,
                                QtGui.QIcon.On)
        self.playIcon.addPixmap(QtGui.QPixmap(f"{self.current_workpath}/images/newsize/pause.png"),
                                QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.ui.run_button.setCheckable(True)
        self.ui.run_button.setIcon(self.playIcon)
        # --- 播放 暂停 停止 --- #
        # --- 侧边栏缩放 --- #
        self.ui.src_menu.clicked.connect(self.scaleMenu)  # hide menu button
        self.ui.src_setting.clicked.connect(self.scalSetting)  # setting button
        # --- 侧边栏缩放 --- #
        # --- 自动加载/动态改变 PT 模型 --- #
        self.pt_Path = f"{self.current_workpath}/ptfiles/"
        os.makedirs(self.pt_Path, exist_ok=True)
        self.pt_list = os.listdir(f'{self.current_workpath}/ptfiles/')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize(f'{self.current_workpath}/ptfiles/' + x))
        # self.solveYoloConflict([f"{self.current_workpath}/ptfiles/" + pt_file for pt_file in self.pt_list]) # Consider if this is still needed
        self.ui.model_box1.clear()
        self.ui.model_box1.addItems(self.pt_list)
        self.ui.model_box2.clear()
        self.ui.model_box2.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.loadModels()) # Use unified loadModels
        self.qtimer_search.start(2000)
        self.ui.model_box1.currentTextChanged.connect(lambda: self.changeModel("left"))
        self.ui.model_box2.currentTextChanged.connect(lambda: self.changeModel("right"))
        # --- 自动加载/动态改变 PT 模型 --- #
        # --- 超参数调整 --- #
        self.ui.iou_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'iou_spinbox'))  # iou box
        self.ui.iou_slider.valueChanged.connect(lambda x: self.changeValue(x, 'iou_slider'))  # iou scroll bar
        self.ui.conf_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'conf_spinbox'))  # conf box
        self.ui.conf_slider.valueChanged.connect(lambda x: self.changeValue(x, 'conf_slider'))  # conf scroll bar
        self.ui.speed_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'speed_spinbox'))  # speed box
        self.ui.speed_slider.valueChanged.connect(lambda x: self.changeValue(x, 'speed_slider'))  # speed scroll bar
        self.ui.line_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'line_spinbox'))  # line box
        self.ui.line_slider.valueChanged.connect(lambda x: self.changeValue(x, 'line_slider'))  # line slider
        # --- 超参数调整 --- #
        # --- 导入 图片/视频、调用摄像头、导入文件夹（批量处理）、调用网络摄像头、结果统计图片、结果统计表格 --- #
        self.ui.src_img.clicked.connect(self.selectFile)
        # 对比模型模式 不支持同时读取摄像头流 (Keep webcam disabled for VS mode as intended)
        # self.src_webcam.clicked.connect(self.selectWebcam)
        self.ui.src_folder.clicked.connect(self.selectFolder)
        self.ui.src_camera.clicked.connect(self.selectRtsp)
        # Result buttons might need adjustment for VS mode if they rely on single model output
        # self.ui.src_result.clicked.connect(self.showResultStatics)
        # self.ui.src_table.clicked.connect(self.showTableResult)
        # --- 导入 图片/视频、调用摄像头、导入文件夹（批量处理）、调用网络摄像头、结果统计图片、结果统计表格 --- #
        # --- 导入模型、 导出结果 --- #
        self.ui.import_button.clicked.connect(self.importModel)
        self.ui.save_status_button.clicked.connect(self.saveStatus)
        self.ui.save_button.clicked.connect(self.saveResult) # saveResult needs adaptation for VS mode
        self.ui.save_button.setEnabled(False)
        # --- 导入模型、 导出结果 --- #
        # --- 视频、图片 预览 --- #
        # Input goes to a shared label, outputs to left/right boxes
        # self.ui.main_inputbox.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.ui.main_leftbox.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.ui.main_rightbox.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        # --- 视频、图片 预览 --- #
        # --- 状态栏 初始化 --- #
        # 状态栏阴影效果 (Assuming UI names like Class_QF1, etc. exist)
        self.shadowStyle(self.ui.mainBody, QColor(0, 0, 0, 38), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Class_QF1, QColor(142, 197, 252), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Target_QF1, QColor(159, 172, 230), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Fps_QF1, QColor(170, 128, 213), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Model_QF1, QColor(162, 129, 247), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Class_QF2, QColor(142, 197, 252), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Target_QF2, QColor(159, 172, 230), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Fps_QF2, QColor(170, 128, 213), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Model_QF2, QColor(162, 129, 247), top_bottom=['top', 'bottom'])
        # 状态栏默认显示
        self.model_name1 = self.ui.model_box1.currentText()  # 获取默认 model
        self.ui.Class_num1.setText('--')
        self.ui.Target_num1.setText('--')
        self.ui.fps_label1.setText('--')
        # ++++++++++++++++++ CHANGE HERE ++++++++++++++++++
        # self.ui.Model_label1.setText(str(self.model_name1).replace(".pt", "")) # Original line
        self.ui.Model_label1.setText("SDES-YOLO") # Changed line
        # ++++++++++++++++++++++++++++++++++++++++++++++++++

        self.model_name2 = self.ui.model_box2.currentText()  # 获取默认 model
        self.ui.Class_num2.setText('--')
        self.ui.Target_num2.setText('--')
        self.ui.fps_label2.setText('--')
        # ++++++++++++++++++ CHANGE HERE ++++++++++++++++++
        # self.ui.Model_label2.setText(str(self.model_name2).replace(".pt", "")) # Original line
        self.ui.Model_label2.setText("SDES-YOLO") # Changed line
        # ++++++++++++++++++++++++++++++++++++++++++++++++++
        # --- 状态栏 初始化 --- #
        self.initThreads()
        # --- 开始 / 停止 --- #
        self.ui.run_button.clicked.connect(self.runorContinue)
        self.ui.stop_button.clicked.connect(self.stopDetect)
        # --- 开始 / 停止 --- #
        # --- Setting栏 初始化 --- #
        self.loadConfig()
        # --- Setting栏 初始化 --- #
        # --- MessageBar Init --- #
        self.showStatus("Welcome to YOLOSHOW VS Mode")
        # --- MessageBar Init --- #

    # 重写 YOLOSHOWBASE 的 loadModels 以更新两个 box
    def loadModels(self):
        try:
            pt_list = os.listdir(f'{self.current_workpath}/ptfiles/')
            pt_list = [file for file in pt_list if file.endswith('.pt')]
            pt_list.sort(key=lambda x: os.path.getsize(f'{self.current_workpath}/ptfiles/' + x))
            if pt_list != self.pt_list:
                self.pt_list = pt_list
                current_text1 = self.ui.model_box1.currentText()
                current_text2 = self.ui.model_box2.currentText()
                self.ui.model_box1.clear()
                self.ui.model_box1.addItems(self.pt_list)
                self.ui.model_box2.clear()
                self.ui.model_box2.addItems(self.pt_list)
                if current_text1 in self.pt_list:
                    self.ui.model_box1.setCurrentText(current_text1)
                if current_text2 in self.pt_list:
                    self.ui.model_box2.setCurrentText(current_text2)
        except Exception as e:
            self.showStatus(f"Error loading models: {e}")

    # 初始化模型线程 (Handles both left and right)
    def initThreads(self):
        self.yolo_threads = YOLOThreadPool()
        # 获取当前Model 类型 key (e.g., yolov8_left)
        model_type_key_left = self.checkCurrentModel(mode="left")
        model_type_key_right = self.checkCurrentModel(mode="right")

        if model_type_key_left and model_type_key_left in MODEL_THREAD_CLASSES:
            if not self.yolo_threads.get(model_type_key_left): # Create only if not exists
                 self.yolo_threads.set(model_type_key_left, MODEL_THREAD_CLASSES[model_type_key_left]())
                 self.initModel(yoloname=model_type_key_left)
        if model_type_key_right and model_type_key_right in MODEL_THREAD_CLASSES:
             if not self.yolo_threads.get(model_type_key_right): # Create only if not exists
                self.yolo_threads.set(model_type_key_right, MODEL_THREAD_CLASSES[model_type_key_right]())
                self.initModel(yoloname=model_type_key_right)


    # 重写 YOLOSHOWBASE 的 initModel 以连接到正确的 UI 元素
    def initModel(self, yoloname=None):
        """
        Initializes and connects signals for a specific YOLO thread (left or right).
        Args:
            yoloname (str): The name of the YOLO thread (e.g., "yolov8_left").
        """
        yolo_thread = self.yolo_threads.get(yoloname)
        if not yolo_thread:
            # This case should ideally not happen if initThreads is called correctly
            self.showStatus(f"Error: Could not find thread instance for {yoloname}")
            return
            # raise ValueError(f"No thread found for '{yoloname}'") # Or raise error

        # Disconnect any existing connections first to prevent duplicates
        try:
            yolo_thread.send_output.disconnect()
            yolo_thread.send_msg.disconnect()
            yolo_thread.send_progress.disconnect()
            yolo_thread.send_fps.disconnect()
            yolo_thread.send_class_num.disconnect()
            yolo_thread.send_target_num.disconnect()
            # Disconnect input signal if it exists (seems less critical here)
            # yolo_thread.send_input.disconnect()
        except (TypeError, RuntimeError):
            pass # Ignore errors if signals were not connected


        # Common settings
        yolo_thread.progress_value = self.ui.progress_bar.maximum()
        # Use main_inputbox for input preview in VS mode
        # yolo_thread.send_input.connect(lambda img: self.showImg(img, self.ui.main_inputbox, 'img'))


        if yoloname.endswith("_left"):
            # 左侧模型加载
            selected_model_file = self.ui.model_box1.currentText()
            if selected_model_file:
                 yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + selected_model_file
            else:
                 self.showStatus("Warning: No model selected for left side.")
                 # Optionally set a default or prevent running

            yolo_thread.send_output.connect(lambda x: self.showImg(x, self.ui.main_leftbox, 'img'))
            # Let left model control the main message bar
            yolo_thread.send_msg.connect(self.showStatus) # Connect directly
            yolo_thread.send_fps.connect(lambda x: self.ui.fps_label1.setText(str(x)))
            yolo_thread.send_class_num.connect(lambda x: self.ui.Class_num1.setText(str(x)))
            yolo_thread.send_target_num.connect(lambda x: self.ui.Target_num1.setText(str(x)))
             # Left model can contribute to progress, but let right control final value?
            # yolo_thread.send_progress.connect(lambda x: self.update_progress(x, "left"))

        elif yoloname.endswith("_right"):
            # 右侧模型加载
            selected_model_file = self.ui.model_box2.currentText()
            if selected_model_file:
                yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + selected_model_file
            else:
                 self.showStatus("Warning: No model selected for right side.")

            yolo_thread.send_output.connect(lambda x: self.showImg(x, self.ui.main_rightbox, 'img'))
            # Right model controls the progress bar value entirely
            yolo_thread.send_progress.connect(lambda x: self.ui.progress_bar.setValue(x))
            yolo_thread.send_fps.connect(lambda x: self.ui.fps_label2.setText(str(x)))
            yolo_thread.send_class_num.connect(lambda x: self.ui.Class_num2.setText(str(x)))
            yolo_thread.send_target_num.connect(lambda x: self.ui.Target_num2.setText(str(x)))
             # Optionally, right model can also send messages if needed for specific errors
             # yolo_thread.send_msg.connect(lambda msg: self.showStatus(f"Right: {msg}"))

    # 重写 YOLOSHOWBASE 的 showStatus 以处理 VS 模式的完成/停止
    def showStatus(self, msg):
        self.ui.message_bar.setText(msg)
        # In VS mode, 'Finish' or 'Stop' might come from either thread.
        # We stop both when either finishes or is stopped.
        if msg == 'Finish Detection' or msg == 'Stop Detection':
             # Check button state BEFORE quitting models, as quitRunningModel might change it
             is_stopping = (msg == 'Stop Detection')
             current_button_checked = self.ui.run_button.isChecked()

             self.quitRunningModel(stop_status=is_stopping) # Stop all threads

             # Update UI elements consistently
             self.ui.run_button.setChecked(False)
             self.ui.save_status_button.setEnabled(True)
             self.ui.progress_bar.setValue(0)
             # Clear only output boxes
             self.ui.main_leftbox.clear()
             self.ui.main_rightbox.clear()
             # Reset stats for both sides
             self.ui.Class_num1.setText('--')
             self.ui.Target_num1.setText('--')
             self.ui.fps_label1.setText('--')
             self.ui.Class_num2.setText('--')
             self.ui.Target_num2.setText('--')
             self.ui.fps_label2.setText('--')

             # Restore input preview if it was cleared by mistake or if desired
             if self.inputPath and not isinstance(self.inputPath, list): # If single input
                try:
                    if any(str(self.inputPath).lower().endswith(ext) for ext in ['.jpg', '.png', '.jpeg', '.bmp', '.dib', '.jpe', '.jp2']):
                        self.showImg(str(self.inputPath), self.ui.main_inputbox, 'path')
                    # Add handling for video first frame if needed
                except Exception as e:
                    print(f"Error restoring input preview: {e}")


    # 导出结果 (Needs Adaptation for VS Mode)
    def saveResult(self):
        # How to save results in VS mode? Save both? Concatenate videos?
        # Current implementation uses the left model's status and path.
        # This needs revision based on desired VS save behavior.

        # Example: Save left model's output only (as currently implemented)
        left_model_key = self.checkCurrentModel(mode="left")
        if not left_model_key:
             self.showStatus("Cannot save: Left model not selected or invalid.")
             return
        left_thread = self.yolo_threads.get(left_model_key)

        if not left_thread or not left_thread.res_status: # Check thread and its status
            self.showStatus("Please run detection and wait for results before saving (Left Model).")
            return

        config_file = f'{self.current_workpath}/config/save.json'
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {} # Create empty config if file doesn't exist
        save_path = config.get('save_path', os.getcwd())

        is_folder = isinstance(self.inputPath, list)
        OutputDir = None # Initialize OutputDir

        if is_folder:
            # Saving folder results in VS mode - complex. Save left model's results?
            # This assumes left_thread handles folder saving internally correctly.
            OutputDir = QFileDialog.getExistingDirectory(
                self, "Save Left Model Results Folder", save_path
            )
            if OutputDir: # Proceed only if a directory was selected
                 self.saveResultProcess(OutputDir, left_model_key, folder=True)
        else:
            # Saving single file result - save left model's output?
            # Suggest adding model side to filename, e.g., "output_left.mp4"
            default_filename = "result_left"
            if isinstance(self.inputPath, str):
                 base, _ = os.path.splitext(os.path.basename(self.inputPath))
                 default_filename = f"{base}_left_result"

            OutputDir, _ = QFileDialog.getSaveFileName(
                self, "Save Left Model Image/Video", os.path.join(save_path, default_filename),
                "Image/Video Type (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi)" # Adjusted filter
            )
            if OutputDir: # Proceed only if a file name was provided
                self.saveResultProcess(OutputDir, left_model_key, folder=False)

        # Update save path only if a save occurred
        if OutputDir:
            # Store the directory containing the saved file/folder
            config['save_path'] = os.path.dirname(OutputDir)
            try:
                config_json = json.dumps(config, ensure_ascii=False, indent=2)
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(config_json)
            except Exception as e:
                self.showStatus(f"Error saving config: {e}")
        else:
             self.showStatus("Save cancelled.")


    # 重新加载模型 (Needs Adaptation for VS Mode - likely unused directly)
    # def resignModel(self, model_name, mode=None):
        # Resigning a single model in VS mode might be complex.
        # ChangeModel handles stopping/starting.
        # pass

    def stopOtherModelProcess(self, model_name_to_stop, model_to_keep=None, side_to_keep=None):
        """Stops a specific model thread."""
        yolo_thread = self.yolo_threads.get(model_name_to_stop)
        if yolo_thread and yolo_thread.isRunning():
            # Disconnect finished signal if connected, handle potential errors
            try:
                yolo_thread.finished.disconnect()
            except (TypeError, RuntimeError):
                pass
            # No automatic restart (resignModel) needed when just stopping others
            # if model_to_keep and side_to_keep:
            #    yolo_thread.finished.connect(lambda: self.check_and_restart_other(model_to_keep, side_to_keep))

            yolo_thread.stop_dtc = True
            self.yolo_threads.stop_thread(model_name_to_stop)
            self.showStatus(f"Stopping thread: {model_name_to_stop}")


    # 停止其他模型 (Stop all except the one specified by current_yoloname)
    def stopOtherModel(self, current_yoloname_to_keep=None):
        """Stops all running YOLO threads except the one specified."""
        for model_name in list(self.yolo_threads.threads_pool.keys()):
            if model_name == current_yoloname_to_keep:
                continue # Skip the one we want to keep/change
            thread_instance = self.yolo_threads.get(model_name)
            if thread_instance and thread_instance.isRunning():
                 self.stopOtherModelProcess(model_name) # Stop the thread
            elif not thread_instance: # Clean up invalid entries
                 self.yolo_threads.remove(model_name)


    # 暂停/继续 另一侧模型 (Simpler approach: runorContinue controls both via is_continue)
    # def PauseAnotherModel(self, mode=None): # Likely not needed
    # def ContinueAnotherModel(self, mode=None): # Likely not needed


    def changeModelProcess(self, yoloname, mode=None):
        """Handles the logic for changing a model on either the left or right side."""
        if not mode or not yoloname:
            return

        # 1. Stop any other running model first
        # We stop ALL others, including the one being replaced if it was running.
        self.stopOtherModel(yoloname) # Stop all except potentially the new one (if key matches)

        # 2. Get or create the thread instance for the new model type key
        if yoloname not in MODEL_THREAD_CLASSES:
             self.showStatus(f"Error: Unknown model type key {yoloname}")
             return

        yolo_thread = self.yolo_threads.get(yoloname)
        model_box = self.ui.model_box1 if mode == "left" else self.ui.model_box2
        selected_pt_file = model_box.currentText()

        if not selected_pt_file:
            self.showStatus(f"No model file selected for {mode} side.")
            # Optionally stop the existing thread if one exists for this side
            if yolo_thread: yolo_thread.stop_dtc = True; self.yolo_threads.stop_thread(yoloname)
            return

        model_file_path = f'{self.current_workpath}/ptfiles/' + selected_pt_file

        if yolo_thread:
             # Thread exists, update its target model file
             yolo_thread.new_model_name = model_file_path
             # Ensure it's stopped if it was running under the old model
             if yolo_thread.isRunning():
                 yolo_thread.stop_dtc = True
                 self.yolo_threads.stop_thread(yoloname)
             # Re-initialize connections (signals might have been disconnected)
             self.initModel(yoloname=yoloname)

        else:
             # Thread doesn't exist, create, set model, and initialize
             self.yolo_threads.set(yoloname, MODEL_THREAD_CLASSES[yoloname]())
             yolo_thread = self.yolo_threads.get(yoloname) # Get the new instance
             yolo_thread.new_model_name = model_file_path
             self.initModel(yoloname=yoloname) # Connect signals

        # Reload common config (iou, conf etc.) - applied to all threads later via changeValue
        self.loadConfig()

        # Update status message
        now_model_file_name = os.path.basename(selected_pt_file)
        self.showStatus(f"Changed model for {mode.capitalize()} to {now_model_file_name}.")

        # Update global variable if still used elsewhere (consider removing dependency)
        # glo_key = 'yoloname1' if mode == 'left' else 'yoloname2'
        # glo.set_value(glo_key, yoloname) # Storing type key like 'yolov8_left'


    # Model 变化
    def changeModel(self, mode=None):
        """Handles model selection change for either left or right side."""
        if mode == "left":
            self.model_name1 = self.ui.model_box1.currentText()
            # ++++++++++++++++++ CHANGE HERE ++++++++++++++++++
            # self.ui.Model_label1.setText(str(self.model_name1).replace(".pt", "")) # Original
            self.ui.Model_label1.setText("SDES-YOLO") # Changed
            # ++++++++++++++++++++++++++++++++++++++++++++++++++
            yolo_type_key = self.checkCurrentModel(mode="left") # Gets e.g., "yolov8_left"
            if yolo_type_key:
                self.changeModelProcess(yolo_type_key, "left")
            else:
                self.showStatus("Unsupported model type selected for left side.")
                # Stop the left thread if it exists and was running
                left_key_old = self.checkCurrentModel(mode="left", old_model_name=self.model_name1) # Get key of previous model
                if left_key_old: self.stopOtherModelProcess(left_key_old)


        elif mode == "right":
            self.model_name2 = self.ui.model_box2.currentText()
             # ++++++++++++++++++ CHANGE HERE ++++++++++++++++++
            # self.ui.Model_label2.setText(str(self.model_name2).replace(".pt", "")) # Original
            self.ui.Model_label2.setText("SDES-YOLO") # Changed
            # ++++++++++++++++++++++++++++++++++++++++++++++++++
            yolo_type_key = self.checkCurrentModel(mode="right") # Gets e.g., "yolov8_right"
            if yolo_type_key:
                self.changeModelProcess(yolo_type_key, "right")
            else:
                self.showStatus("Unsupported model type selected for right side.")
                # Stop the right thread if it exists and was running
                right_key_old = self.checkCurrentModel(mode="right", old_model_name=self.model_name2) # Get key of previous model
                if right_key_old: self.stopOtherModelProcess(right_key_old)


    def runSideModelProcess(self, model_type_key):
        """ Prepares and starts or pauses a single side's model thread. """
        yolo_thread = self.yolo_threads.get(model_type_key)
        if not yolo_thread:
            self.showStatus(f"Cannot run: Thread for {model_type_key} not found.")
            return False # Indicate failure

        yolo_thread.source = self.inputPath # Set input source
        yolo_thread.stop_dtc = False      # Ensure stop flag is reset

        # Determine model file based on side from the key
        side = "left" if model_type_key.endswith("_left") else "right"
        model_box = self.ui.model_box1 if side == "left" else self.ui.model_box2
        selected_pt_file = model_box.currentText()

        if not selected_pt_file:
            self.showStatus(f"Cannot run: No model file selected for {side} side.")
            return False # Indicate failure
        yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + selected_pt_file

        if self.ui.run_button.isChecked():
            yolo_thread.is_continue = True
            if not yolo_thread.isRunning():
                 self.yolo_threads.start_thread(model_type_key)
                 self.showStatus(f"Starting {side} model: {os.path.basename(selected_pt_file)}")
            # else: # Already running, just ensure is_continue is True
                 # self.showStatus(f"Resuming {side} model.")

        else: # Pausing
            yolo_thread.is_continue = False
            # No need to stop the thread, just set flag
            # self.showStatus(f"Pausing {side} model.") # Avoid too many messages
        return True # Indicate success


    # 运行模型 (Run both sides)
    def runModel(self, runbuttonStatus=None):
        self.ui.save_status_button.setEnabled(False)
        if runbuttonStatus is not None:
            self.ui.run_button.setChecked(runbuttonStatus)

        # Check if models are selected and valid
        model_key_left = self.checkCurrentModel(mode="left")
        model_key_right = self.checkCurrentModel(mode="right")

        if not model_key_left or not model_key_right:
             self.showStatus('Cannot start: Select valid models for both sides.')
             if self.ui.run_button.isChecked(): self.ui.run_button.setChecked(False)
             self.ui.save_status_button.setEnabled(True)
             return

        # Run process for both sides
        success_left = self.runSideModelProcess(model_key_left)
        success_right = self.runSideModelProcess(model_key_right)

        if not success_left or not success_right:
             # If either failed to start (e.g., no model file), uncheck button
             if self.ui.run_button.isChecked(): self.ui.run_button.setChecked(False)
             self.ui.save_status_button.setEnabled(True)
             # Stop the one that might have started successfully
             if success_left and model_key_left: self.stopOtherModelProcess(model_key_left)
             if success_right and model_key_right: self.stopOtherModelProcess(model_key_right)
        elif not self.ui.run_button.isChecked():
            self.showStatus("Pause Detection") # General pause message


    # 开始/暂停 预测
    def runorContinue(self):
        if self.inputPath is None:
            self.showStatus("Please select the Image/Video before starting detection...")
            self.ui.run_button.setChecked(False)
            return

        # Optional: Reload global settings if they affect inference directly
        # glo.set_value('yoloname1', self.model_name1) # Less critical if threads read directly
        # glo.set_value('yoloname2', self.model_name2)
        # self.reloadModel() # Avoid frequent reloads unless necessary

        self.runModel() # Handles starting/pausing both sides based on button state

    # 停止识别
    def stopDetect(self):
        self.quitRunningModel(stop_status=True) # Stops all threads in pool
        self.ui.run_button.setChecked(False)
        self.ui.save_status_button.setEnabled(True)
        self.ui.progress_bar.setValue(0)
        # Clear output displays
        self.ui.main_leftbox.clear()
        self.ui.main_rightbox.clear()
        # Reset stats for both sides
        self.ui.Class_num1.setText('--')
        self.ui.Target_num1.setText('--')
        self.ui.fps_label1.setText('--')
        self.ui.Class_num2.setText('--')
        self.ui.Target_num2.setText('--')
        self.ui.fps_label2.setText('--')
        # Keep input preview
        # self.ui.main_inputbox.clear()
        self.showStatus("Detection Stopped.")
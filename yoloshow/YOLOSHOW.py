# -*- coding: utf-8 -*-
from utils import glo
import json
import os
from ui.YOLOSHOWUI import Ui_MainWindow
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFileDialog, QMainWindow
from yoloshow.YOLOThreadPool import YOLOThreadPool
from PySide6.QtCore import QTimer, Qt
from PySide6 import QtCore, QtGui
from yoloshow.YOLOSHOWBASE import YOLOSHOWBASE, MODEL_THREAD_CLASSES
GLOBAL_WINDOW_STATE = True
WIDTH_LEFT_BOX_STANDARD = 80
WIDTH_LEFT_BOX_EXTENDED = 200
WIDTH_LOGO = 60
UI_FILE_PATH = "ui/YOLOSHOWUI.ui"
KEYS_LEFT_BOX_MENU = ['src_menu', 'src_setting', 'src_webcam', 'src_folder', 'src_camera', 'src_vsmode', 'src_setting']

# YOLOSHOW窗口类 动态加载UI文件 和 Ui_mainWindow
class YOLOSHOW(QMainWindow, YOLOSHOWBASE):
    def __init__(self):
        super().__init__()
        self.current_model = None
        self.current_workpath = os.getcwd()
        self.inputPath = None
        self.result_statistic = None
        self.detect_result = None
        # --- 加载UI --- #
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground, True)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无头窗口
        self.initSiderWidget()
        # --- 加载UI --- #
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
        self.ui.model_box.clear()
        self.ui.model_box.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.loadModels())
        self.qtimer_search.start(2000)
        self.ui.model_box.currentTextChanged.connect(self.changeModel)
        # --- 自动加载/动态改变 PT 模型 --- #
        # --- 导入 图片/视频、调用摄像头、导入文件夹（批量处理）、调用网络摄像头、结果统计图片、结果统计表格 --- #
        self.ui.src_img.clicked.connect(self.selectFile)
        self.ui.src_webcam.clicked.connect(self.selectWebcam)
        self.ui.src_folder.clicked.connect(self.selectFolder)
        self.ui.src_camera.clicked.connect(self.selectRtsp)
        self.ui.src_result.clicked.connect(self.showResultStatics)
        self.ui.src_table.clicked.connect(self.showTableResult)
        # --- 导入 图片/视频、调用摄像头、导入文件夹（批量处理）、调用网络摄像头 --- #
        # --- 导入模型、 导出结果 --- #
        self.ui.import_button.clicked.connect(self.importModel)
        self.ui.save_status_button.clicked.connect(self.saveStatus)
        self.ui.save_button.clicked.connect(self.saveResult)
        self.ui.save_button.setEnabled(False)
        # --- 导入模型、 导出结果 --- #
        # --- 视频、图片 预览 --- #
        self.ui.main_leftbox.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.ui.main_rightbox.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        # --- 视频、图片 预览 --- #
        # --- 状态栏 初始化 --- #
        # 状态栏阴影效果
        self.shadowStyle(self.ui.mainBody, QColor(0, 0, 0, 38), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Class_QF, QColor(142, 197, 252), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.classesLabel, QColor(142, 197, 252), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Target_QF, QColor(159, 172, 230), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.targetLabel, QColor(159, 172, 230), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Fps_QF, QColor(170, 128, 213), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.fpsLabel, QColor(170, 128, 213), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.Model_QF, QColor(162, 129, 247), top_bottom=['top', 'bottom'])
        self.shadowStyle(self.ui.modelLabel, QColor(162, 129, 247), top_bottom=['top', 'bottom'])
        # 状态栏默认显示
        self.model_name = self.ui.model_box.currentText()  # 获取默认 model
        self.ui.Class_num.setText('--')
        self.ui.Target_num.setText('--')
        self.ui.fps_label.setText('--')
        # ++++++++++++++++++ CHANGE HERE ++++++++++++++++++
        # self.ui.Model_label.setText(str(self.model_name).replace(".pt", "")) # Original line
        self.ui.Model_label.setText("SDES-YOLO") # Changed line
        # ++++++++++++++++++++++++++++++++++++++++++++++++++
        # --- 状态栏 初始化 --- #
        self.initThreads()
        # --- 超参数调整 --- #
        self.ui.iou_spinbox.valueChanged.connect(
            lambda x: self.changeValue(x, 'iou_spinbox'))  # iou box
        self.ui.iou_slider.valueChanged.connect(lambda x: self.changeValue(x, 'iou_slider'))  # iou scroll bar
        self.ui.conf_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'conf_spinbox'))  # conf box
        self.ui.conf_slider.valueChanged.connect(lambda x: self.changeValue(x, 'conf_slider'))  # conf scroll bar
        self.ui.speed_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'speed_spinbox'))  # speed box
        self.ui.speed_slider.valueChanged.connect(lambda x: self.changeValue(x, 'speed_slider'))  # speed scroll bar
        self.ui.line_spinbox.valueChanged.connect(lambda x: self.changeValue(x, 'line_spinbox'))  # line box
        self.ui.line_slider.valueChanged.connect(lambda x: self.changeValue(x, 'line_slider'))  # line slider
        # --- 超参数调整 --- #
        # --- 开始 / 停止 --- #
        self.ui.run_button.clicked.connect(self.runorContinue)
        self.ui.stop_button.clicked.connect(self.stopDetect)
        # --- 开始 / 停止 --- #
        # --- Setting栏 初始化 --- #
        self.loadConfig()
        # --- Setting栏 初始化 --- #
        # --- MessageBar Init --- #
        self.showStatus("Welcome to YOLOSHOW")
        # --- MessageBar Init --- #

    def initThreads(self):
        self.yolo_threads = YOLOThreadPool()
        # 获取当前Model
        model_name = self.checkCurrentModel()
        if model_name:
            self.yolo_threads.set(model_name, MODEL_THREAD_CLASSES[model_name]())
            self.initModel(yoloname=model_name)

    # 导出结果
    def saveResult(self):
        if not any(thread.res_status for thread in self.yolo_threads.threads_pool.values()):
            self.showStatus("Please select the Image/Video before starting detection...")
            return
        config_file = f'{self.current_workpath}/config/save.json'
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        save_path = config.get('save_path', os.getcwd())
        is_folder = isinstance(self.inputPath, list)
        if is_folder:
            self.OutputDir = QFileDialog.getExistingDirectory(
                self,  # 父窗口对象
                "Save Results in new Folder",  # 标题
                save_path,  # 起始目录
            )
            current_model_name = self.checkCurrentModel()
            self.saveResultProcess(self.OutputDir, current_model_name, folder=True)
        else:
            self.OutputDir, _ = QFileDialog.getSaveFileName(
                self,  # 父窗口对象
                "Save Image/Video",  # 标题
                save_path,  # 起始目录
                "Image/Vide Type (*.jpg *.jpeg *.png *.bmp *.dib  *.jpe  *.jp2 *.mp4)"  # 选择类型过滤项，过滤内容在括号中
            )
            current_model_name = self.checkCurrentModel()
            self.saveResultProcess(self.OutputDir, current_model_name, folder=False)
        # 检查OutputDir是否为空
        if not self.OutputDir:
             self.showStatus('Save Cancelled.')
             return
        config['save_path'] = self.OutputDir
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)

    # 加载 pt 模型到 model_box
    def loadModels(self):
        try:
            pt_list = os.listdir(f'{self.current_workpath}/ptfiles/')
            pt_list = [file for file in pt_list if file.endswith('.pt')]
            pt_list.sort(key=lambda x: os.path.getsize(f'{self.current_workpath}/ptfiles/' + x))
            if pt_list != self.pt_list:
                self.pt_list = pt_list
                current_text = self.ui.model_box.currentText() # Store current selection
                self.ui.model_box.clear()
                self.ui.model_box.addItems(self.pt_list)
                if current_text in self.pt_list: # Restore selection if still exists
                    self.ui.model_box.setCurrentText(current_text)
        except Exception as e:
            self.showStatus(f"Error loading models: {e}")


    def stopOtherModelProcess(self, model_name, current_yoloname):
        yolo_thread = self.yolo_threads.get(model_name)
        if yolo_thread: # Check if thread exists
             # Disconnect previous connection if exists to avoid multiple calls
            try:
                yolo_thread.finished.disconnect()
            except (TypeError, RuntimeError): # Handle cases where it was not connected or already disconnected
                pass
            yolo_thread.finished.connect(lambda: self.resignModel(current_yoloname))
            yolo_thread.stop_dtc = True
            self.yolo_threads.stop_thread(model_name)

    # 停止其他模型
    def stopOtherModel(self, current_yoloname=None):
        for model_name in list(self.yolo_threads.threads_pool.keys()): # Iterate over a copy of keys
            if model_name == current_yoloname:
                continue
            thread_instance = self.yolo_threads.get(model_name)
            if thread_instance and thread_instance.isRunning():
                self.stopOtherModelProcess(model_name, current_yoloname)
            elif not thread_instance: # If thread is in pool but somehow None, remove it
                 self.yolo_threads.remove(model_name)

    # 重新加载模型
    def resignModel(self, model_name):
        # 重载 common 和 yolo 模块
        glo.set_value('yoloname', model_name)
        self.reloadModel()
        if model_name in MODEL_THREAD_CLASSES:
            self.yolo_threads.set(model_name, MODEL_THREAD_CLASSES[model_name]())
            self.initModel(yoloname=model_name)
            self.runModel(True) # Automatically run after resigning
        else:
            self.showStatus(f"Cannot resign: Model type '{model_name}' not recognized.")


    # Model 变化
    def changeModel(self):
        self.model_name = self.ui.model_box.currentText()
        if not self.model_name: # Handle case where model_box might be empty
            self.showStatus("No model selected.")
            # ++++++++++++++++++ CHANGE HERE ++++++++++++++++++
            self.ui.Model_label.setText("--") # Show placeholder if no model
            # ++++++++++++++++++++++++++++++++++++++++++++++++++
            return

        # ++++++++++++++++++ CHANGE HERE ++++++++++++++++++
        # self.ui.Model_label.setText(str(self.model_name).replace(".pt", "")) # Original line
        self.ui.Model_label.setText("SDES-YOLO") # Changed line
        # ++++++++++++++++++++++++++++++++++++++++++++++++++

        model_type_key = self.checkCurrentModel() # Get the base key (e.g., "yolov8")
        if not model_type_key:
            self.showStatus('Unsupported model type selected.')
            self.stopOtherModel() # Stop any running model if the new one is unsupported
            return

        # Stop potentially running incompatible models first
        self.stopOtherModel(model_type_key)

        yolo_thread = self.yolo_threads.get(model_type_key)
        if yolo_thread: # If a thread for this type already exists
            if yolo_thread.isRunning():
                 # If running, update the model path for the next run
                 yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.model_name
                 # Optionally, stop and restart if immediate change is needed, but runorContinue handles this
                 self.showStatus(f"Model set to {os.path.basename(self.model_name)}. Press Play to use.")
            else:
                 # If not running, just update path and it will be used on next run
                 yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.model_name
                 # Ensure parameters are loaded if thread exists but wasn't running
                 self.loadConfig() # Reload config specific to this model type if needed
                 self.showStatus(f"Model changed to {os.path.basename(self.model_name)}.")
        else: # If no thread exists for this type, create and initialize it
            self.yolo_threads.set(model_type_key, MODEL_THREAD_CLASSES[model_type_key]())
            self.initModel(yoloname=model_type_key) # initModel sets the model_name inside the thread too
            self.loadConfig()
            self.showStatus(f"Model changed to {os.path.basename(self.model_name)}.")
            # No need to call runModel here, runorContinue will handle it

    def runModelProcess(self, yolo_name):
        yolo_thread = self.yolo_threads.get(yolo_name)
        if not yolo_thread:
             self.showStatus(f"Error: Thread for {yolo_name} not found.")
             return

        yolo_thread.source = self.inputPath
        yolo_thread.stop_dtc = False
        
        # Ensure the model path inside the thread is up-to-date before starting
        selected_pt_file = self.ui.model_box.currentText()
        if selected_pt_file:
            yolo_thread.new_model_name = f'{self.current_workpath}/ptfiles/' + selected_pt_file
        else:
            self.showStatus("Error: No model file selected in dropdown.")
            return # Don't start if model selection is invalid

        if self.ui.run_button.isChecked():
            yolo_thread.is_continue = True
            # Only start if not already running
            if not yolo_thread.isRunning():
                 self.yolo_threads.start_thread(yolo_name)
            # else: # If already running, is_continue=True ensures it keeps going
            #    pass
            self.showStatus(f'Running Detection with {os.path.basename(yolo_thread.new_model_name)}...')
        else:
            yolo_thread.is_continue = False # Signal pause
            self.showStatus('Pause Detection')


    def runModel(self, runbuttonStatus=None):
        self.ui.save_status_button.setEnabled(False)
        if runbuttonStatus is not None: # Explicitly check for None
            self.ui.run_button.setChecked(runbuttonStatus)

        current_model_type_key = self.checkCurrentModel()
        if current_model_type_key is not None:
            self.runModelProcess(current_model_type_key)
        else:
            self.showStatus('The current model is not supported or not selected.')
            # Ensure button state reflects reality if model is unsupported
            if self.ui.run_button.isChecked():
                self.ui.run_button.setChecked(False)
                self.ui.save_status_button.setEnabled(True) # Re-enable save status if stopped

    # 开始/暂停 预测
    def runorContinue(self):
        if self.inputPath is None:
            self.showStatus("Please select the Image/Video before starting detection...")
            self.ui.run_button.setChecked(False) # Uncheck button if no input
            return

        # Ensure the model selected in the dropdown is valid before proceeding
        current_model_type_key = self.checkCurrentModel()
        if not current_model_type_key:
             self.showStatus('Cannot start: Unsupported model type selected.')
             self.ui.run_button.setChecked(False) # Uncheck button
             return

        # changeModel() might be redundant here if runModelProcess updates the path
        # Calling it ensures consistency if parameters changed
        # self.changeModel() # Let's rely on runModelProcess to set the path for now

        self.runModel() # runModel handles the button state and calls runModelProcess

    # 停止识别
    def stopDetect(self):
        self.quitRunningModel(stop_status=True) # Signal threads to stop
        self.ui.run_button.setChecked(False)     # Update button state
        self.ui.save_status_button.setEnabled(True) # Enable save status button
        self.ui.progress_bar.setValue(0)
        # Clear displays and stats
        # Keep input image in left box, clear only right box (output)
        # self.ui.main_leftbox.clear() # Keep input preview
        self.ui.main_rightbox.clear()
        self.ui.Class_num.setText('--')
        self.ui.Target_num.setText('--')
        self.ui.fps_label.setText('--')
        self.showStatus("Detection Stopped.") # Provide clear feedback
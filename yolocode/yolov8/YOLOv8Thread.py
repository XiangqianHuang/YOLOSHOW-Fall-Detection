import platform
import os.path
import time
import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from pathlib import Path
from yolocode.yolov8.data import load_inference_source
from yolocode.yolov8.data.augment import classify_transforms, LetterBox
from yolocode.yolov8.data.utils import IMG_FORMATS, VID_FORMATS
from yolocode.yolov8.engine.predictor import STREAM_WARNING
from yolocode.yolov8.engine.results import Results
from models.common import AutoBackend
from yolocode.yolov8.utils import callbacks, ops, LOGGER, colorstr, MACOS, WINDOWS
from collections import defaultdict
from yolocode.yolov5.utils.general import increment_path
from yolocode.yolov8.utils.checks import check_imgsz
from yolocode.yolov8.utils.torch_utils import select_device
from concurrent.futures import ThreadPoolExecutor

# +++++++++++++++ ADD THIS IMPORT +++++++++++++++
from utils.email_sender import check_results_and_send_alert
# ++++++++++++++++++++++++++++++++++++++++++++++++

class YOLOv8Thread(QThread):
    # 输入 输出 消息
    send_input = Signal(np.ndarray)
    send_output = Signal(np.ndarray)
    send_msg = Signal(str)
    # 状态栏显示数据 进度条数据
    send_fps = Signal(str)  # fps
    # send_labels = Signal(dict)  # Detected target results (number of each category)
    send_progress = Signal(int)  # Completeness
    send_class_num = Signal(int)  # Number of categories detected
    send_target_num = Signal(int)  # Targets detected
    send_result_picture = Signal(dict)  # Send the result picture
    send_result_table = Signal(list)  # Send the result table
    def __init__(self):
        super(YOLOv8Thread, self).__init__()
        # YOLOSHOW 界面参数设置
        self.current_model_name = None  # The detection model name to use
        self.new_model_name = None  # Models that change in real time
        self.source = None  # input source
        self.stop_dtc = True  # 停止检测
        self.is_continue = True  # continue/pause
        self.save_res = False  # Save test results
        self.iou_thres = 0.45  # iou
        self.conf_thres = 0.25  # conf
        self.speed_thres = 10  # delay, ms
        self.labels_dict = {}  # return a dictionary of results
        self.progress_value = 0  # progress bar
        self.res_status = False  # result status
        self.parent_workpath = None  # parent work path
        self.executor = ThreadPoolExecutor(max_workers=1)  # 只允许一个线程运行
        # YOLOv8 参数设置
        self.model = None
        self.data = 'yolocode/yolov8/cfg/datasets/coco.yaml'  # data_dict
        self.imgsz = 640
        self.device = ''
        self.dataset = None
        self.task = 'detect'
        self.dnn = False
        self.half = False
        self.agnostic_nms = False
        self.stream_buffer = False
        self.crop_fraction = 1.0
        self.done_warmup = False
        self.vid_path, self.vid_writerm, self.vid_cap = None, None, None # Note: vid_writerm is not used in provided code
        self.batch = None
        self.batchsize = 1
        self.project = 'runs/detect'
        self.name = 'exp'
        self.exist_ok = False
        self.vid_stride = 1  # 视频帧率
        self.max_det = 1000  # 最大检测数
        self.classes = None  # 指定检测类别  --class 0, or --class 0 2 3
        self.line_thickness = 3
        self.results_picture = dict()  # 结果图片
        self.results_table = list()  # 结果表格
        self.file_path = None # 文件路径
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        callbacks.add_integration_callbacks(self) # This likely registers callbacks defined elsewhere

    def run(self):
        if not self.model:
            self.send_msg.emit("Loading model: {}".format(os.path.basename(self.new_model_name)))
            self.setup_model(self.new_model_name)
            self.used_model_name = self.new_model_name # Consider renaming to self.current_model_name after setup
        
        source_str = str(self.source) # Use a local variable for the source string for clarity

        # 判断输入源类型 (Using source_str for these checks)
        if isinstance(IMG_FORMATS, str) or isinstance(IMG_FORMATS, tuple): # Ensure IMG_FORMATS is defined
            self.is_file = Path(source_str).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        else:
            self.is_file = Path(source_str).suffix[1:] in (IMG_FORMATS | VID_FORMATS) # Handles if IMG_FORMATS is a set

        self.is_url = source_str.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.webcam = source_str.isnumeric() or source_str.endswith(".streams") or (self.is_url and not self.is_file)
        self.screenshot = source_str.lower().startswith("screen")
        
        self.is_folder = isinstance(self.source, list) # self.source is the original, potentially a list for folders

        if self.save_res:
            self.save_path = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
            self.save_path.mkdir(parents=True, exist_ok=True)  # make dir
        
        if self.is_folder:
            for index, single_source_path in enumerate(self.source): # Iterate through list of file paths
                is_folder_last = True if index + 1 == len(self.source) else False
                self.setup_source(single_source_path) # Setup for each individual file in the folder
                self.detect(is_folder_last=is_folder_last)
                if self.stop_dtc and not is_folder_last : # If stopped early in folder mode, break outer loop
                    break
        else:
            self.setup_source(source_str) # Setup for single file, URL, webcam
            self.detect()

    @torch.no_grad()
    def detect(self, is_folder_last=False):
        # warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True
        self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
        datasets = iter(self.dataset)
        count = 0
        start_time = time.time()  # used to calculate the frame rate
        
        while True:
            if self.stop_dtc:
                if self.is_folder and not is_folder_last: # If in folder mode and not the last file, just break inner loop
                    break 
                self.send_msg.emit('Stop Detection')
                # --- 发送图片和表格结果 --- #
                if self.results_picture: # Only send if not empty
                    self.send_result_picture.emit(self.results_picture)
                    for key, value in self.results_picture.items():
                        self.results_table.append([key, str(value)])
                    self.results_picture = dict() # Clear after sending
                if self.results_table: # Only send if not empty
                    self.send_result_table.emit(self.results_table)
                    self.results_table = list() # Clear after sending
                # --- 发送图片和表格结果 --- #
                # 释放资源
                if self.dataset: # Check if dataset exists
                    self.dataset.running = False  # stop flag for Thread
                    if hasattr(self.dataset, 'threads'):
                        for thread_obj in self.dataset.threads: # Renamed to avoid conflict with QThread
                            if thread_obj.is_alive():
                                thread_obj.join(timeout=1)  # Add timeout
                    if hasattr(self.dataset, 'caps'):
                        for cap_obj in self.dataset.caps:  # Iterate through the stored VideoCapture objects
                            try:
                                cap_obj.release()  # release video capture
                            except Exception as e:
                                LOGGER.warning(f"WARNING ⚠️ Could not release VideoCapture object: {e}")
                cv2.destroyAllWindows()
                if hasattr(self, 'vid_writer') and self.vid_writer and isinstance(self.vid_writer[-1], cv2.VideoWriter):
                    self.vid_writer[-1].release()
                break # Break the while True loop
            
            if self.current_model_name != self.new_model_name: # Check if model needs to be changed
                self.send_msg.emit('Loading Model: {}'.format(os.path.basename(self.new_model_name)))
                self.setup_model(self.new_model_name)
                self.current_model_name = self.new_model_name # Update current model name
                self.done_warmup = False # Reset warmup flag for new model
                datasets = iter(self.dataset) # Re-initialize iterator if dataset might change with model
                
            if self.is_continue:
                current_source_for_msg = self.source
                if self.is_folder:
                    current_source_for_msg = os.path.dirname(self.source[0]) if self.source else "Unknown Folder"
                elif isinstance(self.source, list): # Should not happen if not folder, but as safeguard
                    current_source_for_msg = self.source[0] if self.source else "Unknown List Source"

                if self.is_file and not self.is_folder: # Single file
                    self.send_msg.emit("Detecting File: {}".format(os.path.basename(str(current_source_for_msg))))
                elif self.webcam and not self.is_url:
                    self.send_msg.emit("Detecting Webcam: Camera_{}".format(current_source_for_msg))
                elif self.is_folder:
                    # For folder, message refers to the folder itself. Individual file paths are handled by `path[i]`
                    self.send_msg.emit("Detecting Folder: {}".format(current_source_for_msg))
                elif self.is_url:
                    self.send_msg.emit("Detecting URL: {}".format(current_source_for_msg))
                elif self.screenshot:
                    self.send_msg.emit("Detecting Screenshot")
                else: # Fallback
                    self.send_msg.emit("Detecting: {}".format(current_source_for_msg))

                try:
                    self.batch = next(datasets)
                except StopIteration: # Dataset finished
                    if self.is_folder and not is_folder_last: # More files in folder to process
                        break # Break inner while, go to next file in folder
                    self.send_progress.emit(self.progress_value) # Ensure progress is 100%
                    self.send_msg.emit('Finish Detection')
                     # --- 发送图片和表格结果 --- #
                    if self.results_picture:
                        self.send_result_picture.emit(self.results_picture)
                        for key, value in self.results_picture.items():
                            self.results_table.append([key, str(value)])
                        self.results_picture = dict()
                    if self.results_table:
                        self.send_result_table.emit(self.results_table)
                        self.results_table = list()
                    # --- 发送图片和表格结果 --- #
                    self.res_status = True
                    if self.vid_cap is not None:
                        self.vid_cap.release()
                    if hasattr(self, 'vid_writer') and self.vid_writer and isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()
                    break # Break the while True loop

                path, im0s, s_log_str = self.batch # s_log_str is the logging string from data loader
                
                self.vid_cap = self.dataset.cap if self.dataset.mode == "video" else None
                self.send_input.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                count += 1
                percent = 0
                
                if self.vid_cap:
                    frame_count_total = self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if frame_count_total > 0:
                        percent = int(count / frame_count_total * self.progress_value)
                        self.send_progress.emit(percent)
                    # else: # Live stream or unknown length, progress might not be accurate
                        # percent = self.progress_value # Or handle differently for streams
                        # self.send_progress.emit(percent) # Avoid sending 100% prematurely for streams
                elif not self.webcam and not self.is_url and not self.screenshot : # Image file
                    percent = self.progress_value
                    self.send_progress.emit(percent)


                if count % 5 == 0 and count >= 5:
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                         self.send_fps.emit(str(int(5 / elapsed_time)))
                    start_time = time.time()
                
                with self.dt[0]:
                    im = self.preprocess(im0s)
                with self.dt[1]:
                    preds = self.inference(im)
                with self.dt[2]:
                    self.results = self.postprocess(preds, im, im0s) # List[Results]
                
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    
                    current_item_path_obj = Path(path[i])
                    current_item_source_info = ""

                    if self.screenshot:
                        current_item_source_info = "Desktop Screenshot"
                    elif self.webcam and not self.is_url:
                        current_item_source_info = f"Webcam_{self.source}"
                    elif self.is_url:
                        current_item_source_info = str(self.source) # Ensure it's a string
                    else: # File or file from folder
                        current_item_source_info = current_item_path_obj.name
                    
                    # +++++++++++++++ INTEGRATE EMAIL ALERT +++++++++++++++
                    if self.results and len(self.results) > i and self.results[i]:
                        check_results_and_send_alert([self.results[i]], current_item_source_info)
                    # +++++++++++++++ EMAIL ALERT END +++++++++++++++

                    self.results[i].speed = {
                        "preprocess": self.dt[0].dt * 1e3 / n,
                        "inference": self.dt[1].dt * 1e3 / n,
                        "postprocess": self.dt[2].dt * 1e3 / n,
                    }
                    
                    current_p_path = Path(path[i]) # Use this for clarity
                    current_im0_frame = None if self.source_type.tensor else im0s[i].copy()
                    self.file_path = current_p_path
                    
                    label_str = self.write_results(i, self.results, (current_p_path, im, current_im0_frame))
                    
                    class_nums = 0
                    target_nums = 0
                    self.labels_dict = {} # Reset for current frame/image
                    if 'no detections' not in label_str.lower() and label_str.strip(): # Check if not empty
                        parts = label_str.split(',')
                        for each_target_info in parts:
                            each_target_info = each_target_info.strip()
                            if not each_target_info:
                                continue
                            # Improved parsing: "1 person", "2 dogs"
                            num_label_match = each_target_info.split(' ', 1)
                            if len(num_label_match) == 2 and num_label_match[0].isdigit():
                                nums = int(num_label_match[0])
                                label_name = num_label_match[1].strip()
                                if label_name: # Ensure label_name is not empty
                                    target_nums += nums
                                    class_nums += 1 # Count as one detected class type
                                    self.labels_dict[label_name] = self.labels_dict.get(label_name, 0) + nums
                            elif each_target_info: # Fallback if parsing fails, treat as a label name if not empty
                                # This case might indicate an issue with result.verbose() format
                                # For now, let's assume result.verbose() gives "N label"
                                LOGGER.warning(f"Could not parse target info: '{each_target_info}' from '{label_str}'")


                    self.send_output.emit(self.plotted_img)
                    self.send_class_num.emit(len(self.labels_dict)) # Number of unique classes detected
                    self.send_target_num.emit(target_nums)         # Total number of target instances
                    
                    # Accumulate results for final summary if needed, or send per frame
                    # self.results_picture is updated with self.labels_dict for current frame
                    # This means send_result_picture will emit current frame's stats if called now
                    # The original code sends cumulative results at stop/finish.
                    # For now, let's keep self.results_picture as the current frame's dict for consistency
                    # and the cumulative send logic at stop/finish will handle the overall summary.
                    # However, the original code sets self.results_picture = self.labels_dict
                    # and then at stop/finish, it sends this self.results_picture.
                    # This implies self.results_picture should accumulate if we want a summary.
                    # Let's adjust: accumulate into a temporary dict for the current run.
                    if not hasattr(self, 'current_run_summary_picture'):
                        self.current_run_summary_picture = defaultdict(int)
                    if not hasattr(self, 'current_run_summary_table'):
                        self.current_run_summary_table = []

                    for lbl, num in self.labels_dict.items():
                        self.current_run_summary_picture[lbl] += num
                    # Table entry for current frame (optional, original adds at end)
                    # self.current_run_summary_table.append([current_item_source_info, str(self.labels_dict)])


                    if self.save_res:
                        save_file_path = str(self.save_path / current_p_path.name)
                        self.res_path = self.save_preds(self.vid_cap, i, save_file_path)
                    
                    if self.speed_thres > 0: # Ensure speed_thres is positive
                        time.sleep(self.speed_thres / 1000.0)
                
                # This condition handles single images or last frame of video
                is_stream = self.webcam or self.is_url or self.screenshot
                if not is_stream and ( (self.vid_cap and percent >= self.progress_value) or (not self.vid_cap and self.dataset.mode == 'image') ):
                    if self.is_folder and not is_folder_last: # More files in folder
                        break # Break from while, go to next file in outer loop
                    
                    self.send_progress.emit(self.progress_value) # Ensure 100%
                    self.send_msg.emit('Finish Detection')
                    # --- 发送累积的图片和表格结果 --- #
                    self.send_result_picture.emit(dict(self.current_run_summary_picture))
                    for key, value in self.current_run_summary_picture.items():
                        self.current_run_summary_table.append([key, str(value)])
                    self.send_result_table.emit(list(self.current_run_summary_table))
                    
                    if hasattr(self, 'current_run_summary_picture'): del self.current_run_summary_picture
                    if hasattr(self, 'current_run_summary_table'): del self.current_run_summary_table
                    # --- 发送图片和表格结果 --- #
                    self.res_status = True
                    if self.vid_cap is not None:
                        self.vid_cap.release()
                    if hasattr(self, 'vid_writer') and self.vid_writer and isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()
                    break # Break the while True loop
            else: # Not self.is_continue (paused)
                time.sleep(0.1) # Add a small delay when paused
        
        # Clean up summary attributes if loop exited some other way (e.g. stop_dtc)
        if hasattr(self, 'current_run_summary_picture'):
            # Send pending summary if stopped
            if self.stop_dtc:
                self.send_result_picture.emit(dict(self.current_run_summary_picture))
                temp_table = []
                for key, value in self.current_run_summary_picture.items():
                    temp_table.append([key, str(value)])
                self.send_result_table.emit(temp_table)
            del self.current_run_summary_picture
        if hasattr(self, 'current_run_summary_table'):
             if self.stop_dtc and not self.current_run_summary_picture: # if picture wasn't sent
                self.send_result_table.emit(list(self.current_run_summary_table))
             del self.current_run_summary_table


    def setup_model(self, model_path, verbose=True): # Renamed model to model_path for clarity
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(
            weights=model_path or self.model, # Use model_path
            device=select_device(self.device, verbose=verbose),
            dnn=self.dnn,
            data=self.data,
            fp16=self.half,
            fuse=True, # Default to True as in AutoBackend
            verbose=verbose,
        )
        self.device = self.model.device  # update device
        self.half = self.model.fp16  # update half
        self.model.eval()

    def setup_source(self, source_path_or_id): # Renamed source to source_path_or_id
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.imgsz, stride=self.model.stride, min_dim=2)
        self.transforms = (
            getattr(
                self.model.model if hasattr(self.model, 'model') else self.model, # Handle AutoBackend structure
                "transforms",
                classify_transforms(self.imgsz[0], crop_fraction=self.crop_fraction),
            )
            if self.task == "classify"
            else None
        )
        self.dataset = load_inference_source(
            source=source_path_or_id, # Use renamed parameter
            batch=self.batchsize, # ultralytics uses 'batch', not 'batchsize' arg for load_inference_source
            vid_stride=self.vid_stride,
            buffer=self.stream_buffer,
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and ( # 'stream' attribute not defined in class, assume True if not present
                self.source_type.stream
                or self.source_type.screenshot
                or len(self.dataset) > 1000
                or any(getattr(self.dataset, "video_flag", [False]))
        ):
            LOGGER.warning(STREAM_WARNING)
        
        # Initialize vid_path and vid_writer if not already or if bs changed
        # Assuming bs is mostly 1 for this thread structure.
        if not hasattr(self, 'vid_writer') or len(self.vid_writer) != self.dataset.bs:
            self.vid_path = [None] * self.dataset.bs
            self.vid_writer = [None] * self.dataset.bs
        # self.vid_frame = [None] * self.dataset.bs # vid_frame is not used

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            classes=self.classes,
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        
        results_list = [] # Renamed to avoid conflict with self.results
        for i, pred_item in enumerate(preds): # pred_item are detections for one image
            orig_img_item = orig_imgs[i]
            pred_item[:, :4] = ops.scale_boxes(img.shape[2:], pred_item[:, :4], orig_img_item.shape)
            # path might not be available if batch is not set correctly or if source is tensor
            img_path_str = self.batch[0][i] if self.batch and len(self.batch[0]) > i else "image.jpg"
            results_list.append(Results(orig_img_item, path=img_path_str, names=self.model.names, boxes=pred_item))
        return results_list

    def preprocess(self, im_batch): # Renamed im to im_batch
        """
        Prepares input image before inference.
        Args:
            im_batch (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im_batch, torch.Tensor)
        if not_tensor:
            im_batch = np.stack(self.pre_transform(im_batch)) # Pass im_batch
            im_batch = im_batch[..., ::-1].transpose((0, 3, 1, 2))
            im_batch = np.ascontiguousarray(im_batch)
            im_batch = torch.from_numpy(im_batch)
        
        im_processed = im_batch.to(self.device) # Use a new variable
        im_processed = im_processed.half() if self.model.fp16 else im_processed.float()
        if not_tensor:
            im_processed /= 255.0
        return im_processed

    def inference(self, im_processed, *args, **kwargs): # Renamed im to im_processed
        """Runs inference on a given image using the specified model and arguments."""
        # visualize = kwargs.pop('visualize', False) # Example if visualize was a kwarg
        return self.model(im_processed, augment=False, visualize=False, embed=None, *args, **kwargs) # embed=None for yolo

    def pre_transform(self, im_list): # Renamed im to im_list
        """
        Pre-transform input image before inference.
        Args:
            im_list (List(np.ndarray)): [(h, w, 3) x N] for list.
        Returns:
            (list): A list of transformed images.
        """
        # Ensure model and its attributes are available
        auto_letterbox = hasattr(self.model, 'pt') and self.model.pt 
        stride_val = self.model.stride if hasattr(self.model, 'stride') else 32

        same_shapes = all(x.shape == im_list[0].shape for x in im_list) if im_list else False
        letterbox_transformer = LetterBox(self.imgsz, auto=same_shapes and auto_letterbox, stride=stride_val)
        return [letterbox_transformer(image=x) for x in im_list]

    def save_preds(self, vid_cap_obj, idx, save_file_path_str): # Renamed params for clarity
        """Save video predictions as mp4 at specified path."""
        im0_to_save = self.plotted_img # This is the annotated image
        
        # Determine suffix and fourcc based on OS (already defined in class attributes or could be local)
        # Using local definition for clarity here, though global MACOS/WINDOWS are available
        _is_macos = platform.system() == "Darwin" # Requires import platform
        _is_windows = platform.system() == "Windows" # Requires import platform
        # Fallback if platform import is missing
        # _is_macos = MACOS 
        # _is_windows = WINDOWS


        suffix, fourcc_str = (".mp4", "avc1") if _is_macos else \
                             (".avi", "WMV2") if _is_windows else \
                             (".avi", "MJPG") # Default for other OS (e.g., Linux)

        if self.dataset.mode == "image":
            cv2.imwrite(save_file_path_str, im0_to_save)
            return save_file_path_str
        else:  # 'video' or 'stream'
            # Ensure vid_writer and vid_path are initialized for the current batch size
            if len(self.vid_writer) <= idx: # Resize if needed (shouldn't happen if bs is fixed)
                self.vid_path.extend([None] * (idx - len(self.vid_path) + 1))
                self.vid_writer.extend([None] * (idx - len(self.vid_writer) + 1))

            if self.vid_path[idx] != save_file_path_str:
                self.vid_path[idx] = save_file_path_str
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()
                
                if vid_cap_obj:
                    fps = int(vid_cap_obj.get(cv2.CAP_PROP_FPS))
                    w = int(vid_cap_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0_to_save.shape[1], im0_to_save.shape[0]
                
                # Use Path object for robust path manipulation
                output_path_obj = Path(save_file_path_str).with_suffix(suffix)
                self.vid_writer[idx] = cv2.VideoWriter(
                    str(output_path_obj), cv2.VideoWriter_fourcc(*fourcc_str), fps, (w, h)
                )
            
            self.vid_writer[idx].write(im0_to_save)
            return str(Path(save_file_path_str).with_suffix(suffix)) # Return the path with correct suffix

    def write_results(self, idx, results_list_arg, batch_info): # Renamed params for clarity
        """Write inference results to a file or directory."""
        p_path_obj, im_batch_processed, _ = batch_info # _ is current_im0_frame
        
        log_string = ""
        # im_batch_processed is the preprocessed batch. We need the original image for plotting.
        # results_list_arg[idx] contains the original image.
        
        self.data_path = p_path_obj # This is a Path object
        current_result_obj = results_list_arg[idx] # This is a Results object
        
        log_string += current_result_obj.verbose()
        
        plot_args = {
            "line_width": self.line_thickness,
            "boxes": True, # Assuming you always want boxes
            "conf": True,  # Assuming you always want conf scores
            "labels": True, # Assuming you always want labels
            # "im_gpu": im_batch_processed[idx].to(self.device) if isinstance(im_batch_processed, torch.Tensor) else None # For plotting on GPU if needed
        }
        # result.plot() uses the original image stored within the Results object
        self.plotted_img = current_result_obj.plot(**plot_args) 
        return log_string

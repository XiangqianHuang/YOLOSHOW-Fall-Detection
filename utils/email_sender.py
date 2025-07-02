# utils/email_sender.py
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
import socket
import time
import os

# --- 邮件发送者和接收者配置 (硬编码) ---
SENDER_QQ_EMAIL = ""  # 你的QQ邮箱
QQ_AUTHORIZATION_CODE = ""  # 你的QQ邮箱SMTP授权码
RECEIVER_EMAIL = ""  # 接收警报的邮箱
SMTP_SERVER_QQ = ''
SMTP_PORT_QQ =   # SMTP SSL端口

# --- 全局开关：控制是否发送邮件 ---
ENABLE_EMAIL_SENDING = True  # True 启用, False 禁用

# --- 全局变量用于控制邮件发送频率 ---
last_email_sent_time = 0
EMAIL_COOLDOWN_SECONDS = 300  # 5分钟冷却 (300秒)

# 用于临时存储截图，如果需要的话
TEMP_IMAGE_DIR = "temp_alerts"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)


def send_security_alert_email(subject, body, image_path=None):
    """
    发送安全警报邮件。

    Args:
        subject (str): 邮件主题。
        body (str): 邮件正文。
        image_path (str, optional): 附件图片的路径。默认为None。
    """
    global last_email_sent_time
    global ENABLE_EMAIL_SENDING

    if not ENABLE_EMAIL_SENDING:
        # 如果你的项目禁用了stdout，这个print可能看不到，需要注意
        print("Email sending is currently disabled by global switch in email_sender.py.")
        return False

    current_time = time.time()
    if current_time - last_email_sent_time < EMAIL_COOLDOWN_SECONDS:
        print(f"Email cooldown active. Last email sent {int(current_time - last_email_sent_time)}s ago. Skipping.")
        return False

    msg = MIMEText(body, 'plain', 'utf-8') # 默认为纯文本
    # 如果需要发送图片附件，需要使用 MIMEMultipart，这里暂时简化为纯文本
    # 如果确实需要图片，请告诉我，我们可以修改这部分
    
    msg['From'] = formataddr((Header('YOLOSHOW 安全警报', 'utf-8').encode(), SENDER_QQ_EMAIL))
    msg['To'] = formataddr((Header('管理员', 'utf-8').encode(), RECEIVER_EMAIL))
    msg['Subject'] = Header(subject, 'utf-8')

    smtp_obj = None
    try:
        print(f"Attempting to send security alert email to {RECEIVER_EMAIL}...")
        smtp_obj = smtplib.SMTP_SSL(SMTP_SERVER_QQ, SMTP_PORT_QQ, timeout=20) # 使用SMTP_SSL
        # smtp_obj.set_debuglevel(1) # 取消注释以查看详细的SMTP通信日志
        smtp_obj.login(SENDER_QQ_EMAIL, QQ_AUTHORIZATION_CODE)
        smtp_obj.sendmail(SENDER_QQ_EMAIL, [RECEIVER_EMAIL], msg.as_string())
        print(f"Security alert email sent successfully to {RECEIVER_EMAIL}!")
        last_email_sent_time = current_time
        return True
    except socket.timeout:
        print(f"ERROR sending email: Connection to {SMTP_SERVER_QQ}:{SMTP_PORT_QQ} timed out.")
    except smtplib.SMTPConnectError as e:
        print(f"ERROR sending email: SMTP Connection Error - {e}")
    except smtplib.SMTPAuthenticationError as e:
        print(f"ERROR sending email: SMTP Authentication Error - {e}. Check email/auth code.")
    except smtplib.SMTPSenderRefused as e:
        print(f"ERROR sending email: Sender refused - {e}.")
    except smtplib.SMTPRecipientsRefused as e:
        print(f"ERROR sending email: Recipients refused - {e}.")
    except smtplib.SMTPDataError as e:
        print(f"ERROR sending email: Data error - {e}.")
    except smtplib.SMTPException as e:
        print(f"ERROR sending email: SMTP Error - {type(e).__name__}: {e}")
    except Exception as e:
        print(f"ERROR sending email: An unexpected error occurred - {type(e).__name__}: {e}")
    finally:
        if smtp_obj:
            try:
                smtp_obj.quit()
            except Exception:
                pass # 忽略退出时的错误
    return False


def check_results_and_send_alert(results, source_info="N/A"):
    """
    检查YOLO检测结果，如果检测到任何物体，则尝试发送邮件。
    Args:
        results: YOLO模型的推理结果。通常是 ultralytics.engine.results.Results 对象或其列表。
        source_info (str): 关于检测源的信息 (例如文件名, 摄像头ID)。
    """
    detected_objects_count = 0
    if not results:
        return

    # Ultralytics YOLOv8 results 通常是一个列表，即使是单张图片也是包含一个元素的列表
    # 每个元素是 ultralytics.engine.results.Results 对象
    # 这个对象有一个 .boxes 属性，其 .xyxy .cls .conf 等包含了检测信息
    # 我们可以通过 len(results[0].boxes) 或 results[0].boxes.shape[0] 来获取检测数量
    
    current_results = []
    if isinstance(results, list):
        current_results = results
    else:
        current_results = [results] # 包装成列表以便统一处理

    for res in current_results:
        if hasattr(res, 'boxes') and res.boxes is not None:
            # `res.boxes.shape[0]` 是检测到的边界框数量
            # 或者更简单地用 `len(res.boxes)`
            num_detections_in_res = len(res.boxes)
            if num_detections_in_res > 0:
                detected_objects_count += num_detections_in_res
                # 如果只想在第一次检测到时触发，可以在这里提前判断并发送
                # print(f"Detected {num_detections_in_res} objects in current result from source: {source_info}")

    if detected_objects_count > 0:
        print(f"Total {detected_objects_count} object(s) detected in source: {source_info}. Preparing email alert.")
        
        email_subject = f"安全警报：在 {source_info} 检测到跌倒"
        email_body = (
            f"YOLOSHOW 系统警报：\n\n"
            f"在时间: {time.strftime('%Y-%m-%d %H:%M:%S')} \n"
            f"于源: {source_info}\n"
            f"检测到 {detected_objects_count} 个跌倒。\n\n"
            f"请检查相关监控。"
        )
        
        # 尝试发送邮件
        send_security_alert_email(email_subject, email_body)
        # 注意：如果需要发送图片附件，这里需要先保存图片，然后将路径传给 send_security_alert_email
        # 例如:
        # if hasattr(results[0], 'plot'): # plot() 方法返回带标注的图像
        #    annotated_img = results[0].plot()
        #    img_filename = f"alert_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        #    img_path = os.path.join(TEMP_IMAGE_DIR, img_filename)
        #    cv2.imwrite(img_path, annotated_img) # 需要导入 cv2
        #    send_security_alert_email(email_subject, email_body, image_path=img_path)
        # else:
        #    send_security_alert_email(email_subject, email_body)

    # else:
        # print(f"No objects detected in source: {source_info}") # 可以用于调试

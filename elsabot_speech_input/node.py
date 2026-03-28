import os
from threading import Timer
import json
import queue

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from ament_index_python.packages import get_package_share_directory

from speech_action_interfaces.action import Recognize
from speech_action_interfaces.msg import Wakeword
from std_msgs.msg import Bool, String, Int32

from .server_client import SpeechInputServerClient

class SpeechInput(Node):
    def __init__(self):
        super().__init__('speech_input_node')

        self.speech_server_host_port = '127.0.0.1:8800'
        self.wake_word_model_dir = os.path.join(get_package_share_directory('elsabot_speech_input'), 'wakewords/')
        self.wake_word_model_dir_server = '/jetson_ws/src/elsabot_speech_input/wakewords/'

        self.get_logger().info(f'Wake word model dir: {self.wake_word_model_dir}')
        self.get_logger().info(f'Wake word model dir (server): {self.wake_word_model_dir_server}')

        self.stt_server = ActionServer(self, Recognize, 'recognize', self.stt_execute_callback, cancel_callback=self.stt_cancel_callback)

        self.pub_listening = self.create_publisher(Bool, '/speech_detect/listening', 10)
        self.pub_vad = self.create_publisher(Bool, '/speech_detect/vad', 10)
        self.pub_aoa = self.create_publisher(Int32, '/speech_detect/aoa', 10)
        self.pub_wakeword = self.create_publisher(Wakeword, '/speech_detect/wakeword', 10)

        self.sub_speaking = self.create_subscription(Bool, '/head/speaking', self.speaking_callback, 2);

        self.speech_server_client = SpeechInputServerClient(self.get_logger(), self.speech_server_host_port, self.wakeword_callback, self.vad_callback,
                                        self.speech_recog_finished_callback, self.speech_recog_failed_callback)
    
        self.set_status_timer()
        self.stt_results_queue = queue.Queue()

        self.get_logger().info('SpeechInput client initialized')

        self.speech_server_client.load_wake_words(self.wake_word_model_dir, self.wake_word_model_dir_server)

    def clear_stt_results_queue(self):
        while True:
            try:
                self.stt_results_queue.get_nowait()
            except Exception:
                break

    def stt_execute_callback(self, goal_handle):
        self.clear_stt_results_queue()
        self.speech_server_client.start_speech_recognizer(goal_handle.request.timeout)
        # Wait for the result
        r = self.stt_results_queue.get()

        result = Recognize.Result()    
        if r["result"] == "success":
            result.text = r["text"]
        elif r["result"] == "failed":
            result.text = "ERROR"
        elif r["result"] == "cancelled":
            return                        

        goal_handle.succeed()
        return result

    def stt_cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        self.stt_results_queue.put({"result": "cancelled"})
        goal_handle.canceled()

    def speaking_callback(self, msg):
        self.get_logger().info(f'Received speaking: mode: {msg.data}')

    def vad_callback(self, active):
        self.get_logger().info(f'VAD change: active: {active}')
        self.report_vad(active)

    def wakeword_callback(self, wakeword):
        self.get_logger().info(f'Wakeword: {wakeword}')
        self.report_wakeword(wakeword)

    def speech_recog_finished_callback(self, text):
        self.get_logger().info(f'Got speech recognizer result: {text}')
        self.stt_results_queue.put({"result": "success", "text": text})

    def speech_recog_failed_callback(self, error):
        self.get_logger().info(f'Got speech recognizer result: {error}')
        self.stt_results_queue.put({"result": "failed", "error": error})

    def report_listening(self, is_listening):
        msg = Bool()
        msg.data = is_listening
        self.pub_listening.publish(msg)

    def report_vad(self, vad_active):
        msg = Bool()
        msg.data = vad_active
        self.pub_vad.publish(msg)

    def report_aoa(self, aoa):
        msg = Int32()
        msg.data = aoa
        self.pub_aoa.publish(msg)

    def report_wakeword(self, wakeword):
        msg = Wakeword()
        msg.stamp = self.get_clock().now().to_msg()
        msg.word = wakeword
        msg.angle = self.last_aoa
        self.pub_wakeword.publish(msg)

    def set_status_timer(self):
        self.timer = Timer(0.5, self.report_status)
        self.timer.start()
        self.check_angle_of_arrival()

    def report_status(self):
        self.set_status_timer()
        self.check_angle_of_arrival()

    def check_angle_of_arrival(self):
        # fix - implement
        self.last_aoa = 0
        return



def main(args=None):
    rclpy.init(args=args)
    server_node = SpeechInput()

    rclpy.spin(server_node)
    server_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
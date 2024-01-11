import cv2
import numpy as np
import streamlit as st
import threading
import time

# カメラストリームを取得するクラス
class CameraStream(threading.Thread):
    def __init__(self, fps):
        super(CameraStream, self).__init__()
        self.cap = cv2.VideoCapture(0)
        self.fps = fps
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        while not self.stopped():
            # FPSの設定
            interval = 1.0 / self.fps

            # 1秒ごとにフレーム画像を更新する
            _, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Streamlit上の画像を更新する
            st_frame.image(gray)

            # スリープ時間の計算
            t1 = cv2.getTickCount()
            t2 = cv2.getTickCount()
            time_diff = (t2 - t1) / cv2.getTickFrequency()

            # スリープ
            if time_diff < interval:
                time.sleep(interval - time_diff)

        # カメラの解放
        self.cap.release()

# Streamlitアプリの設定
st.title("カメラストリーム")
st.sidebar.title("設定")
st.sidebar.header("カメラ設定")

# FPSを設定するスライダーの追加
fps = st.sidebar.slider("FPS", 1, 30, 15)

# カメラストリームを取得するインスタンスを生成
camera_stream = CameraStream(fps)

# Start/Stopボタンを追加
if st.sidebar.button("Start"):
    camera_stream.start()

if st.sidebar.button("Stop"):
    camera_stream.stop()

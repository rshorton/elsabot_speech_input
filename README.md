# Elsabot Speech Input Processor

This package implements a ROS2 node that implements speech input processing using local wakeword and speech-to-text processing.

The input processor is implemented using two processes.  The first is an HTTP client that runs in this ROS2 node which makes STT requests to a server running in a second process. The second process is implemented by the code found in the 'server' folder.  That code expects to be run from within a Docker container (for Nvidia Jetson in the Elsabot case) that provides an installation of OpenWakeword and Faster-Whisper.  See the jetson_support repo for the docker file and script (run_stt_tts.py) used to start that container.

A websocket connection is used between the client and server to receive wakeword events, STT processing status, and STT result.

The client uses the Elsabot Audio Output node to play cueing sounds to indicate when STT starts recording and when it has stopped recording.

Regarding the wakeword model used by OpenWakeword, there are various Jupyter notebooks available with the steps necessary to train a wakeword.  Good luck getting one of those to run.  Just pay the $4 or so and use https://openwakeword.com/ to get a well-trained model for your custom wakeword.


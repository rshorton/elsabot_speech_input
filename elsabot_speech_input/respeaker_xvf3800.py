# Supports newer reSpeaker USB 4MIC Array based on xvf3800
# This code uses the reSpeaker control repo from Seeed.

import usb.core
import usb.util

from .reSpeaker_XVF3800_USB_4MIC_ARRAY.python_control.xvf_host import ReSpeaker

class Respeaker_xvf3800():
    def __init__(self, logger):
        self.logger = logger
        self.log_prefix = f'{self.__class__.__name__}:'
        self.seeed_mic_dev = None

    def find(self, vid=0x2886, pid=0x001A):
        dev = usb.core.find(idVendor=vid, idProduct=pid)
        if not dev:
            return
        return ReSpeaker(dev)

    def init(self):
        self.seeed_mic_dev = self.find()
        if not self.seeed_mic_dev:
            self.logger.error(f'Error, failed to find Seeed Mic device (xvf3800)')
            return False
        return self.configure_seed_mic_dev()

    def configure_seed_mic_dev(self):
        return True

    def read_mic_array_aoa(self):
        val = self.seeed_mic_dev.read('DOA_VALUE')
        return val[0]

    def read_mic_array_vad(self):
        val = self.seeed_mic_dev.read('DOA_VALUE')
        return val[1]

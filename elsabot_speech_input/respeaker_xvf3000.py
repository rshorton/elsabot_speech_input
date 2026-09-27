# Supports original reSpeaker USB 4MIC Array based on xvf3000

from .tuning import Tuning, find

class Respeaker_xvf3000():
    def __init__(self, logger):
        self.logger = logger
        self.log_prefix = f'{self.__class__.__name__}:'
        self.seeed_mic_dev = None

    def init(self):
        self.seeed_mic_dev = find()
        if self.seeed_mic_dev is None:
            self.logger.error(f'Error, failed to find Seeed Mic device (xvf3000)')
            return False
        return self.configure_seed_mic_dev()

    def configure_seed_mic_dev(self):
        self.seeed_mic_dev.write('GAMMAVAD_SR', 2)
        self.seeed_mic_dev.write('AGCGAIN', 15)
        self.seeed_mic_dev.write('AGCONOFF', 0)
        return True

    def read_mic_array_aoa(self):
        return self.seeed_mic_dev.read('DOAANGLE')

    def read_mic_array_vad(self):
        return self.seeed_mic_dev.read('VOICEACTIVITY')

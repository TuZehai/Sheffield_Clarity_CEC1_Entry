"""Global configuration."""

import os
import logging
import configparser
import json


class ClarityConfig(configparser.ConfigParser):
    def getlist(self, section, key, fallback=None):
        value = self.get(section, key, fallback=fallback)
        if isinstance(value, str):
            value = json.loads(value)
        return value

    def __init__(self, config_filename):

        super(ClarityConfig, self).__init__(
            allow_no_value=True, inline_comment_prefixes=("#")
        )

        if config_filename and os.path.exists(config_filename):
            self.clarity_cfg = os.path.abspath(config_filename)
            self.read(self.clarity_cfg)
        else:
            logging.info("Config file not present: using inbuilt defaults")
        FORMAT = "%(levelname)s:%(funcName)s: %(message)s"
        logging_level = self.get("clarity", "LOGGING_LEVEL", fallback="INFO")
        logging.basicConfig(level=logging_level, format=FORMAT)

        self.fs = self.getint("clarity", "CLARITY_FS", fallback=44100)
        self.latency = self.getfloat("clarity", "LATENCY", fallback=0.0)
        self.output_gain_constant = self.getfloat(
            "clarity", "OUTPUT_GAIN_CONSTANT", fallback=1.0
        )
        self.pre_duration = self.getfloat("clarity", "PRE_DURATION", fallback=1.0)
        self.post_duration = self.getfloat("clarity", "POST_DURATION", fallback=1.0)
        self.tail_duration = self.getfloat("clarity", "TAIL_DURATION", fallback=0.0)
        self.ramp_duration = self.getfloat("clarity", "RAMP_DURATION", fallback=0.5)

        self.default_cfs = self.getlist(
            "clarity",
            "DEFAULT_CFS",
            fallback=[125, 250, 500, 1000, 2000, 4000, 6000, 8000],
        )
        self.noisegatelevels = self.getlist(
            "clarity", "NOISEGATELEVELS", fallback=[38, 38, 36, 37, 32, 26, 23, 22, 8]
        )

        self.noisegateslope = self.getint("clarity", "NOISEGATESLOPE", fallback=0)
        self.cr_level = self.getint("clarity", "CR_LEVEL", fallback=0)
        self.max_output_level = self.getint("clarity", "MAX_OUTPUT_LEVEL", fallback=100)
        self.ahr = self.getint("clarity", "AHR", fallback=20)
        self.cfg_file = self.get(
            "clarity", "CFG_FILE", fallback="prerelease_combination3_smooth"
        )
        self.calib_dB_SPL = self.getint("clarity", "CALIB_DB_SPL", fallback=65)
        self.ref_RMS_dB = self.getfloat("clarity", "REF_RMS_DB", fallback=-31.2)
        self.equiv0dBSPL = self.getfloat("clarity", "equiv0dBSPL", fallback=100)
        self.calib = self.getboolean("clarity", "CALIB", fallback=False)
        self.N_listeners_scene = self.getint("clarity", "N_LISTENERS_SCENE", fallback=3)
        #        self.speech_SNRS = self.getlist("clarity", "SPEECHINT_SNRS", fallback=[0, 10])
        #        self.nonspeech_SNRS = self.getlist(
        #            "clarity", "NONSPEECHINT_SNRS", fallback=[0, 10]
        #        )
        self.addnoise = self.getint("clarity", "ADDNOISE", fallback=0)


config_filename = None
if "CLARITY_ROOT" in os.environ:
    config_filename = f"{os.environ['CLARITY_ROOT']}/clarity.cfg"
CONFIG = ClarityConfig(config_filename)

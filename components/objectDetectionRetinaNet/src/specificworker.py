#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2020 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from genericworker import *
from dnnlib.api import *
import numpy as np


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 2000
        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        print('SpecificWorker destructor')

    def setParams(self, params):
        try:
            # configure RetinaNet
            self.retinanet = ObjectDetector(weights_path=params["weights_file"], cls_path=params["cls_file"])
            # output directory
            self.output_dir = params["output_dir"]
        except Exception as e:
            print("Error reading config params")
            print(e)
            return False
        return True

    @QtCore.Slot()
    def compute(self):
        print('SpecificWorker.compute...')
        return True

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== Methods for Component Implements ==================
    # ===================================================================

    #
    # IMPLEMENTATION of processImage method from YoloServer interface
    #
    def YoloServer_processImage(self, img):
        # extract RGB image
        image = np.frombuffer(img.image, np.uint8).reshape(img.height, img.width, img.depth)
        # perform network inference
        detections = self.ObjectDetector.draw_caption(image, self.output_dir)
        # process output detections
        ret_detections = []
        for obj in detections:
            box = RoboCompYoloServer.Box(name=obj[0], left=obj[1], top=obj[2], right=obj[3], bot=obj[4], prob=obj[5])
            ret_detections.append(box)
        return RoboCompYoloServer.Objects(ret_detections)

    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompYoloServer you can use this types:
    # RoboCompYoloServer.TImage
    # RoboCompYoloServer.Box


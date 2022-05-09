import cv2
import onnxruntime
import numpy as np
from importlib_metadata import metadata
from utils import *
import copy
class ONNXModel():
    def __init__(self, onnx_path):
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

    def infer(self, image_numpy):
        scores = self.onnx_session.run([self.output_name],
                                       {self.input_name: image_numpy})[0]
        scores = do_nms(scores)
        return scores

if __name__ == "__main__":
    import torch
    import copy
    import time
    import time
    import os
    anchor_ys = np.linspace(1, 0, num=72)
    onnxmodel = ONNXModel('lane.onnx')
    tt1 = time.time()
    img = "1.jpg"
    image = cv2.imread(img)
    image_raw = copy.deepcopy(image)
    img = cv2.resize(image, (640, 360))
    image = img.transpose([2, 0, 1]).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    t_fer = time.time()
    scores = onnxmodel.infer(image)
    decode = []
    pre = post_process(image_raw, scores, anchor_ys)
    decode.append(pre)
    or_h, or_w = image_raw.shape[:2]
    lines, N = border_label(decode[0], or_w, or_h)
    for indx, line in lines.items():
        for curr_p, next_p in zip(line['points'][:-1], line['points'][1:]):
            image_ = cv2.line(image_raw, tuple(curr_p), tuple(next_p), color=(0, 255, 0), thickness=4)
        xy_G = (int((line['e_point'][0] + line['s_point'][0]) / 2),
                int((line['e_point'][1] + line['s_point'][1]) / 2))
        text = 'border: ' + str(line['code'])
        image_ = cv2.putText(image_, text, xy_G,
                             cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(0, 0, 255))
        image_ = cv2.putText(image_, 'conf:' + str(line['conf_set']), (xy_G[0], xy_G[1] + 50),
                             cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(0, 255, 255))

    cv2.imwrite("out.jpg", image_)

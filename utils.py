import cv2
import onnxruntime
import numpy as np
from PIL import Image ,ImageDraw ,ImageFont, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import math

from scipy.interpolate import InterpolatedUnivariateSpline


class Lane:
    def __init__(self, points=None, invalid_value=-2., metadata=None):
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(points[:, 1], points[:, 0], k=min(3, len(points) - 1))
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01

        self.metadata = metadata or {}

    def __repr__(self):
        return '[Lane]\n' + str(self.points) + '\n[/Lane]'

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)

        lane_xs[(lane_ys < self.min_y) | (lane_ys > self.max_y)] = self.invalid_value
        return lane_xs

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter < len(self.points):
            self.curr_iter += 1
            return self.points[self.curr_iter - 1]
        self.curr_iter = 0
        raise StopIteration
        
def do_nms(proposals, conf_threshold=0.5, nms_thres=50., nms_topk=4):
    scores = proposals[:, 1]
    above_threshold = scores > conf_threshold
    proposals = proposals[above_threshold]
    if proposals.shape[0] == 0:
        return None
    scores = scores[above_threshold]
    keep, num_to_keep, _ = Lane_nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
    keep = keep[:num_to_keep]
    proposals = proposals[keep]
    return proposals

def Lane_nms(proposals,scores,overlap=50, top_k=4):
        keep_index = []
        sorted_score = np.sort(scores)[-1] # from big to small 
        indices = np.argsort(-scores) # from big to small 
        
        r_filters = np.zeros(len(scores))

        for i,indice in enumerate(indices):
            if r_filters[i]==1: # continue if this proposal is filted by nms before
                continue
            keep_index.append(indice)
            if len(keep_index)>top_k: # break if more than top_k
                break
            if i == (len(scores)-1):# break if indice is the last one
                break
            sub_indices = indices[i+1:]
            for sub_i,sub_indice in enumerate(sub_indices):
                r_filter = Lane_IOU(proposals[indice,:],proposals[sub_indice,:],overlap)
                if r_filter: r_filters[i+1+sub_i]=1 
        num_to_keep = len(keep_index)
        keep_index = list(map(lambda x: x.item(), keep_index))
        return keep_index, num_to_keep, None
        
def Lane_IOU(parent_box, compared_box, threshold):
        '''
        calculate distance one pair of proposal lines
        return True if distance less than threshold 
        '''
        n_offsets=72
        n_strips = n_offsets - 1

        start_a = (parent_box[2] * n_strips + 0.5).astype(int) # add 0.5 trick to make int() like round  
        start_b = (compared_box[2] * n_strips + 0.5).astype(int)
        start = max(start_a,start_b)
        end_a = start_a + parent_box[4] - 1 + 0.5 - (((parent_box[4] - 1)<0).astype(int))
        end_b = start_b + compared_box[4] - 1 + 0.5 - (((compared_box[4] - 1)<0).astype(int))
        end = min(min(end_a,end_b),71)
        if (end - start)<0:
            return False
        dist = 0
        for i in range(5+start,5 + end.astype(int)):
            if i>(5+end):
                 break
            if parent_box[i] < compared_box[i]:
                dist += compared_box[i] - parent_box[i]
            else:
                dist += parent_box[i] - compared_box[i]
        return dist < (threshold * (end - start + 1))
def post_process(img, proposals, anchor_ys,n_offsets=72):
    n_strips = n_offsets - 1
    lanes = []
    for lane in proposals:
        lane_xs = lane[5:]/640
        start = int(round(lane[2].item() * n_strips))
        length = int(round(lane[4].item()))
        end = start + length - 1
        end = min(end, len(anchor_ys) - 1)
        mask = ~((((lane_xs[:start] >= 0.) &
                   (lane_xs[:start] <= 1.))[::-1].cumprod()[::-1]).astype(np.bool_))
        lane_xs[end + 1:] = -2
        lane_xs[:start][mask] = -2
        lane_ys = anchor_ys[lane_xs >= 0]
        lane_xs = lane_xs[lane_xs >= 0]
        lane_xs = np.flip(lane_xs,axis=0)
        lane_ys = np.flip(lane_ys,axis=0)
        if len(lane_xs) <= 1:
            continue
        points = np.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),axis=1).squeeze(2)
        lane = Lane(points,
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
        lanes.append(lane)
    return lanes
def lin_f(xy_s, xy_e):
    a = (xy_e[1] - xy_s[1]) / (xy_e[0] - xy_s[0])
    b = xy_e[1] - a * xy_e[0]
    return a, b

def verify_lane(xy_c, lanes):
    """Verify which lane a given point C belongs to

    """

    lane_num = None

    dist_set = []  # distances from the point to lane's borders
    lane_codes = []
    func_set = []

    for _, lane in lanes.items():
        z_pt, _ = Z_point(xy_c, lane['func'][0], lane['func'][1])
        dist = np.linalg.norm(np.asarray(xy_c)-np.asarray(lane['s_point']))  # 求范数
        dist_set.append(dist)
        func_set.append(lane['func'])
        lane_codes.append(lane['code'])

    sort_inds = list(np.argsort(dist_set))
    func_1 = func_set[sort_inds[0]]
    sign_1 = np.sign(xy_c[1] - func_1[0] * xy_c[0] - func_1[1])
    func_2 = func_set[sort_inds[1]]
    sign_2 = np.sign(xy_c[1] - func_2[0] * xy_c[0] - func_2[1])

    if sign_1 * sign_2 > 0:
        lane_num = min(lane_codes[sort_inds[0]], lane_codes[sort_inds[1]])

    return lane_num

def Z_point(xy_g, a, b):
    """Find a Z point so that  GZ is
        perpendicular to the line defined by y = ax + b
    :param: xy_g: tuple
        G point's coordinate

    :param: a: float
    :param: b: float

    """
    x_z = (xy_g[0] + a * xy_g[1] - a * b) / (1 + a ** 2)
    y_z = a * x_z + b

    # Distance GZ
    dist = np.sqrt((xy_g[1] - y_z) ** 2 + (xy_g[0] - x_z) ** 2) * np.sign(x_z)

    xy_z = (x_z, y_z)

    return xy_z, dist

def lin_f(xy_s, xy_e):
    """Find a function representing for a line:
                    y = ax + b
    :param: xy_s: tuple
        start point coordinate of the line
    :param: xy_s: tuple
        end point coordinate of the line

    """
    a = (xy_e[1] - xy_s[1]) / (xy_e[0] - xy_s[0])
    b = xy_e[1] - a * xy_e[0]
    return a, b

#在图片上写中文
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def border_label(pred, im_w, im_h):
    lines = {}
    point_set = []
    dist_set = []
    func_set = []  # line functions
    s_point_set = []  # line's start point
    e_point_set = []  # line's end point
    conf_point_set=[]
    N = 0
    for i, lane in enumerate(pred):
        points = lane.points
        conf_lane = (round(float(lane.metadata['conf']), 4))
        conf_point_set.append(conf_lane)
        N += points.shape[0]
        points[:, 0] *= im_w
        points[:, 1] *= im_h
        points = points.round().astype(int)
        # plot line between start and end points
        s_point = tuple(points[0, :])
        e_point = tuple(points[-1, :])
        s_point_set.append(s_point)
        e_point_set.append(e_point)
        # find a function, y = ax +b, represents for the line
        a, b = lin_f(s_point, e_point)
        point_set.append(points)
        dist_set.append(e_point[0])  # end-point's x coordinate
        func_set.append((a, b))

    sort_inds = np.argsort(dist_set)
    for i, ind in enumerate(sort_inds):
        lines[i] = {}
        lines[i]['points'] = point_set[ind]
        lines[i]['func'] = func_set[ind]
        lines[i]['code'] = i
        lines[i]['dist'] = dist_set[ind]
        lines[i]['s_point'] = s_point_set[ind]
        lines[i]['e_point'] = e_point_set[ind]
        lines[i]['conf_set'] =conf_point_set[ind]
    return lines, N
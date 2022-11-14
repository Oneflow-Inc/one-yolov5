# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import oneflow as flow
import oneflow.nn as nn

from utils.metrics import bbox_iou
from utils.oneflow_utils import de_parallel


# 标签平滑
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = flow.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - flow.exp((dx - 1) / (self.alpha + 1e-4))
        loss = loss * alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = flow.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = flow.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = loss * alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = flow.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = flow.abs(true - pred_prob) ** self.gamma
        loss = loss * (alpha_factor * modulating_factor)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


# 计算损失(分类损失+置信度损失+框坐标回归损失)
class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=flow.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=flow.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = (
            BCEcls,
            BCEobj,
            1.0,
            h,
            autobalance,
        )
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        # 初始化各个部分损失
        lcls = flow.zeros(1, device=self.device)  # class loss
        lbox = flow.zeros(1, device=self.device)  # box loss
        lobj = flow.zeros(1, device=self.device)  # object loss
        # 获得标签分类,边框,索引，anchors
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx

            # tobj = flow.zeros( pi.shape[:4] , dtype=pi.dtype, device=self.device)  # target obj
            tobj = flow.zeros((pi.shape[:4]), dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires flow 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = flow.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox = lbox + (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = flow.full_like(pcls, self.cn, device=self.device)  # targets

                    # t[range(n), tcls[i]] = self.cp
                    t[flow.arange(n, device=self.device), tcls[i]] = self.cp

                    lcls = lcls + self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in flow.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj = lobj + (obji * self.balance[i])  # obj loss

            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, flow.cat((lbox, lobj, lcls)).detach()

    # ---------------------------------------------------------
    # build_tangets函数用于获得在训练时计算loss函数所需要的目标框，即被认为是正样本与yolov3/v4的不同:yolov5支持跨网格预测
    # 对于任何一个bbox，三个输出预测特征层都可能有先验框anchors匹配;该函数输出的正样本框比传入的targets （GT框）数目多
    # 具体处理过程:
    # (1)对于任何一层计算当前bbox和当前层anchor的匹配程度，不采用iou，而是shape比例;如果anchor和bbox的宽高比差距大于4，则认为不匹配，此时忽略相应的bbox，即当做背景;
    # (2)然后对bbox计算落在的网格所有anchors都计算loss(并不是直接和GT框比较计算loss)
    # 注意此时落在网格不再是一个，而是附近的多个，这样就增加了正样本数，可能存在有些bbox在三个尺度都预测的情况另外，
    # yolov5也没有conf分支忽略阈值(ignore_thresh)的操作，而yoloy3/v4有。
    # --------------------------------------------------------

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets

        tcls, tbox, indices, anch = [], [], [], []
        gain = flow.ones(7, device=self.device)  # normalized to gridspace gain
        # ai.shape = (na,nt) 生成anchor索引
        # anchor索引，后面有用，用于表示当前bbox和当前层的哪个anchor匹配
        ai = flow.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        targets = flow.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        # 设置网格中心偏移量
        g = 0.5  # bias
        # 附近的4个框
        off = (
            flow.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets
        # 对每个检测层进行处理
        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape

            # gain[2:6] = flow.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
            gain[2:6] = flow.tensor(p[i].shape, device=self.device)[[3, 2, 3, 2]].float()  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = flow.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                if j.numel() > 0:
                    j = flow.stack((flow.ones_like(j), j, k, l, m))
                    t = t.repeat((5, 1, 1))[j]
                    offsets = (flow.zeros_like(gxy)[None] + off[:, None])[j]
                else:
                    t = targets[0]
                    offsets = 0
                #     j = flow.stack((k, l, m))
                # t = t.repeat((5, 1, 1))[j]
                # offsets = (flow.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors

            # a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            a, (b, c) = (
                a.contiguous().long().view(-1),
                bc.contiguous().long().T,
            )  # anchors, image, class

            # gij = (gxy - offsets).long()
            gij = (gxy - offsets).contiguous().long()

            gi, gj = gij.T  # grid indices

            # Append

            # indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            gi = gi.clamp(0, shape[3] - 1)
            gj = gj.clamp(0, shape[2] - 1)
            indices.append((b, a, gj, gi))  # image, anchor, grid

            tbox.append(flow.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

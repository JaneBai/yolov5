# Loss functions

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        '''
		对于目标obj损失来说无疑是用BCELoss(二分类损失)
		YOLOV5中使用的分类损失是BCEloss,对每个类别作sigmoid（而不是softmax）,
		对于每个类别来说可以是一个2分类任务.
		其中nn.BCEWithLogitsLoss自带sigmoid
		'''
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        """
        对标签做平滑，eps=0就代表不做标签平滑，那么默认cp=1,cn=0
        后续对正类别赋值cp, 负类别赋值cn
        """
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        # 如果g的值为0，则代表不使用focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        '''
		balance用来设置三个特征图对应输出的损失系数
		从左到右对应大特征图(检测小物体)到小特征图(检测大物体),也就是说小物体的损失权重更大.
		当然模型不一定要三个特征图，其他情况比如四个五个特征图，同样可以设置相应的损失权重
		'''
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # 将box标签和anchor进行匹配，其中FPN特征图为三层，分别为降采样8，16，32
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        """
        Args:
            p: 网络输出，List[torch.tensor * 3], p[i].shape = (b, 3, h, w, nc+5), hw分别为特征图的长宽,b为batch-size
            targets: targets.shape = (nt, 6) , 6=icxywh,i表示第一张图片，c为类别，然后为坐标xywh
            model: 模型
        Returns:
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # gain是为了后续的target(na,nt,7)中的归一化的xywh转为特征图中的网格坐标，其中7表示：i c x y w h ai
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        # ai.shape = (na, nt) 生成anchor索引
        # 需要na个尺度，都进行训练，那么标签就需要复制na个，ai就代表每个尺度的索引，默认里面的值为0，1，2
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # targets.shape = (na, nt, 7)
        '''
        targets.repeat(na, 1, 1):  (na,nt,6)
        ai[:, :, None]:  (na,nt,1)  --广播--> (na,nt,1)
        targets:  (na,nt,7)    7: i c x y w h ai
        '''
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        
        # 设置偏移量
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # 对每个检测层进行处理
        for i in range(self.nl):
            '''
            self.anchors[i] ：(3,3,2) 就是hyp.yaml里面看到的anchors的形状,
            3个尺度,3个形状的anchor, 2表示anchor的宽高
            
            anchors: (3,2)  就是当前尺度对应的3个anchor

            原来targets中xywh的值都是0~1之间，也就是原图中，归一化的坐标. 
            对 i c x y w h ai  相对应位置乘上 [1,1,特征图宽,特征图高,特征图宽，特征图高,1]
            
            得到的xywh就是特征图中的位置了,至于为什么是这种线性关系，
            为什么这样乘能得到特征图中的位置，最后会给出解释.
            '''
            anchors = self.anchors[i]
            # 得到特征图的坐标系数
            """
            p[i].shape = (b, 3, h, w，nc+5), hw分别为特征图的长宽
            gain = [1, 1, w, h, w, h, 1]
            """
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain # 原始归一化坐标转化为当前特征图中的网格坐标
            if nt:
                # Matches
                """
                预测的wh与anchor的wh做匹配，筛选掉比值大于hyp['anchor_t']的(这应该是yolov5的创新点)，从而更好的回归(与新的边框回归方式有关)
                由于yolov3回归wh采用的是out=exp(in)，这很危险，因为out=exp(in)可能会无穷大，就会导致失控的梯度，不稳定，NaN损失并最终完全失去训练；
                (当然原yolov3采用的是将targets进行反算来求in与网络输出的结果，就问题不大，但采用iou loss，就需要将网络输出算成out来进行loss求解，所以会面临这个问题)；
                所以作者采用新的wh回归方式:
                (wh.sigmoid() * 2) ** 2 * anchors[i], 原来yolov3为anchors[i] * exp(wh)
                将标签框与anchor的倍数控制在0~4之间；
                hyp.scratch.yaml中的超参数anchor_t=4，所以也是通过此参数来判定anchors与标签框契合度
                """
                # 计算比值ratio
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio

                """
                筛选满足1 / hyp['anchor_t'] < targets_wh/anchor_wh < hyp['anchor_t']的框;
                由于wh回归公式中将标签框与anchor的倍数控制在0~4之间，所以这样筛选之后也会浪费一些输出空间；
                由于分给每个特征金字塔层的anchor尺度都不一样，这里根据标签wh与anchor的wh的比例分配标签，
                就相当于把不同尺度的GT分配给了不同的特征层来回归；
                """
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # yolov5不再通过iou来分配标签，而仅仅使用网格分配
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # 晒选过后的 t.shape = (M, 7), M为筛选过后的数量
                t = t[j]  # filter

                # Offsets
                # 得到box中心点坐标xy(相对于图像左上角的), (M, 2)
                gxy = t[:, 2:4]  # grid xy
                # 得到box中心点相对于特征图右下角的坐标，（M，2）
                gxi = gain[[2, 3]] - gxy  # inverse
                # ((gxy % 1. < g) & (gxy > 1.)).T shape为(2, M)
                # jklm shape为(M, )
                """
                把相对于各个网格左上角x<0.5,y<0.5和相对于右下角的x<0.5,y<0.5的框提取出来；
                也就是j,k,l,m，在选取gij(也就是标签框分配给的网格的时候)对这四个部分的框都做一个偏移(减去上面的off),也就是下面的gij = (gxy - offsets).long()操作；
                再将这四个部分的框与原始的gxy拼接在一起，总共就是五个部分；
                也就是说：①将每个网格按照2x2分成四个部分，每个部分的框不仅采用当前网格的anchor进行回归，也采用该部分相邻的两个网格的anchor进行回归；
                原yolov3就仅仅采用当前网格的anchor进行回归；
                估计是用来缓解网格效应，但由于v5没发论文，所以也只是推测，yolov4也有相关解决网格效应的措施，是通过对sigmoid输出乘以一个大于1的系数；
                这也与yolov5新的边框回归公式相关；
                由于①，所以中心点回归也从yolov3的0~1的范围变成-0.5~1.5的范围；
                所以中心点回归的公式变为：
                xy.sigmoid() * 2. - 0.5 + cx
                每个标签框采用了更多的anchor来回归，笔者亲测这个操作是增加了召回率，但准确率有所下降
                总体来说map会有所增加
                """
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # b为batch中哪一张图片的索引，c为类别
            b, c = t[:, :2].long().T  # image, class
            # 中心点回归标签
            gxy = t[:, 2:4]  # grid xy
            # 长宽回归标签
            gwh = t[:, 4:6]  # grid wh
            # 对应于原yolov3中，gij = gxy.long()
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            # a为anchor的索引
            a = t[:, 6].long()  # anchor indices
            #添加索引，方便计算损失的时候取出对应位置的输出
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            # 获取(x,y)相对于网格
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

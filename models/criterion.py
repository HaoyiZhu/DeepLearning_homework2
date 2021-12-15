import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class CELoss(nn.Layer):
    def __init__(self, epsilon=None, weighted=False):
        super(CELoss, self).__init__()
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon
        self.weighted = weighted

    def _labelsmoothing(self, target, class_num):
        if len(target.shape) == 1 or target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]
        if self.epsilon is not None:
            class_num = x.shape[-1]
            label[label.argmax(-1) < 5] = self._labelsmoothing(label[label.argmax(-1) < 5], class_num)
            soft_label = True
        else:
            if label.shape[-1] == x.shape[-1]:
                soft_label = True
            else:
                soft_label = False
        weight = None
        if self.weighted:
            weight = label.sum(0)
            weight[weight < 1] = 1e10
            weight = 1 / weight
            weight = weight / weight.sum() * 10

        loss = F.cross_entropy(x, label=label, weight=weight, soft_label=soft_label)
        loss = loss.mean()
        return loss
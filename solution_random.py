import random

class AlgSolution: # 类名不能变

    def __init__(self):
        pass

    def reset(self, reference_text, reference_image):
        # 每次进行新的测试数据推理，会先调用reset函数；
        # reference_text和reference_image分别为该数据的参考文本和参考图像，如果为None则代表没有
        # reference_image为base64编码的png图像
        # 此函数完成一些状态初始化、缓存清空之类的功能
        pass

    def predicts(self, ob, success):
        # ob: 以base64编码的png图像，分辨率为320*320，代表当前时刻的观测图像
        # success：bool变量，True/False代表上一个指令执行成功/失败
        # 大部分时候success都为True，当选手执行"carry"操作抬起伤员时，如果不在距离允许的范围内，该动作可能会失败

        linear = random.randint(-100, 100)
        angular = random.randint(-30,30)

        action = {
            'angular': angular,  # [-30, 30]
            'velocity': linear,  # [-100, 100],
            'viewport': 0,  # {0: stay, 1: look up, 2: look down},
            'interaction': 0,  # {0: stand, 1: jump, 2: crouch, 3: carry, 4: drop, 5: open door}
        }
        return action
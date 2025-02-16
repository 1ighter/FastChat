import random
from typing import List, Dict


class AdTitleGenerator:
    def __init__(self):
        self.features = {
            "style": ["强网感", "玩梗文化", "夸张对比", "场景化", "情感共鸣", "权威背书"],
            "structure": ["分层结构", "符号强化", "数字突显", "对比结构", "递进句式", "地域限定"],
            "interaction": ["悬念设置", "参与感营造", "数据说服", "时效刺激", "利益承诺", "群体认同"]
        }
        self.industry_rules = {
            "美妆": {"适配特征": ["权威背书", "数据说服", "场景化"], "修饰词": ["必备", "神器"]},
            "科技": {"适配特征": ["数字突显", "权威背书", "分层结构"], "修饰词": ["黑科技", "革新"]},
            "教育": {"适配特征": ["情感共鸣", "递进句式", "群体认同"], "修饰词": ["秘籍", "突破"]}
        }
        self.enhancers = ["加入网络流行语", "创造对比冲突", "使用场景化描述", "添加权威数据", "制造紧迫感"]

    def _select_features(self, category: str, count: int = 1) -> List[str]:
        """随机选择特征（参考资料<a target="_blank" href="https://developer.baidu.com/article/details/2795219" class="hitref" data-title="Python随机抽样的三种方法及自定义封装函数实现-百度开发..." data-snippet='在Python中,随机抽样是一种常用的数据处理方法。以下是三种常用的随机抽样方法:random.choice、random.sample和numpy.random.choice。接下来,我们将自定义封装...' data-url="https://developer.baidu.com/article/details/2795219">8</a>）"""
        return random.sample(self.features[category], k=min(count, len(self.features[category])))

    def _evaluate_adaptation(self, industry: str, features: List[str]) -> str:
        """行业适配评估（参考资料<a target="_blank" href="https://tech.meituan.com/2020/01/23/meituan-delivery-machine-learning.html" class="hitref" data-title="一站式机器学习平台建设实践 - 美团技术团队" data-snippet='1）首先在获取数据阶段，支持在线和离线两个层面的处理，分别通过采样、过滤、归一化、标准化等手段生产实时和离线特征，并推送到在线的特征库，供线上服务使用 ...' data-url="https://tech.meituan.com/2020/01/23/meituan-delivery-machine-learning.html">14</a><a target="_blank" href="https://geek.zshipu.com/post/%E4%BA%92%E8%81%94%E7%BD%91/%E9%A2%84%E4%BC%B0%E5%9C%A8%E5%8A%A8%E6%80%81%E6%A0%B7%E5%BC%8F%E5%BB%BA%E6%A8%A1%E5%92%8C%E7%89%B9%E5%BE%81%E8%A1%A8%E8%BE%BE%E5%AD%A6%E4%B9%A0%E6%96%B9%E9%9D%A2%E7%9A%84%E8%BF%9B%E5%B1%95/" class="hitref" data-title="预估在动态样式建模和特征表达学习方面的进展 - 知识铺的博客" data-snippet='一个广告要展示的时候，我们第一层要选Layout，确定样式的布局；选定Layout 之后，要选定每个容器中放哪个物料，从图片到描述，到标题等等，整个计算就是一个多连 ...' data-url="https://geek.zshipu.com/post/%E4%BA%92%E8%81%94%E7%BD%91/%E9%A2%84%E4%BC%B0%E5%9C%A8%E5%8A%A8%E6%80%81%E6%A0%B7%E5%BC%8F%E5%BB%BA%E6%A8%A1%E5%92%8C%E7%89%B9%E5%BE%81%E8%A1%A8%E8%BE%BE%E5%AD%A6%E4%B9%A0%E6%96%B9%E9%9D%A2%E7%9A%84%E8%BF%9B%E5%B1%95/">18</a>）"""
        if industry not in self.industry_rules:
            return "常规适配"
        return "高度契合" if any(f in self.industry_rules[industry]["适配特征"] for f in features) else "常规适配"

    def generate_titles(self,
                        primary_industry: str,
                        secondary_industry: str,
                        seo_keywords: List[str],
                        num_titles: int = 10) -> List[str]:
        """
        生成广告标题主方法（参考资料<a target="_blank" href="https://blog.csdn.net/fufulove/article/details/142312908" class="hitref" data-title="利用Python进行在线广告推荐和优化毕业设计源码_python编..." data-snippet='来自北京大学的张晓磊等人发表的《Python-based Advertising Optimization》一文,提出了一种基于内容的在线广告优化算法。该算法利用了Python中的pandas、nump...' data-url="https://blog.csdn.net/fufulove/article/details/142312908">2</a><a target="_blank" href="https://developer.nvidia.com/zh-cn/blog/nvidia-tensorrt-llm-ad-generation/" class="hitref" data-title="NVIDIA TensorRT-LLM 在推荐广告及搜索广告的生成式召回的加速实践" data-snippet='在建模封装层，通过TensorRT-LLM 实现LLM 模型的构建与优化。然后将LLM 无缝整合至现有生态系统，利用Python 与TensorFlow API 实现端到端推理图的构建。' data-url="https://developer.nvidia.com/zh-cn/blog/nvidia-tensorrt-llm-ad-generation/">12</a>）
        :param primary_industry: 一级行业
        :param secondary_industry: 二级行业
        :param seo_keywords: SEO关键词列表
        :param num_titles: 生成数量
        :return: 广告标题列表
        """
        results = []
        for _ in range(num_titles):
            # 随机特征选择（每个类别1-2个）
            selected_features = {
                "style": self._select_features("style", random.randint(1, 2)),
                "structure": self._select_features("structure", random.randint(1, 2)),
                "interaction": self._select_features("interaction", random.randint(1, 2))
            }

            # 行业适配评估
            adaptation = self._evaluate_adaptation(
                primary_industry,
                sum(selected_features.values(), [])
            )

            # 构建prompt
            prompt = f"""基于以下信息，生成一个广告标题:seo关键词为{' '.join(seo_keywords)},
                风格特征为{','.join(selected_features['style'])},
                结构结构为{','.join(selected_features['structure'])},
                引导用户行为为{','.join(selected_features['interaction'])},
                一级行业为{primary_industry},
                二级行业为{secondary_industry}。
                创意增强：{'/'.join(random.sample(self.enhancers, 2))}"""

            # f"与行业适配程度为{adaptation}",

            # 此处应调用大模型API生成实际标题（模拟演示）
            results.append(prompt)

        return results


# 使用示例（参考资料<a target="_blank" href="https://blog.51cto.com/topic/79a01a9e252399c.html" class="hitref" data-title="python 将模型封装成服务_51CTO博客" data-snippet='51CTO博客已为您找到关于python 将模型封装成服务的相关内容,包含IT学习相关文档代码介绍、相关教程视频课程,以及python 将模型封装成服务问答内容。更多python 将模型封装成...' data-url="https://blog.51cto.com/topic/79a01a9e252399c.html">3</a><a target="_blank" href="https://blog.51cto.com/topic/c8e3dbee2d13be0.html" class="hitref" data-title="python在线封装_51CTO博客" data-snippet='51CTO博客已为您找到关于python在线封装的相关内容,包含IT学习相关文档代码介绍、相关教程视频课程,以及python在线封装问答内容。更多python在线封装相关解答可以来51CTO博客...' data-url="https://blog.51cto.com/topic/c8e3dbee2d13be0.html">5</a>）
if __name__ == "__main__":
    generator = AdTitleGenerator()
    titles = generator.generate_titles(
        primary_industry="美妆",
        secondary_industry="护肤品",
        seo_keywords=["天然成分", "抗衰老"],
        num_titles=3
    )
    for idx, title in enumerate(titles, 1):
        print(f"标题{idx}: {title}")

import random
from typing import List, Dict


class AdTitleGenerator:
    def __init__(self):
        self.features = {
            "style": ["强网感", "玩梗文化", "夸张对比", "场景化", "情感共鸣", "权威背书"],
            "structure": ["分层结构", "符号强化", "数字突显", "对比结构", "递进句式", "地域限定"],
            "interaction": ["悬念设置", "参与感营造", "数据说服", "时效刺激", "利益承诺", "群体认同"]
        }

        self.enhancers = ["加入网络流行语", "创造对比冲突", "使用场景化描述", "添加权威数据", "制造紧迫感"]

    def _select_features(self, category: str, count: int = 1) -> List[str]:
        """随机选择特征（参考资料<a target="_blank" href="https://developer.baidu.com/article/details/2795219" class="hitref" data-title="Python随机抽样的三种方法及自定义封装函数实现-百度开发..." data-snippet='在Python中,随机抽样是一种常用的数据处理方法。以下是三种常用的随机抽样方法:random.choice、random.sample和numpy.random.choice。接下来,我们将自定义封装...' data-url="https://developer.baidu.com/article/details/2795219">8</a>）"""
        return random.sample(self.features[category], k=min(count, len(self.features[category])))


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

    def generate_titles_by_title(self,
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


            # 构建prompt
            prompts = [
                "请根据以下标题生成一个类似的标题：",
                "请根据以下标题进行扩展：",
                "请用不同的表达方式改写以下标题：",
                "请用更吸引人的方式改写以下标题：",
                "请根据以下标题，生成一个更具创意的标题："
            ]

            # f"与行业适配程度为{adaptation}",

            # 此处应调用大模型API生成实际标题（模拟演示）

            results.append(random.choice(prompts) + seo_keywords[0])

        return results

    def generate_titles_by_industry(self,
                        primary_industry: str,
                        secondary_industry: str,
                        seo_keywords: List[str],
                        num_titles: int = 10) -> List[str]:
        """
        根据行业生成广告标题
        :param primary_industry:
        :param secondary_industry:
        :param num_titles:
        :return:
        """
        results = []
        for _ in range(num_titles):
            prompts =  [
    f"请根据以下行业生成一条广告标题：一级行业是{primary_industry}，二级行业是{secondary_industry}。要求包含数字和痛点刺激，采用'X个必知避坑'结构，15字内生成b站爆款标题。",
    f"请根据以下行业生成一条广告标题：一级行业是{primary_industry}，二级行业是{secondary_industry}。使用'2024最新'等时效词+技术创新关键词，模仿行业报告体例生成。",
    f"请根据以下行业生成一条广告标题：一级行业是{primary_industry}，二级行业是{secondary_industry}。制造认知反差，采用'居然/原来'转折词+emoji符号生成b站高赞标题。",
    f"请根据以下行业生成一条广告标题：一级行业是{primary_industry}，二级行业是{secondary_industry}。按'99%人不知道的XX秘籍'结构生成b站SEO标题，突出信息稀缺性。",
    f"请根据以下行业生成一条广告标题：一级行业是{primary_industry}，二级行业是{secondary_industry}。结合场景化痛点，使用'遇到XX怎么办？'设问句式生成b站问答标题。",
    f"请根据以下行业生成一条广告标题：一级行业是{primary_industry}，二级行业是{secondary_industry}。采用'XXX vs XXX'对比结构生成电商商品标题，突出性能优势。",
    f"请根据以下行业生成一条广告标题：一级行业是{primary_industry}，二级行业是{secondary_industry}。按'手把手/保姆级'关键词生成新手教程标题，强调零基础速成。",
    f"请根据以下行业生成一条广告标题：一级行业是{primary_industry}，二级行业是{secondary_industry}。用'所有XX人都该看'绝对词+互动句式生成微博话题标题。",
    f"请根据以下行业生成一条广告标题：一级行业是{primary_industry}，二级行业是{secondary_industry}。生成速卖通商品智能标题。",
    f"请根据以下行业生成一条广告标题：一级行业是{primary_industry}，二级行业是{secondary_industry}。采用情感营销策略生成朋友圈文案标题，触发用户共鸣。"
        ]

            results.append(random.choice(prompts))

        return results


# 使用示例（参考资料<a target="_blank" href="https://blog.51cto.com/topic/79a01a9e252399c.html" class="hitref" data-title="python 将模型封装成服务_51CTO博客" data-snippet='51CTO博客已为您找到关于python 将模型封装成服务的相关内容,包含IT学习相关文档代码介绍、相关教程视频课程,以及python 将模型封装成服务问答内容。更多python 将模型封装成...' data-url="https://blog.51cto.com/topic/79a01a9e252399c.html">3</a><a target="_blank" href="https://blog.51cto.com/topic/c8e3dbee2d13be0.html" class="hitref" data-title="python在线封装_51CTO博客" data-snippet='51CTO博客已为您找到关于python在线封装的相关内容,包含IT学习相关文档代码介绍、相关教程视频课程,以及python在线封装问答内容。更多python在线封装相关解答可以来51CTO博客...' data-url="https://blog.51cto.com/topic/c8e3dbee2d13be0.html">5</a>）
if __name__ == "__main__":
    generator = AdTitleGenerator()
    titles = generator.generate_titles_by_industry(
        primary_industry="美妆",
        secondary_industry="护肤品",
        seo_keywords=["茶吧机牌子哪个好"],
        num_titles=10
    )
    for idx, title in enumerate(titles, 1):
        print(f"标题{idx}: {title}")

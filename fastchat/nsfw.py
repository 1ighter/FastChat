class nsfw:
    def __init__(self):
        self.ban_words = ['充气妹', '平成元年', '改号软件', '芳香疗法', '股权激励',
                          '光州', '最新技术', '刷微博粉丝中差评', '出售微博粉丝', '短信群发器',
                          '小攻', '夕阳红', '广场屠杀', '王丹', '同人文', '灵数', '撩拨',
                          '呼死你', '调教', '掌相', '以诺派魔法', '假公文材料',
                          '求约', '投资移民', '暴乱', '6月4', '避孕药', '民主女神',
                          '公积金提取', '电子称干扰', '屠杀', '政治风波', '妹控',
                          '一元购', '高频以太', '代考', '微商', '共同妄想', 'BoysLove', 'boyslove', 'bl',
                          '清除负面新闻', 'X日女友', '小艾斯', '接触魔法', '能量治疗', '个股推荐',
                          '宗教用品', '崔健', '偷电', '64', 'ASMR', '颅内高潮', 'asmr', '长安街', '三十多年前',
                          '算卦', 'K12', '李鹏', '轮回转世', '个人证件代办', '代练', '首都战车',
                          '刘华清', '89风云', '骚', '虚拟账号交易', 'feiji杯', '撸Sir', '足控',
                          '民主无价', '维多利亚公园', '学生绝食', '语C', '催债', '一撸到底', '丝袜控',
                          '摄像机拍下的这个画面', '同学们我们来晚了', '农药', '调戏', '奔现', '以太体',
                          '黄雀行动', 'SM', '泣血的春天', '射', '性别鉴定', '安全预防', '吾尔开希', '除臭',
                          '濒死体验', '青年节次月', '牛B', '择日', '广场学生', '黑丝控', '卖肾', '虚拟装备交易',
                          '中美自由结合体', '灵应板', '学生肉饼', '节电器', '磕音', 'YAOI', '境外劳务', '记忆力培训',
                          '没有烟抽的日子', '为爱鼓掌', '纯天然', '人体器官买卖', '办证刻章', '淦', '阻挡解放军坦克',
                          '试药', '代缴社保', '你妈逼', '学联', '纯灵性', '脚控', '灵摆', '撸蛇', '求签', '平反',
                          '矮穷挫', '国殇之柱', '坦克人', '天安门', '基础网格', '代办积分落户', '宠物出售', '境外置业',
                          '心理培训', '戒网瘾', '麻将室', '个人代购店铺', '周舵', '对不起同学们', '这是我的职责',
                          '6在左4在右', '小受', '一夜情', '黑客软件', '李小琳', '珠宝鉴定', '青少年性教育', '戒毒',
                          '敏感年份', '校园贷', '刚加苏丹国魔法', '黑客技术', '网络信誉等级作弊', '鲍彤', '不存在的一天',
                          '代开发票', '减肥馆', '小卡片', '舔狗', '不存在的那一天', '邓小平', '烟花爆竹', '诋毁他人信息服务',
                          '求偶', '无副作用', '心灵传输', '广场事件', '增高药', '肉搏', '项小吉', '裸', '坦克第1师', '撸',
                          '棋牌室', '刷单', '断背', '国家珍稀植物', '维园', '88+1', '星座占卜', '挑逗', '抖M', '男权',
                          '赌石行为', '李小鹏', '六月的广场', '和谐能量', '搜索引擎排名作弊', '刘霞', 'GV','gv', '疫苗',
                          '广场上数千人的鲜血', '一块红布', 'NFT', 'nft', '百年空椅', '原油交易', '卢恩魔法', '419',
                          '硬了', 'LSP', '公积金代缴', '灰魔法', '债权转让', '健美', '灵动', '福利姬', '挡车的歹徒',
                          '八嘎', 'FJ杯', '正太控', '代发论文', '逼格', '陆肆', '学生被碾', '咨询积分落户', '坦克',
                          '混沌魔法', '假如我们的铁蹄', '念写', '8的平方', '调情', '香港人权组织', '肾源', '情感挽回',
                          '敏感的日子', '试爱', '封建迷信', '伪基站', '强权压不了悲伤', '区块链', '断袖', '外挂',
                          '他们不在乎我们', '心灵感应', '屌', '妈妈我没有做错', '当年的学生', '网赚提现', '灵视',
                          '啪啪', '众筹', '民国78年', '语爱', '药到病除', '足浴按摩', '38集团军', '试婚', '蕾丝',
                          '信用卡套现', '股票配资', '遍地烛光', '塔罗魔法', 'KKP', 'kkp', '89', 'G点', 'g点', '丧葬殡仪', '一元夺宝',
                          '9875321事件', '戒毒器械', '假身份证', 'VIIV', '票据转让', '自由论', '六月首都', '35年前',
                          '郑圣哲', '治愈率', '胡耀邦', '自闭症', '网赚招聘', '89年事件', '姓名学', '酒水/药品回收',
                          '八九暴乱', '出体经验', '广场肉饼', '8963+1', '百合', '灵体投射', '三十八军', '求爱',
                          '赵紫阳', '无效退款', '黑丝', '18禁', '坦克屠城', '假发票', '流产', '境外房产', '三十八集团军',
                          '股权培训', '这是我的责任', '催收', '女用情趣', '矿物治疗', '北京知识界联合会', '致富学',
                          '八嘎压路', '精神力疗法', '我要上广场', '美乳', '丰胸', '首都各界联席会议', '胎儿性别鉴定',
                          '学生贷', '电子捕鱼器', '网赚兼职', '灵媒', '挑弄', '中国人权', '手机游戏代充', '三焦',
                          '极化疗法', '境外产子', '色彩治疗', '柴玲', 'xp雷达', '早恋叛逆', '广场静坐', '基情四射',
                          '处男', '比特币', '前世回溯', '弟控', 'diao', '广场坦克', '吸粉', '拦坦克', '最高技术',
                          '宠物盲盒', '代孕', '亲子鉴定', '6月的那件事', 'POS机', '远程疗法', 'P2P', 'p2p', '凯尔特尔',
                          '干', '性别控制药物', '抑郁症治疗', '证件挂靠', '广场运动', '抖S', '运势测算', '民主纪念碑',
                          '疗效最佳', '人体器官', '历史不会被忘记', '代替魔法', '镇压', '捕鱼游戏', '非法节电', '刘仲敬',
                          'xxx广场', '民主烈士', 'Jan-90', '可软可硬', '装逼', '文爱', '白嫖', '濒临灭绝动物', '信仰治疗',
                          '遥视', '血腥的一夜', '修锁开锁', 'jz', '春夏之交', '支联会', '资质转让', '广场请愿', '李志',
                          '广场上的鲜血', '戒毒药品', '养生馆', '地磅干扰', '89年的北京', '碾人', '股票杠杆', '触物感知',
                          '六月四日', '北京的示威者', '成功学', '动物磁性治疗', '珠宝赌石', '激凸', '三十五年前', '黄牛',
                          '纳音', '8平方', 'my duty', '广场的游魂', '19禁', '传染性魔法', '成人', '劈腿', '日历里不存在',
                          '青楼', '沙雕', '代还服务', '以太模板', '神入疗法', '集资', '木樨地', '广场压人', '私人侦探',
                          '静坐绝食', '前世记忆', '兄控', '38军', '情趣内衣', '六月飞霜', '赵昕', '搞基', '低俗交友', '情趣', '6.4',
                          '租游戏账号', '学生运动', '各地聚会北京', '小爱慕', '北京饭店', '考试作弊服务', '办假证信息',
                          '国家级新药', '侯德健', '佐药治疗', '五四青年节延后一个月', 'BL', 'bl', '私服', '本子', '撸撸杯',
                          '消失的一天', '刘晓波', '男用情趣', '热血在广场', '情商学', '飞机杯', '假财务票据', '违章代缴费',
                          '89动乱', '国家保护野生动物', '无刺激', '紫微斗数', '活体出售', '掷爻', 'R18', 'r18', 'pos机','攻受', '六四',
                          '代办户口', '螳臂当车', '药品回收', '虚拟货币', '冲锋枪擦了', '64勋章', '干支纳音', '浑沌魔法',
                          '假资质证明', '女权', '假肢生产装配', '屌丝', '股权咨询', '御姐控', '里番', '妈妈我饿了，但我没胃口',
                          '人偶魔法', '六月起义', '视爱', '根治', '斐济杯', '夏日飞雪', '侍寝', '公路上开坦克', '约爱', '刷钻',
                          'POS机销售', '做你的', '刷信用', '胎儿鉴定', '海外月子中心', '酒吧', '双手能量疗法', '军用设备', '北京屠夫',
                          '徐勤先', 'GL', '坦克上街', '处方药', '牛逼','天花板',"最佳", "最具","最爱","最赚","最优","最优秀","最好",
                          "最大","最大程度","最高","最高级","最高档","最奢侈","最低","最低级","最低价","最底","最便宜","史上最低价",
                          "最流行","最受欢迎","最时尚","最聚拢","最符合","最舒适","最先","最先进","最先进科学","最先进加工工艺","最先享受",
                          "最后","最后一波","最新","最新科技","最新科学,第一","中国第一","全网第一","销量第一","排名第一","唯一",
                          "第一品牌","NO.1","TOP.1","独一无二","全国第一","一流","一天","仅此一","最后一波","全国","国家级","国家级产品",
                          "全球级","宇宙级","世界级","顶级","顶尖","尖端","顶级工艺","顶级享受","高级","极品",
                          "极佳","终极","极致,首个","首选","独家","独家配方","首发","首款","全国销量冠军",
                          "国家级产品","国家","国家领导人","填补国内空白","中国驰名","国际品质,大牌","金牌","名牌","王牌",
                          "领袖品牌","世界领先","遥遥领先","领导者","缔造者","创新品牌","领先上市","巨星","著名","掌门人",
                          "至尊","巅峰","奢侈","优秀","资深","领袖","之王","王者","冠军","史无前例","前无古人","永久","万能","祖传","特效","无敌","纯天然","100%",
                          "高档","正品","真皮","超赚","精准","老字号","中国驰名商标","特供","专供","专家推荐","质量免检","无需国家质量检测","免抽检","国家领导人推荐",
                          "国家机关推荐","涉嫌欺诈消费者","恭喜获奖","全民免单","点击有惊喜","点击获取","点击转身","点击试穿",
                          "点击翻转","领取奖品","秒杀","抢爆","再不抢就没了","不会更便宜了","没有他就","错过就没机会了","万人疯抢","全民疯抢","卖疯了",
                          "限时必须具体时间，今日","今天","几天几夜","倒计时","趁现在","就","仅限","周末","周年庆","特惠趴","购物大趴","闪购","品牌团","精品团",
                          "单品团，严禁使用随时结束","随时涨价","马上降价"]
    def banwords(self):
        return self.ban_words
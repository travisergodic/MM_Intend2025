import pandas as pd


df_image_scene = pd.read_csv("../assets/image_scene.csv")
df_intend = pd.read_csv("../assets/intend.csv")



IMAGE_SCENE_LABEL_TO_DESC = {
    k:v for k, v in zip(df_image_scene["label"].tolist(), df_image_scene["description"].tolist())
}

INTEND_LABEL_TO_DESC = {
    k:v for k, v in zip(df_intend["label"].tolist(), df_intend["description"].tolist())
}


IMAGE_SCENCE_CLASSES = [
    '商品分类选项',
    '商品头图',
    '商品详情页截图',
    '下单过程中出现异常（显示购买失败浮窗）',
    '订单详情页面',
    '支付页面',
    '消费者与客服聊天页面',
    '评论区截图页面',
    '物流页面-物流列表页面',
    '物流页面-物流跟踪页面',
    '物流页面-物流异常页面',
    '退款页面',
    '退货页面',
    '换货页面',
    '购物车页面',
    '店铺页面',
    '活动页面',
    '优惠券领取页面',
    '账单/账户页面',
    '个人信息页面',
    '投诉举报页面',
    '实物拍摄(含售后)',
    '外部APP截图',
    '平台介入页面',
    '其他类别图片'
]

INTEND_CLASSES = [
    '反馈密封性不好',
    '是否好用',
    '是否会生锈',
    '排水方式',
    '包装区别',
    '发货数量',
    '反馈用后症状',
    '商品材质',
    '功效功能',
    '是否易褪色',
    '适用季节',
    '能否调光',
    '版本款型区别',
    '单品推荐',
    '用法用量',
    '控制方式',
    '上市时间',
    '商品规格',
    '信号情况',
    '养护方法',
    '套装推荐',
    '何时上货',
    '气泡'
]

IMAGE_SCENE_LABEL_TO_NUMBER = {name:idx+1 for idx, name in enumerate(IMAGE_SCENCE_CLASSES)}


INTEND_LABEL_TO_NUMBER = {name:idx+1 for idx, name in enumerate(INTEND_CLASSES)}


MANUAL_CHAIN_OF_THOUGHT={
    "c5aaa403-e3aa-4441-8ea7-20bd48c5bbc7-381": "页面显示商品退货运费保障服务信息，包含会员权益说明，屬於商品详情页各个部分的截图。图像标籤：商品详情页截图。",
    "23ef81b2-65b0-48e3-ba6b-48e91e798986-695": "订单信息和物流详情显示在页面上，。图像标籤：订单详情页面"
}
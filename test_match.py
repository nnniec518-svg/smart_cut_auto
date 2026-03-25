#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试本地生活优化匹配"""
from core.sequence_planner import SequencePlanner
import json
import logging

logging.basicConfig(level=logging.INFO)

# 文案脚本
script = """不用确定我们京东315活动3C、家电政府补贴至高15%
我再问一下
不用问,美的四大权益都可以享受
权益一
全屋智能家电套购
送至高1499元豪礼
权益二
空气机及无风感系列空调一价全包
至高省1399元
权益三
购指定型号享最高
价值3200元局改服务
权益四
上抖音团购领
美的专属优惠券
活动时间：2026年3月1日-3月31日
美的火三月福利，今天带你们薅小天鹅洗烘套装的羊毛，错过等一年
美的50代150 还可以叠加京东活动真的是太划算了
首选这套本色洗烘套装
蓝氧护色黑科技太懂女生了——白T恤越洗越亮，彩色衣服不串色，洗完烘完直接能穿，不用再担心晒不干有异味
现在领50代150的券，叠加京东315补贴，算下来比平时便宜大几百
重点来了
京东315还有新房全屋全套5折抢，老房焕新补贴直接省到家
买小天鹅 先用券减100，再享政府补贴，双重优惠叠加，这便宜不占白不占
我在京东电器旗舰店等你们
想买洗烘套装的赶紧来，记得领50代150的券，叠加补贴真的超划算～"""

# 初始化规划器
planner = SequencePlanner(db_path="storage/materials.db")

# 执行规划
edl = planner.plan(script)

# 保存结果
with open("temp/sequence.json", "w", encoding="utf-8") as f:
    json.dump({
        "edl": edl,
        "total_clips": len(edl)
    }, f, ensure_ascii=False, indent=2)

print(f"\n=== 匹配完成 ===")
print(f"总片段数: {len(edl)}")
print(f"A_ROLL: {sum(1 for e in edl if e.get('track_type') == 'A_ROLL')}")
print(f"B_ROLL: {sum(1 for e in edl if e.get('track_type') == 'B_ROLL')}")
print(f"缺失: {sum(1 for e in edl if e.get('missing'))}")
print(f"\n详细结果已保存到 temp/sequence.json")

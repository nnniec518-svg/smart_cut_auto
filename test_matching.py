import sys
sys.path.insert(0, r'c:\Users\nnniec\Program\smart_cut_auto')

from core.planner import SequencePlanner
from db.models import Database

script = """不用确定我们京东315活动3C、家电政府补贴至高15%
我再问一下
不用问,美的四大权益都可以享受
权益一
全屋智能家电套购
送至高1499元豪礼"""

db = Database()
planner = SequencePlanner(db)

print("="*50)
print("开始匹配测试")
print("="*50)

# 执行匹配
result = planner.plan(script)

print("\n" + "="*50)
print(f"匹配结果: {len(result)} 个片段")
print("="*50)

# 统计
a_roll_count = sum(1 for r in result if r.get('track_type') == 'A_ROLL')
b_roll_count = sum(1 for r in result if r.get('track_type') == 'B_ROLL')

print(f"A_ROLL: {a_roll_count}")
print(f"B_ROLL: {b_roll_count}")
print(f"总计: {len(result)}")

# 打印详情
print("\n详情:")
for i, r in enumerate(result):
    print(f"{i+1}. [{r.get('track_type')}] {r.get('text', '')[:30]} -> {r.get('video_path', '')[:40]}")
    print(f"   similarity={r.get('similarity', 0):.3f}, reason={r.get('reason', '')}")

db.close()

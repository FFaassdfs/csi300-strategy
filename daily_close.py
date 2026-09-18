"""
每日收盘数据刷新: 拉取行情入库 + 计算指标/信号
供计划任务 15:10 调用 (2026-09-10 修复: 不再调用 send_advice,
收盘邮件由 16:00 独立计划任务发送, 避免一天两封重复邮件)
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(script, *args):
    print(f'--- {script} ---')
    subprocess.run([PY, os.path.join(PROJECT_ROOT, script), *args], cwd=PROJECT_ROOT)


def main():
    # 刷新数据入库 (邮件交给 16:00 计划任务 send_advice.py close)
    run('auto_refresh.py')


if __name__ == '__main__':
    main()

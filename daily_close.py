"""
每日收盘全流程: 刷新数据 → 生成信号 → 发送邮件建议
供计划任务 15:10 调用
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
    # 1. 刷新数据入库
    run('auto_refresh.py')
    # 2. 发送策略邮件
    run('send_advice.py')


if __name__ == '__main__':
    main()

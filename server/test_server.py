import requests
import time
import sys
from threading import Thread
from contextlib import contextmanager

BASE_URL = "http://127.0.0.1:8000"

# 自定义进度条样式
class ProgressBar:
    def __init__(self, total, width=40):
        self.total = total
        self.width = width

    def update(self, current):
        progress = int(self.width * current / self.total)
        bar = '█' * progress + '-' * (self.width - progress)
        percent = f"{100 * current / self.total:.1f}%"
        sys.stdout.write(f"\r[{bar}] {percent}")
        sys.stdout.flush()


# 测试工具类
class ServiceTester:
    def __init__(self):
        self.start_url = f"{BASE_URL}/start-task"
        self.check_url = f"{BASE_URL}/check-status/"


    def test_async_task(self, message, task_id = 1):
        """完整测试流程"""
        print("触发异步任务...")
        response = requests.post(self.start_url, message)
        res = response.json()
        task_id = res['task_id']

        task_status = requests.get(self.check_url + task_id)
        print(task_status.json()['status'])



if __name__ == "__main__":
    tester = ServiceTester()

    # 自动启动服务测试（仅限开发环境）
    # 注意：生产环境应单独启动服务
    tester.test_async_task(message="test message", task_id = 1)

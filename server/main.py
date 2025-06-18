import uvicorn
from fastapi import FastAPI
from celery_config import celery_app
import time

from pydantic import BaseModel

class TestRequest(BaseModel):
    messages: str
    info: str
app = FastAPI()

# 定义 Celery 任务
@celery_app.task(bind=True)
def long_running_task(self, TestRequest):
    """模拟长时间运行的任务"""
    duration = TestRequest.duration
    for i in range(TestRequest.duration):
        time.sleep(1)
        self.update_state(
            state="PROGRESS",
            meta={"current": i+1, "total": duration}
        )
    return {"result": "任务完成", "duration": duration}

# FastAPI 路由
@app.post("/start-task")
async def start_task(duration: int = 10):
    # 异步触发 Celery 任务
    task = long_running_task.delay(duration)
    return {"task_id": task.id}

@app.get("/check-status/{task_id}")
async def check_status(task_id: str):
    # 获取任务状态
    result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result
    }

if __name__ == '__main__':
    # 添加以下代码
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
    print()
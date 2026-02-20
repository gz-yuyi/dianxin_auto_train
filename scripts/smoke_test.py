#!/usr/bin/env python3
"""
冒烟测试脚本：端到端测试训练任务和推理 API

测试流程：
1. 后台启动 worker 和 server
2. 调用训练接口，使用 assets 目录中的数据进行 LoRA 训练
3. 等待训练完成
4. 对 inference 所有接口进行冒烟测试
5. 停止所有后台进程
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

# 配置
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
ASSETS_DIR = PROJECT_ROOT / "assets"
API_HOST = "127.0.0.1"
API_PORT = 8001  # 使用非默认端口避免冲突
BASE_URL = f"http://{API_HOST}:{API_PORT}"
TRAINING_TIMEOUT = 600  # 训练超时时间（秒）
POLL_INTERVAL = 5  # 状态轮询间隔（秒）

# 颜色输出
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def log_info(msg: str) -> None:
    print(f"{GREEN}[INFO]{RESET} {msg}")


def log_error(msg: str) -> None:
    print(f"{RED}[ERROR]{RESET} {msg}", file=sys.stderr)


def log_warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def find_training_data() -> tuple[str, str] | None:
    """查找训练数据文件"""
    # 优先使用 1施工单位类型 数据集
    dataset_dir = ASSETS_DIR / "1施工单位类型"
    if dataset_dir.exists():
        train_file = dataset_dir / "施工单位类型_训练集.xlsx"
        test_file = dataset_dir / "施工单位类型_测试集.xlsx"
        if train_file.exists():
            return str(train_file), str(test_file) if test_file.exists() else str(train_file)
    
    # 查找其他可用的数据集
    for subdir in ASSETS_DIR.iterdir():
        if subdir.is_dir():
            xlsx_files = list(subdir.glob("*.xlsx"))
            if xlsx_files:
                # 找包含"训练"的文件，否则使用第一个
                train_files = [f for f in xlsx_files if "训练" in f.name]
                test_files = [f for f in xlsx_files if "测试" in f.name]
                train = train_files[0] if train_files else xlsx_files[0]
                test = test_files[0] if test_files else train
                return str(train), str(test)
    
    return None


def start_services() -> tuple[subprocess.Popen, subprocess.Popen, Path, Path]:
    """后台启动 worker 和 server，输出重定向到日志文件"""
    log_info("启动 Worker 进程...")
    
    # 创建日志目录
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    worker_log = log_dir / "smoke_test_worker.log"
    server_log = log_dir / "smoke_test_server.log"
    
    log_info(f"Worker 日志: {worker_log}")
    log_info(f"Server 日志: {server_log}")
    
    # 使用进程组启动，方便后续终止，输出重定向到文件
    worker_log_fh = open(worker_log, "w", encoding="utf-8")
    worker_proc = subprocess.Popen(
        [sys.executable, "main.py", "worker"],
        cwd=PROJECT_ROOT,
        stdout=worker_log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )
    # 保存文件句柄以便后续关闭
    worker_proc._log_fh = worker_log_fh
    
    log_info("启动 API Server 进程...")
    server_log_fh = open(server_log, "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        [sys.executable, "main.py", "api", "--host", API_HOST, "--port", str(API_PORT)],
        cwd=PROJECT_ROOT,
        stdout=server_log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )
    # 保存文件句柄以便后续关闭
    server_proc._log_fh = server_log_fh
    
    # 等待服务启动
    log_info("等待服务启动...")
    time.sleep(5)
    
    # 检查服务是否正常运行
    for _ in range(30):
        try:
            resp = requests.get(f"{BASE_URL}/docs", timeout=2)
            if resp.status_code == 200:
                log_info("API Server 已就绪")
                break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    else:
        raise RuntimeError("API Server 启动失败")
    
    return worker_proc, server_proc, worker_log, server_log


def stop_services(worker_proc: subprocess.Popen, server_proc: subprocess.Popen) -> None:
    """停止所有后台进程"""
    log_info("停止后台进程...")
    
    for proc, name in [(server_proc, "Server"), (worker_proc, "Worker")]:
        try:
            if os.name != "nt":
                # 使用进程组信号终止整个进程组
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            
            # 等待进程结束
            proc.wait(timeout=5)
            log_info(f"{name} 进程已停止")
        except subprocess.TimeoutExpired:
            log_warn(f"{name} 进程未能及时停止，强制终止...")
            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception as e:
            log_error(f"停止 {name} 进程时出错: {e}")
        finally:
            # 关闭日志文件句柄
            if hasattr(proc, '_log_fh'):
                proc._log_fh.close()


def print_log_files(worker_log: Path, server_log: Path) -> None:
    """打印日志文件内容"""
    print("\n" + "=" * 80)
    print("WORKER 日志")
    print("=" * 80)
    if worker_log.exists():
        try:
            with open(worker_log, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # 只打印最后 2000 字符避免输出过长
                if len(content) > 2000:
                    print("... (跳过开头部分)")
                    print(content[-2000:])
                else:
                    print(content)
        except Exception as e:
            print(f"读取 Worker 日志失败: {e}")
    else:
        print("Worker 日志文件不存在")
    
    print("\n" + "=" * 80)
    print("SERVER 日志")
    print("=" * 80)
    if server_log.exists():
        try:
            with open(server_log, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # 只打印最后 2000 字符避免输出过长
                if len(content) > 2000:
                    print("... (跳过开头部分)")
                    print(content[-2000:])
                else:
                    print(content)
        except Exception as e:
            print(f"读取 Server 日志失败: {e}")
    else:
        print("Server 日志文件不存在")
    print("=" * 80)


def create_training_task(train_file: str, test_file: str) -> str:
    """创建训练任务"""
    log_info("创建 LoRA 训练任务...")
    
    # 注意：测试集没有标签列，不使用单独验证集文件
    # 使用 train_val_split 从训练集划分验证集
    payload = {
        "model_name_cn": "冒烟测试模型",
        "model_name_en": "smoke_test_model",
        "training_data_file": train_file,
        "validation_data_file": None,
        "base_model": "bert-base-chinese",
        "hyperparameters": {
            "learning_rate": 5e-5,
            "epochs": 2,  # 冒烟测试使用较少 epoch
            "batch_size": 32,
            "max_sequence_length": 128,
            "precision": "fp32",
            "gradient_accumulation_steps": 1,
            "early_stopping_enabled": False,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.001,
            "early_stopping_metric": "val_accuracy",
            "random_seed": 42,
            "train_val_split": 0.2,
            "text_column": "文本内容",
            "label_column": "标签列",
            "sheet_name": None,
            "validation_sheet_name": None,
            "lora": {
                "enabled": True,
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ["query", "value"]
            }
        },
        "callback_url": None
    }
    
    resp = requests.post(
        f"{BASE_URL}/training/tasks",
        json=payload,
        timeout=30
    )
    resp.raise_for_status()
    
    task_id = resp.json()["task_id"]
    log_info(f"训练任务已创建: {task_id}")
    return task_id


def wait_for_training(task_id: str) -> bool:
    """等待训练完成"""
    log_info(f"等待训练完成（最长 {TRAINING_TIMEOUT} 秒）...")
    
    start_time = time.time()
    while time.time() - start_time < TRAINING_TIMEOUT:
        resp = requests.get(f"{BASE_URL}/training/tasks/{task_id}", timeout=10)
        resp.raise_for_status()
        
        data = resp.json()
        status = data["status"]
        
        if status == "completed":
            log_info("训练任务完成")
            return True
        elif status in ("failed", "stopped"):
            log_error(f"训练任务失败: {data.get('error_message', 'Unknown error')}")
            return False
        
        # 显示进度
        progress = data.get("progress")
        if progress and progress.get("current_epoch"):
            log_info(f"训练中... Epoch {progress['current_epoch']}/{progress.get('total_epochs', '?')}")
        
        time.sleep(POLL_INTERVAL)
    
    log_error("训练任务超时")
    return False


def get_model_dir_from_task(task_id: str) -> str | None:
    """从任务结果中获取模型目录"""
    resp = requests.get(f"{BASE_URL}/training/tasks/{task_id}", timeout=10)
    resp.raise_for_status()
    
    data = resp.json()
    artifacts = data.get("artifacts")
    if artifacts:
        # 从 lora_adapter_path 或 model_path 推断模型目录
        lora_path = artifacts.get("lora_adapter_path")
        if lora_path:
            return Path(lora_path).parent.name
        model_path = artifacts.get("model_path")
        if model_path:
            return Path(model_path).parent.name
    
    return None


def test_inference_api(model_dir: str) -> bool:
    """测试推理 API"""
    log_info("开始测试推理 API...")
    all_passed = True
    
    # 1. 测试加载模型
    log_info("测试 1/6: 加载 LoRA 模型...")
    try:
        resp = requests.post(
            f"{BASE_URL}/inference/models/load",
            json={"model_dir": model_dir, "max_length": 128},
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        assert data["status"] == "loaded", f"加载失败: {data}"
        model_id = data["model_id"]
        log_info(f"✓ 模型加载成功: {model_id}")
    except Exception as e:
        log_error(f"✗ 模型加载失败: {e}")
        return False
    
    # 2. 测试分类推理
    log_info("测试 2/6: 分类推理...")
    try:
        resp = requests.post(
            f"{BASE_URL}/inference/predict",
            json={
                "model_id": model_id,
                "texts": ["这是一个测试文本", "这是另一个测试文本"],
                "top_n": 3
            },
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        assert "labels" in data, "响应中缺少 labels"
        assert "top_n" in data, "响应中缺少 top_n"
        assert len(data["labels"]) == 2, f"预期返回 2 个标签，实际返回 {len(data['labels'])}"
        log_info(f"✓ 分类推理成功，预测标签: {data['labels']}")
    except Exception as e:
        log_error(f"✗ 分类推理失败: {e}")
        all_passed = False
    
    # 3. 测试可用模型列表
    log_info("测试 3/6: 可用模型列表...")
    try:
        resp = requests.get(f"{BASE_URL}/inference/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        assert "models" in data, "响应中缺少 models"
        assert data["total"] > 0, "模型列表为空"
        log_info(f"✓ 模型列表查询成功，共 {data['total']} 个模型")
    except Exception as e:
        log_error(f"✗ 模型列表查询失败: {e}")
        all_passed = False
    
    # 4. 测试模型查询
    log_info("测试 4/6: 模型查询...")
    try:
        resp = requests.post(
            f"{BASE_URL}/inference/models/query",
            json={"model_ids": [model_id, "non_existent_model"]},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        assert "models" in data, "响应中缺少 models"
        # 应该只返回存在的模型
        found_model_ids = [m["model_id"] for m in data["models"]]
        assert model_id in found_model_ids, f"查询结果中未找到模型 {model_id}"
        log_info(f"✓ 模型查询成功")
    except Exception as e:
        log_error(f"✗ 模型查询失败: {e}")
        all_passed = False
    
    # 5. 测试推理服务状态查询
    log_info("测试 5/6: 推理服务状态查询...")
    try:
        resp = requests.get(f"{BASE_URL}/inference/status", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        assert "service_status" in data, "响应中缺少 service_status"
        assert "workers" in data, "响应中缺少 workers"
        log_info(f"✓ 服务状态查询成功，状态: {data['service_status']}, Workers: {data['total_workers']}")
    except Exception as e:
        log_error(f"✗ 服务状态查询失败: {e}")
        all_passed = False
    
    # 6. 测试卸载模型
    log_info("测试 6/6: 卸载 LoRA 模型...")
    try:
        resp = requests.post(
            f"{BASE_URL}/inference/models/unload",
            json={"model_id": model_id},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        assert data["status"] == "unloaded", f"卸载失败: {data}"
        log_info(f"✓ 模型卸载成功")
    except Exception as e:
        log_error(f"✗ 模型卸载失败: {e}")
        all_passed = False
    
    return all_passed


def main() -> int:
    """主函数"""
    log_info("=" * 60)
    log_info("开始冒烟测试")
    log_info("=" * 60)
    
    # 查找训练数据
    data_files = find_training_data()
    if not data_files:
        log_error("未找到训练数据文件")
        return 1
    
    train_file, test_file = data_files
    log_info(f"训练数据: {train_file}")
    log_info(f"验证数据: {test_file}")
    
    worker_proc = None
    server_proc = None
    worker_log = None
    server_log = None
    
    try:
        # 1. 启动服务
        worker_proc, server_proc, worker_log, server_log = start_services()
        
        # 2. 创建训练任务
        task_id = create_training_task(train_file, test_file)
        
        # 3. 等待训练完成
        if not wait_for_training(task_id):
            log_error("训练任务未成功完成")
            print_log_files(worker_log, server_log)
            return 1
        
        # 4. 获取模型目录
        model_dir = get_model_dir_from_task(task_id)
        if not model_dir:
            log_error("无法从任务结果中获取模型目录")
            print_log_files(worker_log, server_log)
            return 1
        log_info(f"模型目录: {model_dir}")
        
        # 5. 测试推理 API
        if not test_inference_api(model_dir):
            log_error("部分推理 API 测试失败")
            print_log_files(worker_log, server_log)
            return 1
        
        log_info("=" * 60)
        log_info("冒烟测试全部通过！")
        log_info("=" * 60)
        return 0
        
    except Exception as e:
        log_error(f"测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        # 发生异常时打印日志
        if worker_log and server_log:
            print_log_files(worker_log, server_log)
        return 1
        
    finally:
        # 6. 停止服务
        if worker_proc and server_proc:
            stop_services(worker_proc, server_proc)


if __name__ == "__main__":
    sys.exit(main())

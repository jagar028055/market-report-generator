"""
タスクマネージャー - 非同期処理の統合管理
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
from enum import Enum, auto
import uuid

from .async_data_fetcher import AsyncDataFetcher
from .async_chart_generator import AsyncChartGenerator
from .async_report_generator import AsyncReportGenerator
from ..config import get_system_config
from ..utils.exceptions import TaskManagerError, TaskExecutionError
from ..utils.error_handler import ErrorHandler


class TaskStatus(Enum):
    """タスクステータス"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


class TaskPriority(Enum):
    """タスク優先度"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class TaskResult:
    """タスク結果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.start_time and self.end_time:
            self.execution_time = (self.end_time - self.start_time).total_seconds()


@dataclass
class Task:
    """タスク定義"""
    task_id: str
    name: str
    coro: Callable
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


class TaskManager:
    """非同期タスクマネージャー"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.system_config = get_system_config()
        self.error_handler = ErrorHandler(self.logger)
        
        # 非同期コンポーネント
        self.data_fetcher = AsyncDataFetcher(self.logger)
        self.chart_generator = AsyncChartGenerator(logger=self.logger)
        self.report_generator = AsyncReportGenerator(logger=self.logger)
        
        # タスク管理
        self.tasks: Dict[str, Task] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # 並行処理設定
        self.max_concurrent_tasks = self.system_config.MAX_CONCURRENT_REQUESTS
        self.max_workers = self.system_config.MAX_WORKERS
        
        # セマフォとロック
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self.task_lock = asyncio.Lock()
        
        # 統計情報
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "cancelled_tasks": 0,
            "total_execution_time": 0.0
        }
        
        self.logger.info(f"Initialized TaskManager with {self.max_concurrent_tasks} concurrent tasks")
    
    async def add_task(self, 
                      name: str, 
                      coro: Callable,
                      priority: TaskPriority = TaskPriority.MEDIUM,
                      timeout: Optional[float] = None,
                      dependencies: List[str] = None,
                      metadata: Dict[str, Any] = None) -> str:
        """タスクを追加"""
        
        async with self.task_lock:
            task_id = str(uuid.uuid4())
            
            task = Task(
                task_id=task_id,
                name=name,
                coro=coro,
                priority=priority,
                timeout=timeout,
                dependencies=dependencies or [],
                metadata=metadata or {}
            )
            
            self.tasks[task_id] = task
            self.task_results[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.PENDING
            )
            
            self.stats["total_tasks"] += 1
            self.logger.info(f"Added task {task_id}: {name}")
            
            return task_id
    
    async def add_data_fetch_task(self, 
                                 name: str = "data_fetch",
                                 priority: TaskPriority = TaskPriority.HIGH,
                                 timeout: Optional[float] = 120.0,
                                 **kwargs) -> str:
        """データ取得タスクを追加"""
        
        async def data_fetch_coro():
            return await self.data_fetcher.fetch_all_data(**kwargs)
        
        return await self.add_task(
            name=name,
            coro=data_fetch_coro,
            priority=priority,
            timeout=timeout,
            metadata={"type": "data_fetch", "kwargs": kwargs}
        )
    
    async def add_chart_generation_task(self, 
                                       chart_data: Dict[str, Any],
                                       name: str = "chart_generation",
                                       priority: TaskPriority = TaskPriority.MEDIUM,
                                       timeout: Optional[float] = 180.0,
                                       dependencies: List[str] = None) -> str:
        """チャート生成タスクを追加"""
        
        async def chart_gen_coro():
            return await self.chart_generator.generate_all_charts(chart_data)
        
        return await self.add_task(
            name=name,
            coro=chart_gen_coro,
            priority=priority,
            timeout=timeout,
            dependencies=dependencies,
            metadata={"type": "chart_generation", "data_size": len(chart_data)}
        )
    
    async def add_report_generation_task(self, 
                                        name: str = "report_generation",
                                        priority: TaskPriority = TaskPriority.LOW,
                                        timeout: Optional[float] = 300.0,
                                        dependencies: List[str] = None,
                                        **kwargs) -> str:
        """レポート生成タスクを追加"""
        
        async def report_gen_coro():
            return await self.report_generator.generate_complete_report(**kwargs)
        
        return await self.add_task(
            name=name,
            coro=report_gen_coro,
            priority=priority,
            timeout=timeout,
            dependencies=dependencies,
            metadata={"type": "report_generation", "kwargs": kwargs}
        )
    
    async def execute_task(self, task_id: str) -> TaskResult:
        """単一タスクを実行"""
        
        if task_id not in self.tasks:
            raise TaskManagerError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        result = self.task_results[task_id]
        
        # 依存関係をチェック
        if task.dependencies:
            for dep_id in task.dependencies:
                if dep_id not in self.task_results:
                    raise TaskManagerError(f"Dependency {dep_id} not found")
                
                dep_result = self.task_results[dep_id]
                if dep_result.status != TaskStatus.COMPLETED:
                    raise TaskManagerError(f"Dependency {dep_id} not completed")
        
        async with self.semaphore:
            try:
                result.status = TaskStatus.RUNNING
                result.start_time = datetime.now()
                
                self.logger.info(f"Starting task {task_id}: {task.name}")
                
                # タスクを実行
                if task.timeout:
                    task_result = await asyncio.wait_for(
                        task.coro(),
                        timeout=task.timeout
                    )
                else:
                    task_result = await task.coro()
                
                result.result = task_result
                result.status = TaskStatus.COMPLETED
                result.end_time = datetime.now()
                
                self.stats["completed_tasks"] += 1
                self.stats["total_execution_time"] += result.execution_time or 0
                
                self.logger.info(f"Completed task {task_id}: {task.name} in {result.execution_time:.2f}s")
                
            except asyncio.TimeoutError:
                result.status = TaskStatus.TIMEOUT
                result.end_time = datetime.now()
                result.error = TaskExecutionError(f"Task {task_id} timed out")
                
                self.stats["failed_tasks"] += 1
                self.logger.error(f"Task {task_id} timed out")
                
            except asyncio.CancelledError:
                result.status = TaskStatus.CANCELLED
                result.end_time = datetime.now()
                result.error = TaskExecutionError(f"Task {task_id} was cancelled")
                
                self.stats["cancelled_tasks"] += 1
                self.logger.warning(f"Task {task_id} was cancelled")
                
            except Exception as e:
                result.status = TaskStatus.FAILED
                result.end_time = datetime.now()
                result.error = e
                
                self.stats["failed_tasks"] += 1
                self.error_handler.handle_error(e, {"task_id": task_id, "task_name": task.name})
                self.logger.error(f"Task {task_id} failed: {e}")
        
        return result
    
    async def execute_all_tasks(self) -> Dict[str, TaskResult]:
        """すべてのタスクを実行"""
        
        start_time = time.time()
        self.logger.info(f"Starting execution of {len(self.tasks)} tasks")
        
        # 優先度順にタスクをソート
        sorted_tasks = sorted(
            self.tasks.values(),
            key=lambda t: (t.priority.value, t.created_at),
            reverse=True
        )
        
        # 依存関係を解決してタスクを実行
        executed_tasks = set()
        pending_tasks = {task.task_id: task for task in sorted_tasks}
        
        while pending_tasks:
            # 実行可能なタスクを探す
            ready_tasks = []
            for task_id, task in pending_tasks.items():
                if all(dep_id in executed_tasks for dep_id in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # デッドロックを検出
                remaining_tasks = list(pending_tasks.keys())
                raise TaskManagerError(f"Circular dependency detected in tasks: {remaining_tasks}")
            
            # 実行可能なタスクを並行実行
            tasks_to_execute = []
            for task in ready_tasks:
                tasks_to_execute.append(self.execute_task(task.task_id))
            
            # 実行
            results = await asyncio.gather(*tasks_to_execute, return_exceptions=True)
            
            # 結果を処理
            for task, result in zip(ready_tasks, results):
                executed_tasks.add(task.task_id)
                pending_tasks.pop(task.task_id)
                
                if isinstance(result, Exception):
                    self.logger.error(f"Task execution failed: {result}")
        
        execution_time = time.time() - start_time
        self.logger.info(f"Completed execution of all tasks in {execution_time:.2f} seconds")
        
        return self.task_results.copy()
    
    async def execute_task_group(self, task_ids: List[str]) -> Dict[str, TaskResult]:
        """タスクグループを実行"""
        
        start_time = time.time()
        self.logger.info(f"Starting execution of task group: {task_ids}")
        
        # タスクを並行実行
        tasks = [self.execute_task(task_id) for task_id in task_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果をまとめる
        group_results = {}
        for task_id, result in zip(task_ids, results):
            if isinstance(result, Exception):
                group_results[task_id] = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=result
                )
            else:
                group_results[task_id] = result
        
        execution_time = time.time() - start_time
        self.logger.info(f"Completed task group execution in {execution_time:.2f} seconds")
        
        return group_results
    
    async def cancel_task(self, task_id: str) -> bool:
        """タスクをキャンセル"""
        
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.cancel()
            
            # 結果を更新
            if task_id in self.task_results:
                self.task_results[task_id].status = TaskStatus.CANCELLED
                self.task_results[task_id].end_time = datetime.now()
            
            self.logger.info(f"Cancelled task {task_id}")
            return True
        
        return False
    
    async def cancel_all_tasks(self) -> int:
        """すべてのタスクをキャンセル"""
        
        cancelled_count = 0
        for task_id in list(self.running_tasks.keys()):
            if await self.cancel_task(task_id):
                cancelled_count += 1
        
        self.logger.info(f"Cancelled {cancelled_count} tasks")
        return cancelled_count
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """タスクステータスを取得"""
        
        if task_id in self.task_results:
            return self.task_results[task_id].status
        
        return None
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """タスク結果を取得"""
        
        return self.task_results.get(task_id)
    
    def get_all_results(self) -> Dict[str, TaskResult]:
        """すべてのタスク結果を取得"""
        
        return self.task_results.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        
        stats = self.stats.copy()
        
        # 現在の状態を追加
        stats.update({
            "pending_tasks": len([r for r in self.task_results.values() if r.status == TaskStatus.PENDING]),
            "running_tasks": len([r for r in self.task_results.values() if r.status == TaskStatus.RUNNING]),
            "average_execution_time": (
                stats["total_execution_time"] / stats["completed_tasks"] 
                if stats["completed_tasks"] > 0 else 0
            ),
            "success_rate": (
                stats["completed_tasks"] / stats["total_tasks"] * 100 
                if stats["total_tasks"] > 0 else 0
            )
        })
        
        return stats
    
    def clear_completed_tasks(self) -> int:
        """完了したタスクをクリア"""
        
        completed_ids = [
            task_id for task_id, result in self.task_results.items()
            if result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        ]
        
        for task_id in completed_ids:
            self.tasks.pop(task_id, None)
            self.task_results.pop(task_id, None)
        
        self.logger.info(f"Cleared {len(completed_ids)} completed tasks")
        return len(completed_ids)
    
    def cleanup(self):
        """リソースをクリーンアップ"""
        
        # 実行中のタスクをキャンセル
        for task in self.running_tasks.values():
            task.cancel()
        
        # コンポーネントをクリーンアップ
        self.data_fetcher.cleanup()
        self.chart_generator.cleanup()
        self.report_generator.cleanup()
        
        # エラー履歴をクリア
        self.error_handler.clear_history()
        
        # タスクデータをクリア
        self.tasks.clear()
        self.task_results.clear()
        self.running_tasks.clear()
        
        self.logger.info("TaskManager cleanup completed")


# 便利な関数
async def create_complete_report_workflow(task_manager: TaskManager, **kwargs) -> List[str]:
    """完全なレポート生成ワークフローを作成"""
    
    # データ取得タスク
    data_task_id = await task_manager.add_data_fetch_task(
        name="fetch_market_data",
        priority=TaskPriority.HIGH,
        **kwargs
    )
    
    # チャート生成タスク（データ取得後）
    chart_task_id = await task_manager.add_task(
        name="generate_charts",
        coro=lambda: task_manager.chart_generator.generate_all_charts({}),
        priority=TaskPriority.MEDIUM,
        dependencies=[data_task_id]
    )
    
    # レポート生成タスク（すべて完了後）
    report_task_id = await task_manager.add_report_generation_task(
        name="generate_report",
        priority=TaskPriority.LOW,
        dependencies=[data_task_id, chart_task_id],
        **kwargs
    )
    
    return [data_task_id, chart_task_id, report_task_id]


async def create_data_only_workflow(task_manager: TaskManager, **kwargs) -> List[str]:
    """データのみの取得ワークフローを作成"""
    
    data_task_id = await task_manager.add_data_fetch_task(
        name="fetch_market_data_only",
        priority=TaskPriority.HIGH,
        **kwargs
    )
    
    return [data_task_id]


async def create_chart_only_workflow(task_manager: TaskManager, chart_data: Dict[str, Any]) -> List[str]:
    """チャートのみの生成ワークフローを作成"""
    
    chart_task_id = await task_manager.add_chart_generation_task(
        chart_data=chart_data,
        name="generate_charts_only",
        priority=TaskPriority.MEDIUM
    )
    
    return [chart_task_id]


# 使用例
async def run_complete_report_example():
    """完全なレポート生成の使用例"""
    
    task_manager = TaskManager()
    
    try:
        # ワークフローを作成
        workflow_tasks = await create_complete_report_workflow(task_manager)
        
        # すべてのタスクを実行
        results = await task_manager.execute_all_tasks()
        
        # 統計情報を表示
        stats = task_manager.get_statistics()
        print(f"Task execution completed: {stats}")
        
        return results
        
    except Exception as e:
        logging.error(f"Error in run_complete_report_example: {e}")
        raise
    
    finally:
        task_manager.cleanup()


# 同期インターフェース
def run_task_manager_sync(workflow_func: Callable, *args, **kwargs):
    """同期インターフェースでタスクマネージャーを実行"""
    
    async def _run():
        task_manager = TaskManager()
        try:
            return await workflow_func(task_manager, *args, **kwargs)
        finally:
            task_manager.cleanup()
    
    return asyncio.run(_run())
"""
进程内消息中间件 - 类 ROS2 话题机制

在 Windows 无 ROS2 的场景下，用进程内 Pub/Sub 解耦 Qt GUI 与 MuJoCo 仿真。

设计要点:
- Topic: 一个话题，支持多发布者/多订阅者，回调在发布者线程执行
- MessageBroker: 全局单例，管理所有话题
- 线程安全: 所有操作加锁保护
- Qt 集成: 订阅回调中通过 pyqtSignal 转到 GUI 线程
"""
import threading
from typing import Any, Callable, Dict, List, Optional


class Topic:
    """一个话题，支持多发布者/多订阅者"""

    def __init__(self, name: str, max_queue: int = 2):
        self._name = name
        self._lock = threading.Lock()
        self._latest: Any = None
        self._subscribers: List[Callable] = []
        self._max_queue = max_queue

    def publish(self, message: Any):
        """发布消息（非阻塞，直接调用所有订阅者回调）

        注意: 回调在发布者线程中执行。如果回调是 Qt 的 signal.emit，
        那么它是线程安全的（Qt 自动投递到目标线程的事件循环）。
        """
        with self._lock:
            self._latest = message
            callbacks = list(self._subscribers)

        for callback in callbacks:
            try:
                callback(message)
            except Exception as e:
                print(f"[Topic:{self._name}] subscriber callback error: {e}")

    def subscribe(self, callback: Callable):
        """订阅话题，回调在发布者线程执行"""
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable):
        """取消订阅"""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def get_latest(self) -> Any:
        """获取最新消息（用于轮询模式）"""
        with self._lock:
            return self._latest

    def clear_latest(self):
        """清除最新消息缓存，不触发订阅者回调"""
        with self._lock:
            self._latest = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


class MessageBroker:
    """进程内消息中间件 - 类 ROS2 话题机制

    用法:
        broker = MessageBroker.instance()
        topic = broker.create_topic("/simu/images")
        topic.subscribe(my_callback)
        topic.publish(my_data)
    """

    _instance: Optional["MessageBroker"] = None
    _init_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "MessageBroker":
        """获取全局单例"""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """重置单例（仅用于测试或彻底重启）"""
        with cls._init_lock:
            if cls._instance is not None:
                cls._instance.clear_all()
                cls._instance = None

    def __init__(self):
        self._topics: Dict[str, Topic] = {}
        self._lock = threading.Lock()

    def create_topic(self, name: str, max_queue: int = 2) -> Topic:
        """创建或获取话题"""
        with self._lock:
            if name not in self._topics:
                self._topics[name] = Topic(name, max_queue)
            return self._topics[name]

    def get_topic(self, name: str) -> Optional[Topic]:
        """获取已有话题（不创建）"""
        with self._lock:
            return self._topics.get(name)

    def publish(self, topic_name: str, message: Any):
        """发布消息到指定话题"""
        topic = self._topics.get(topic_name)
        if topic is not None:
            topic.publish(message)

    def subscribe(self, topic_name: str, callback: Callable):
        """订阅指定话题（如果话题不存在则自动创建）"""
        with self._lock:
            if topic_name not in self._topics:
                self._topics[topic_name] = Topic(topic_name)
        self._topics[topic_name].subscribe(callback)

    def unsubscribe(self, topic_name: str, callback: Callable):
        """取消订阅"""
        topic = self._topics.get(topic_name)
        if topic is not None:
            topic.unsubscribe(callback)

    def get_latest(self, topic_name: str) -> Any:
        """获取指定话题的最新消息"""
        topic = self._topics.get(topic_name)
        return topic.get_latest() if topic else None

    def clear_all(self):
        """清除所有话题和缓存（仿真重启时调用）"""
        with self._lock:
            self._topics.clear()

    def clear_topic(self, topic_name: str):
        """清除指定话题的缓存（不触发订阅者回调）"""
        topic = self._topics.get(topic_name)
        if topic is not None:
            topic.clear_latest()

    def list_topics(self) -> List[str]:
        """列出所有话题名称"""
        with self._lock:
            return list(self._topics.keys())

    def topic_info(self) -> Dict[str, int]:
        """返回所有话题及其订阅者数量"""
        with self._lock:
            return {name: t.subscriber_count for name, t in self._topics.items()}

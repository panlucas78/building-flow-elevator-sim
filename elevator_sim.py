"""楼宇 Flow 仿真核心：客流输入、乘客生命周期、调度策略和实验评估。

阅读顺序：SimulationConfig → generate_requests → Simulator.dispatch →
Elevator.run/_serve → Simulator.run → run_experiment/benchmark。
所有策略只看到已到达的楼层与方向呼梯；目的地在上车后才进入停站计划。
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import statistics
import zipfile
from dataclasses import asdict, dataclass, field, replace
from itertools import groupby
from typing import Callable, Iterable

import simpy


# 【策略开关】五种组合用于比较 ETA、楼层关联与预测各自带来的影响。
POLICIES = {
    "nearest": "传统就近 + LOOK",
    "eta": "预计接客时间 + LOOK",
    "slp": "ETA + 楼层关联",
    "forecast": "ETA + 预测预定位",
    "adaptive": "ETA + 楼层关联 + 预测",
}
DEFAULT_POLICIES = ("nearest", "eta", "adaptive")
SCENARIOS = {
    "up_peak": "早间上行高峰",
    "down_peak": "傍晚下行高峰",
    "mixed": "教学办公混合客流",
    "class_change": "课间楼层间客流",
}
MAX_REQUESTS = 25000
MAX_CSV_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class SimulationConfig:
    """实验参数集中管理；frozen 防止某个策略在运行中改写共同条件。"""
    floors: int = 12
    elevators: int = 3
    capacity: int = 12
    duration: int = 3600
    travel_seconds: float = 3.0
    door_open_seconds: float = 2.0
    door_close_seconds: float = 2.0
    boarding_seconds: float = 0.6
    alighting_seconds: float = 0.4
    base_rate: float = 0.05
    peak_rate: float = 0.75
    peak_center: float = 1200.0
    peak_width: float = 300.0
    scenario: str = "up_peak"
    seed: int = 42
    dispatch_interval: float = 1.0
    sample_interval: float = 2.0
    forecast_horizon: float = 60.0
    rebalance_interval: float = 30.0
    max_wait_priority: float = 120.0  # 空车优先接长期等待者的阈值，不是等待上限承诺。
    slp_bonus_seconds: float = 4.0

    def __post_init__(self):
        # 入口校验：提前拒绝非法容量、楼层、时间及 NaN，避免异常进入事件循环。
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("随机种子必须是非负整数")
        for name, low, high in (("floors", 2, 60), ("elevators", 1, 12),
                                ("capacity", 1, 40), ("duration", 1, 14400)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(f"{name} 必须是 {low}~{high} 的整数")
        for name in ("travel_seconds", "peak_width", "dispatch_interval", "sample_interval",
                     "forecast_horizon", "rebalance_interval", "max_wait_priority"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须大于 0 且为有限数值")
        for name in ("door_open_seconds", "door_close_seconds", "boarding_seconds",
                     "alighting_seconds", "base_rate", "peak_rate", "slp_bonus_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 不能为负数、NaN 或无穷大")
        if not math.isfinite(self.peak_center) or not 0 <= self.peak_center <= self.duration:
            raise ValueError("高峰中心必须位于仿真时间范围内")
        if self.scenario not in SCENARIOS:
            raise ValueError("未知客流场景")
        if self.base_rate + self.peak_rate > 10:
            raise ValueError("峰值客流不能超过每秒 10 人")


@dataclass(frozen=True)
class Request:
    """不可变的输入请求：编号、到达时间、起点和终点；所有策略共享。"""
    id: int
    arrival: float
    origin: int
    destination: int

    @property
    def direction(self) -> int:
        return 1 if self.destination > self.origin else -1


@dataclass
class Passenger:
    """每个策略独立维护的乘客状态，用三个时间戳还原完整服务过程。"""
    request: Request
    elevator_id: int | None = None
    assigned_at: float | None = None
    boarded_at: float | None = None
    delivered_at: float | None = None

    def row(self, horizon: float) -> dict:
        # 未上车者只记录截至 horizon 的已观察等待，不能当成最终等待时间。
        req = self.request
        return {
            **asdict(req), "elevator_id": self.elevator_id,
            "assigned_at": self.assigned_at, "boarded_at": self.boarded_at,
            "delivered_at": self.delivered_at,
            "status": "已送达" if self.delivered_at is not None else (
                "乘梯中" if self.boarded_at is not None else "等待中"),
            "wait_seconds": None if self.boarded_at is None else self.boarded_at - req.arrival,
            "observed_wait_seconds": (self.boarded_at if self.boarded_at is not None else horizon) - req.arrival,
            "ride_seconds": None if self.delivered_at is None else self.delivered_at - self.boarded_at,
            "journey_seconds": None if self.delivered_at is None else self.delivered_at - req.arrival,
        }


def arrival_rate(config: SimulationConfig, time: float) -> float:
    """基础客流叠加一个高斯形高峰；单位为人/秒，是合成场景假设。"""
    return config.base_rate + config.peak_rate * math.exp(-((time - config.peak_center) / config.peak_width) ** 2)


def _poisson(rng: random.Random, rate: float) -> int:
    """用局部随机数发生器抽取一秒内的泊松到达人数。"""
    threshold, product, count = math.exp(-rate), 1.0, 0
    while product > threshold:
        product *= rng.random()
        count += 1
    return max(0, count - 1)


def generate_requests(config: SimulationConfig) -> tuple[Request, ...]:
    """先生成整批客流，再让各策略分别回放，避免随机输入不同造成假优势。"""
    rng = random.Random(config.seed)
    requests = []
    upper = list(range(2, config.floors + 1))
    teaching = list(range(2, max(2, (config.floors + 1) // 2) + 1))
    for second in range(config.duration):
        count = _poisson(rng, arrival_rate(config, second + 0.5))
        for _ in range(count):
            draw = rng.random()
            if config.scenario == "up_peak":
                mode = "up" if draw < 0.85 else "inter"
            elif config.scenario == "down_peak":
                mode = "down" if draw < 0.85 else "inter"
            elif config.scenario == "mixed":
                mode = "up" if draw < 0.4 else ("down" if draw < 0.8 else "inter")
            else:
                mode = "class" if draw < 0.8 else ("up" if draw < 0.9 else "down")
            # 场景只改变起终点分布；同一次实验的物理参数和到达请求保持一致。
            if mode == "up":
                origin, destination = 1, rng.choice(upper)
            elif mode == "down":
                origin, destination = rng.choice(upper), 1
            else:
                pool = teaching if mode == "class" and len(teaching) > 1 else list(range(1, config.floors + 1))
                origin = rng.choice(pool)
                destination = rng.choice([floor for floor in pool if floor != origin])
            arrival = min(config.duration - 0.000001, round(second + rng.random(), 6))
            if len(requests) >= MAX_REQUESTS:
                raise ValueError("一次实验最多支持 25000 名乘客，请缩短仿真时间或降低客流")
            requests.append(Request(len(requests), arrival, origin, destination))
    return tuple(sorted(requests, key=lambda req: (req.arrival, req.id)))


def validate_requests(requests: Iterable[Request], config: SimulationConfig) -> tuple[Request, ...]:
    """校验自生成或导入请求，并统一排序；拒绝重复编号、越界楼层和时间。"""
    requests = tuple(requests)
    if len(requests) > MAX_REQUESTS:
        raise ValueError("一次实验最多支持 25000 名乘客，请缩短仿真时间或降低客流")
    seen = set()
    for req in requests:
        if not isinstance(req.id, int) or isinstance(req.id, bool) or req.id in seen:
            raise ValueError("乘客 id 必须是唯一整数")
        seen.add(req.id)
        if not math.isfinite(req.arrival) or not 0 <= req.arrival < config.duration:
            raise ValueError(f"乘客 {req.id} 的到达时间必须在 [0, 仿真时长) 内")
        if any(not isinstance(floor, int) or isinstance(floor, bool) or not 1 <= floor <= config.floors
               for floor in (req.origin, req.destination)) or req.origin == req.destination:
            raise ValueError(f"乘客 {req.id} 的起终点楼层无效")
    return tuple(sorted(requests, key=lambda req: (req.arrival, req.id)))


def requests_from_csv(data: bytes, config: SimulationConfig) -> tuple[Request, ...]:
    """把 UTF-8 CSV 转换为同一套 Request，复用模型而不另写导入仿真逻辑。"""
    if len(data) > MAX_CSV_BYTES:
        raise ValueError("CSV 文件不能超过 5 MiB，请减少记录或删除无关列")
    reader = None
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")), strict=True)
        headers = [name.strip() for name in reader.fieldnames] if reader.fieldnames else []
        if not headers or not {"id", "arrival", "origin", "destination"}.issubset(headers):
            raise ValueError("CSV 必须包含 id,arrival,origin,destination 四列，编码为 UTF-8")
        if any(not name for name in headers) or len(headers) != len(set(headers)):
            raise ValueError("CSV 表头不能重复或为空，否则会覆盖或丢失数据")
        reader.fieldnames = headers
        requests = []
        for row in reader:
            line = reader.line_num
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV 第 {line} 行的列数与表头不一致，请检查多余逗号或缺失单元格")
            if len(requests) >= MAX_REQUESTS:
                raise ValueError("CSV 超过 25000 条请求")
            try:
                requests.append(Request(int(row["id"]), float(row["arrival"]),
                                        int(row["origin"]), int(row["destination"])))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV 第 {line} 行存在无效数字") from exc
        return validate_requests(requests, config)
    except UnicodeDecodeError as exc:
        raise ValueError("请将 CSV 另存为 UTF-8 编码") from exc
    except csv.Error as exc:
        line = reader.line_num if reader is not None else 1
        raise ValueError(f"CSV 第 {line} 行格式无效或字段过长，请检查引号、分隔符和字段长度") from exc


def request_fingerprint(requests: Iterable[Request]) -> str:
    """对标准化请求计算 SHA-256；相同指纹便于核对实验输入是否一致。"""
    data = json.dumps([asdict(req) for req in requests], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class DemandPredictor:
    """短时需求预测：只用已发生的呼梯，按起始楼层做双窗口平滑与趋势外推。"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.history: list[Request] = []

    def observe(self, request: Request):
        # 仅由到达事件调用；历史数据不会提前加入未来请求。
        self.history.append(request)

    def forecast(self, now: float) -> list[float]:
        current, previous = [0] * (self.config.floors + 1), [0] * (self.config.floors + 1)
        if now <= 0:
            return [0.0] * (self.config.floors + 1)
        # 最近 0~60 秒和之前 60~120 秒分别计数；目的地不参与预测。
        for req in reversed(self.history):
            if req.arrival < now - 120:
                break
            if req.arrival > now:
                continue
            (current if req.arrival >= now - 60 else previous)[req.origin] += 1
        duration = min(60.0, now)
        old_duration = min(60.0, max(0.0, now - 60))
        result = [0.0] * (self.config.floors + 1)
        for floor in range(1, self.config.floors + 1):
            recent = current[floor] / duration
            old = previous[floor] / old_duration if old_duration else recent
            # 较新窗口占比更高，趋势经阻尼后外推，最后截为非负人数。
            level = 0.7 * recent + 0.3 * old
            trend = 0.25 * (recent - old) if old_duration >= 20 else 0.0
            result[floor] = max(0.0, level + 0.5 * trend) * self.config.forecast_horizon
        return result


class FloorRelations:
    """楼层关联 SLP：分区人工先验与已完成行程的 OD 频次融合。

    OD 指起点和终点；此处是可解释启发式评分，不是训练好的机器学习模型。
    """
    def __init__(self, floors: int):
        self.floors = floors
        self.cut = max(2, (floors + 1) // 2)
        self.od = [[0] * (floors + 1) for _ in range(floors + 1)]

    def prior(self, a: int, b: int) -> float:
        if a == b:
            return 1.0
        if a == 1 or b == 1:
            return 0.8
        if a <= self.cut and b <= self.cut:
            return 0.6
        if a > self.cut and b > self.cut:
            return 0.4
        return 0.1

    def observe_completed(self, request: Request):
        # 只有送达之后才学习起终点关系，保持常规方向呼梯的信息边界。
        self.od[request.origin][request.destination] += 1

    def weight(self, a: int, b: int) -> float:
        row = self.od[a]
        return self.prior(a, b) if not max(row) else 0.7 * self.prior(a, b) + 0.3 * row[b] / max(row)

    def affinity(self, origin: int, stops: set[int]) -> float:
        # 使用平均值而非累加，避免已有停站越多就无条件得到越高加分。
        return statistics.fmean(self.weight(origin, floor) for floor in stops) if stops else 0.0


class Elevator:
    """单台电梯状态机：记录位置、方向、待接乘客、车内乘客和门控动作。"""
    def __init__(self, sim: "Simulator", elevator_id: int):
        self.sim, self.env, self.config = sim, sim.env, sim.config
        self.id, self.floor, self.direction = elevator_id, 1, 0
        self.state = "IDLE"
        self.assigned: list[int] = []  # 已分配但尚未上车的乘客编号。
        self.onboard: list[int] = []  # 已上车且尚未送达的乘客编号。
        self.park_target: int | None = None
        self.moving_from = self.moving_to = None
        self.motion_start = self.motion_end = self.phase_end = 0.0
        self.edge_empty = True
        self.distance = self.empty_distance = 0.0
        self.door_cycles = 0
        self.wakeup = self.env.event()

    @property
    def position(self) -> float:
        # 运行途中用时间比例插值得到连续位置，供回放和截止距离统计使用。
        if self.moving_to is None:
            return float(self.floor)
        fraction = min(1.0, max(0.0, (self.env.now - self.motion_start) / self.config.travel_seconds))
        return self.moving_from + (self.moving_to - self.moving_from) * fraction

    @property
    def workload(self) -> int:
        return len(self.onboard) + len(self.assigned)

    def notify(self):
        # 用 SimPy 事件唤醒空闲电梯；无需真实线程或持续空转轮询。
        if not self.wakeup.triggered:
            self.wakeup.succeed()

    def stops(self, include_full_pickups: bool = False) -> set[int]:
        # 目的地只从 onboard 读取；未上车乘客只贡献接客楼层。
        stops = {self.sim.passengers[pid].request.destination for pid in self.onboard}
        if include_full_pickups or len(self.onboard) < self.config.capacity:
            stops.update(self.sim.passengers[pid].request.origin for pid in self.assigned)
        return stops

    def _oldest(self) -> Passenger | None:
        return min((self.sim.passengers[pid] for pid in self.assigned),
                   key=lambda p: (p.request.arrival, p.request.id), default=None)

    def _urgent(self) -> Passenger | None:
        oldest = self._oldest()
        return oldest if (oldest and not self.onboard and
                          self.env.now - oldest.request.arrival >= self.config.max_wait_priority) else None

    def _boarding_direction(self) -> int | None:
        """判断当前楼层能否接人：容量未满、方向相符，或已到反向时机。"""
        if len(self.onboard) >= self.config.capacity:
            return None
        here = [self.sim.passengers[pid] for pid in self.assigned
                if self.sim.passengers[pid].request.origin == self.floor]
        if not here:
            return None
        urgent = self._urgent()
        if urgent:
            return urgent.request.direction if urgent.request.origin == self.floor else None
        if any(p.request.direction == self.direction for p in here):
            return self.direction
        ahead = any((floor - self.floor) * self.direction > 0 for floor in self.stops())
        if self.direction == 0 or not ahead:
            return min(here, key=lambda p: (p.request.arrival, p.request.id)).request.direction
        return None

    def _next_target(self, stops: set[int]) -> int:
        """LOOK 选站：优先沿当前方向到最近目标，前方无目标时折返。"""
        urgent = self._urgent()
        if urgent:
            return urgent.request.origin
        if self.direction > 0:
            ahead = [floor for floor in stops if floor > self.floor]
            return min(ahead) if ahead else max(stops)
        if self.direction < 0:
            ahead = [floor for floor in stops if floor < self.floor]
            return max(ahead) if ahead else min(stops)
        return min(stops, key=lambda floor: (abs(floor - self.floor), floor))

    def _phase(self, state: str, seconds: float):
        # timeout 推进的是仿真时间，不会按该秒数让电脑真的等待。
        self.state, self.phase_end = state, self.env.now + seconds
        yield self.env.timeout(seconds)

    def _serve(self):
        """一次停站按开门、先下后上、关门执行，逐人计时且严格限制容量。"""
        self.door_cycles += 1
        yield from self._phase("OPENING", self.config.door_open_seconds)
        for pid in list(self.onboard):
            passenger = self.sim.passengers[pid]
            if passenger.request.destination == self.floor:
                yield from self._phase("ALIGHTING", self.config.alighting_seconds)
                self.onboard.remove(pid)
                passenger.delivered_at = float(self.env.now)
                self.sim.relations.observe_completed(passenger.request)
                self.sim.log("送达", self, pid, f"乘客到达 {self.floor} 楼")
        direction = self._boarding_direction()
        if direction is not None:
            self.direction = direction
            while len(self.onboard) < self.config.capacity:
                candidates = [self.sim.passengers[pid] for pid in self.assigned
                              if self.sim.passengers[pid].request.origin == self.floor
                              and self.sim.passengers[pid].request.direction == direction]
                if not candidates:
                    break
                passenger = min(candidates, key=lambda p: (p.request.arrival, p.request.id))
                yield from self._phase("BOARDING", self.config.boarding_seconds)
                # 上车完成时从候梯任务转入车内名单，避免重复派单与重复上车。
                self.assigned.remove(passenger.request.id)
                self.onboard.append(passenger.request.id)
                passenger.boarded_at = float(self.env.now)
                self.sim.log("上车", self, passenger.request.id,
                             f"等待 {passenger.boarded_at - passenger.request.arrival:.1f} 秒")
        yield from self._phase("CLOSING", self.config.door_close_seconds)

    def run(self):
        """SimPy 进程：服务本站、选择目标、逐层运行，无任务则等待唤醒。"""
        while True:
            drop_here = any(self.sim.passengers[pid].request.destination == self.floor for pid in self.onboard)
            if drop_here or self._boarding_direction() is not None:
                yield from self._serve()
                continue
            stops = self.stops() - {self.floor}
            if stops:
                target = self._next_target(stops)
                self.park_target = None
            elif self.park_target is not None and self.park_target != self.floor:
                target = self.park_target
            else:
                self.state, self.direction, self.park_target = "IDLE", 0, None
                yield self.wakeup
                self.wakeup = self.env.event()
                continue
            self.direction = 1 if target > self.floor else -1
            self.state = "MOVING"
            self.moving_from, self.moving_to = self.floor, self.floor + self.direction
            self.motion_start, self.motion_end = self.env.now, self.env.now + self.config.travel_seconds
            self.edge_empty = not self.onboard  # 预定位路段同样计入空载距离。
            yield self.env.timeout(self.config.travel_seconds)
            self.distance += 1.0
            self.empty_distance += float(self.edge_empty)
            self.floor = self.moving_to
            self.moving_from = self.moving_to = None

    def estimate_pickup(self, request: Request) -> float:
        """估计接客时间 ETA：考虑剩余动作、方向、折返、停站和任务积压。

        只使用呼梯起点和方向；这是用于排序的近似值，不是精确到达承诺。
        """
        start = self.moving_to if self.moving_to is not None else self.floor
        delay = max(0.0, self.motion_end - self.env.now) if self.moving_to is not None else max(0.0, self.phase_end - self.env.now)
        stops = self.stops(include_full_pickups=True) | {request.origin}
        distance = abs(start - request.origin)
        if self.direction > 0 and (request.origin < start or (request.direction < 0 and max(stops) > request.origin)):
            turn = max(start, max(stops))
            distance = turn - start + abs(turn - request.origin)
        elif self.direction < 0 and (request.origin > start or (request.direction > 0 and min(stops) < request.origin)):
            turn = min(start, min(stops))
            distance = start - turn + abs(turn - request.origin)
        intermediate = min(len(stops - {request.origin}), max(0, distance - 1))
        stop_time = self.config.door_open_seconds + self.config.door_close_seconds
        overflow = max(0, self.workload + 1 - self.config.capacity)
        return (delay + distance * self.config.travel_seconds + intermediate * stop_time
                + len(self.assigned) * self.config.boarding_seconds
                + overflow * 2 * self.config.travel_seconds + self.config.door_open_seconds)

    def distance_at_horizon(self) -> tuple[float, float]:
        # 截止时即使没走完一层，也计入已完成的部分路程，避免漏算成本。
        fraction = 0.0 if self.moving_to is None else abs(self.position - self.moving_from)
        return self.distance + fraction, self.empty_distance + fraction * self.edge_empty

    def snapshot(self) -> dict:
        return {"id": self.id, "floor": self.floor, "position": self.position,
                "direction": self.direction, "state": self.state, "load": len(self.onboard),
                "assigned": len(self.assigned), "target": self.moving_to or self.park_target,
                "stops": sorted(self.stops())}


@dataclass
class SimulationResult:
    """单个策略的完整输出：指标、逐客记录、回放帧、事件及预测。"""
    policy: str
    metrics: dict
    passengers: list[dict]
    snapshots: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    forecasts: list[dict] = field(default_factory=list)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * quantile
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


class Simulator:
    """单策略实验容器；每个策略有全新的时钟、电梯、乘客和预测历史。"""
    def __init__(self, config: SimulationConfig, requests: Iterable[Request], policy: str, record: bool = True):
        if policy not in POLICIES:
            raise ValueError(f"未知调度策略：{policy}")
        self.config, self.requests, self.policy = config, validate_requests(requests, config), policy
        self.record, self.env = record, simpy.Environment()
        self.passengers: dict[int, Passenger] = {}
        self.elevators = [Elevator(self, index) for index in range(config.elevators)]
        self.predictor, self.relations = DemandPredictor(config), FloorRelations(config.floors)
        self.events, self.snapshots, self.forecasts = [], [], []
        self.last_rebalance = -config.rebalance_interval
        self.last_forecast_record = -20.0

    def log(self, event: str, elevator: Elevator | None = None, pid: int | None = None, detail: str = ""):
        if self.record:
            self.events.append({"time": round(float(self.env.now), 6), "event": event,
                                "elevator": None if elevator is None else elevator.id,
                                "passenger": pid, "detail": detail})

    def _arrivals(self):
        # 请求按时进入可见状态；同一时刻的到达先全部登记，再统一派单。
        for arrival, group in groupby(self.requests, key=lambda req: req.arrival):
            yield self.env.timeout(arrival - self.env.now)
            for req in group:
                self.passengers[req.id] = Passenger(req)
                self.predictor.observe(req)
                self.log("到达", pid=req.id, detail=f"{req.origin} 楼 {'上行' if req.direction > 0 else '下行'}")
            self.dispatch()

    def dispatch(self):
        """给尚未分配的已到达乘客选车；评分越小越优，平局规则固定。"""
        unassigned = sorted((p for p in self.passengers.values() if p.elevator_id is None),
                            key=lambda p: (p.request.arrival, p.request.id))
        for passenger in unassigned:
            # 所有策略共享容量筛选；若都积压则仍可分配，真正上车时再严格限载。
            candidates = [e for e in self.elevators if e.workload < self.config.capacity] or self.elevators
            scores = []
            for elevator in candidates:
                distance = abs(elevator.position - passenger.request.origin)
                eta = elevator.estimate_pickup(passenger.request)
                affinity = self.relations.affinity(passenger.request.origin, elevator.stops(True))
                # 就近比较楼层距离；其余比较秒数。SLP 加分也以秒为单位。
                score = distance if self.policy == "nearest" else eta
                if self.policy in ("slp", "adaptive"):
                    score -= self.config.slp_bonus_seconds * affinity
                scores.append((score, elevator.workload, elevator.id, eta, affinity, elevator))
            _, _, _, eta, affinity, best = min(scores, key=lambda row: row[:3])
            # 一人只分配一次并由同一电梯负责；不会每秒覆盖原分配结果。
            passenger.elevator_id, passenger.assigned_at = best.id, float(self.env.now)
            best.assigned.append(passenger.request.id)
            best.park_target = None
            best.notify()
            self.log("分配", best, passenger.request.id,
                     f"预计接客 {eta:.1f} 秒；当前任务 {best.workload}；楼层关联 {affinity:.2f}")
        if self.env.now - self.last_forecast_record >= 20:
            if self.record:
                prediction = self.predictor.forecast(self.env.now)
                row = {"time": float(self.env.now), "by_floor": prediction,
                       "horizon_seconds": self.config.forecast_horizon,
                       "predicted_next_horizon": sum(prediction)}
                # 仅在确为 60 秒预测时保留旧字段，兼容已有默认实验读取方式。
                if self.config.forecast_horizon == 60:
                    row["predicted_next_60"] = sum(prediction)
                self.forecasts.append(row)
            self.last_forecast_record = self.env.now
        if self.policy in ("forecast", "adaptive") and self.env.now - self.last_rebalance >= self.config.rebalance_interval:
            self._rebalance()
            self.last_rebalance = self.env.now

    def _rebalance(self):
        """把真正空闲且无任务的车提前移向预测需求楼层，不打断载客服务。"""
        if self.env.now < 60 or len(self.predictor.history) < 3:
            return
        prediction = self.predictor.forecast(self.env.now)
        if sum(prediction) < 0.5:
            return
        coverage = [0] * (self.config.floors + 1)
        for elevator in self.elevators:
            anchor = elevator.park_target or round(elevator.position)
            coverage[anchor] += 1
        for elevator in self.elevators:
            if elevator.state != "IDLE" or elevator.workload:
                continue
            coverage[elevator.floor] -= 1
            # 覆盖越多则边际收益越低，路程越长成本越高，减少集中和无效空跑。
            def benefit(floor):
                return prediction[floor] / (1 + coverage[floor]) - 0.08 * abs(floor - elevator.floor)
            target = max(range(1, self.config.floors + 1), key=lambda floor: (benefit(floor), -abs(floor - elevator.floor), -floor))
            if target != elevator.floor and benefit(target) > benefit(elevator.floor) + 0.2:
                elevator.park_target = target
                elevator.notify()
                self.log("预定位", elevator, detail=f"根据已发生的呼梯预测，空闲电梯前往 {target} 楼")
            coverage[elevator.park_target or elevator.floor] += 1

    def _dispatcher(self):
        while True:
            yield self.env.timeout(self.config.dispatch_interval)
            self.dispatch()

    def _snapshot(self):
        waiting = [0] * (self.config.floors + 1)
        riding = delivered = 0
        for passenger in self.passengers.values():
            if passenger.delivered_at is not None:
                delivered += 1
            elif passenger.boarded_at is not None:
                riding += 1
            else:
                waiting[passenger.request.origin] += 1
        self.snapshots.append({"time": float(self.env.now), "waiting": sum(waiting), "riding": riding,
                               "delivered": delivered, "by_floor": waiting,
                               "elevators": [e.snapshot() for e in self.elevators]})

    def _monitor(self):
        # 回放最多约 2000 个采样间隔；只记录快照，不参与调度决策。
        while True:
            self._snapshot()
            yield self.env.timeout(max(self.config.sample_interval, self.config.duration / 2000))

    def run(self) -> SimulationResult:
        """启动各进程并运行到截止时刻，保留未上车与乘梯中的真实状态。"""
        for elevator in self.elevators:
            self.env.process(elevator.run())
        self.env.process(self._arrivals())
        self.env.process(self._dispatcher())
        if self.record:
            self.env.process(self._monitor())
        # 到达区间为 [0,T)；恰在 T 的内部事件未处理，不额外清空队列。
        self.env.run(until=self.config.duration)
        if self.record:
            self._snapshot()
        rows = [p.row(self.config.duration) for p in self.passengers.values()]
        # 平均等待/P95 只含已上车者，必须同时展示未服务人数，防止统计误读。
        waits = [row["wait_seconds"] for row in rows if row["wait_seconds"] is not None]
        rides = [row["ride_seconds"] for row in rows if row["ride_seconds"] is not None]
        journeys = [row["journey_seconds"] for row in rows if row["journey_seconds"] is not None]
        observed = [row["observed_wait_seconds"] for row in rows]
        distances = [e.distance_at_horizon() for e in self.elevators]
        delivered, boarded, total = len(journeys), len(waits), len(rows)
        metrics = {
            "total": total, "boarded": boarded, "delivered": delivered,
            "waiting": total - boarded, "riding": boarded - delivered,
            "boarding_rate": 100.0 * boarded / total if total else 0.0,
            "completion_rate": 100.0 * delivered / total if total else 0.0,
            "avg_wait": statistics.fmean(waits) if waits else None,
            "p95_wait": _percentile(waits, 0.95), "max_wait": max(waits) if waits else None,
            "avg_ride": statistics.fmean(rides) if rides else None,
            "avg_journey": statistics.fmean(journeys) if journeys else None,
            "observed_wait": statistics.fmean(observed) if observed else None,
            "avg_queue": sum(observed) / self.config.duration,  # 等待人数时间积分 / 时长。
            "total_distance": sum(pair[0] for pair in distances),
            "empty_distance": sum(pair[1] for pair in distances),
            "door_cycles": sum(e.door_cycles for e in self.elevators),
            "oldest_unserved_wait": max((self.config.duration - p.request.arrival
                                         for p in self.passengers.values() if p.boarded_at is None), default=0.0),
        }
        # 守恒检查：每个已到达乘客只能处于候梯、乘梯或已送达中的一种状态。
        assert total == metrics["waiting"] + metrics["riding"] + delivered
        return SimulationResult(self.policy, metrics, rows, self.snapshots, self.events, self.forecasts)


@dataclass
class Experiment:
    """一次公平对比：共用配置与请求，收集各策略结果并附带复现信息。"""
    config: SimulationConfig
    requests: tuple[Request, ...]
    results: dict[str, SimulationResult]
    input_source: str = "合成客流"

    @property
    def fingerprint(self) -> str:
        return request_fingerprint(self.requests)

    def summary(self) -> dict:
        return {"schema": "elevator-experiment/v2", "config": asdict(self.config),
                "input_source": self.input_source, "request_sha256": self.fingerprint,
                "request_count": len(self.requests),
                "results": {policy: result.metrics for policy, result in self.results.items()},
                "methodology": {
                    "information": "只在到达后观察楼层和方向，目的地在上车后加入路径；所有策略共享同一请求和物理参数",
                    "avg_wait": "仅已上车乘客，必须与上车率及未服务人数同时阅读",
                    "observed_wait": "包含未上车乘客截至仿真结束的等待，属于截断观察值，不能代替最终等待时间",
                    "avg_queue": "所有乘客已观察等待时间之和 / 仿真时长，单位为人",
                    "distance": "楼层跨度，包含仿真结束时未完成路段的已行驶部分；不等同实际耗电量",
                    "forecast": f"仅基于过去120秒呼梯的双窗口平滑与阻尼趋势，预测未来{self.config.forecast_horizon:g}秒；无训练模型和未来客流输入",
                    "slp": "分区人工先验与已送达乘客OD频次融合；最大调度加分受 slp_bonus_seconds 限制",
                    "boundary": "到达时间属于[0,T)，运行到T截止；未服务及乘梯中人数保留，不自动清空队列",
                }}


def run_experiment(config: SimulationConfig, policies: Iterable[str] = DEFAULT_POLICIES,
                   requests: Iterable[Request] | None = None, record: bool = True) -> Experiment:
    """统一实验入口；先固定客流，再为每个策略建立独立仿真环境。"""
    policies = tuple(dict.fromkeys(policies))
    if not policies or any(policy not in POLICIES for policy in policies):
        raise ValueError("至少选择一种有效调度策略")
    # 公平对比的关键：generate_requests 在策略循环外只调用一次。
    trace = validate_requests(generate_requests(config) if requests is None else requests, config)
    results = {policy: Simulator(config, trace, policy, record).run() for policy in policies}
    return Experiment(config, trace, results, "合成客流" if requests is None else "导入请求")


def csv_bytes(rows: list[dict], fieldnames: list[str] | None = None) -> bytes:
    """导出带 UTF-8 BOM 的 CSV，便于 Excel 正确识别中文。"""
    output = io.StringIO(newline="")
    names = fieldnames or (list(rows[0]) if rows else [])
    writer = csv.DictWriter(output, fieldnames=names)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def export_experiment(experiment: Experiment) -> bytes:
    """把参数、输入指纹、请求、指标、逐客日志和回放打包，方便复核。"""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("experiment.json", json.dumps(experiment.summary(), ensure_ascii=False, indent=2, allow_nan=False))
        archive.writestr("requests.csv", csv_bytes([asdict(req) for req in experiment.requests], ["id", "arrival", "origin", "destination"]))
        archive.writestr("metrics.csv", csv_bytes([{"policy": policy, "name": POLICIES[policy], **result.metrics}
                                                  for policy, result in experiment.results.items()]))
        for policy, result in experiment.results.items():
            archive.writestr(f"passengers_{policy}.csv", csv_bytes(result.passengers,
                list(Passenger(Request(0, 0.0, 1, 2)).row(experiment.config.duration))))
            archive.writestr(f"events_{policy}.csv", csv_bytes(result.events, ["time", "event", "elevator", "passenger", "detail"]))
            archive.writestr(f"replay_{policy}.json", json.dumps(result.snapshots, ensure_ascii=False, allow_nan=False))
    return output.getvalue()


def benchmark(config: SimulationConfig, seeds: Iterable[int], policies: Iterable[str] = DEFAULT_POLICIES,
              on_progress: Callable[[int, int, int], None] | None = None) -> dict:
    """重复多个随机种子并保留原始结果，用模块组合比较观察收益与代价。"""
    policies, seeds = tuple(dict.fromkeys(policies)), tuple(seeds)
    if not seeds or len(seeds) > 50:
        raise ValueError("随机种子数量必须为 1~50")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("随机种子必须是非负整数")
    if len(seeds) != len(set(seeds)):
        raise ValueError("批量实验的随机种子不能重复，重复回放不构成独立样本")
    runs = []
    for seed in seeds:
        experiment = run_experiment(replace(config, seed=seed), policies, record=False)
        runs.append({"seed": seed, "request_count": len(experiment.requests),
                     "request_sha256": experiment.fingerprint,
                     "results": {policy: result.metrics for policy, result in experiment.results.items()}})
        if on_progress is not None:
            on_progress(len(runs), len(seeds), seed)
    summary = {}
    for policy in policies:
        summary[policy] = {}
        for metric in ("avg_wait", "p95_wait", "completion_rate", "waiting", "empty_distance", "total_distance"):
            values = [run["results"][policy][metric] for run in runs if run["results"][policy][metric] is not None]
            summary[policy][metric] = {"n": len(values), "mean": statistics.fmean(values) if values else None,
                                       "std": statistics.stdev(values) if len(values) > 1 else None}
    paired = None
    if "nearest" in policies and "adaptive" in policies:
        differences = [run["results"]["nearest"]["avg_wait"] - run["results"]["adaptive"]["avg_wait"]
                       for run in runs if all(run["results"][policy]["avg_wait"] is not None
                                              for policy in ("nearest", "adaptive"))]
        if differences:
            mean = statistics.fmean(differences)
            # 正态近似区间只作探索；小样本和已上车人群差异限制了结论强度。
            margin = 1.96 * statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else None
            paired = {"n": len(differences), "mean_wait_reduction_seconds": mean,
                      "normal_approx_95_ci": None if margin is None else [mean - margin, mean + margin],
                      "note": "近似区间，小样本仅作探索；等待统计为各策略已上车人群，须同时比较上车率和完成率"}
    return {"schema": "elevator-benchmark/v2", "config": asdict(config), "seeds": seeds,
            "summary": summary, "paired_wait_comparison": paired, "runs": runs}

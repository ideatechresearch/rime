import time, math, random
import logging
from functools import partial
from typing import Callable, Optional
from datetime import datetime, timedelta


def merge_partial(task, *args, **kwargs):
    if isinstance(task, partial):
        return partial(task.func, *(task.args + args), **{**task.keywords, **kwargs})
    return partial(task, *args, **kwargs)


def exec_tasks(tasks: list) -> list:
    """同步执行到期任务列表，返回结果列表。异常不中断，打印后继续。"""
    results = []
    for t in tasks:
        try:
            results.append(t())
        except Exception as e:
            print(f"Task {t.func.__name__} execution failed with error: {e}")
            results.append(e)
    return results


class TimeContext:
    def __init__(self):
        self.current: float = 0.0  # 虚拟累计秒
        self.tick_time: datetime = datetime.now()

    def tick(self, dt: float) -> float:
        self.current += dt
        self.tick_time += timedelta(seconds=dt)
        return self.current

    def after_tick(self, wheel: "TimeWheel"):
        self.tick(wheel.tick_duration)
        return self.tick_time

    def clear(self):
        self.current = 0.0
        self.tick_time = datetime.now()

    @property
    def now(self) -> float:
        '''避免时间倒退,加速模式当前虚拟时间 tick_time'''
        return max(self.tick_time.timestamp(), time.time())

    @property
    def time_str(self) -> str:
        return self.tick_time.strftime('%Y-%m-%d %H:%M:%S')

    def delay_sec(self, execute_time: datetime) -> float:
        return (execute_time - self.tick_time).total_seconds()


class TimeWheel:
    """到期任务的执行策略通过 on_exec 回调开放出去——调用方可以选择直接执行、收集后批量处理、丢到线程池等，时间轮本身不持有任何调度策略"""

    def __init__(self, slots: int, tick_duration: int | float, name: str = "tw",
                 context: TimeContext = None, next_wheel: "TimeWheel" = None):
        assert slots > 0 and tick_duration > 0
        self.slots = slots
        self.tick_duration = tick_duration
        self.name = name
        self.current_pos: int = 0
        self.wheel = [[] for _ in range(slots)]
        self.next_wheel: Optional["TimeWheel"] = next_wheel
        self.context: TimeContext = context or TimeContext()

    def set_next_wheel(self, next_wheel: "TimeWheel"):
        self.next_wheel = next_wheel

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"

    @property
    def elapsed_time(self) -> float:
        """当前时间轮已推进的秒数（相对本层）"""
        return self.current_pos * self.tick_duration

    @property
    def wheel_span(self) -> float:
        """当前时间轮总秒数（相对本层）一圈总秒数 circle_seconds"""
        return self.slots * self.tick_duration

    @property
    def task_count(self) -> int:
        return sum(len(slot) for slot in self.wheel)

    def slots_delay(self, rounds: int, pos: int) -> int:
        offset_slots = (pos - self.current_pos) % self.slots
        return rounds * self.slots * self.tick_duration + offset_slots * self.tick_duration

    def clear(self) -> list[tuple]:
        tasks = []
        for slot in self.wheel:
            tasks.extend(slot)
            slot.clear()
        self.current_pos = 0
        return tasks

    def add_task(self, delay: float | int, task: Callable, *args, **kwargs) -> tuple[int, float]:
        """多层图,时间轮是层级图（树 + 环）,任务是节点"""
        if delay < 0:
            delay = 0.0

        if delay > self.wheel_span and self.next_wheel:  # 超过本层 span，上放到上层轮
            return self.next_wheel.add_task(delay, task, *args, **kwargs)
        target = self.context.current + delay  # execute_ts
        ticks = int(delay // self.tick_duration)  # offset
        rounds = ticks // self.slots  # 当前层内轮数
        wrapped_task = merge_partial(task, *args, **kwargs)  # bound_func
        pos = int((self.current_pos + ticks) % self.slots)
        self.wheel[pos].append((rounds, target, wrapped_task))
        delay_sec = self.slots_delay(rounds, pos)
        return delay_sec, target

    def tick_once(self) -> tuple[list, list]:
        """
        执行一次 tick。返回 (finished_levels, exec_tasks)。
        不推进 context — 调用方负责在执行 exec_tasks 之后调用 context.tick()。
        """
        remaining_tasks = []
        exec_list = []
        tasks = self.wheel[self.current_pos]
        now = self.context.current
        for rounds, target, task in tasks:
            remaining_delay = target - now
            if rounds > 0:
                remaining_tasks.append((rounds - 1, target, task))
            elif remaining_delay > self.tick_duration:
                self.add_task(remaining_delay, task)
            else:
                exec_list.append(task)

        self.wheel[self.current_pos] = remaining_tasks
        self.current_pos = (self.current_pos + 1) % self.slots

        finished_levels = []
        if self.current_pos == 0:
            finished_levels.append(self.name)
            if self.next_wheel:  # 级联上层任务
                upper_finished, upper_exec = self.next_wheel.tick_once()
                finished_levels.extend(upper_finished)
                exec_list.extend(upper_exec)
                self.cascade_tasks(offset=0)
        return finished_levels, exec_list

    def cascade_tasks(self, offset: int = None) -> int:
        """把上层时间轮当前槽的任务下放到本层"""
        if not self.next_wheel:
            return 0

        remaining_tasks = []
        if offset:
            pos = int((self.next_wheel.current_pos + offset) % self.next_wheel.slots)
        else:
            pos = self.next_wheel.current_pos
        now = self.context.current
        tasks = self.next_wheel.wheel[pos]
        for rounds, target, task in tasks:
            remaining_delay = target - now
            remaining_tasks.append((remaining_delay, task))
        self.next_wheel.wheel[pos] = []

        for remaining_delay, task in remaining_tasks:
            self.add_task(remaining_delay, task)

        return len(remaining_tasks)

    def move_to_upper(self):
        if not self.next_wheel:
            return
        if not self.task_count:
            return
        now = self.context.current
        for slot_index, slot in enumerate(self.wheel):
            for rounds, target, task in slot:
                remaining_delay = target - now
                self.next_wheel.add_task(remaining_delay, task)
            slot.clear()

    def accelerate(self, tick_steps: int = 1):
        """推进指定步数，yield (step_index, advanced_seconds, finished_levels, exec_tasks)"""
        if tick_steps <= 0:
            raise ValueError("steps must be > 0")

        advanced = self.context.current
        for i in range(int(tick_steps)):
            finished, exec_list = self.tick_once()
            self.context.tick(self.tick_duration)
            yield i, self.context.current - advanced, finished, exec_list

    def run_ticks(self, tick_steps: int, on_exec: Callable[[list], None] = None):
        """同步推进指定步数，到期任务通过 on_exec 回调处理（默认直接执行）。"""
        for i, advanced, finished, exec_list in self.accelerate(tick_steps):
            if exec_list:
                if on_exec:
                    on_exec(exec_list)
                else:
                    exec_tasks(exec_list)


class HierarchicalTimeWheel:
    def __init__(self):
        self.context = TimeContext()
        self.second_wheel = TimeWheel(60, 1, "second", context=self.context)
        self.minute_wheel = TimeWheel(60, self.second_wheel.wheel_span, "minute", self.context)
        self.hour_wheel = TimeWheel(24, self.minute_wheel.wheel_span, "hour", self.context)
        self.day_wheel = TimeWheel(365, self.hour_wheel.wheel_span, "day", self.context)

        self.second_wheel.set_next_wheel(self.minute_wheel)
        self.minute_wheel.set_next_wheel(self.hour_wheel)
        self.hour_wheel.set_next_wheel(self.day_wheel)

        self.levels = {
            self.second_wheel.name: self.second_wheel,
            self.minute_wheel.name: self.minute_wheel,
            self.hour_wheel.name: self.hour_wheel,
            self.day_wheel.name: self.day_wheel
        }
        self._running = False

    @property
    def clock(self) -> dict:
        """
        多层时间轮的全局相位（类似钟表指针联动,带下级偏移）
        每层值范围在 [0,1)，已包含下层轮的偏移
        """
        levels = sorted(self.levels.items(), key=lambda item: item[1].wheel_span)
        phase = {name: wheel.current_pos / wheel.slots for name, wheel in levels}
        for i in range(len(levels) - 1):
            child_name, _ = levels[i]
            parent_name, parent_wheel = levels[i + 1]
            phase[parent_name] += phase[child_name] / parent_wheel.slots
        for k in phase:
            phase[k] = phase[k] % 1.0
        return phase

    @property
    def elapsed_time(self) -> float:
        total_seconds = (
                self.second_wheel.elapsed_time +
                self.minute_wheel.elapsed_time +
                self.hour_wheel.elapsed_time +
                self.day_wheel.elapsed_time
        )
        return total_seconds

    @property
    def task_count(self) -> int:
        """统计所有层的任务总数"""
        total = (
                self.second_wheel.task_count +
                self.minute_wheel.task_count +
                self.hour_wheel.task_count +
                self.day_wheel.task_count
        )
        return total

    def add_task(self, delay: float | int, task: Callable, *args, jitter_percent: float | None = None,
                 **kwargs) -> tuple[int, float]:
        """
        添加定时任务
        :param delay: 延迟秒数
        :param task: 要执行的任务(函数)
        :param jitter_percent:百分比 Jitter,-5% ～ +5% 的对称抖动,防 overfitting
        """
        if jitter_percent:
            jitter_percent = max(0.0, min(1.0, jitter_percent))
            jitter = delay * (random.random() * (2 * jitter_percent) - jitter_percent)
            delay += jitter
        for name, w in self.levels.items():
            if delay <= w.wheel_span:
                return w.add_task(delay, task, *args, **kwargs)
        return self.day_wheel.add_task(delay, task, *args, **kwargs)

    def add_tasks(self, delay: float | int, tasks: list[tuple[Callable, tuple, dict]], window: float | int = None,
                  jitter_percent: float | None = None) -> list:
        """
        批量添加任务，并在 [base_delay - window, base_delay + window] 区间内平摊任务。
        """
        batch_size = len(tasks)
        if batch_size == 0:
            return []
        if window is None:
            window = delay * (1.0 - math.exp(-math.log(batch_size)))

        step = (window * 2.0) / batch_size
        targets = []
        for i, (task, args, kwargs) in enumerate(tasks):
            final_delay = delay - window + i * step
            delay_sec, target = self.add_task(final_delay, task, *args, jitter_percent=jitter_percent, **kwargs)
            targets.append(target)

        return targets

    def add_task_absolute(self, execute_time: datetime | float, task, *args, **kwargs) -> tuple[int, float]:
        """添加定时任务，可传入 datetime 或时间戳"""
        if isinstance(execute_time, datetime):
            execute_time = execute_time.timestamp()
        delay = max(0.0, execute_time - self.context.now)
        return self.add_task(delay, task, *args, **kwargs)

    def add_job(self, func, trigger=None, args: list | tuple = None, kwargs: dict = None, **trigger_args):
        '''trigger:the alias name of the trigger (e.g. ``date``, ``interval`` or ``cron``)'''
        args = args or ()
        kwargs = kwargs or {}
        final_kwargs = {**trigger_args, **kwargs}
        dispatcher = {
            'interval': self.add_periodic_task,
            'cron': self.add_daily_task,
            'date': self.add_task_absolute,
        }
        method = dispatcher.get(trigger, self.add_task)
        return method(task=func, *args, **final_kwargs)

    def add_daily_task(self, hour: int, minute: int, task, *args, **kwargs):
        """每天固定时刻执行任务"""
        handle = {"cancel": False}

        def wrapper():
            if handle["cancel"]:
                return
            task(*args, **kwargs)
            next_run = wrapper.next_run + timedelta(days=1)
            wrapper.next_run = next_run
            self.add_task_absolute(next_run, wrapper)
            print(f'daily task:{task.__name__} run_time={self.context.time_str} next_run={next_run}')

        now = max(self.context.tick_time, datetime.now())
        run_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if run_time < now:
            run_time += timedelta(days=1)
        wrapper.next_run = run_time
        self.add_task_absolute(run_time, wrapper)
        print(f'daily task:{task.__name__} run_time={run_time}')
        return handle

    def add_periodic_task(self, interval: int, task, *args, **kwargs):
        """周期任务，在虚拟时间上周期触发，不与 tick 步长绑定"""
        handle = {"cancel": False}

        def wrapper():
            if handle["cancel"]:
                return
            task(*args, **kwargs)
            now = self.context.current
            if wrapper.next_target < now:
                wrapper.next_target = now
            wrapper.next_target += interval
            delay = max(wrapper.next_target - now - self.second_wheel.tick_duration, 0)
            self.add_task(delay, wrapper)

        wrapper.next_target = self.context.current + interval
        self.add_task(interval, wrapper)
        return handle

    def tick(self, level: str = "second", on_exec: Callable[[list], None] = None) -> list:
        """
        同步推进时间轮。返回所有 finished_levels。
        到期任务通过 on_exec 回调处理（默认直接同步执行）。
        """
        wheel = self.levels.get(level)
        if not wheel:
            print(f"[TimeWheel] Invalid level: {level}")
            return []

        delta = time.time() - self.context.tick_time.timestamp()
        if delta < 0:
            logging.warning(f"[TimeWheel] super tick,last:{self.context.time_str}")
            return []

        missed = int(delta // wheel.tick_duration)
        if missed > 1:
            logging.warning(f"[TimeWheel] (missed={missed}),last:{self.context.time_str}")

        finished_all = []
        for _ in range(missed):
            for name, w in self.levels.items():
                if w is wheel:
                    break
                w.move_to_upper()

            finished, exec_list = wheel.tick_once()
            wheel.context.tick(wheel.tick_duration)
            finished_all.extend(finished)

            if exec_list:
                if on_exec:
                    on_exec(exec_list)
                else:
                    exec_tasks(exec_list)

        return finished_all

    def super_tick(self, virtual_seconds: float, level: str = "second",
                   yield_every: int = 100,
                   on_exec: Callable[[list], None] = None):
        """
        手动加速时间推进，同步版本。
        :param virtual_seconds: 要推进的虚拟未来时间（秒）
        :param level: 推进的层级
        :param on_exec: 到期任务回调，默认直接执行
        :return: 推进统计 dict
        """
        if virtual_seconds <= 0:
            raise ValueError("virtual_seconds must be > 0")
        wheel = self.levels.get(level)
        if wheel is None:
            raise ValueError(f"invalid level: {level}")

        total_ticks = int(virtual_seconds / wheel.tick_duration)
        advanced_virtual: float = 0.0
        start_real = time.monotonic()

        for i, advanced, finished, exec_list in wheel.accelerate(total_ticks):
            advanced_virtual = advanced
            if exec_list:
                if on_exec:
                    on_exec(exec_list)
                else:
                    exec_tasks(exec_list)

        real_used = time.monotonic() - start_real
        return {
            "virtual_advanced": advanced_virtual,
            "real_used": real_used,
            "speed_factor": virtual_seconds / real_used if real_used > 0 else float('inf'),
            "virtual_tick": wheel.context.current,
            "virtual_elapsed": self.elapsed_time,
            "virtual_time": self.context.tick_time
        }

    def run_loop(self, level: str = "second", on_exec: Callable[[list], None] = None):
        """同步阻塞运行时间轮，每 tick_duration 间隔推进一次。适合独立线程使用。"""
        wheel = self.levels.get(level)
        if wheel is None:
            raise ValueError(f"invalid level: {level}")

        self._running = True
        print(f"[{wheel.name}] started (tick={wheel.tick_duration}s)")

        next_tick = time.monotonic()
        try:
            while self._running:
                now = time.monotonic()
                missed = max(1, int((now - next_tick) / wheel.tick_duration) + 1)
                for _ in range(missed):
                    finished, exec_list = wheel.tick_once()
                    self.context.tick(wheel.tick_duration)
                    if exec_list:
                        if on_exec:
                            on_exec(exec_list)
                        else:
                            exec_tasks(exec_list)

                next_tick += missed * wheel.tick_duration
                sleep_time = max(0.0001, next_tick - time.monotonic())
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            print(f"[{wheel.name}] stopped")

    def stop(self):
        self._running = False
        print(f'[TimeWheel] stopped at {self.context.time_str}(elapsed={self.elapsed_time}s)')


if __name__ == "__main__":

    def print_time(msg):
        time.sleep(0.1)  # 模拟耗时操作
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


    # ---- 1. 单层 TimeWheel：手动 tick 推进 ----
    print("=== 单层 TimeWheel ===")
    tw = TimeWheel(60, 1, "second")
    tw.add_task(3, lambda: print_time("3秒后执行"))
    tw.add_task(5, lambda: print_time("5秒后执行"))
    for i in range(7):
        finished, exec_list = tw.tick_once()
        tw.context.tick(tw.tick_duration)
        if exec_list:
            exec_tasks(exec_list)
        print(f"  tick {i + 1}: pos={tw.current_pos}, exec={len(exec_list)}")
    print(f"  elapsed={tw.elapsed_time}s\n")

    # ---- 2. HierarchicalTimeWheel：super_tick 加速推进 ----
    print("=== 层级时间轮 super_tick ===")
    htw = HierarchicalTimeWheel()


    def vprint(msg):
        print(f"  [{htw.context.time_str}] {msg}")


    now = datetime.now()
    htw.add_task_absolute(now + timedelta(seconds=10), vprint, "10秒后执行")
    htw.add_task_absolute(now + timedelta(minutes=2), vprint, "2分钟后执行")
    htw.add_task_absolute(time.time() + 5, vprint, "5秒后")
    htw.add_task(65, vprint, "65秒后")
    htw.add_task(100, vprint, "100秒后")
    htw.add_task(3600, vprint, "一小时后")
    htw.add_task(3600 * 3, vprint, "3小时后")
    htw.add_daily_task(12, 0, vprint, "每天12:00")
    htw.add_daily_task(18, 30, vprint, "每天18:30")
    print(f"  task_count={htw.task_count}")

    # 加速推进 300000 虚拟秒（约3.5天），到期任务直接同步执行
    result = htw.super_tick(300000)
    print(f"  super_tick result: speed={result['speed_factor']:.0f}x, "
          f"virtual_time={result['virtual_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  elapsed={htw.elapsed_time}s, task_count={htw.task_count}\n")

    # ---- 3. on_exec 回调：自定义到期任务处理 ----
    print("=== on_exec 回调收集模式 ===")
    htw2 = HierarchicalTimeWheel()
    collected = []


    def on_exec_collect(tasks):
        """不直接执行，收集到期任务供后续处理"""
        collected.extend(tasks)


    htw2.add_task(5, lambda: print_time("5s task"))
    htw2.add_task(10, lambda: print_time("10s task"))
    result = htw2.super_tick(15, on_exec=on_exec_collect)
    print(f"  collected {len(collected)} tasks, speed={result['speed_factor']:.0f}x")
    # 手动执行收集的任务
    exec_tasks(collected)
    print()

    # ---- 4. 周期任务 add_periodic_task ----
    print("=== 周期任务 ===")
    htw3 = HierarchicalTimeWheel()
    counter = [0]


    def periodic_tick():
        counter[0] += 1
        print(f"  periodic #{counter[0]} at {htw3.context.time_str}")


    htw3.add_periodic_task(60, periodic_tick)  # 每60秒触发
    result = htw3.super_tick(300)  # 推进5分钟
    print(f"  triggered {counter[0]} times in 5min, elapsed={htw3.elapsed_time}s\n")

    # ---- 5. add_tasks 批量 + jitter ----
    print("=== 批量任务 + jitter ===")
    htw4 = HierarchicalTimeWheel()
    batch = [(lambda i=i: print_time(f"batch-{i}")) for i in range(5)]
    tasks_arg = [(f, (), {}) for f in batch]
    htw4.add_tasks(30, tasks_arg, window=10, jitter_percent=0.05)
    print(f"  task_count={htw4.task_count}")
    result = htw4.super_tick(60)
    print(f"  after super_tick: task_count={htw4.task_count}\n")

    # ---- 6. clock 多尺度相位 ----
    print("=== clock 相位 ===")
    htw5 = HierarchicalTimeWheel()
    htw5.super_tick(3661)  # 推进1小时1分1秒
    print(f"  phase: {htw5.clock}")
    print(f"  elapsed={htw5.elapsed_time}s")

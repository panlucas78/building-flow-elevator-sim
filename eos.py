"""楼宇 Flow 界面与入口。

main 负责参数、结果和回放；batch_main 负责批量实验；launch 负责本地启动。
仿真业务集中在 elevator_sim.py，本文件从模型读取结果后展示。
"""
from __future__ import annotations

import argparse
import bisect
import json
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from elevator_sim import (
    DEFAULT_POLICIES, POLICIES, SCENARIOS, Experiment, SimulationConfig,
    benchmark, csv_bytes, export_experiment, requests_from_csv, run_experiment,
)

ROOT = Path(__file__).resolve().parent  # 按脚本位置定位文件，避免从不同目录启动出错。
COLORS = {"nearest": "#8292A4", "eta": "#5278D0", "slp": "#DB8A3B",
          "forecast": "#8E6BBD", "adaptive": "#167D8D"}
SHORT_NAMES = {"nearest": "就近 + LOOK", "eta": "ETA + LOOK", "slp": "ETA + SLP",
               "forecast": "ETA + 预测", "adaptive": "SLP + 预测"}
STATES = {"IDLE": "空闲", "MOVING": "运行", "OPENING": "开门", "CLOSING": "关门",
          "BOARDING": "上客", "ALIGHTING": "下客"}


def building_svg(snapshot: dict, config: SimulationConfig) -> str:
    """把一帧仿真快照画成楼宇 SVG；只插入内部数值，不插入上传文本或脚本。"""
    row_height = min(38.0, max(20.0, 650 / config.floors))
    height, width = config.floors * row_height + 84, 160 + config.elevators * 90
    def y(floor):
        return height - 38 - (floor - 1) * row_height
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="电梯楼宇回放" '
             f'viewBox="0 0 {width} {height}" style="width:100%;max-height:730px;font-family:Segoe UI,Microsoft YaHei,sans-serif">',
             f'<rect width="{width}" height="{height}" rx="16" fill="#ffffff"/>',
             '<text x="18" y="25" font-size="12" fill="#738398">楼层 / 候梯人数</text>']
    for floor in range(1, config.floors + 1):
        line_y = y(floor)
        if floor % 2 == 0:
            parts.append(f'<rect x="10" y="{line_y-row_height/2}" width="{width-20}" height="{row_height}" fill="#F5F8FB"/>')
        parts.append(f'<line x1="15" y1="{line_y+row_height/2}" x2="{width-15}" y2="{line_y+row_height/2}" stroke="#E7EDF3"/>')
        parts.append(f'<text x="20" y="{line_y+4}" font-size="12" fill="#51677D">{floor:02d} F</text>')
        waiting = snapshot["by_floor"][floor]
        if waiting:
            parts.append(f'<rect x="64" y="{line_y-8}" width="{min(48, 8+waiting*3)}" height="16" rx="8" fill="#FDE9D3"/>')
            parts.append(f'<text x="70" y="{line_y+4}" font-size="11" fill="#A96219">{waiting}</text>')
    for car in snapshot["elevators"]:
        x = 145 + car["id"] * 90
        parts.append(f'<rect x="{x}" y="40" width="62" height="{height-60}" rx="10" fill="#EAF0F5" fill-opacity="0.6"/>')
        parts.append(f'<text x="{x+31}" y="25" text-anchor="middle" font-size="12" font-weight="600" fill="#344E64">E{car["id"]+1}</text>')
        color = "#D7923E" if car["load"] >= config.capacity else ("#167D8D" if car["state"] != "IDLE" else "#8292A4")
        car_y = y(car["position"])
        parts.append(f'<rect x="{x+3}" y="{car_y-13}" width="56" height="26" rx="7" fill="{color}"/>')
        arrow = "↑" if car["direction"] > 0 else ("↓" if car["direction"] < 0 else "·")
        parts.append(f'<text x="{x+31}" y="{car_y+4}" text-anchor="middle" font-size="11" fill="white">{arrow} {car["load"]}/{config.capacity}</text>')
    parts.append('</svg>')
    return "".join(parts)


def main():
    """Streamlit 主页面：收集配置、调用模型，再展示图表与交互回放。"""
    # 界面依赖延迟导入，命令行批量实验不需要启动网页或加载图表组件。
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="楼宇 Flow · 电梯调度实验室", page_icon="🏢", layout="wide")
    st.markdown("""<style>
    .block-container{padding-top:1.8rem;padding-bottom:3rem;max-width:1540px}
    h1,h2,h3{letter-spacing:-.025em}
    [data-testid="stMetric"]{background:white;border:1px solid #E3EAF1;border-radius:15px;padding:18px 20px;min-height:126px}
    [data-testid="stMetricLabel"]{color:#61758A;font-size:13px}
    [data-testid="stMetricValue"]{font-size:30px;color:#183044}
    [data-testid="stSidebar"]{border-right:1px solid #E4EBF2}
    .hero{background:linear-gradient(115deg,#102D42,#174B59);border-radius:20px;padding:28px 32px;color:#F5FAFD;margin-bottom:18px}
    .hero .eyebrow{font-size:11px;letter-spacing:3px;color:#8CD0CF;margin-bottom:9px}
    .hero h1{color:#FFF;font-size:32px;font-weight:650;padding:0;margin:0 0 10px}
    .hero p{font-size:14px;color:#BDCFD8;margin:0;line-height:1.7}
    .badge{display:inline-block;font-size:11px;background:#FFFFFF16;color:#B5E3DC;border:1px solid #FFFFFF24;border-radius:30px;padding:4px 10px;margin-top:16px;margin-right:7px}
    [data-baseweb="tab-list"]{gap:24px;border-bottom:1px solid #E1E8EF;margin-top:14px}
    [data-baseweb="tab"]{font-size:14px;padding-left:0;padding-right:0}
    div.stButton>button{border-radius:10px}
    </style>""", unsafe_allow_html=True)

    @st.cache_data(show_spinner=False, max_entries=8)
    def compute(values: dict, policies: tuple[str, ...], imported: bytes | None):
        # 参数、策略和文件字节共同构成缓存键；相同输入可复用确定性结果。
        cfg = SimulationConfig(**values)
        requests = requests_from_csv(imported, cfg) if imported is not None else None
        return run_experiment(cfg, policies, requests)

    with st.sidebar:
        st.markdown("### 楼宇 Flow")
        st.caption("客流自适应 · 电梯集群调度")
        # 表单提交后才重算，避免拖动参数时反复运行完整实验。
        with st.form("parameters"):
            st.markdown("#### 01 / 建筑与客流")
            scenario = st.selectbox("客流场景", list(SCENARIOS), format_func=SCENARIOS.get)
            a, b = st.columns(2)
            floors = a.number_input("楼层数", 2, 30, 12)
            elevators = b.number_input("电梯数", 1, 6, 3)
            capacity = st.slider("单台载客容量（人）", 1, 24, 12)
            minutes = st.select_slider("仿真时长（分钟）", options=[10, 20, 30, 45, 60, 90, 120], value=60)
            peak_percent = st.slider("高峰位置（占总时长 %）", 10, 90, 33, step=1)
            base_rate = st.slider("基础到达率（人 / 秒）", 0.0, 1.0, 0.05, step=0.01)
            peak_rate = st.slider("高峰额外到达率（人 / 秒）", 0.0, 3.0, 0.75, step=0.05)
            peak_width = st.slider("高峰宽度参数（秒）", 60, 900, 300, step=30,
                                   help="高斯峰参数；距中心一个宽度时，额外到达率约为峰值的 36.8%。")
            st.markdown("#### 02 / 运行与策略")
            travel = st.slider("每跨一层行驶时间（秒）", 1.0, 8.0, 3.0, step=0.5)
            policies = st.multiselect("对比策略", list(POLICIES), default=list(DEFAULT_POLICIES), format_func=POLICIES.get)
            seed = st.number_input("随机种子", 0, 2147483647, 42, help="相同参数与种子会生成完全相同的请求。")
            with st.expander("门控与调度参数"):
                opening = st.number_input("开门时间（秒）", 0.0, 10.0, 2.0, step=0.5)
                closing = st.number_input("关门时间（秒）", 0.0, 10.0, 2.0, step=0.5)
                boarding = st.number_input("每人上车时间（秒）", 0.0, 3.0, 0.6, step=0.1)
                alighting = st.number_input("每人下车时间（秒）", 0.0, 3.0, 0.4, step=0.1)
                slp_bonus = st.number_input("楼层关联最大加分（秒）", 0.0, 15.0, 4.0, step=0.5)
            with st.expander("导入自己的客流"):
                imported_file = st.file_uploader("UTF-8 CSV 请求文件", type="csv")
                st.caption("列名：id, arrival, origin, destination。最多25000人、5 MiB。arrival 单位为秒，须小于仿真时长。导入后以文件为准。")
            submitted = st.form_submit_button("运行公平对比", type="primary", width="stretch")
        st.caption("SimPy 离散事件引擎 · 本地软件仿真")

    config = SimulationConfig(
        floors=int(floors), elevators=int(elevators), capacity=capacity, duration=minutes * 60,
        scenario=scenario, seed=int(seed), peak_center=minutes * 60 * peak_percent / 100,
        peak_width=peak_width, base_rate=base_rate, peak_rate=peak_rate, travel_seconds=travel,
        door_open_seconds=opening, door_close_seconds=closing, boarding_seconds=boarding,
        alighting_seconds=alighting, slp_bonus_seconds=slp_bonus,
    )
    uploaded_bytes = imported_file.getvalue() if imported_file is not None else None
    # 页面交互会触发脚本重跑；session_state 保存已完成实验，回放时无需重算。
    if submitted or "experiment" not in st.session_state:
        if not policies:
            st.error("请至少选择一种调度策略。")
            st.stop()
        try:
            with st.spinner("生成共享请求，运行各策略并记录乘客轨迹…"):
                st.session_state.experiment = compute(asdict(config), tuple(policies), uploaded_bytes)
            # 输入改变后重置旧回放和批量结果，防止新旧实验混用。
            for key in ("replay_time", "replay_policy", "replay_playing", "batch_result", "experiment_zip"):
                st.session_state.pop(key, None)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    experiment: Experiment = st.session_state.experiment
    active = experiment.config
    st.markdown("""<div class="hero"><div class="eyebrow">BUILDING FLOW / DISPATCH LAB</div>
    <h1>让每一次等待，都有据可查。</h1>
    <p>面向复合型楼宇的客流自适应电梯调度实验室 · 比较策略、回放运行、追踪每位乘客</p>
    <span class="badge">同一批客流公平对比</span><span class="badge">乘客全程可追踪</span>
    <span class="badge">固定种子可复现</span></div>""", unsafe_allow_html=True)
    st.caption(f"{SCENARIOS[active.scenario]}  /  {active.floors} 层 · {active.elevators} 台 · 每台 {active.capacity} 人"
               f"  /  {active.duration // 60} 分钟  /  种子 {active.seed}  /  {experiment.input_source} {len(experiment.requests)} 人"
               f"  /  请求指纹 {experiment.fingerprint[:12]}")
    if config != active:
        st.info("当前显示的是上一次已完成的实验。点击侧栏「运行公平对比」应用新参数。")
    focus = "adaptive" if "adaptive" in experiment.results else next(iter(experiment.results))
    metrics = experiment.results[focus].metrics
    reference = experiment.results.get("nearest")
    def value(number, suffix="", decimals=1):
        return "—" if number is None else f"{number:,.{decimals}f}{suffix}"
    def wait_delta(metric):
        baseline = reference.metrics[metric] if reference else None
        return None if focus == "nearest" or baseline is None or metrics[metric] is None else f"{metrics[metric] - baseline:+.1f} 秒"
    st.markdown(f"##### 当前观察：{POLICIES[focus]}")
    cards = st.columns(4)
    cards[0].metric("平均等待 · 已上车", value(metrics["avg_wait"], " 秒"), wait_delta("avg_wait"), delta_color="inverse")
    cards[1].metric("P95 等待 · 已上车", value(metrics["p95_wait"], " 秒"), wait_delta("p95_wait"), delta_color="inverse")
    cards[2].metric("已完成运送", f'{metrics["delivered"]} 人', f'{metrics["completion_rate"]:.1f}% 完成率', delta_color="off")
    cards[3].metric("结束时等待", f'{metrics["waiting"]} 人', f'{metrics["riding"]} 人仍在乘梯', delta_color="off")
    st.caption("等待时间差值相对就近策略；负数表示当前策略等待更短。")
    if any(result.metrics["waiting"] for result in experiment.results.values()):
        st.warning("部分乘客在仿真截止时尚未上车。平均等待和 P95 仅统计已上车人群，请结合未服务人数、上车率和完成率判断。")
    if not experiment.requests:
        st.info("本次没有生成乘客，可提高到达率或更换随机种子；空客流不会计算等待改善比例。")

    tabs = st.tabs(["对比总览", "楼宇回放", "乘客与调度", "批量验证", "模型说明"])

    def chart_layout(fig, title, y_title):
        fig.update_layout(title=dict(text=title, font=dict(size=16)), height=360,
                          margin=dict(l=15, r=15, t=95, b=35), template="plotly_white",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Microsoft YaHei, Segoe UI, sans-serif", color="#51677D", size=12),
                          legend=dict(orientation="h", y=1.15, x=0, font=dict(size=10)), hovermode="x unified",
                          xaxis_title="仿真时间（秒）", yaxis_title=y_title)
        return fig

    def metrics_rows():
        return [{"策略": POLICIES[policy], "平均等待（秒）": result.metrics["avg_wait"],
                 "P95等待（秒）": result.metrics["p95_wait"], "上车率（%）": result.metrics["boarding_rate"],
                 "完成率（%）": result.metrics["completion_rate"], "结束时等待（人）": result.metrics["waiting"],
                 "空载距离（层）": result.metrics["empty_distance"], "总距离（层）": result.metrics["total_distance"],
                 "全体已观察等待（秒）": result.metrics["observed_wait"]}
                for policy, result in experiment.results.items()]

    # 【结果总览】所有表格和差值直接取自本次仿真，不预设策略一定改善。
    with tabs[0]:
        left, right = st.columns(2)
        queue, delivered = go.Figure(), go.Figure()
        for policy, result in experiment.results.items():
            times = [snap["time"] for snap in result.snapshots]
            line = dict(color=COLORS[policy], width=2.5)
            queue.add_trace(go.Scatter(x=times, y=[s["waiting"] for s in result.snapshots], mode="lines", name=SHORT_NAMES[policy], line=line))
            delivered.add_trace(go.Scatter(x=times, y=[s["delivered"] for s in result.snapshots], mode="lines", name=SHORT_NAMES[policy], line=line))
        left.plotly_chart(chart_layout(queue, "楼层候梯队列", "等待人数（人）"), width="stretch")
        right.plotly_chart(chart_layout(delivered, "累计完成运送", "送达人数（人）"), width="stretch")
        st.markdown("#### 同一批乘客，完整比较")
        st.dataframe(pd.DataFrame(metrics_rows()).round(2), hide_index=True, width="stretch")
        st.caption("平均等待 / P95：已上车乘客。全体已观察等待：未上车者计至截止时刻，是截断值。距离单位为楼层跨度，不表示实际耗电量。")
        left, right = st.columns([1.25, 1])
        bins = list(range((active.duration + 59) // 60))
        counts = [0] * len(bins)
        od = [[0] * active.floors for _ in range(active.floors)]
        # 起终点热力图是实验结束后的输入分析，不代表调度器提前知道目的地。
        for req in experiment.requests:
            counts[min(len(counts)-1, int(req.arrival // 60))] += 1
            od[req.origin-1][req.destination-1] += 1
        arrival_chart = go.Figure(go.Bar(x=[minute * 60 for minute in bins], y=counts, marker_color="#BBDCD9", name="实际到达"))
        prediction = experiment.results[focus].forecasts
        horizon = active.forecast_horizon
        arrival_chart.add_trace(go.Scatter(x=[row["time"] for row in prediction], y=[row["predicted_next_horizon"] if "predicted_next_horizon" in row else row["predicted_next_60"] for row in prediction],
                                          name=f"预测未来{horizon:g}秒人数", line=dict(color="#D58C43", width=2)))
        left.plotly_chart(chart_layout(arrival_chart, "到达客流与滚动预测", f"人数（实际60秒 / 预测{horizon:g}秒）"), width="stretch")
        left.caption(f"柱形为当前分钟（60秒）实际到达；折线为该时刻预测的未来{horizon:g}秒人数。预测仅使用当时已发生的呼梯，两个窗口并非一一对齐。")
        heatmap = go.Figure(go.Heatmap(z=od, x=list(range(1, active.floors+1)), y=list(range(1, active.floors+1)),
                                     colorscale=[[0, "#F2F6FA"], [1, "#167D8D"]], colorbar=dict(title="人")))
        heatmap.update_layout(title="本次实验起终点分布", height=320, margin=dict(l=45, r=10, t=55, b=35),
                              xaxis_title="目的楼层", yaxis_title="起始楼层", paper_bgcolor="rgba(0,0,0,0)")
        right.plotly_chart(heatmap, width="stretch")
        right.caption("此图为实验结束后的输入分析。调度器在乘客上车前只读取呼梯楼层和方向。")
        a, b, c = st.columns(3)
        # ZIP 与本次实验生命周期绑定；配置或策略重算成功后清除，避免仅按请求指纹误用旧结果。
        if "experiment_zip" not in st.session_state:
            st.session_state.experiment_zip = export_experiment(experiment)
        a.download_button("下载完整实验包 ZIP", st.session_state.experiment_zip, "elevator_experiment.zip", "application/zip", width="stretch")
        b.download_button("下载共享请求 CSV", csv_bytes([asdict(req) for req in experiment.requests], ["id", "arrival", "origin", "destination"]),
                          "requests.csv", "text/csv", width="stretch")
        c.download_button("下载指标 CSV", csv_bytes(metrics_rows()), "metrics.csv", "text/csv", width="stretch")

    # 【楼宇回放】读取保存的快照；播放速度只影响展示，不改变仿真结果。
    with tabs[1]:
        playback_timer_active = st.session_state.get("replay_playing", False)
        @st.fragment(run_every=0.7 if playback_timer_active else None)
        def replay():
            selected = st.selectbox("回放策略", list(experiment.results), format_func=POLICIES.get,
                                    index=list(experiment.results).index(focus), key="replay_policy")
            frames = experiment.results[selected].snapshots
            times = [frame["time"] for frame in frames]
            st.session_state.setdefault("replay_playing", False)
            st.session_state.setdefault("replay_time", times[0])
            controls = st.columns([1, 1, 1, 2])
            def toggle():
                st.session_state.replay_playing = not st.session_state.replay_playing
            def rewind():
                st.session_state.replay_playing = False
                st.session_state.replay_time = times[0]
            def step():
                st.session_state.replay_playing = False
                # 二分查找下一帧，单步推进与不等间隔快照也能正确配合。
                index = min(len(times)-1, bisect.bisect_right(times, st.session_state.replay_time))
                st.session_state.replay_time = times[index]
            controls[0].button("暂停" if st.session_state.replay_playing else "播放", on_click=toggle, width="stretch")
            controls[1].button("回到起点", on_click=rewind, width="stretch")
            controls[2].button("下一步", on_click=step, width="stretch")
            speed = controls[3].select_slider("每次推进（仿真秒）", [2, 10, 30, 60], value=10)
            if st.session_state.replay_playing:
                next_index = min(len(times)-1, bisect.bisect_left(times, st.session_state.replay_time + speed))
                st.session_state.replay_time = times[next_index]
                if next_index == len(times)-1:
                    st.session_state.replay_playing = False
            def pause():
                st.session_state.replay_playing = False
            st.select_slider("仿真时间（秒）", options=times, key="replay_time", on_change=pause)
            snapshot = frames[bisect.bisect_left(times, st.session_state.replay_time)]
            left, right = st.columns([2.1, 1])
            with left:
                st.image(building_svg(snapshot, active), width="stretch", alt="楼层与电梯运行状态")
            with right:
                st.markdown(f'#### {snapshot["time"] / 60:.1f} / {active.duration / 60:.0f} 分钟')
                st.metric("楼层候梯", f'{snapshot["waiting"]} 人')
                st.metric("乘梯中 / 已送达", f'{snapshot["riding"]} / {snapshot["delivered"]} 人')
                cars = [{"电梯": f'E{car["id"]+1}', "状态": STATES[car["state"]],
                         "楼层": round(car["position"], 1), "载客": car["load"], "待接": car["assigned"]}
                        for car in snapshot["elevators"]]
                st.dataframe(pd.DataFrame(cars), hide_index=True, width="stretch")
                st.caption("橙色为满载，青色为运行或门控，灰色为空闲。车厢内显示方向与载客量。")
            # 播放/暂停/到达终点时重建 fragment，暂停期间停止后台周期刷新。
            if st.session_state.replay_playing != playback_timer_active:
                st.rerun()
        replay()

    # 【追踪记录】按乘客时间戳及调度日志解释结果，完整数据可从 ZIP 获取。
    with tabs[2]:
        selected = st.selectbox("查看策略", list(experiment.results), format_func=POLICIES.get, key="detail_policy")
        result = experiment.results[selected]
        status = st.radio("乘客状态", ["全部", "等待中", "乘梯中", "已送达"], horizontal=True)
        rows = [{**row, "elevator_id": "待分配" if row["elevator_id"] is None else f'E{row["elevator_id"]+1}'}
                for row in result.passengers if status == "全部" or row["status"] == status]
        names = {"id": "乘客", "arrival": "到达秒", "origin": "起点", "destination": "终点", "elevator_id": "电梯",
                 "assigned_at": "分配秒", "boarded_at": "上车秒", "delivered_at": "送达秒", "status": "状态",
                 "wait_seconds": "等待秒", "observed_wait_seconds": "已观察等待秒", "ride_seconds": "乘梯秒", "journey_seconds": "全程秒"}
        st.dataframe(pd.DataFrame(rows[:1000]).rename(columns=names).round(2), hide_index=True, width="stretch")
        st.caption(f"符合条件 {len(rows)} 人，页面最多显示 1000 人；实验包包含完整记录。")
        st.markdown("#### 调度与服务日志")
        kinds = st.multiselect("日志类型", ["到达", "分配", "上车", "送达", "预定位"], default=["分配", "预定位"])
        events = [{**row, "elevator": "—" if row["elevator"] is None else f'E{row["elevator"]+1}'}
                  for row in result.events if row["event"] in kinds]
        st.dataframe(pd.DataFrame(events[-500:]).rename(columns={"time": "时刻(秒)", "event": "事件", "elevator": "电梯",
                                                               "passenger": "乘客", "detail": "说明"}), hide_index=True, width="stretch")
        st.caption("页面显示最近 500 条筛选日志，完整日志在实验包内。ETA 是考虑方向、停站与任务量的估计值。")

    # 【批量验证】每个种子内共享请求，不同种子之间改变客流以观察稳定性。
    with tabs[3]:
        st.markdown("#### 多随机种子与模块消融")
        st.write("每个种子生成一批请求，该批请求由全部选定策略共享。保留各次原始结果，观察收益是否稳定。")
        with st.form("batch"):
            count = st.slider("重复次数", 2, 30, 5)
            batch_policies = st.multiselect("批量策略", list(POLICIES), default=list(POLICIES), format_func=POLICIES.get)
            batch_clicked = st.form_submit_button("运行批量验证")
        if batch_clicked:
            if not batch_policies:
                st.error("请至少选择一种策略。")
            elif experiment.input_source != "合成客流":
                st.info("批量验证针对合成客流。请先在侧栏移除 CSV 并运行一次合成实验；导入文件的重复回放是确定性的。")
            else:
                progress = st.progress(0.0, text=f"已完成 0 / {count} 个随机种子")
                def show_progress(completed, total, completed_seed):
                    progress.progress(completed / total, text=f"已完成 {completed} / {total} 个随机种子（种子 {completed_seed}）")
                try:
                    with st.spinner(f"正在比较 {count} 个随机种子 × {len(batch_policies)} 种策略…"):
                        report = benchmark(active, range(active.seed, active.seed + count), batch_policies,
                                           on_progress=show_progress)
                    st.session_state.batch_result = report
                except ValueError as exc:
                    st.error(f"本次批量验证失败：{exc}。下方如有结果，仍为上一次成功的批量实验。")
                finally:
                    progress.empty()
        if "batch_result" in st.session_state:
            report = st.session_state.batch_result
            table = []
            for policy, summary in report["summary"].items():
                wait = summary["avg_wait"]
                table.append({"策略": POLICIES[policy], "平均等待（秒）": wait["mean"], "等待标准差": wait["std"],
                              "平均完成率（%）": summary["completion_rate"]["mean"],
                              "平均未上车人数": summary["waiting"]["mean"], "平均空载距离（层）": summary["empty_distance"]["mean"]})
            st.dataframe(pd.DataFrame(table).round(2), hide_index=True, width="stretch")
            paired = report["paired_wait_comparison"]
            if paired:
                st.write(f'就近相对自适应的已上车平均等待差：{paired["mean_wait_reduction_seconds"]:+.2f} 秒；正数表示自适应等待更短。')
                interval = paired["normal_approx_95_ci"]
                if interval:
                    st.caption(f"配对差值的正态近似 95% 区间：[{interval[0]:.2f}, {interval[1]:.2f}] 秒。小样本仅作探索，且需同时比较上车率和完成率。")
            st.download_button("下载批量验证 JSON", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"),
                               "benchmark.json", "application/json")

    with tabs[4]:
        st.markdown("#### 模型如何工作")
        st.markdown("""
1. **乘客请求**：按场景生成泊松到达，或导入 CSV；每人有独立起终点与时间戳。
2. **共同物理规则**：每层固定运行时间、开关门耗时、逐人上下车、先下后上、容量约束。
3. **共同信息边界**：调度器只观察已经发生的楼层与方向呼梯；目的地在上车后进入车厢停靠计划。
4. **共同 LOOK 行驶**：优先沿当前方向服务前方目标，没有前方目标时反向。空车对等待过久的已分配请求优先接客。
5. **楼层关联**：低区教学 / 高区办公的人工关联先验，与已送达乘客的 OD 频次融合；加分有上限。
6. **短时预测**：过去两个60秒窗口的楼层到达率做平滑与阻尼趋势外推，预测未来60秒需求，只给空闲电梯做预定位。
7. **公平评估**：所有策略回放相同请求；结束时等待和乘梯中的人保留，完整参数、指纹和日志可以导出。
""")
        st.info("这是软件研究与演示模型。客流场景为合成假设；预测和楼层关联采用可解释启发式方法。改进策略可能在部分场景退化，页面按实际结果展示。")
        st.markdown("#### 指标与边界")
        for name, description in experiment.summary()["methodology"].items():
            labels = {"information": "信息边界", "avg_wait": "平均等待", "observed_wait": "全体已观察等待", "avg_queue": "平均队长",
                      "distance": "运行距离", "forecast": "预测", "slp": "楼层关联", "boundary": "截止规则"}
            st.markdown(f"**{labels[name]}**：{description}")
        st.caption("当前未模拟加速度、人员重量与群组、真实电机能耗、故障及紧急控制，也未连接硬件。")
        st.markdown("#### 开源基础")
        st.markdown("仿真引擎：[SimPy / MIT](https://simpy.readthedocs.io/)。界面：[Streamlit / Apache 2.0](https://github.com/streamlit/streamlit)。"
                    "图表：[Plotly.py / MIT](https://github.com/plotly/plotly.py)。更多来源、许可和改造边界见项目中的 `OPEN_SOURCE_NOTICES.md`。")
        st.json(asdict(active), expanded=False)


def batch_main():
    """解析命令行参数，运行多种子对比并把 JSON/CSV 保存到指定目录。"""
    parser = argparse.ArgumentParser(description="电梯调度多种子公平对比")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="up_peak")
    parser.add_argument("--output", default=str(ROOT / "reports"))
    args = parser.parse_args()
    if not 1 <= args.seeds <= 50:
        parser.error("--seeds 必须为 1~50")
    try:
        cfg = SimulationConfig(duration=args.duration, peak_center=args.duration / 3,
                               seed=args.seed, scenario=args.scenario)
        report = benchmark(cfg, range(args.seed, args.seed + args.seeds), POLICIES)
    except ValueError as exc:
        parser.error(str(exc))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    rows = [{"seed": run["seed"], "request_sha256": run["request_sha256"], "policy": policy,
             "name": POLICIES[policy], **metrics} for run in report["runs"] for policy, metrics in run["results"].items()]
    (output / "benchmark.csv").write_bytes(csv_bytes(rows))
    print(json.dumps({"saved_to": str(output), "seeds": args.seeds, "summary": report["summary"]}, ensure_ascii=False, indent=2))


def launch():
    """使用当前 Python 环境启动本地 Streamlit，自动寻找可用端口。"""
    for port in range(8501, 8521):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "eos.py"),
                                "--server.address", "127.0.0.1", "--server.port", str(port)], cwd=ROOT)
    raise RuntimeError("8501~8520 端口均已占用，请关闭旧的仿真窗口后重试。")


if __name__ == "__main__":
    # 三种入口：--batch 批量实验；python eos.py 自动启动；streamlit run 渲染页面。
    if "--batch" in sys.argv:
        batch_main()
    else:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx(suppress_warning=True) is None:
            raise SystemExit(launch())
        main()

# 开源来源与复用说明

## 实际使用的组件

| 组件 | 本次验收版本 | 许可 | 来源与用途 |
|---|---|---|---|
| SimPy | 4.1.2 | MIT | https://simpy.readthedocs.io/ ，通过 Environment、事件、生成器进程与 timeout 实现仿真时钟、到达、门控和运行过程 |
| Streamlit | 1.67.0 | Apache 2.0 | https://github.com/streamlit/streamlit ，本地参数界面、数据表、下载及回放控制 |
| Plotly.py | 6.9.0 | MIT | https://github.com/plotly/plotly.py ，交互式时序图与起终点热力图 |
| pandas | 2.3.3 | BSD 3-Clause | https://github.com/pandas-dev/pandas ，界面数据表 |

完整直接及间接依赖版本见 `requirements.lock.txt`。各依赖包附带的许可证、版权与通知保留在 `.venv/Lib/site-packages` 的包目录或 dist-info 目录中；重新分发环境或依赖时须保留相应通知。

## 检索和选型

改造前比较了 Python 仿真框架和现成电梯项目：

- SimPy：Python 离散事件仿真框架，与原有技术栈一致，因此用于实际仿真内核。
- elevator-core：https://github.com/andymai/elevator-core ，Rust 电梯内核，README 声明 MIT 或 Apache 2.0 许可；乘客生命周期、策略接口和确定性回放具有参考价值，接入 Rust 会增加本次 Python 项目的构建成本。
- Elevator Saga：https://github.com/magwo/elevatorsaga ，MIT，适合浏览器可视化研究，仓库声明已不再积极维护。

业务层的乘客模型、LOOK 路径、ETA 近似评分、SLP 分区先验、因果预测与实验导出在本文件夹实现；SimPy 提供进程和事件运行时。SLP 和预测均为可解释的启发式方法，不应描述为已经训练或已经在真实楼宇验证的模型。
